// One native process owns scene projection, ordered Tile lists and the SDK.
// Frozen render helpers supply the established FLK1 protocol, not a new bitstream.
#include <limits>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#define HETEROGS_RENDER_LIBRARY
#include "../initialize/render_resident.cpp"
#undef HETEROGS_RENDER_LIBRARY
namespace projection {
#define main unused_projection_main
#include "attributes.cpp"
#undef main
}
#include "cached_projection.hpp"

using DeviceBuffer=decltype(std::declval<icraft::xrt::ZG330Device>().defaultMemRegion().malloc(64,0,64));
struct FrameStats {
    double project_ms=0,sort_ms=0,engine_ms=0,pack_ms=0,upload_ms=0,wait_ms=0,read_ms=0;
    uint64_t input_bytes=0,output_bytes=0,cycles=0;
};

struct EngineWorkspace {
    DeviceBuffer memory;
    size_t capacity=0;
    std::vector<Word> payload[2],received,packed;
    std::vector<Pixel> pixels;
    std::vector<uint32_t> tags;
    uint32_t sequence=0;

    bool compact_payload=false;
    void execute(const Scene& s,unsigned batch,FrameStats& stats) {
        Board board;
        if(compact_payload) {
            // One DMA record per visible Gaussian; reuse it across Tile lists.
            packed.resize(s.gs.size());
            for(size_t i=0;i<s.gs.size();++i)std::memcpy(packed[i].data(),&s.gs[i].x,36);
        }
        size_t maxwords=0;
        for(unsigned b=0;b<s.tiles;b+=batch) {
            size_t n=std::min(batch,s.tiles-b);
            for(unsigned t=b;t<std::min(b+batch,s.tiles);++t)n+=s.ranges[t][1]-s.ranges[t][0];
            maxwords=std::max(maxwords,n);
        }
        const size_t inbytes=maxwords*64,outbytes=batch*4096,stride=inbytes+outbytes;
        if(capacity<stride*2) {
            // Board() confirmed idle; old allocations are never reused in flight.
            memory=board.dev.defaultMemRegion().malloc(stride*2,0,64);
            capacity=stride*2;
        }
        const uint64_t base=memory->begin.addr();
        if(base%64||base+stride*2>0xffffffffULL)throw std::runtime_error("DDR address range");
        for(auto& v:payload)v.reserve(maxwords);
        received.resize(batch*64);pixels.resize(size_t(s.w)*s.h);
        const unsigned jobs=(s.tiles+batch-1)/batch;
        tags.resize(jobs);
        auto launch=[&](unsigned j) {
            unsigned slot=j&1,b=j*batch,e=std::min(b+batch,s.tiles);
            auto& v=payload[slot];
            size_t position=0;
            if(compact_payload) {
                size_t words=e-b;
                for(unsigned t=b;t<e;++t)words+=s.ranges[t][1]-s.ranges[t][0];
                v.resize(words);
            }else v.clear();
            for(unsigned t=b;t<e;++t) {
                auto range=s.ranges[t];Word h{};
                unsigned x=(t%((s.w+15)/16))*16,y=(t/((s.w+15)/16))*16;
                put32(h,0,range[1]-range[0]);put16(h,4,x);put16(h,6,y);
                put16(h,8,std::min(16u,s.w-x)|(std::min(16u,s.h-y)<<5));
                for(int k=0;k<3;++k)put16(h,10+2*k,tohalf(s.bg[k]));
                if(compact_payload)v[position++]=h;else v.push_back(h);
                for(unsigned k=range[0];k<range[1];++k) {
                    if(compact_payload)v[position++]=packed[s.ids[k]];
                    else {Word q{};std::memcpy(q.data(),&s.gs[s.ids[k]].x,36);v.push_back(q);}
                }
            }
            auto begin=Clock::now();
            memory.write(slot*stride,reinterpret_cast<char*>(v.data()),v.size()*64);
            stats.upload_ms+=elapsed(begin,Clock::now())/1000;
            stats.input_bytes+=v.size()*64;
            if(++sequence==0)++sequence;
            tags[j]=sequence;
            board.submit(base+slot*stride,v.size(),base+slot*stride+inbytes,e-b,sequence);
        };
        launch(0);
        for(unsigned j=0;j<jobs;++j) {
            if(j+1<jobs)launch(j+1);
            auto begin=Clock::now();board.wait_done();
            stats.wait_ms+=elapsed(begin,Clock::now())/1000;
            unsigned b=j*batch,tiles=std::min(batch,s.tiles-b);
            if(board.read(0x30)!=tags[j]||board.read(0x48)||
               board.read(0x38)!=payload[j&1].size()||board.read(0x3c)!=tiles*64)
                throw std::runtime_error("DMA completion/counts");
            stats.cycles+=board.read(0x34);
            board.write(0x2c,2);
            begin=Clock::now();
            memory.read(reinterpret_cast<char*>(received.data()),(j&1)*stride+inbytes,tiles*4096);
            stats.read_ms+=elapsed(begin,Clock::now())/1000;
            stats.output_bytes+=tiles*4096;
            for(unsigned t=0;t<tiles;++t)for(unsigned p=0;p<256;++p) {
                const uint8_t* raw=received[t*64+p/4].data()+(p%4)*16;
                uint16_t h[4];uint32_t last,tag;
                std::memcpy(h,raw,8);std::memcpy(&last,raw+8,4);std::memcpy(&tag,raw+12,4);
                if(tag!=t*256+p)throw std::runtime_error("DMA pixel tag");
                unsigned tile=b+t,x=(tile%((s.w+15)/16))*16+p%16,y=(tile/((s.w+15)/16))*16+p/16;
                if(x<s.w&&y<s.h) {
                    Pixel q{fromhalf(h[0]),fromhalf(h[1]),fromhalf(h[2]),fromhalf(h[3]),last};
                    if(!std::isfinite(q.r)||!std::isfinite(q.g)||!std::isfinite(q.b)||!std::isfinite(q.t))
                        throw std::runtime_error("nonfinite hardware pixel");
                    pixels[y*s.w+x]=q;
                }
            }
        }
    }
};

struct LiveScene {
    std::vector<float> rows;
    std::vector<PreparedGaussian> prepared;
    std::vector<uint32_t> selected,remap,cursors;
    std::vector<projection::Attribute> attrs;
    std::vector<uint64_t> keys,scratch;
    Scene scene{};
    unsigned count=0,budget=0,threads=1;
    bool cached=true,comparison_sort=false,uniform_preview=false,depth_layout=false;

    void install(std::vector<float> next,unsigned n) {
        std::vector<PreparedGaussian> properties(n);
        for(unsigned i=0;i<n;++i) {
            for(unsigned k=0;k<62;++k)
                if(!std::isfinite(next[size_t(i)*62+k]))throw std::runtime_error("nonfinite model");
            properties[i]=prepare_gaussian(next.data()+size_t(i)*62);
        }
        std::vector<uint32_t> ids(n);std::iota(ids.begin(),ids.end(),0);
        if(budget&&budget<n&&uniform_preview) {
            ids.resize(budget);
            const float area=float(n)/float(budget);
            for(unsigned i=0;i<budget;++i) {
                ids[i]=uint64_t(i)*n/budget;
                // Explicit preview approximation: compensate reduced sampling
                // density by a broader footprint, keeping per-Gaussian opacity.
                for(auto& row:properties[ids[i]].world)for(float& v:row)v*=area;
            }
        }else if(budget&&budget<n) {
            // Explicit lossy profile: rank scene-space opacity-weighted surface area.
            auto better=[&](unsigned a,unsigned b) {
                if(properties[a].importance!=properties[b].importance)
                    return properties[a].importance>properties[b].importance;
                return a<b;
            };
            std::nth_element(ids.begin(),ids.begin()+budget,ids.end(),better);
            ids.resize(budget);std::sort(ids.begin(),ids.end());
        }
        rows.swap(next);prepared.swap(properties);selected.swap(ids);count=n;
        attrs.resize(n);remap.resize(n);keys.reserve(selected.size());
    }

    void project_and_sort(const projection::Camera& c,FrameStats& stats) {
        auto begin=Clock::now();
        int failed=0;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:failed)
        for(size_t j=0;j<selected.size();++j) {
            auto i=selected[j];attrs[i]={};
            try {
                if(cached)project_prepared(rows.data()+size_t(i)*62,prepared[i],c,attrs[i]);
                else projection::project(rows.data()+size_t(i)*62,c,attrs[i]);
            }catch(...){failed=1;}
        }
        if(failed)throw std::runtime_error("projection failed");
        stats.project_ms=elapsed(begin,Clock::now())/1000;
        begin=Clock::now();
        auto& s=scene;s.w=c.w;s.h=c.h;s.tile=16;s.n=count;s.bg={0,0,0};
        const unsigned nx=(c.w+15)/16,ny=(c.h+15)/16;s.tiles=nx*ny;
        s.gs.clear();s.gs.reserve(selected.size());keys.clear();
        s.ranges.assign(s.tiles,{0,0});cursors.assign(s.tiles,0);
        for(unsigned i:selected) {
            const auto& a=attrs[i];if(!a.valid)continue;
            if(!depth_layout) {
                remap[i]=s.gs.size();Gaussian g{};g.id=i;
                std::memcpy(&g.x,a.q,sizeof(a.q));s.gs.push_back(g);
            }
            uint32_t depth;std::memcpy(&depth,&a.q[9],4);
            keys.push_back((uint64_t(depth)<<32)|i);
            for(unsigned y=a.min_y;y<a.max_y;++y)for(unsigned x=a.min_x;x<a.max_x;++x)++cursors[y*nx+x];
        }
        if(comparison_sort)std::sort(keys.begin(),keys.end());
        else {
            // Positive FP32 depth preserves bit order. Four stable byte passes
            // preserve original Gaussian ID ties because input IDs are ascending.
            scratch.resize(keys.size());
            for(unsigned pass=0;pass<4;++pass) {
                std::array<uint32_t,256> counts{};
                const unsigned shift=32+8*pass;
                for(uint64_t key:keys)++counts[(key>>shift)&255];
                uint32_t offset=0;
                for(auto& count:counts){uint32_t n=count;count=offset;offset+=n;}
                for(uint64_t key:keys)scratch[counts[(key>>shift)&255]++]=key;
                keys.swap(scratch);
            }
        }
        uint64_t total=0;
        for(unsigned t=0;t<s.tiles;++t) {
            const unsigned size=cursors[t];
            if(size)s.ranges[t]={uint32_t(total),uint32_t(total+size)};
            cursors[t]=total;total+=size;
            if(total>50000000)throw std::runtime_error("Tile list capacity");
        }
        s.ids.resize(total);
        for(uint64_t key:keys) {
            const uint32_t i=uint32_t(key);const auto& a=attrs[i];
            uint32_t mapped=remap[i];
            if(depth_layout) {
                mapped=s.gs.size();Gaussian g{};g.id=i;
                std::memcpy(&g.x,a.q,sizeof(a.q));s.gs.push_back(g);
            }
            for(unsigned y=a.min_y;y<a.max_y;++y)for(unsigned x=a.min_x;x<a.max_x;++x)
                s.ids[cursors[y*nx+x]++]=mapped;
        }
        s.active=s.gs.size();s.entries=s.ids.size();
        stats.sort_ms=elapsed(begin,Clock::now())/1000;
    }
};

int main(int argc,char** argv) {
    std::ios::sync_with_stdio(false);
    try {
        LiveScene model;unsigned batch=8;bool cpu=false,compact_payload=false;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--uncached")model.cached=false;
            else if(a=="--comparison-sort")model.comparison_sort=true;
            else if(a=="--uniform-preview")model.uniform_preview=true;
            else if(a=="--depth-layout")model.depth_layout=true;
            else if(a=="--compact-payload")compact_payload=true;
            else if(a=="--cpu")cpu=true;
            else if((a=="--threads"||a=="--max-gaussians"||a=="--batch")&&i+1<argc) {
                unsigned v=std::stoul(argv[++i]);
                if(a=="--threads")model.threads=v;else if(a=="--batch")batch=v;else model.budget=v;
            }else throw std::runtime_error("unsupported live renderer option");
        }
        if(!model.threads||model.threads>4||!batch||batch>32||model.budget>1000000)
            throw std::runtime_error("live renderer bounds");
        if(model.uniform_preview&&(!model.cached||!model.budget))
            throw std::runtime_error("uniform preview requires cached projection and an explicit budget");
        omp_set_dynamic(0);selected_mode=2;Board::resident=true;
        if(!cpu){Board board;}
        const int camera_fd=memfd_create("gs-live-camera",MFD_CLOEXEC);
        if(camera_fd<0)throw std::runtime_error("camera memfd");
        const std::string camera_path="/proc/self/fd/"+std::to_string(camera_fd);
        {
            EngineWorkspace engine;engine.compact_payload=compact_payload;std::vector<uint8_t> rgb;
            std::cout<<"LIVE_READY 1"<<std::endl;
            std::string line;
            while(std::getline(std::cin,line)) {
                std::istringstream cmd(line);std::string action,path,extra;
                cmd>>action;if(action=="QUIT")break;
                if(action=="LOAD"||action=="LOADROWS") {
                    auto begin=Clock::now();unsigned n=0;std::vector<float> rows;
                    if(action=="LOAD") {
                        if(!(cmd>>std::quoted(path))||(cmd>>extra))throw std::runtime_error("LOAD path");
                        rows=projection::load_model(path.c_str(),n);
                    }else {
                        if(!(cmd>>n)||(cmd>>extra)||!n||n>1000000)throw std::runtime_error("LOADROWS count");
                        rows.resize(size_t(n)*62);
                        if(!std::cin.read(reinterpret_cast<char*>(rows.data()),rows.size()*4))
                            throw std::runtime_error("LOADROWS length");
                    }
                    model.install(std::move(rows),n);
                    std::cout<<"SCENE "<<n<<" "<<model.selected.size()<<" "<<elapsed(begin,Clock::now())/1000<<std::endl;
                }else if(action=="RENDER"||action=="RENDER_RAW") {
                    if(cmd>>extra||!model.count)throw std::runtime_error("RENDER needs scene");
                    std::array<char,136> bytes;
                    if(!std::cin.read(bytes.data(),bytes.size()))throw std::runtime_error("camera length");
                    if(lseek(camera_fd,0,SEEK_SET)!=0||write(camera_fd,bytes.data(),bytes.size())!=ssize_t(bytes.size()))
                        throw std::runtime_error("camera write");
                    auto begin=Clock::now();auto camera=projection::load_camera(camera_path.c_str());
                    FrameStats stats;model.project_and_sort(camera,stats);
                    auto started=Clock::now();
                    if(cpu)engine.pixels=render(model.scene,Mode::dense,model.threads,false).pixels;
                    else engine.execute(model.scene,batch,stats);
                    stats.engine_ms=elapsed(started,Clock::now())/1000;started=Clock::now();
                    rgb.resize(engine.pixels.size()*3);
                    for(size_t i=0;i<engine.pixels.size();++i) {
                        const float colors[3]={engine.pixels[i].r,engine.pixels[i].g,engine.pixels[i].b};
                        for(int k=0;k<3;++k) {
                            if(!std::isfinite(colors[k]))throw std::runtime_error("nonfinite pixel");
                            rgb[3*i+k]=uint8_t(std::nearbyint(std::clamp(double(colors[k]),0.,1.)*255.));
                        }
                    }
                    stats.pack_ms=elapsed(started,Clock::now())/1000;
                    const size_t rawbytes=action=="RENDER_RAW"?16+engine.pixels.size()*sizeof(Pixel):0;
                    rusage usage{};getrusage(RUSAGE_SELF,&usage);
                    std::cout<<std::setprecision(12)<<"FRAME {\"width\":"<<camera.w<<",\"height\":"<<camera.h
                        <<",\"rgb_bytes\":"<<rgb.size()<<",\"raw_bytes\":"<<rawbytes
                        <<",\"project_ms\":"<<stats.project_ms<<",\"sort_ms\":"<<stats.sort_ms
                        <<",\"engine_ms\":"<<stats.engine_ms<<",\"pack_ms\":"<<stats.pack_ms
                        <<",\"native_ms\":"<<elapsed(begin,Clock::now())/1000
                        <<",\"upload_ms\":"<<stats.upload_ms<<",\"wait_ms\":"<<stats.wait_ms
                        <<",\"read_ms\":"<<stats.read_ms<<",\"input_bytes\":"<<stats.input_bytes
                        <<",\"output_bytes\":"<<stats.output_bytes<<",\"hardware_cycles\":"<<stats.cycles
                        <<",\"active\":"<<model.scene.active<<",\"tile_entries\":"<<model.scene.entries
                        <<",\"selected\":"<<model.selected.size()<<",\"rss_kib\":"<<usage.ru_maxrss<<"}\n";
                    std::cout.write(reinterpret_cast<char*>(rgb.data()),rgb.size());
                    if(rawbytes) {
                        std::cout.write("GSSOUT01",8);std::cout.write(reinterpret_cast<char*>(&camera.w),4);
                        std::cout.write(reinterpret_cast<char*>(&camera.h),4);
                        std::cout.write(reinterpret_cast<char*>(engine.pixels.data()),engine.pixels.size()*sizeof(Pixel));
                    }
                    std::cout.flush();if(!std::cout)throw std::runtime_error("frame delivery");
                }else throw std::runtime_error("unknown live command");
            }
        }
        close(camera_fd);
        if(!cpu){icraft::xrt::Device::Close(Board::retained_device);close(Board::retained_fd);}
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
