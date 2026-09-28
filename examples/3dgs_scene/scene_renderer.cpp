// Prepared complete-frame rasterization; reference images are never read.
#define GSC_REPLAY_LIBRARY
#include "replay.cpp"
#ifndef CPU_ONLY
#include "batch.hpp"
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <omp.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sys/resource.h>

namespace scene {
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
double process_us(const rusage&r){return 1e6*(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+r.ru_utime.tv_usec+r.ru_stime.tv_usec;}
template<class T>void take(std::istream&f,T*p,size_t n){
    if(!f.read(reinterpret_cast<char*>(p),n*sizeof(T)))throw std::runtime_error("truncated input");
}
struct Gaussian{uint32_t id;float x,y,a,b,c,opacity,r,g,blue,depth;};
struct Pixel{float r=0,g=0,b=0,t=1;uint32_t last=0;};
static_assert(sizeof(Gaussian)==44&&sizeof(Pixel)==20,"unsupported binary layout");
struct Input{
    uint32_t w,h,tile,n,active,entries,tiles;std::array<float,3>bg;
    std::vector<Gaussian>gs;std::vector<std::array<uint32_t,2>>ranges;std::vector<uint32_t>ids;
};
Input load(const char*path){
    std::ifstream f(path,std::ios::binary);char magic[8];take(f,magic,8);
    if(std::memcmp(magic,"GSSCN001",8))throw std::runtime_error("input ABI mismatch");
    Input s;uint32_t h[7];take(f,h,7);
    s.w=h[0];s.h=h[1];s.tile=h[2];s.n=h[3];s.active=h[4];s.entries=h[5];s.tiles=h[6];
    take(f,s.bg.data(),3);
    if(s.w<1||s.h<1||s.w>2048||s.h>2048||s.tile!=16||s.n>1000000||s.active>s.n||
       s.entries>50000000||s.tiles!=((s.w+15)/16)*((s.h+15)/16))throw std::runtime_error("input limits");
    s.gs.resize(s.active);s.ranges.resize(s.tiles);s.ids.resize(s.entries);
    take(f,s.gs.data(),s.gs.size());take(f,s.ranges.data(),s.ranges.size());take(f,s.ids.data(),s.ids.size());
    if(f.peek()!=EOF)throw std::runtime_error("trailing input");
    for(auto b:s.bg)if(!std::isfinite(b)||b<0||b>1)throw std::runtime_error("background");
    for(size_t i=0;i<s.gs.size();++i){
        const auto&q=s.gs[i];const float values[]={q.x,q.y,q.a,q.b,q.c,q.opacity,q.r,q.g,q.blue,q.depth};
        for(float x:values)if(!std::isfinite(x))throw std::runtime_error("nonfinite Gaussian");
        if(q.a<=0||q.c<=0||q.depth<=0||q.opacity<0||q.opacity>1||q.r<0||q.g<0||q.blue<0||
           q.r>16||q.g>16||q.blue>16||(i&&q.id<=s.gs[i-1].id))throw std::runtime_error("Gaussian contract");
    }
    uint64_t total=0;
    for(auto range:s.ranges){
        if(range[1]<range[0]||range[1]>s.entries)throw std::runtime_error("range bounds");
        float previous=0;total+=range[1]-range[0];
        for(uint32_t j=range[0];j<range[1];++j){
            if(s.ids[j]>=s.active||s.gs[s.ids[j]].depth<previous)throw std::runtime_error("depth order");
            previous=s.gs[s.ids[j]].depth;
        }
    }
    if(total!=s.entries)throw std::runtime_error("range coverage");
    return s;
}
uint32_t quantize(float x,uint32_t scale){return uint32_t(std::floor(double(x)*scale+.5));}
struct Counters{
    uint64_t pairs=0,power_skip=0,alpha_skip=0,qualified=0,contributions=0,early=0;
    uint64_t commands=0,reads=0,writes=0,batches=0,core=0,execution=0;
    double load_us=0,wait_us=0,readback_us=0;
    std::vector<uint8_t>visited,accepted;
    explicit Counters(size_t n):visited(n),accepted(n){}
};
struct Frame{
    std::vector<Pixel>pixels;std::vector<std::array<uint32_t,6>>states;
    uint64_t visited=0,accepted=0;Counters counts;double prepare_us=0,render_us=0,total_us=0,cpu_us=0;
    explicit Frame(const Input&s):pixels(s.w*s.h),states(s.w*s.h),counts(0){}
};
#ifndef CPU_ONLY
struct Fpga{
    int lock;icraft::xrt::Device device;std::unique_ptr<Hardware>hw;std::unique_ptr<BatchHardware>batch;
    Fpga(){
        lock=::open("/run/lock/fpga43dgs-gsc1.lock",O_CREAT|O_RDWR|O_CLOEXEC,0600);
        if(lock<0||flock(lock,LOCK_EX|LOCK_NB))throw std::runtime_error("hardware in use");
        device=icraft::xrt::Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
        hw=std::make_unique<Hardware>(device.cast<icraft::xrt::ZG330Device>());
        batch=std::make_unique<BatchHardware>(*hw);
    }
    ~Fpga(){batch.reset();hw.reset();icraft::xrt::Device::Close(device);::close(lock);}
};
#else
struct Fpga{};
#endif

Frame render(const Input&s,const std::string&mode,int threads,Fpga*fpga){
    Frame frame(s);std::vector<Counters>counter;
    for(int i=0;i<threads;++i)counter.emplace_back(s.active);
    std::vector<std::array<uint32_t,3>> colors(s.active);
    std::vector<Cmd> commands;commands.reserve(256);std::vector<size_t>destinations;destinations.reserve(256);
    std::vector<State>results;results.reserve(256);
    rusage ru0{},ru1{};getrusage(RUSAGE_SELF,&ru0);auto begin=Clock::now();
    if(mode!="float")for(size_t i=0;i<s.gs.size();++i){
        const auto&q=s.gs[i];colors[i]={quantize(q.r,1u<<24),quantize(q.g,1u<<24),quantize(q.blue,1u<<24)};
    }
    auto prepared=Clock::now();
    Cmd background={2,0,quantize(s.bg[0],1u<<24),quantize(s.bg[1],1u<<24),quantize(s.bg[2],1u<<24),0};
    auto store=[&](size_t pixel,const State&state){
        for(int k=0;k<6;++k)frame.states[pixel][k]=state[k];
        frame.pixels[pixel]={float(state[0])/float(1u<<24),float(state[1])/float(1u<<24),
            float(state[2])/float(1u<<24),float(state[3])/float(ONE),state[4]};
    };
    auto flush=[&](){
        if(commands.empty())return;
#ifndef CPU_ONLY
        fpga->batch->run(commands,results);auto&b=*fpga->batch;auto&c=counter[0];
        if(results.size()!=destinations.size())throw std::runtime_error("batch output count");
        for(size_t i=0;i<results.size();++i){
            if(results[i][5]&2)throw std::runtime_error("FPGA arithmetic contract error");
            store(destinations[i],results[i]);
        }
        c.commands+=commands.size();c.reads+=b.reads;c.writes+=b.writes;c.batches+=b.batches;
        c.core+=b.core;c.execution+=b.execution;c.load_us+=b.load_us;c.wait_us+=b.wait_us;c.readback_us+=b.readback_us;
        commands.clear();destinations.clear();
#else
        throw std::runtime_error("FPGA unavailable in CPU-only binary");
#endif
    };
    auto append=[&](const Cmd&cmd,size_t pixel){
        commands.push_back(cmd);if(cmd[0]==2)destinations.push_back(pixel);
        if(commands.size()==256)flush();
    };
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic,1) if(threads>1)
    for(int tile=0;tile<int(s.tiles);++tile){
        auto&c=counter[omp_get_thread_num()];
        uint32_t x0=(tile%((s.w+15)/16))*16,y0=(tile/((s.w+15)/16))*16;
        auto range=s.ranges[tile];
        for(uint32_t y=y0;y<std::min(y0+16,s.h);++y)for(uint32_t x=x0;x<std::min(x0+16,s.w);++x){
            size_t pixel=y*s.w+x;Pixel p;Reference ref;uint32_t t=ONE;
            if(mode=="fpga")append({0,0,0,0,0,0},pixel);
            else if(mode=="fixed")ref.step({0,0,0,0,0,0});
            for(uint32_t j=range[0];j<range[1];++j){
                uint32_t index=s.ids[j],ordinal=j-range[0]+1;const auto&q=s.gs[index];
                ++c.pairs;c.visited[index]=1;
                float dx=q.x-float(x),dy=q.y-float(y);
                float power=-.5f*(q.a*dx*dx+q.c*dy*dy)-q.b*dx*dy;
                if(power>0){++c.power_skip;continue;}
                float alpha=std::min(.99f,q.opacity*std::exp(power));
                if(alpha<1.f/255.f){++c.alpha_skip;continue;}
                ++c.qualified;
                if(mode=="float"){
                    float next=p.t*(1.f-alpha);if(next<.0001f){++c.early;break;}
                    p.r+=q.r*alpha*p.t;p.g+=q.g*alpha*p.t;p.b+=q.blue*alpha*p.t;p.t=next;p.last=ordinal;
                }else{
                    auto rgb=colors[index];auto a=quantize(alpha,ONE);Cmd cmd={1,a,rgb[0],rgb[1],rgb[2],ordinal};
                    if(mode=="fpga"){
                        append(cmd,pixel);auto next=t-mul(t,std::min(a,1063004406u));
                        if(next<107375){++c.early;break;}t=next;
                    }else{
                        ref.step(cmd);if(ref.done){++c.early;break;}
                    }
                }
                ++c.contributions;c.accepted[index]=1;
            }
            if(mode=="float"){
                p.r+=p.t*s.bg[0];p.g+=p.t*s.bg[1];p.b+=p.t*s.bg[2];frame.pixels[pixel]=p;
            }else if(mode=="fpga")append(background,pixel);
            else store(pixel,ref.step(background));
        }
    }
    if(mode=="fpga")flush();
    auto end=Clock::now();getrusage(RUSAGE_SELF,&ru1);
    frame.prepare_us=us(begin,prepared);frame.render_us=us(prepared,end);frame.total_us=us(begin,end);
    frame.cpu_us=process_us(ru1)-process_us(ru0);
    for(size_t i=0;i<s.active;++i){
        bool v=false,a=false;for(auto&c:counter){v|=c.visited[i];a|=c.accepted[i];}
        frame.visited+=v;frame.accepted+=a;
    }
    for(auto&c:counter){auto&t=frame.counts;
        t.pairs+=c.pairs;t.power_skip+=c.power_skip;t.alpha_skip+=c.alpha_skip;t.qualified+=c.qualified;
        t.contributions+=c.contributions;t.early+=c.early;t.commands+=c.commands;t.reads+=c.reads;
        t.writes+=c.writes;t.batches+=c.batches;t.core+=c.core;t.execution+=c.execution;
        t.load_us+=c.load_us;t.wait_us+=c.wait_us;t.readback_us+=c.readback_us;
    }
    return frame;
}

int main(int argc,char**argv){
    if(argc!=7)throw std::runtime_error("scene_renderer scene.bin prefix float|fixed|fpga threads repeats warmup");
    auto input=load(argv[1]);std::string prefix=argv[2],mode=argv[3];
    int threads=std::stoi(argv[4]),repeats=std::stoi(argv[5]),warmup=std::stoi(argv[6]);
    if((mode!="float"&&mode!="fixed"&&mode!="fpga")||threads<1||threads>4||
       repeats<1||repeats>100||warmup<0||warmup>10||(mode=="fpga"&&threads!=1))throw std::runtime_error("invalid arguments");
    std::unique_ptr<Fpga>fpga;if(mode=="fpga")fpga=std::make_unique<Fpga>();
    std::ofstream timing(prefix+"_timing.csv");timing<<std::setprecision(12);
    timing<<"sample,prepare_us,render_us,total_us,process_cpu_us,visited_pairs,qualified,contributions,early_pixels,unique_visited,unique_contributors,commands,batches,register_reads,register_writes,core_cycles,execution_cycles,load_us,wait_us,readback_us\n";
    std::vector<Pixel>first;std::vector<std::array<uint32_t,6>>first_states;Frame result(input);
    for(int rep=-warmup;rep<repeats;++rep){
        result=render(input,mode,threads,fpga.get());auto&c=result.counts;
        if(rep==-warmup){first=result.pixels;first_states=result.states;}
        if(std::memcmp(first.data(),result.pixels.data(),first.size()*sizeof(Pixel))||first_states!=result.states)
            throw std::runtime_error("repeat output drift");
        if(rep>=0)timing<<rep<<','<<result.prepare_us<<','<<result.render_us<<','<<result.total_us<<','<<result.cpu_us<<','
            <<c.pairs<<','<<c.qualified<<','<<c.contributions<<','<<c.early<<','<<result.visited<<','<<result.accepted<<','
            <<c.commands<<','<<c.batches<<','<<c.reads<<','<<c.writes<<','<<c.core<<','<<c.execution<<','
            <<c.load_us<<','<<c.wait_us<<','<<c.readback_us<<'\n';
        timing.flush();std::cout<<"FRAME "<<rep<<" mode="<<mode<<" threads="<<threads<<" total_ms="<<result.total_us/1000<<std::endl;
    }
    std::ofstream out(prefix+".bin",std::ios::binary);out.write("GSSOUT01",8);
    out.write(reinterpret_cast<const char*>(&input.w),4);out.write(reinterpret_cast<const char*>(&input.h),4);
    out.write(reinterpret_cast<const char*>(result.pixels.data()),result.pixels.size()*sizeof(Pixel));
    if(mode!="float"){
        std::ofstream raw(prefix+"_states.u32",std::ios::binary);
        raw.write(reinterpret_cast<const char*>(result.states.data()),result.states.size()*24);
    }
    rusage ru{};getrusage(RUSAGE_SELF,&ru);std::ofstream meta(prefix+"_meta.json");
    meta<<"{\"mode\":\""<<mode<<"\",\"threads\":"<<threads<<",\"repeats\":"<<repeats<<",\"warmup\":"<<warmup
        <<",\"scene_gaussians\":"<<input.n<<",\"active\":"<<input.active<<",\"pixels\":"<<input.w*input.h
        <<",\"rss_kib\":"<<ru.ru_maxrss<<",\"output_stable\":true,\"npu\":false,\"dma\":false}\n";
    if(!out||!timing||!meta)throw std::runtime_error("output write failure");
    return 0;
}
}
int main(int argc,char**argv)try{return scene::main(argc,argv);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
