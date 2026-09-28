// The frozen CPU base remains callable from this binary for paired experiments.
#define main frozen_cpu_main
#include "cat_reference.cpp"
#undef main
#include <icraft-xrt/dev/zg330_device.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint16_t tohalf(float f){__fp16 h=f;uint16_t u;std::memcpy(&u,&h,2);return u;}
float fromhalf(uint16_t u){__fp16 h;std::memcpy(&h,&u,2);return float(h);}
using Word=std::array<uint8_t,64>;
void put32(Word&w,int b,uint32_t v){std::memcpy(w.data()+b,&v,4);}
void put16(Word&w,int b,uint16_t v){std::memcpy(w.data()+b,&v,2);}
struct Event{unsigned job,begin,tiles;double prepare0,prepare1,upload0,upload1,submit,complete,read0,read1;uint32_t cyc,reads,writes,overlap,starve,busy_before_upload,busy_after_upload;};
struct Board {
 int fd;icraft::xrt::Device device;icraft::xrt::ZG330Device dev;
 Board(bool require_idle=true){
  fd=open("/run/lock/fpga43dgs-gsc1.lock",O_CREAT|O_RDWR|O_CLOEXEC,0600);
  if(fd<0||flock(fd,LOCK_EX|LOCK_NB))throw std::runtime_error("Board in use");
  device=icraft::xrt::Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");dev=device.cast<icraft::xrt::ZG330Device>();
  if(read(0)!=0x464c4b30||read(4)!=0x10000)throw std::runtime_error("FLK0 ABI absent; no writes performed");
  if(require_idle&&read(8)!=0)throw std::runtime_error("Engine not clean/idle; inspect before retry");
 }
 uint32_t read(unsigned a){return dev.defaultRegRegion().read(0x400c0200+a,false);}
 void write(unsigned a,unsigned v){dev.defaultRegRegion().write(0x400c0200+a,v,false);}
 void wait_ack(){auto limit=Clock::now()+std::chrono::seconds(5);for(;;){auto s=read(8);if(s&8)throw std::runtime_error("Descriptor rejected");if(!(s&16))return;if(Clock::now()>limit)throw std::runtime_error("Request CDC timeout");}}
 void submit(uint32_t ib,uint32_t words,uint32_t ob,uint32_t tiles,uint32_t id){
  wait_ack();write(0x10,ib);write(0x14,words);write(0x18,ob);write(0x1c,tiles*64);write(0x20,tiles);write(0x24,id);write(0x28,0);write(0x2c,1);
 }
 void wait_done(){auto limit=Clock::now()+std::chrono::seconds(30);for(;;){auto s=read(8);if(s&8)throw std::runtime_error("Engine rejected request");if(s&4)return;if(Clock::now()>limit)throw std::runtime_error("Kernel timeout; reset the engine/board before any further allocation or retry");}}
 ~Board(){icraft::xrt::Device::Close(device);close(fd);}
};
int main(int argc,char**argv)try{
 if(argc>1&&std::string(argv[1])=="cpu")return frozen_cpu_main(argc-1,argv+1);
 if(argc==2&&std::string(argv[1])=="status"){
  Board b(false);for(unsigned a:{0u,4u,8u,0x30u,0x34u,0x38u,0x3cu,0x40u,0x44u,0x48u})std::cout<<std::hex<<"offset="<<a<<" value="<<b.read(a)<<'\n';return 0;
 }
 if(argc==5&&std::string(argv[1])=="raw"){
  std::ifstream f(argv[2],std::ios::binary|std::ios::ate);auto bytes=f.tellg();f.seekg(0);
  unsigned tiles=std::stoul(argv[3]);if(bytes<=0||uint64_t(bytes)%64||bytes>100000000||!tiles||tiles>256)throw std::runtime_error("Raw input bounds");
  std::vector<Word>in(size_t(bytes)/64),out(tiles*64);if(!f.read((char*)in.data(),bytes))throw std::runtime_error("Raw input read");
  size_t at=0;for(unsigned t=0;t<tiles;t++){if(at>=in.size())throw std::runtime_error("Raw header missing");uint32_t count;std::memcpy(&count,in[at].data(),4);at+=1+count;}if(at!=in.size())throw std::runtime_error("Raw record count");
  Board b;size_t output_offset=uint64_t(bytes)+64,total_bytes=output_offset+out.size()*64+64;
  auto mem=b.dev.defaultMemRegion().malloc(total_bytes,0,64);uint64_t addr=mem->begin.addr();
  if(addr%64||addr+total_bytes>0xffffffffULL)throw std::runtime_error("Raw DMA addresses");
  Word guard{};guard.fill(0xa5);mem.write(bytes,(char*)guard.data(),64);mem.write(total_bytes-64,(char*)guard.data(),64);
  mem.write(0,(char*)in.data(),bytes);b.submit(addr,in.size(),addr+output_offset,tiles,1);b.wait_done();
  if(b.read(0x30)!=1||b.read(0x38)!=in.size()||b.read(0x3c)!=out.size()||b.read(0x48))throw std::runtime_error("Raw accounting");
  std::cout<<"input_records="<<b.read(0x38)<<" output_records="<<b.read(0x3c)<<" hardware_cycles="<<b.read(0x34)<<" prefetch_overlap="<<b.read(0x40)<<'\n';
  mem.read((char*)out.data(),output_offset,out.size()*64);b.write(0x2c,2);
  Word before{},after{};mem.read((char*)before.data(),bytes,64);mem.read((char*)after.data(),total_bytes-64,64);
  if(before!=guard||after!=guard)throw std::runtime_error("Raw output guard overwritten");
  std::cout<<"output_guards=PASS\n";
  std::ofstream g(argv[4],std::ios::binary);g.write((char*)out.data(),out.size()*64);if(!g)throw std::runtime_error("Raw output write");return 0;
 }
 if(argc!=7)throw std::runtime_error("usage: render scene.bin prefix serial|pipeline tiles_per_job repeats warmup");
 auto s=load(argv[1]);std::string prefix=argv[2],mode=argv[3];unsigned batch=std::stoul(argv[4]);int reps=std::stoi(argv[5]),warm=std::stoi(argv[6]);
 if((mode!="serial"&&mode!="pipeline")||!batch||batch>32||reps<1||reps>10||warm<0||warm>2)throw std::runtime_error("Run limits");
 unsigned jobs=(s.tiles+batch-1)/batch;size_t maxwords=0;
 for(unsigned b=0;b<s.tiles;b+=batch){size_t n=std::min(batch,s.tiles-b);for(unsigned t=b;t<std::min(b+batch,s.tiles);t++)n+=s.ranges[t][1]-s.ranges[t][0];maxwords=std::max(maxwords,n);}
 Board board;size_t inbytes=maxwords*64,outbytes=batch*4096,stride=inbytes+outbytes;
 auto mem=board.dev.defaultMemRegion().malloc(stride*2,0,64);uint64_t base=mem->begin.addr();
 if(base%64||base+stride*2>0xffffffffULL)throw std::runtime_error("DDR allocation not representable");
 std::vector<Word> payload[2];for(auto&v:payload)v.reserve(maxwords);
 std::vector<Word> received(batch*64);
 std::vector<std::array<uint16_t,9>> packed(s.active);
 std::vector<Pixel> pixels(s.w*s.h),previous;
 std::ofstream timing(prefix+"_timing.csv"),trace(prefix+"_trace.csv");
 timing<<"sample,total_us,prepare_us,upload_us,wait_us,read_us,cpu_us,input_bytes,output_bytes,jobs,hardware_cycles,prefetch_overlap_cycles\n";
 trace<<"sample,job,tile_begin,tiles,prepare0_us,prepare1_us,upload0_us,upload1_us,submit_us,complete_us,read0_us,read1_us,cycles,reads,writes,prefetch_overlap_cycles,unserved_kernel_read_requests,busy_before_upload,busy_after_upload\n";
 timing<<std::setprecision(12);trace<<std::setprecision(12);
 uint32_t sequence=0;
 for(int rep=-warm;rep<reps;rep++){
  rusage ru0{},ru1{};getrusage(RUSAGE_SELF,&ru0);auto begin=Clock::now();
  for(size_t i=0;i<s.gs.size();i++){auto&q=s.gs[i];const float f[]={q.x,q.y,q.a,q.b,q.c,q.opacity,q.r,q.g,q.blue};for(int k=0;k<9;k++)packed[i][k]=tohalf(f[k]);}
  double prepare=elapsed(begin,Clock::now()),upload=0,waiting=0,readback=0;uint64_t totalin=0,totalout=0,cycles=0,overlap=0;
  std::vector<Event>events(jobs);std::vector<uint32_t>ids(jobs);
  auto launch=[&](unsigned j){
   unsigned slot=j&1,b=j*batch,e=std::min(b+batch,s.tiles);auto&ev=events[j];ev={};ev.job=j;ev.begin=b;ev.tiles=e-b;
   ev.prepare0=elapsed(begin,Clock::now());auto&v=payload[slot];v.clear();
   for(unsigned t=b;t<e;t++){
    auto range=s.ranges[t];Word h{};unsigned x=(t%((s.w+15)/16))*16,y=(t/((s.w+15)/16))*16;
    put32(h,0,range[1]-range[0]);put16(h,4,x);put16(h,6,y);put16(h,8,std::min(16u,s.w-x)|(std::min(16u,s.h-y)<<5));
    for(int k=0;k<3;k++)put16(h,10+2*k,tohalf(s.bg[k]));v.push_back(h);
    for(unsigned k=range[0];k<range[1];k++){Word q{};std::memcpy(q.data(),packed[s.ids[k]].data(),18);v.push_back(q);}
   }
   ev.prepare1=elapsed(begin,Clock::now());prepare+=ev.prepare1-ev.prepare0;
   ev.busy_before_upload=board.read(8);
   ev.upload0=elapsed(begin,Clock::now());mem.write(slot*stride,(char*)v.data(),v.size()*64);ev.upload1=elapsed(begin,Clock::now());upload+=ev.upload1-ev.upload0;totalin+=v.size()*64;
   ev.busy_after_upload=board.read(8);
   ids[j]=++sequence;board.submit(base+slot*stride,v.size(),base+slot*stride+inbytes,e-b,ids[j]);ev.submit=elapsed(begin,Clock::now());
  };
  launch(0);
  for(unsigned j=0;j<jobs;j++){
   if(mode=="pipeline"&&j+1<jobs)launch(j+1);
   auto wb=Clock::now();board.wait_done();waiting+=elapsed(wb,Clock::now());auto&ev=events[j];ev.complete=elapsed(begin,Clock::now());
   if(board.read(0x30)!=ids[j]||board.read(0x48))throw std::runtime_error("Job tag or DMA error");
   ev.cyc=board.read(0x34);ev.reads=board.read(0x38);ev.writes=board.read(0x3c);ev.overlap=board.read(0x40);ev.starve=board.read(0x44);
   if(ev.reads!=payload[j&1].size()||ev.writes!=ev.tiles*64)throw std::runtime_error("DMA counts mismatch");
   cycles+=ev.cyc;overlap+=ev.overlap;
   // ACK frees the engine; the completed output bank remains host-owned until
   // readback finishes. Queued work writes the other bank concurrently.
   board.write(0x2c,2);
   ev.read0=elapsed(begin,Clock::now());mem.read((char*)received.data(),(j&1)*stride+inbytes,ev.tiles*4096);ev.read1=elapsed(begin,Clock::now());readback+=ev.read1-ev.read0;totalout+=ev.tiles*4096;
   for(unsigned t=0;t<ev.tiles;t++)for(unsigned p=0;p<256;p++){
    const uint8_t*raw=received[t*64+p/4].data()+(p%4)*16;uint16_t h[4];uint32_t last,tag;
    std::memcpy(h,raw,8);std::memcpy(&last,raw+8,4);std::memcpy(&tag,raw+12,4);
    if(tag!=t*256+p)throw std::runtime_error("Output ordering/tag mismatch");
    unsigned tile=ev.begin+t,x=(tile%((s.w+15)/16))*16+p%16,y=(tile/((s.w+15)/16))*16+p/16;
    if(x<s.w&&y<s.h){Pixel out{fromhalf(h[0]),fromhalf(h[1]),fromhalf(h[2]),fromhalf(h[3]),last};
     if(!std::isfinite(out.r)||!std::isfinite(out.g)||!std::isfinite(out.b)||!std::isfinite(out.t))throw std::runtime_error("Nonfinite hardware pixel");pixels[y*s.w+x]=out;}
   }
   if(mode=="serial"&&j+1<jobs)launch(j+1);
  }
  auto end=Clock::now();getrusage(RUSAGE_SELF,&ru1);
  if(!previous.empty()&&std::memcmp(previous.data(),pixels.data(),pixels.size()*sizeof(Pixel)))throw std::runtime_error("Repeat drift");previous=pixels;
  double cpu=1e6*((ru1.ru_utime.tv_sec+ru1.ru_stime.tv_sec)-(ru0.ru_utime.tv_sec+ru0.ru_stime.tv_sec))+(ru1.ru_utime.tv_usec+ru1.ru_stime.tv_usec)-(ru0.ru_utime.tv_usec+ru0.ru_stime.tv_usec);
  if(rep>=0){timing<<rep<<','<<elapsed(begin,end)<<','<<prepare<<','<<upload<<','<<waiting<<','<<readback<<','<<cpu<<','<<totalin<<','<<totalout<<','<<jobs<<','<<cycles<<','<<overlap<<'\n';
   for(auto&e:events)trace<<rep<<','<<e.job<<','<<e.begin<<','<<e.tiles<<','<<e.prepare0<<','<<e.prepare1<<','<<e.upload0<<','<<e.upload1<<','<<e.submit<<','<<e.complete<<','<<e.read0<<','<<e.read1<<','<<e.cyc<<','<<e.reads<<','<<e.writes<<','<<e.overlap<<','<<e.starve<<','<<e.busy_before_upload<<','<<e.busy_after_upload<<'\n';}
  std::cout<<"sample="<<rep<<" total_us="<<elapsed(begin,end)<<" input_bytes="<<totalin<<" hardware_cycles="<<cycles<<std::endl;
 }
 std::ofstream out(prefix+".bin",std::ios::binary);out.write("GSSOUT01",8);out.write((char*)&s.w,4);out.write((char*)&s.h,4);out.write((char*)pixels.data(),pixels.size()*sizeof(Pixel));
 if(!out||!timing||!trace)throw std::runtime_error("Write results failed");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
