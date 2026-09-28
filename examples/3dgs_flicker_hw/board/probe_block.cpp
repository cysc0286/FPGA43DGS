// Verify the existing vendor PL-DDR block path, without changing the bitstream.
// Legacy adder has a 16-bit address and a count+1 DMA request: reserve guard
// records and verify them, rather than allowing the extra request outside alloc.
#include <icraft-xrt/dev/zg330_device.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>
using namespace icraft::xrt;
using C=std::chrono::steady_clock;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
int main(){try{
 auto d=Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
 auto dev=d.cast<ZG330Device>();
 auto rd=[&](unsigned a){return dev.defaultRegRegion().read(0x400c0000+a,false);};
 auto wr=[&](unsigned a,unsigned v){dev.defaultRegRegion().write(0x400c0000+a,v,false);};
 if(rd(0xc0)!=0x20230628||rd(0x100)!=0x47534231)throw std::runtime_error("Unexpected platform");
 for(unsigned n: {16u,64u,256u}){
  unsigned size=(n+2)*64;
  auto m=dev.defaultMemRegion().malloc(size*2,0,64);
  uint64_t addr=m->begin.addr();
  if(addr%64||addr+2*size>65536)throw std::runtime_error("Legacy address range cannot safely represent allocation");
  std::vector<uint8_t> in(size,0),out(size,0),sentinel(size,0xa5);
  for(unsigned i=0;i<n+2;i++){uint32_t v=0x12340000+i;std::memcpy(in.data()+64*i,&v,4);}
  m.write(size,(char*)sentinel.data(),size);
  auto a=C::now();m.write(0,(char*)in.data(),size);auto b=C::now();
  m.read((char*)out.data(),0,size);if(out!=in)throw std::runtime_error("Block roundtrip mismatch");
  auto c=C::now();wr(4,(n<<16)|unsigned(addr));wr(8,(n<<16)|unsigned(addr+size));wr(0,1);
  auto limit=C::now()+std::chrono::seconds(2);
  while(!rd(0x80)){if(C::now()>limit)throw std::runtime_error("DMA timeout");}
  // Legacy done is raised before all writes are necessarily visible. Explicit
  // tail settle is diagnostic only, NOT an acceptable new-renderer protocol.
  std::this_thread::sleep_for(std::chrono::milliseconds(2));auto e=C::now();
  m.read((char*)out.data(),size,size);auto f=C::now();
  unsigned changed=0;
  for(unsigned i=0;i<n+2;i++){
   uint32_t v;std::memcpy(&v,out.data()+64*i,4);
   if(v==0x12340001+i)++changed;
   else if(i<n)throw std::runtime_error("Adder result mismatch");
  }
  bool guard=true;for(unsigned i=(n+1)*64;i<size;i++)guard&=out[i]==0xa5;
  if(!guard)throw std::runtime_error("DMA crossed allocated guard");
  std::cout<<"{\"n\":"<<n<<",\"allocation\":"<<addr<<",\"input_bytes\":"<<size
   <<",\"write_us\":"<<us(a,b)<<",\"submit_wait_settle_us\":"<<us(c,e)
   <<",\"read_us\":"<<us(e,f)<<",\"processed_records\":"<<changed<<",\"guard_pass\":true}"<<std::endl;
 }
 for(unsigned size: {65536u,1048576u,8388608u}){
  auto m=dev.defaultMemRegion().malloc(size,0,64);
  std::vector<uint8_t> in(size),out(size);uint32_t seed=0x20260926;
  for(auto&v:in){seed=seed*1664525u+1013904223u;v=seed>>24;}
  for(unsigned repeat=0;repeat<5;repeat++){
   in[repeat]^=0x5a;auto a=C::now();m.write(0,(char*)in.data(),size);auto b=C::now();
   m.read((char*)out.data(),0,size);auto c=C::now();
   if(in!=out)throw std::runtime_error("Large block mismatch");
   std::cout<<"{\"kind\":\"large_block\",\"bytes\":"<<size<<",\"repeat\":"<<repeat
    <<",\"write_us\":"<<us(a,b)<<",\"read_us\":"<<us(b,c)<<",\"exact\":true}"<<std::endl;
  }
 }
 Device::Close(d);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
