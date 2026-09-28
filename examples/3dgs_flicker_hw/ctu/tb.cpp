#include "ctu.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
int main(int argc,char**argv){
 if(argc==3){
  std::ifstream f(argv[1],std::ios::binary);unsigned long long words[8];std::vector<Word> input;
  while(f.read((char*)words,64)){Word q=0;for(int k=0;k<8;k++)q.range(k*64+63,k*64)=words[k];input.push_back(q);}
  if(!f.eof()||input.empty())return 10;
  for(unsigned mode=0;mode<=4;mode++){
   hls::stream<Word> in,out;for(const Word&q:input)in.write(q);
   flicker_ctu(in,out,input.size(),mode);
   std::ofstream g((std::string(argv[2])+"_mode"+std::to_string(mode)+".raw").c_str(),std::ios::binary);
   unsigned n=0;while(!out.empty()){Word q=out.read(),original=q;original.range(227,224)=0;
    if(n>=input.size()||original!=input[n++])return 11;
    for(int k=0;k<8;k++)words[k]=q.range(k*64+63,k*64).to_uint64();g.write((char*)words,64);
   }
   if(!g||n!=input.size()||!in.empty())return 12;
  }
  std::cout<<"PASS: actual Gaussian CTU "<<input.size()<<" inputs x five modes, exact order and attributes"<<std::endl;return 0;
 }
 for(unsigned mode=0;mode<=4;mode++){
  hls::stream<Word> in,out;unsigned expected[120];
  for(unsigned i=0;i<120;i++){
   unsigned kind=i%4,ox=(i%3)*128,oy=(i%2)*64;bool spiky=(i/4)&1;
   float x=kind==0?3:(kind==1?0:(kind==2?5:1));float y=kind==2?5:(kind==3?1:0);
   const float fields[]={x+ox,y+oy,16,0,spiky?256.f:16.f,.5f,.2f,.3f,.4f};
   Word q=0;for(int k=0;k<9;k++)q.range(k*16+15,k*16)=pack_half(half(fields[k]));
   q.range(159,144)=ox;q.range(175,160)=oy;q.range(180,176)=kind==3?2:8;q.range(185,181)=kind==3?2:8;
   q[186]=spiky;q.range(223,192)=i;in.write(q);
   bool dense=mode==1||(mode==3&&!spiky)||(mode==4&&spiky);
   expected[i]=mode==0?(kind==3?1:15):(kind==0?(dense?1:0):(kind==2?0:1));
  }
  flicker_ctu(in,out,120,mode);
  for(unsigned i=0;i<120;i++){
   if(out.empty())return 1;Word q=out.read();
   if(q.range(223,192)!=i||q.range(227,224)!=expected[i]){
    std::cerr<<"mask failure mode="<<mode<<" index="<<i<<" actual="<<q.range(227,224)<<" expected="<<expected[i]<<std::endl;return 2;
   }
  }
  if(!in.empty()||!out.empty())return 3;
 }
 // Empty stream is a first-class case; all internal stages must drain.
 hls::stream<Word> in,out;flicker_ctu(in,out,0,1);if(!out.empty())return 4;
 std::cout<<"PASS: CTU 600 Gaussian sub-tile requests, five modes, edge clamps, ordered masks and empty stream"<<std::endl;
 return 0;
}
