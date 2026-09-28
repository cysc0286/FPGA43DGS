// Differential contract for the actual split stage, independent of rendering.
#include "pipeline.hpp"
#include "split_baseline.hpp"
#include <vector>
#include <iostream>
void flicker_split_probe(hls::stream<Word>&,hls::stream<Word>&,unsigned,unsigned);
static unsigned seed=1907;
static float random_unit(){seed=1664525u*seed+1013904223u;return float(seed>>8)/16777216.f;}
int main(){
 std::vector<Word> records;
 const unsigned tiles=5;
 for(unsigned t=0;t<tiles;t++){
  unsigned n=t==0?0:17,ox=t*16,oy=32;
  Word h=0;h.range(31,0)=n;h.range(47,32)=ox;h.range(63,48)=oy;
  h.range(68,64)=t==3?5:16;h.range(73,69)=t==4?2:16;
  h.range(127,80)=0x326632663266ULL;records.push_back(h);
  for(unsigned j=0;j<n;j++){
   float a=.002f+random_unit()*2.f,c=.002f+random_unit()*2.f;
   float b=(random_unit()-.5f)*hls::sqrtf(a*c);
   if(j==0){a=0;b=0;c=0;} // Conservative invalid-conic fallback.
   if(j==1){a=-1;b=0;c=1;}
   const float f[9]={float(ox)-24+random_unit()*64, float(oy)-24+random_unit()*64,a,b,c,random_unit(),random_unit(),random_unit(),random_unit()};
   Word q=0;for(unsigned k=0;k<9;k++)q.range(k*32+31,k*32)=fp_struct<float>(f[k]).data();records.push_back(q);
  }
 }
 for(unsigned mode=0;mode<6;mode++){
  hls::stream<Word> in,refin,out,refout;
  for(Word q:records){in.write(q);refin.write(q);}
  baseline_split(refin,refout,tiles,mode);flicker_split_probe(in,out,tiles,mode);
  unsigned count=0;
  while(!refout.empty()){
   Word expected=refout.read();
   if(out.empty()||out.read()!=expected){std::cerr<<"FAIL split mode="<<mode<<" record="<<count<<std::endl;return 1;}
   count++;
  }
  if(!out.empty()||!in.empty()||!refin.empty())return 2;
  std::cout<<"SPLIT_EXACT mode="<<mode<<" records="<<count<<std::endl;
 }
 std::cout<<"PASS split original differential, six modes, empty/edge/invalid conic/order"<<std::endl;
 return 0;
}
