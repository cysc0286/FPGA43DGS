#include "renderer.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
int main(int argc,char**argv){
 if(argc==4){
  std::ifstream f(argv[1],std::ios::binary);std::ofstream g(argv[2],std::ios::binary);
  hls::stream<Word> in,out;unsigned long long words[8];
  while(f.read((char*)words,64)){Word w=0;for(int k=0;k<8;k++)w.range(k*64+63,k*64)=words[k];in.write(w);}
  if(!f.eof()||!g)return 10;
  flicker_render(in,out,std::atoi(argv[3]),0);
  while(!out.empty()){Word w=out.read();for(int k=0;k<8;k++)words[k]=w.range(k*64+63,k*64).to_uint64();g.write((char*)words,64);}
  if(!in.empty()||!g)return 11;
  std::cout<<"PASS: actual Gaussian stream HLS C reference"<<std::endl;return 0;
 }
 hls::stream<Word> in,out;
 // Includes an empty tile, an active terminal contribution and early exit.
 for(unsigned t=0;t<3;t++){
  Word hdr=0;hdr.range(31,0)=t==0?0:(t==1?1:8);hdr.range(68,64)=16;hdr.range(73,69)=16;
  in.write(hdr);
  for(unsigned j=0;j<(t==0?0:(t==1?1:8));j++){
   Word q=0;const float fields[]={0,0,1,0,1,.5f,.25f,.5f,.75f};
   for(int k=0;k<9;k++)q.range(16*k+15,16*k)=pack_half(half(fields[k]));in.write(q);
  }
 }
 flicker_render(in,out,3,0);
 unsigned n=0;
 while(!out.empty()){
  Word w=out.read();for(unsigned k=0;k<4;k++){
   ap_uint<128>p=w.range(128*k+127,128*k);
   unsigned tag=p.range(127,96);if(tag!=n*4+k)return 1;
   if(tag==0 && p.range(63,48)!=pack_half(half(1)))return 2;
   if(tag==256){
    if(p.range(15,0)!=pack_half(half(.125f))||p.range(31,16)!=pack_half(half(.25f))||p.range(47,32)!=pack_half(half(.375f))||p.range(95,64)!=1)return 3;
   }
  }++n;
 }
 if(n!=192||!in.empty())return 4;
 std::cout<<"PASS: FP16 parameter-input renderer 3 tiles 768 pixels"<<std::endl;return 0;
}
