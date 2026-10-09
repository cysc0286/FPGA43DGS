#include "pipeline.hpp"
#include "coord_half.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#ifndef FLK_PIPEGS_HGR
#define FLK_PIPEGS_HGR 0
#endif

#ifndef FLK_HGR_DPC
#define FLK_HGR_DPC 0
#endif

static Word encode(const float*fields){Word q=0;for(int k=0;k<9;k++)q.range(32*k+31,32*k)=fp_struct<float>(fields[k]).data();return q;}
static Word old(Word q){Word w=0;for(int k=0;k<9;k++)w.range(16*k+15,16*k)=pack_half(half(unpack_float(q,k*32)));return w;}
int main(int argc,char**argv){
 // Exhaust the entire 17-bit domain, beyond the actual coordinate bound.
 // This checks zero, normalization, every tie, exponent carry, and overflow.
 for(unsigned coordinate=0;coordinate<131072;coordinate++){
  if(coord_half_bits(coordinate)!=pack_half(half(coordinate))){
   std::cerr<<"FAIL exact coordinate conversion "<<coordinate<<std::endl;
   return 20;
  }
 }
 std::cout<<"PASS: all 131072 coordinate conversions exact"<<std::endl;
 if(argc==5){
  std::ifstream f(argv[1],std::ios::binary);std::ofstream g(argv[2],std::ios::binary);
  unsigned long long words[8];hls::stream<Word> in,out;
  while(f.read((char*)words,64)){Word q=0;for(int k=0;k<8;k++)q.range(k*64+63,k*64)=words[k];in.write(q);}
  if(!f.eof()||!g)return 10;
  flicker_render_pipeline(in,out,std::atoi(argv[3]),std::atoi(argv[4]));
  unsigned n=0;while(!out.empty()){Word q=out.read();for(int k=0;k<8;k++)words[k]=q.range(k*64+63,k*64).to_uint64();g.write((char*)words,64);n++;}
  if(!in.empty()||n!=64*std::atoi(argv[3])||!g)return 11;
  std::cout<<"PASS: pipeline real input mode="<<argv[4]<<" output_records="<<n<<std::endl;return 0;
 }
 std::vector<Word> input,legacy;
 // Empty tile, terminal contribution, early stop, two edge shapes, and all
 // four sub-tiles. Every selected pixel sees the same original depth order.
 for(unsigned t=0;t<7;t++){
  unsigned n=t==0?0:(t==1?1:(t>=5?24:7)),width=t==3?5:16,height=t==4?2:16;
  Word h=0;h.range(31,0)=n;h.range(47,32)=t*16;h.range(63,48)=32;
  h.range(68,64)=width;h.range(73,69)=height;
  for(int k=0;k<3;k++)h.range(80+k*16+15,80+k*16)=pack_half(half(.1f*(k+1)));
  input.push_back(h);legacy.push_back(h);
  for(unsigned j=0;j<n;j++){
   // Fully saturated Tile followed by a fresh transparent Tile verifies
   // drain/reset, and changed late colors detect contributions after stop.
   const float fields[]={float(t*16+8),40,t>=5?.0001f:.05f,0,t>=5?.0001f:.05f,
                         t==5?.99f:(t==6?.02f:.9f),j<4?.2f:.8f,.4f,j<4?.8f:.2f};
   Word q=encode(fields);input.push_back(q);legacy.push_back(old(q));
  }
 }
 hls::stream<Word> gin,gout;for(Word q:legacy)gin.write(q);flicker_render(gin,gout,7,0);
 std::vector<Word>golden;while(!gout.empty())golden.push_back(gout.read());
 unsigned dpc_mismatches=0;
 double rgb_squared_error=0,rgb_max_error=0;unsigned rgb_samples=0,nonfinite=0;
 for(unsigned mode=0;mode<6;mode++){
  hls::stream<Word>in,out;for(Word q:input)in.write(q);
  flicker_render_pipeline(in,out,7,mode);unsigned n=0;
  while(!out.empty()){
   Word q=out.read();if(n>=golden.size())return 1;
   if(mode==0&&q!=golden[n]){
#if FLK_DPC_DIAGNOSTIC || FLK_HGR_DPC
    ++dpc_mismatches;
    if(dpc_mismatches<=4)std::cerr<<"HGR_DPC_DIAGNOSTIC mismatch record="<<n<<std::endl;
#else
    std::cerr<<"FAIL baseline byte mismatch record="<<n<<std::endl;return 2;
#endif
   }
   if(mode==0){
    for(int pixel=0;pixel<4;pixel++)for(int channel=0;channel<4;channel++){
     unsigned offset=pixel*128+channel*16;
     double actual=float(unpack_half(q,offset)),expected=float(unpack_half(golden[n],offset));
     if(!std::isfinite(actual))nonfinite++;
     if(channel<3){double delta=std::fabs(actual-expected);rgb_squared_error+=delta*delta;
      if(delta>rgb_max_error)rgb_max_error=delta;rgb_samples++;}
    }
   }
   for(int k=0;k<4;k++){ap_uint<128> p=q.range(k*128+127,k*128);if(p.range(127,96)!=n*4+k)return 3;}
   n++;
  }
  if(n!=golden.size()||!in.empty())return 4;
 }
#if FLK_DPC_DIAGNOSTIC || FLK_HGR_DPC
 std::cerr<<"HGR_DPC_DIAGNOSTIC total_mode0_mismatches="<<dpc_mismatches<<std::endl;
 std::cerr<<"HGR_DPC_DIAGNOSTIC rgb_max_abs="<<rgb_max_error
  <<" rgb_rmse="<<std::sqrt(rgb_squared_error/rgb_samples)
  <<" nonfinite="<<nonfinite<<std::endl;
#endif
 if(nonfinite)return 12;
 hls::stream<Word>in,out;flicker_render_pipeline(in,out,0,0);if(!out.empty())return 5;
#if FLK_DPC_DIAGNOSTIC || FLK_HGR_DPC
 std::cout<<"PASS: DPC protocol only; numerical differences reported, image quality NOT accepted"<<std::endl;
#else
 std::cout<<"PASS: pipeline mode 0 exact FLK0, six modes, seven tiles, edge/order/empty/drain/reset"<<std::endl;
#endif
 return 0;
}
