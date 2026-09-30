#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <utils/x_hls_utils.h>
#include <iostream>
void exp_probe(hls::stream<ap_uint<16> >&,hls::stream<ap_uint<16> >&,unsigned);
int main(){
 hls::stream<ap_uint<16> > in,out;
 // All nonpositive finite FP16 codes plus negative infinity. Positive zero is
 // added separately; NaNs are excluded by the caller's pw <= 0 comparison.
 for(unsigned m=0;m<=0x7c00;m++)in.write(0x8000|m);
 in.write(0);exp_probe(in,out,0x7c02);
 for(unsigned m=0;m<=0x7c01;m++){
  unsigned code=m==0x7c01?0:(0x8000|m);
  half x=fp_struct<half>(ap_uint<16>(code)).to_half();
  ap_uint<16> expected=fp_struct<half>(hls::half_exp(x)).data(),actual=out.read();
  if(expected!=actual){std::cerr<<"mismatch "<<code<<" expected "<<expected<<" actual "<<actual<<std::endl;return 1;}
 }
 if(!in.empty()||!out.empty())return 2;
 std::cout<<"PASS exhaustive 31746 nonpositive FP16 inputs, exact HLS exp bits"<<std::endl;
 return 0;
}
