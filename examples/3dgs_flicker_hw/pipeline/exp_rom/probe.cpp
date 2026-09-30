#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <utils/x_hls_utils.h>
#if FLK_EXACT_EXP_ROM == 2
#include "table_compact.hpp"
#elif FLK_EXACT_EXP_ROM == 1
#include "table.hpp"
#endif
void exp_probe(hls::stream<ap_uint<16> >& input,hls::stream<ap_uint<16> >& output,unsigned count){
#pragma HLS INTERFACE ap_fifo port=input
#pragma HLS INTERFACE ap_fifo port=output
#pragma HLS INTERFACE ap_ctrl_hs port=return
 for(unsigned i=0;i<count;i++){
#pragma HLS PIPELINE II=1
  half x=fp_struct<half>(input.read()).to_half();
#ifdef FLK_EXACT_EXP_ROM
  half y=exact_negative_exp<0>(x);
#else
  half y=hls::half_exp(x);
#endif
  output.write(fp_struct<half>(y).data());
 }
}
