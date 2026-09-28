#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <utils/x_hls_utils.h>
typedef ap_uint<512> Word;
inline half unpack_half(Word w,int bit){ap_uint<16>b=w.range(bit+15,bit);return fp_struct<half>(b).to_half();}
inline ap_uint<16> pack_half(half v){return fp_struct<half>(v).data();}
void flicker_render(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode);
