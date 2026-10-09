#pragma once
#include "../hls/renderer.hpp"
typedef ap_uint<128> PixelBits;
inline unsigned kind(Word q){return q.range(511,510);}
inline float unpack_float(Word q,int bit){ap_uint<32> x=q.range(bit+31,bit);return fp_struct<float>(x).to_float();}
void flicker_render_pipeline(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode);
void framed_ctu(hls::stream<Word>& input,hls::stream<Word>& output,unsigned mode);
