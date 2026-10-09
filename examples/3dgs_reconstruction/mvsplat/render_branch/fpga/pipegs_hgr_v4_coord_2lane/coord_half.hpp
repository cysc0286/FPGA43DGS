#pragma once

// A lane coordinate is an unsigned 16-bit origin plus an offset of 0..7.
// All such sums are exactly representable in FP32. This direct converter
// preserves the original FP32 -> FP16 round-to-nearest, ties-to-even result,
// including overflow to infinity, without instantiating a generic FP32 unit.
static ap_uint<16> coord_half_bits(ap_uint<17> value) {
#pragma HLS INLINE
 ap_uint<17> normalized=value;
 ap_uint<5> exponent=16;
 if(normalized.range(16,1)==0){normalized<<=16;exponent-=16;}
 if(normalized.range(16,9)==0){normalized<<=8;exponent-=8;}
 if(normalized.range(16,13)==0){normalized<<=4;exponent-=4;}
 if(normalized.range(16,15)==0){normalized<<=2;exponent-=2;}
 if(!normalized[16]){normalized<<=1;exponent-=1;}
 ap_uint<12> significand=normalized.range(16,6);
 bool round_up=normalized[5] && (normalized.range(4,0)!=0 || normalized[6]);
 significand+=round_up;
 if(significand[11]){significand>>=1;exponent+=1;}
 ap_uint<16> result=0;
 result.range(9,0)=significand.range(9,0);
 result.range(14,10)=ap_uint<5>(exponent+15);
 if(exponent>=16)result=0x7c00;
 if(value==0)result=0;
 return result;
}

static half coord_half(unsigned value) {
#pragma HLS INLINE
 return unpack_half16(coord_half_bits(ap_uint<17>(value)));
}
