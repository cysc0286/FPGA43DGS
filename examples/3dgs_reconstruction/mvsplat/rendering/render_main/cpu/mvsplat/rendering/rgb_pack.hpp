// RGB8 packaging of finite FP16-origin FPGA pixels. Half values in [0,1]
// times 255 are exactly representable in FP32; ties-to-even therefore matches
// the established double/nearbyint conversion. Never use for arbitrary FP32
// CPU pixels without an independent rounding analysis.
#pragma once
#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

template<class PixelType>
inline void pack_half_rgb(const PixelType* pixels,size_t count,uint8_t* rgb) {
    static_assert(offsetof(PixelType,g)==4 && offsetof(PixelType,b)==8 &&
                  offsetof(PixelType,t)==12,"RGB SIMD input layout changed");
#if defined(__aarch64__)
    if(std::fegetround()==FE_TONEAREST) {
        const auto zero=vdupq_n_f32(0.f),one=vdupq_n_f32(1.f),scale=vdupq_n_f32(255.f);
        const auto largest=vdupq_n_f32(std::numeric_limits<float>::max());
        for(size_t i=0;i<count;++i) {
            const auto channels=vld1q_f32(&pixels[i].r);
            const auto finite=vcleq_f32(vabsq_f32(channels),largest);
            if(!(vgetq_lane_u32(finite,0)&&vgetq_lane_u32(finite,1)&&vgetq_lane_u32(finite,2)))
                throw std::runtime_error("nonfinite pixel");
            const auto clipped=vminq_f32(vmaxq_f32(channels,zero),one);
            const auto bytes=vcvtnq_u32_f32(vmulq_f32(clipped,scale));
            rgb[3*i]=uint8_t(vgetq_lane_u32(bytes,0));
            rgb[3*i+1]=uint8_t(vgetq_lane_u32(bytes,1));
            rgb[3*i+2]=uint8_t(vgetq_lane_u32(bytes,2));
        }
        return;
    }
#endif
    // Respect a caller's non-default floating-point rounding environment.
    for(size_t i=0;i<count;++i) {
        const float channels[3]={pixels[i].r,pixels[i].g,pixels[i].b};
        for(unsigned k=0;k<3;++k) {
            if(!std::isfinite(channels[k]))throw std::runtime_error("nonfinite pixel");
            rgb[3*i+k]=uint8_t(std::nearbyint(std::clamp(double(channels[k]),0.,1.)*255.));
        }
    }
}
