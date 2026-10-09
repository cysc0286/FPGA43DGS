#include <array>
#include <cstring>
#include <iostream>
#include "../rgb_pack.hpp"
struct Pixel {float r,g,b,t;uint32_t last;};
int main() {
#if !defined(__aarch64__)
    std::cerr<<"Run exhaustive half and NEON validation on AArch64\n";return 2;
#else
    unsigned checked=0;
    for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        if(std::fesetround(mode))return 3;
        for(unsigned bits=0;bits<65536;++bits) {
            const uint16_t u=uint16_t(bits);__fp16 half;std::memcpy(&half,&u,2);
            const float value=float(half);Pixel p{value,value,value,1,0};std::array<uint8_t,3> rgb{};
            bool failed=false;try{pack_half_rgb(&p,1,rgb.data());}catch(const std::runtime_error&){failed=true;}
            if(!std::isfinite(value)){if(!failed)return 4;continue;}
            if(failed)return 5;
            const auto expected=uint8_t(std::nearbyint(std::clamp(double(value),0.,1.)*255.));
            if(rgb[0]!=expected||rgb[1]!=expected||rgb[2]!=expected)return 6;
            ++checked;
        }
    }
    std::fesetround(FE_TONEAREST);
    std::cout<<"PASS RGB half patterns in four rounding modes; finite="<<checked<<"\n";
    return 0;
#endif
}
