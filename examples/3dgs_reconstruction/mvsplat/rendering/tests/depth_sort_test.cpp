#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "../depth_sort.hpp"

template<unsigned Bits>
void check(const std::vector<uint64_t>& input, const std::vector<uint64_t>& expected) {
    auto actual = input;
    std::vector<uint64_t> scratch;
    sort_positive_depth<Bits>(actual, scratch);
    if(actual != expected) throw std::runtime_error("depth/ID ordering changed");
    // Reusing the same buffers must also work after a different-size scene.
    actual.clear(); sort_positive_depth<Bits>(actual, scratch);
    actual = input; sort_positive_depth<Bits>(actual, scratch);
    if(actual != expected) throw std::runtime_error("scratch reuse changed order");
}

int main() {
    uint32_t state = 16383;
    unsigned cases = 0;
    for(unsigned n: {0u,1u,17u,32768u,100000u}) for(unsigned kind=0;kind<4;++kind) {
        std::vector<uint64_t> input;
        for(unsigned id=0;id<n;++id) {
            state = state*1664525u+1013904223u;
            uint32_t bits = 0x3e800000u+(state%0x04000000u);
            if(kind==1)bits=0x40000000u+(id%9);
            if(kind==2)bits=0x40800000u-id;
            if(kind==3)bits=0x3f800000u;
            input.push_back((uint64_t(bits)<<32)|id);
        }
        auto expected=input;
        std::sort(expected.begin(),expected.end());
        check<8>(input,expected); check<11>(input,expected); check<16>(input,expected);
        ++cases;
    }
    std::cout << "PASS: " << cases << " cases, all three radix widths, tie ordering and scratch reuse\n";
}
