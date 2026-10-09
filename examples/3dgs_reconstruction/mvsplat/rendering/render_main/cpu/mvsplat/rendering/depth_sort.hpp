// Stable positive-FP32 depth sort. Low 32 bits preserve the source Gaussian ID.
// Input IDs are already ascending, so stability preserves the original ties.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

template<unsigned Bits>
inline void sort_positive_depth(std::vector<uint64_t>& keys,
                                std::vector<uint64_t>& scratch) {
    static_assert(Bits == 8 || Bits == 11 || Bits == 16, "radix width");
    constexpr unsigned bins = 1u << Bits;
    scratch.resize(keys.size());
    std::array<uint32_t,bins> counts;
    for(unsigned offset = 0; offset < 32; offset += Bits) {
        counts.fill(0);
        const unsigned shift = 32 + offset;
        for(uint64_t key: keys) ++counts[(key >> shift) & (bins-1)];
        uint32_t start = 0;
        for(auto& count: counts) { const uint32_t n = count; count = start; start += n; }
        for(uint64_t key: keys) scratch[counts[(key >> shift) & (bins-1)]++] = key;
        keys.swap(scratch);
    }
}
