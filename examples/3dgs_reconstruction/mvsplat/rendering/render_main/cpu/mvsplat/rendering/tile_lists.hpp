// Each worker owns a contiguous depth-ordered range. Per-Tile prefix offsets
// preserve that order while workers write disjoint output segments.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

template<class Attribute>
inline void parallel_tile_lists(const std::vector<uint64_t>& keys,
                                const std::vector<Attribute>& attrs,
                                const std::vector<uint32_t>& remap,
                                unsigned nx, unsigned ny, unsigned workers,
                                std::vector<std::array<uint32_t,2>>& ranges,
                                std::vector<uint32_t>& ids,
                                std::vector<uint32_t>& workspace) {
    const unsigned tiles = nx*ny;
    workspace.assign(size_t(workers)*tiles, 0);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(unsigned w=0; w<workers; ++w) {
        auto* counts = workspace.data()+size_t(w)*tiles;
        const size_t first=keys.size()*w/workers, last=keys.size()*(w+1)/workers;
        for(size_t k=first; k<last; ++k) {
            const auto& a=attrs[uint32_t(keys[k])];
            for(unsigned y=a.min_y; y<a.max_y; ++y)
                for(unsigned x=a.min_x; x<a.max_x; ++x) ++counts[y*nx+x];
        }
    }
    ranges.assign(tiles,{0,0});
    uint64_t total=0;
    for(unsigned t=0; t<tiles; ++t) {
        const uint32_t begin=uint32_t(total);
        for(unsigned w=0; w<workers; ++w) {
            auto& count=workspace[size_t(w)*tiles+t];
            const uint32_t n=count;
            count=uint32_t(total); total+=n;
            if(total>50000000) throw std::runtime_error("Tile list capacity");
        }
        if(total!=begin) ranges[t]={begin,uint32_t(total)};
    }
    ids.resize(size_t(total));
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(unsigned w=0; w<workers; ++w) {
        auto* cursors=workspace.data()+size_t(w)*tiles;
        const size_t first=keys.size()*w/workers, last=keys.size()*(w+1)/workers;
        for(size_t k=first; k<last; ++k) {
            const uint32_t i=uint32_t(keys[k]); const auto& a=attrs[i];
            for(unsigned y=a.min_y; y<a.max_y; ++y)
                for(unsigned x=a.min_x; x<a.max_x; ++x)
                    ids[cursors[y*nx+x]++]=remap[i];
        }
    }
}
