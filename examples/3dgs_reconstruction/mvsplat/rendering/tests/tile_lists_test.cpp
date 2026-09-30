#include "../tile_lists.hpp"
#include <iostream>
#include <random>

struct Box { unsigned min_x,min_y,max_x,max_y; };

int main() {
    std::mt19937 rng(73); unsigned cases=0;
    for(unsigned n: {0u,1u,17u,32768u}) for(unsigned nx: {1u,8u,9u}) {
        const unsigned ny=5;
        std::vector<Box> attrs(n); std::vector<uint32_t> remap(n);
        std::vector<uint64_t> keys(n);
        for(unsigned i=0;i<n;++i) {
            const unsigned x=rng()%nx,y=rng()%ny;
            attrs[i]={x,y,x+1+unsigned(rng()%(nx-x)),y+1+unsigned(rng()%(ny-y))};
            remap[i]=n-1-i; keys[i]=(uint64_t(rng())<<32)|i;
        }
        std::sort(keys.begin(),keys.end());
        std::vector<uint32_t> reference;
        std::vector<std::array<uint32_t,2>> expected(nx*ny,{0,0});
        for(unsigned t=0;t<nx*ny;++t) {
            const uint32_t begin=reference.size(),x=t%nx,y=t/nx;
            for(auto key:keys) { const auto i=uint32_t(key); const auto& a=attrs[i];
                if(x>=a.min_x&&x<a.max_x&&y>=a.min_y&&y<a.max_y)reference.push_back(remap[i]);
            }
            if(reference.size()!=begin)expected[t]={begin,uint32_t(reference.size())};
        }
        for(unsigned workers: {1u,2u,3u,4u}) {
            std::vector<uint32_t> ids,workspace;
            std::vector<std::array<uint32_t,2>> ranges;
            for(unsigned repeat=0;repeat<2;++repeat) {
                parallel_tile_lists(keys,attrs,remap,nx,ny,workers,ranges,ids,workspace);
                if(ranges!=expected||ids!=reference)throw std::runtime_error("Parallel Tile order mismatch");
            }
            ++cases;
        }
    }
    std::cout<<"PASS parallel Tile lists "<<cases<<" cases, including buffer reuse\n";
}
