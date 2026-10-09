// Build the full tile candidate lists from board-computed attributes.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Attribute {
    float q[10];
    uint16_t min_x, min_y, max_x, max_y;
    uint32_t valid;
};
struct Gaussian {
    uint32_t id;
    float q[10];
};
static_assert(sizeof(Attribute)==52 && sizeof(Gaussian)==44,"scene ABI changed");

template <typename T> void read_exact(std::ifstream& f,T* dst,size_t n) {
    if(!f.read(reinterpret_cast<char*>(dst),n*sizeof(T)))throw std::runtime_error("truncated attributes");
}

int main(int argc,char**argv)try {
    if(argc!=3)throw std::runtime_error("usage: group_sort input.attr output.scene");
    auto start=Clock::now();
    std::ifstream f(argv[1],std::ios::binary);
    char magic[8];read_exact(f,magic,8);
    if(std::memcmp(magic,"FLKATR01",8))throw std::runtime_error("attribute ABI mismatch");
    uint32_t h[4];read_exact(f,h,4);
    const uint32_t count=h[0],w=h[1],height=h[2],expected=h[3];
    if(!count||count>1000000||!w||!height||w>2048||height>2048)
        throw std::runtime_error("attribute dimensions");
    std::vector<Attribute> attr(count);
    read_exact(f,attr.data(),count);
    if(f.peek()!=EOF)throw std::runtime_error("trailing attribute bytes");
    auto loaded=Clock::now();
    const uint32_t nx=(w+15)/16,ny=(height+15)/16,tiles=nx*ny;
    std::vector<std::vector<uint32_t>> buckets(tiles);
    std::vector<uint32_t> remap(count,UINT32_MAX);
    std::vector<Gaussian> active;
    active.reserve(expected);
#if defined(GLOBAL_SORT)
    std::vector<uint32_t> sorted_ids;
    sorted_ids.reserve(expected);
#elif defined(PACKED_SORT)
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(expected);
#endif
    for(uint32_t i=0;i<count;++i) {
        const auto& a=attr[i];
        if(!a.valid)continue;
        if(a.valid!=1||a.min_x>=a.max_x||a.min_y>=a.max_y||a.max_x>nx||a.max_y>ny)
            throw std::runtime_error("invalid active tile rectangle");
        for(float x:a.q)if(!std::isfinite(x))throw std::runtime_error("nonfinite attribute");
        if(!(a.q[9]>.2f))throw std::runtime_error("nonpositive visible depth");
        remap[i]=active.size();
        Gaussian g{};g.id=i;std::memcpy(g.q,a.q,sizeof(g.q));active.push_back(g);
#ifdef GLOBAL_SORT
        sorted_ids.push_back(i);
#elif defined(PACKED_SORT)
        uint32_t depth_bits;
        std::memcpy(&depth_bits,&a.q[9],sizeof(depth_bits));
        sorted_keys.push_back((uint64_t(depth_bits)<<32)|i);
#else
        for(uint32_t y=a.min_y;y<a.max_y;++y)
            for(uint32_t x=a.min_x;x<a.max_x;++x)
                buckets[y*nx+x].push_back(i);
#endif
    }
    if(active.size()!=expected)throw std::runtime_error("active count mismatch");
    auto grouped=Clock::now();
#ifdef GLOBAL_SORT
    // One depth sort serves every Tile; stable ties follow original PLY ID.
    std::stable_sort(sorted_ids.begin(),sorted_ids.end(),[&](uint32_t a,uint32_t b){
        return attr[a].q[9]<attr[b].q[9];
    });
    auto globally_sorted=Clock::now();
    for(uint32_t id:sorted_ids) {
        const auto& a=attr[id];
        for(uint32_t y=a.min_y;y<a.max_y;++y)
            for(uint32_t x=a.min_x;x<a.max_x;++x)
                buckets[y*nx+x].push_back(id);
    }
#elif defined(PACKED_SORT)
    std::sort(sorted_keys.begin(),sorted_keys.end());
    auto globally_sorted=Clock::now();
    for(uint64_t key:sorted_keys) {
        uint32_t id=uint32_t(key);
        const auto& a=attr[id];
        for(uint32_t y=a.min_y;y<a.max_y;++y)
            for(uint32_t x=a.min_x;x<a.max_x;++x)
                buckets[y*nx+x].push_back(id);
    }
#endif
    std::vector<std::array<uint32_t,2>> ranges(tiles);
    std::vector<uint32_t> ids;
    for(uint32_t t=0;t<tiles;++t) {
        auto& bucket=buckets[t];
        if(bucket.empty())continue;
#if defined(GLOBAL_SORT) || defined(PACKED_SORT)
        // Already ordered by the global stable sort.
#elif defined(RADIX_SORT)
        // Positive IEEE-754 depth bit patterns have the same order as floats.
        // Stable passes retain original Gaussian ID order for equal depths.
        std::vector<uint32_t> scratch(bucket.size());
        for(unsigned pass=0;pass<4;++pass) {
            std::array<uint32_t,256> offsets{};
            auto digit=[&](uint32_t id) {
                uint32_t bits;
                std::memcpy(&bits,&attr[id].q[9],sizeof(bits));
                return (bits>>(pass*8))&255u;
            };
            for(uint32_t id:bucket)++offsets[digit(id)];
            uint32_t next=0;
            for(auto& offset:offsets){uint32_t count=offset;offset=next;next+=count;}
            for(uint32_t id:bucket)scratch[offsets[digit(id)]++]=id;
            bucket.swap(scratch);
        }
#else
        std::stable_sort(bucket.begin(),bucket.end(),[&](uint32_t a,uint32_t b){
            return attr[a].q[9]<attr[b].q[9];
        });
#endif
        ranges[t]={uint32_t(ids.size()),uint32_t(ids.size()+bucket.size())};
        for(uint32_t id:bucket)ids.push_back(remap[id]);
    }
    auto sorted=Clock::now();
    std::ofstream out(argv[2],std::ios::binary);
    if(!out)throw std::runtime_error("scene output open");
    out.write("GSSCN001",8);
    uint32_t header[7]={w,height,16,count,uint32_t(active.size()),uint32_t(ids.size()),tiles};
    float bg[3]={0,0,0};
    out.write(reinterpret_cast<const char*>(header),sizeof(header));
    out.write(reinterpret_cast<const char*>(bg),sizeof(bg));
    out.write(reinterpret_cast<const char*>(active.data()),active.size()*sizeof(Gaussian));
    out.write(reinterpret_cast<const char*>(ranges.data()),ranges.size()*sizeof(ranges[0]));
    out.write(reinterpret_cast<const char*>(ids.data()),ids.size()*sizeof(uint32_t));
    if(!out)throw std::runtime_error("scene output write");out.close();
    auto done=Clock::now();
    auto ms=[](Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();};
    std::cout<<std::fixed<<std::setprecision(3)<<"points="<<count<<" active="<<active.size()
             <<" entries="<<ids.size()<<" read_ms="<<ms(start,loaded)
#if defined(GLOBAL_SORT) || defined(PACKED_SORT)
             <<" group_ms="<<ms(globally_sorted,sorted)<<" sort_ms="<<ms(grouped,globally_sorted)
#else
             <<" group_ms="<<ms(loaded,grouped)<<" sort_ms="<<ms(grouped,sorted)
#endif
             <<" write_ms="<<ms(sorted,done)<<" total_ms="<<ms(start,done)<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
