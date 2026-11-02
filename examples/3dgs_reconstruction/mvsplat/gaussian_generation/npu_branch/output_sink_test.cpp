// SDK-free host contract/serialization benchmark; never an NPU timing result.
#include "output_sink.h"
#include <array>
#include <chrono>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

static void require(bool passed,const char* reason) {
    if(!passed)throw std::runtime_error(reason);
}
static void contracts() {
    std::array<char,10> storage;storage.fill('G');
    MgsOutputSink sink(storage.data()+1,8);std::ostream out(&sink);
    out.write("abcd",4);out.put('e');out.write("fgh",3);out.flush();
    require(out.good() && sink.size()==8,"Exact output size");
    require(std::string(storage.data()+1,8)=="abcdefgh","Mixed-write payload");
    out.seekp(1);out.write("BC",2);
    require(out.good() && std::string(storage.data()+1,8)=="aBCdefgh","Bounded overwrite");
    out.seekp(0,std::ios::end);out.put('x');
    require(out.fail() && storage.front()=='G' && storage.back()=='G',"Overflow guards");
    std::array<char,4> short_storage{};
    MgsOutputSink short_sink(short_storage.data(),4);std::ostream short_out(&short_sink);
    short_out.write("abcdefg",7);
    require(short_out.fail() && short_sink.size()==4,"Short write must fail");
    MgsOutputSink gap(short_storage.data(),4);std::ostream gap_out(&gap);
    gap_out.seekp(3);gap_out.put('x');
    require(gap_out.fail() && gap.size()==0,"Unwritten hole must fail");
    MgsOutputSink partial(short_storage.data(),4);std::ostream partial_out(&partial);
    partial_out.write("ab",2);
    require(partial_out.good() && partial.size()==2,"Partial size preserved");
    partial_out.seekp(-1,std::ios::beg);
    require(partial_out.fail(),"Negative seek must fail");
    MgsOutputSink empty(short_storage.data(),0);std::ostream empty_out(&empty);
    empty_out.put('x');require(empty_out.fail(),"Zero capacity guard");
    std::cout<<"contracts_passed=true\n";
}
static void benchmark(size_t bytes,int count) {
    using Clock=std::chrono::steady_clock;
    std::vector<char> payload(bytes),target(bytes);
    for(size_t i=0;i<bytes;++i)payload[i]=static_cast<char>(i%251);
    std::vector<double> old_times,new_times;
    // Alternate order to avoid systematically warming one mode first.
    for(int i=-2;i<count;++i)for(int j=0;j<2;++j) {
        const bool direct=((i+j)%2)==0;
        const auto start=Clock::now();
        if(direct) {
            MgsOutputSink sink(target.data(),bytes);std::ostream out(&sink);
            for(size_t k=0;k<bytes;k+=4096)
                out.write(payload.data()+k,static_cast<std::streamsize>(std::min(size_t(4096),bytes-k)));
            require(out.good() && sink.size()==bytes,"Direct length");
        } else {
            std::ostringstream out(std::ios::out|std::ios::binary);
            for(size_t k=0;k<bytes;k+=4096)
                out.write(payload.data()+k,static_cast<std::streamsize>(std::min(size_t(4096),bytes-k)));
            const auto text=out.str();std::memcpy(target.data(),text.data(),text.size());
        }
        const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
        require(target==payload,"Benchmark bytes differ");
        if(i>=0)(direct?new_times:old_times).push_back(seconds);
    }
    auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    std::cout<<"bytes="<<bytes<<" rounds="<<count<<" old_seconds="<<median(old_times)
             <<" direct_seconds="<<median(new_times)<<" payload_exact=true\n";
}
int main() {
    try {contracts();benchmark(1024*1024,20);benchmark(2*84*128*128*4,20);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    return 0;
}
