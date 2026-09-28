// Input-only CPU/PL replay. Golden state stays on the PC.
#ifndef CPU_ONLY
#include <icraft-xrt/dev/zg330_device.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

using Cmd=std::array<uint32_t,6>;
using State=std::array<uint32_t,8>;
constexpr uint32_t ONE=1u<<30,MAX_RGB=16u<<24;
uint32_t mul(uint32_t a,uint32_t b){return (uint64_t(a)*b+(1ull<<29))>>30;}
struct Reference {
    uint32_t r=0,g=0,b=0,t=ONE,last=0,seen=0;bool done=false,finished=false,error=false;
    State step(const Cmd& c){
        auto op=c[0],a=c[1],cr=c[2],cg=c[3],cb=c[4],ord=c[5];error=false;
        if(op==0){r=g=b=last=seen=0;t=ONE;done=finished=false;}
        else if((op!=1 && op!=2)||finished||cr>MAX_RGB||cg>MAX_RGB||cb>MAX_RGB||
                (op==1&&(a>ONE||ord==0||ord<=seen)))error=true;
        else if(op==2){r+=mul(cr,t);g+=mul(cg,t);b+=mul(cb,t);finished=true;}
        else {
            seen=ord;
            if(!done && a>=4210753){
                auto w=mul(t,std::min(a,1063004406u));auto nt=t-w;
                if(nt<107375)done=true;
                else{r+=mul(cr,w);g+=mul(cg,w);b+=mul(cb,w);t=nt;last=ord;}
            }
        }
        return {r,g,b,t,last,uint32_t(2*error+4*done+8*finished),0,0};
    }
};

#ifndef CPU_ONLY
struct Hardware {
    icraft::xrt::ZG330Device fpai; uint32_t sequence;
    explicit Hardware(icraft::xrt::ZG330Device d):fpai(d),sequence(0){
        if(read(0x9c)!=0x47534331 || read(0xc0)!=0x20230628)
            throw std::runtime_error("GSC1 capability absent: no commands submitted");
        if(read(0x98)&1)throw std::runtime_error("PL is already busy");
        sequence=read(0x94);
    }
    uint32_t read(uint32_t offset){return fpai.defaultRegRegion().read(0x400c0000+offset,false);}
    void write(uint32_t offset,uint32_t v){fpai.defaultRegRegion().write(0x400c0000+offset,v,false);}
    void submit(const Cmd& c){
        write(0x0c,c[1]);write(0x10,c[2]);write(0x14,c[3]);write(0x20,c[4]);
        write(0x24,c[5]);write(0x28,c[0]);if(++sequence==0)++sequence;
        write(0x2c,sequence);
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(read(0x94)!=sequence){
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("PL completion timeout");
        }
    }
    State state(){
        State s={read(0x84),read(0x88),read(0x8c),read(0x90),read(0xa0),read(0x98),read(0xa4),read(0xa8)};
        if(read(0x94)!=sequence)throw std::runtime_error("Completion changed during readback");
        return s;
    }
};
#endif

#ifndef GSC_REPLAY_LIBRARY
int main(int argc,char**argv){
    try{
        if(argc!=6)throw std::runtime_error("usage: replay cpu|fpga input.txt actual.txt trace|tile repeats");
        std::string backend=argv[1],mode=argv[4];int repeats=std::stoi(argv[5]);
        if((backend!="cpu"&&backend!="fpga")||(mode!="trace"&&mode!="tile")||repeats<1||repeats>100)
            throw std::runtime_error("invalid mode or repeats");
        std::ifstream in(argv[2]);if(!in)throw std::runtime_error("input not found");
        std::vector<Cmd> commands;std::string line,extra;
        while(std::getline(in,line)){
            std::istringstream f(line);Cmd cmd{};
            for(auto&v:cmd){uint64_t x;if(!(f>>std::hex>>x)||x>UINT32_MAX)throw std::runtime_error("bad input");v=x;}
            if(f>>extra)throw std::runtime_error("extra input field");
            commands.push_back(cmd);
        }
        if(in.bad()||commands.empty()||commands.front()[0]!=0)throw std::runtime_error("input must start with CLEAR");
        if(mode=="tile")for(const auto&c:commands)if(c[0]>2)throw std::runtime_error("invalid tile opcode");
        std::vector<State> output;output.reserve(commands.size());std::vector<double> times;
        std::vector<uint64_t> cycles, hashes;
        auto execute=[&](auto&& submit,auto&& state){
            for(int repeat=-1;repeat<repeats;++repeat){ // one untimed warm-up
                output.clear();uint64_t total_cycles=0;
                auto start=std::chrono::steady_clock::now();
                for(const auto&c:commands){
                    submit(c);
                    if(mode=="trace"||c[0]==2){auto s=state();output.push_back(s);if(c[0]==2)total_cycles+=s[7];}
                }
                auto end=std::chrono::steady_clock::now();
                uint64_t hash=14695981039346656037ull;
                for(const auto&s:output)for(size_t f=0;f<6;++f)for(unsigned shift=0;shift<32;shift+=8){
                    hash^=(s[f]>>shift)&255u;hash*=1099511628211ull;
                }
                if(!hashes.empty() && hash!=hashes.front())throw std::runtime_error("Nonidentical repeated output");
                if(repeat>=0){times.push_back(std::chrono::duration<double,std::milli>(end-start).count());cycles.push_back(total_cycles);hashes.push_back(hash);}
            }
        };
        if(backend=="cpu"){
            Reference ref;State s{};
            execute([&](const Cmd&c){s=ref.step(c);},[&](){return s;});
        }else{
#ifdef CPU_ONLY
            throw std::runtime_error("CPU-only build");
#else
            auto dev=icraft::xrt::Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
            try{Hardware hw(dev.cast<icraft::xrt::ZG330Device>());
                execute([&](const Cmd&c){hw.submit(c);},[&](){return hw.state();});
            }catch(...){icraft::xrt::Device::Close(dev);throw;}
            icraft::xrt::Device::Close(dev);
#endif
        }
        std::ofstream out(argv[3]);if(!out)throw std::runtime_error("output not writable");
        for(const auto&s:output){for(size_t i=0;i<s.size();++i)out<<std::hex<<s[i]<<(i+1==s.size()?'\n':' ');}
        out.flush();if(!out)throw std::runtime_error("output write failed");
        std::cout<<"{\"backend\":\""<<backend<<"\",\"mode\":\""<<mode<<"\",\"commands\":"<<commands.size()
                 <<",\"output_records\":"<<output.size()<<",\"warmups\":1,\"elapsed_ms\":[";
        for(size_t i=0;i<times.size();++i)std::cout<<(i?",":"")<<times[i];
        std::cout<<"],\"pixel_cycles\":[";
        for(size_t i=0;i<cycles.size();++i)std::cout<<(i?",":"")<<cycles[i];
        std::cout<<"],\"state_hashes_fnv1a64\":[";
        for(size_t i=0;i<hashes.size();++i)std::cout<<(i?",":"")<<'\"'<<std::hex<<hashes[i]<<'\"';
        std::cout<<"],\"timing_scope\":\"input resident; excludes parsing/open/file IO/hash; includes submit/wait/readback\"}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
#endif
