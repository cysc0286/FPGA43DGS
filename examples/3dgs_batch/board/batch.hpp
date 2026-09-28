// Included after replay.cpp (GSC_REPLAY_LIBRARY). Uses only the established SDK.
#pragma once
#include <thread>
struct BatchHardware {
    Hardware& sdk;uint32_t task=0;uint64_t reads=0,writes=0,core=0,execution=0,batches=0;
    double load_us=0,wait_us=0,readback_us=0;
    using C=std::chrono::steady_clock;
    static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
    explicit BatchHardware(Hardware& h):sdk(h){
        if(read(0x100)!=0x47534231||read(0x104)!=0x10000||read(0x108)!=0x01000100)
            throw std::runtime_error("GSB1 capability/ABI missing: refusing writes");
        if(read(0x10c)&1)throw std::runtime_error("Batch engine already owned");
        write(0x120,16);idle();
    }
    uint32_t read(uint32_t a){++reads;return sdk.read(a);}
    void write(uint32_t a,uint32_t v){++writes;sdk.write(a,v);}
    uint32_t wait(uint32_t mask,uint32_t value){
        auto deadline=C::now()+std::chrono::seconds(2);
        for(;;){auto s=read(0x10c);
            if(s&0xf10)throw std::runtime_error("Batch protocol/reject status "+std::to_string(s));
            if((s&mask)==value)return s;
            if(C::now()>deadline)throw std::runtime_error("Batch wait timeout");
        }
    }
    void idle(){wait(1,0);}
    void reset_metrics(){reads=writes=core=execution=batches=0;load_us=wait_us=readback_us=0;}
    void run(const std::vector<Cmd>& cmds,std::vector<State>& out,bool trace=false,size_t chunk=256){
        if(chunk<1||chunk>256)throw std::runtime_error("Invalid batch size");
        out.clear();reset_metrics();
        for(size_t begin=0;begin<cmds.size();begin+=chunk){
            auto a=C::now();size_t end=std::min(begin+chunk,cmds.size()),outputs=0;
            idle();
            for(size_t i=begin;i<end;++i){
                for(auto word:cmds[i])write(0x110,word);
                if(trace||cmds[i][0]==2)++outputs;
            }
            if(++task==0)++task;
            write(0x114,task);write(0x118,end-begin);write(0x11c,trace);write(0x120,1);
            auto b=C::now();wait(2,2);auto c=C::now();
            if(read(0x12c)!=task||read(0x130)!=end-begin||read(0x134)!=outputs)
                throw std::runtime_error("Batch accounting mismatch");
            // Read latched cycle counters once per batch; included in host timing.
            core+=uint64_t(read(0x160))|(uint64_t(read(0x164))<<32);
            execution+=uint64_t(read(0x168))|(uint64_t(read(0x16c))<<32);
            for(size_t i=0;i<outputs;++i){
                wait(4,4);State s{};
                for(unsigned f=0;f<8;++f)s[f]=read(0x140+4*f);
                out.push_back(s);write(0x120,2);
            }
            write(0x120,4);idle();auto d=C::now();
            load_us+=us(a,b);wait_us+=us(b,c);readback_us+=us(c,d);++batches;
        }
    }
};
