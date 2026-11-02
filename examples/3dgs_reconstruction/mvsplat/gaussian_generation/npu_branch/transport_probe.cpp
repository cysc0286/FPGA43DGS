// Diagnostic only: bounded SDK memory round trips above the renderer guard.
// No register access, firmware change, network acceptance, or speed claim.
// Run only with all NPU workers stopped: SDK allocations are process-local.
#include <icraft-xrt/core/device.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace icraft::xrt;
int main() {
    auto device=Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
    bool passed=true;
    {
        auto region=device.defaultMemRegion();
        auto guard=region.malloc(64ULL*1024*1024,true,64);
        auto chunk=region.malloc(20ULL*1024*1024,true,64);
        std::vector<char> sent(chunk->byte_size),received(chunk->byte_size);
        for(size_t size:{64ULL,4096ULL,65536ULL,1048576ULL,8388608ULL,20971520ULL}) {
            for(int pattern=0;pattern<2;++pattern) {
                for(size_t i=0;i<size;++i)sent[i]=char((i*73+i/257+pattern*101)&255);
                chunk.write(0,sent.data(),size);
                for(int read=0;read<2;++read) {
                    std::fill(received.begin(),received.begin()+size,char(0x5a));
                    chunk.read(received.data(),0,size);
                    size_t wrong=0,first=size;
                    for(size_t i=0;i<size;++i)if(sent[i]!=received[i]){++wrong;first=std::min(first,i);}
                    passed &= wrong==0;
                    std::cout<<"{\"bytes\":"<<size<<",\"pattern\":"<<pattern
                             <<",\"read\":"<<read<<",\"wrong_bytes\":"<<wrong
                             <<",\"first_wrong\":"<<first<<"}"<<std::endl;
                }
            }
        }
    }
    Device::Close(device);
    return passed?0:1;
}
