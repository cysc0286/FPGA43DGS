// Unary static partitions: one shared device, retained sessions, FP32 host ABI.
// SDK output conversion uses SFB; raw Tensor::read is not a host-layout oracle.
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xir/core/network.h>
#include <icraft-xir/ops/hard_op.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace icraft::xrt;
using namespace icraft::xir;
using Clock=std::chrono::steady_clock;
static thread_local std::string error_text;
struct Owner {
    HostDevice host=HostDevice::Default();
    Device device=Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
    ~Owner() { try { Device::Close(device); } catch (...) {} }
};
static std::weak_ptr<Owner> owner;
static std::shared_ptr<Owner> shared_owner() {
    auto value=owner.lock();
    if (!value) { value=std::make_shared<Owner>(); owner=value; }
    return value;
}
static int compute_count(const HardOpNode* op) {
    int count=(op->name=="icraft::xir::Conv2dNode" || op->name=="icraft::xir::MatmulNode")?1:0;
    for (const auto& child:op->sub_hard_ops) count+=compute_count(child.as<HardOpNode>());
    return count;
}
struct MgsContext {
    std::shared_ptr<Owner> device_owner;
    Network network;
    Session session;
    Tensor input;
    size_t in_bytes,out_bytes;
    int bindings=0;
    MgsContext(const char* json,const char* raw,size_t ib,size_t ob): in_bytes(ib),out_bytes(ob) {
        network=Network::CreateFromJsonFile(std::filesystem::path(json));
        network.lazyLoadParamsFromFile(std::filesystem::path(raw));
        if (network.inputs().size()!=1 || network.outputs().size()!=1) throw std::runtime_error("Unary ABI required");
        for (int i=0;i<2;++i) {
            auto type=(i==0?network.inputs()[0]:network.outputs()[0]).tensorType();
            if (!type->element_dtype.isFP32() || type.bytes()!=(i==0?ib:ob)) throw std::runtime_error("Host ABI mismatch");
        }
        device_owner=shared_owner();
        session=Session::Create<zg330::ZG330Backend,HostBackend>(network,{device_owner->device,device_owner->host});
        session.apply();
        for (const auto& entry:session.getForwards()) {
            const auto& op=std::get<0>(entry);
            if (op->typeKey()=="icraft::xir::HardOpNode") {
                if (!std::get<1>(entry).is<zg330::ZG330Backend>()) throw std::runtime_error("Host compute fallback");
                bindings+=compute_count(op.as<HardOpNode>());
            } else if (op->typeKey()!="icraft::xir::InputNode" && op->typeKey()!="icraft::xir::OutputNode")
                throw std::runtime_error("Unexpected runtime operation");
        }
        if (bindings==0) throw std::runtime_error("NPU compute binding missing");
        input=Tensor(network.inputs()[0]);input.mallocOn(device_owner->host.defaultMemRegion());
    }
};
extern "C" const char* mgs_error() {return error_text.c_str();}
extern "C" int mgs_abi_version() {return 2;}
extern "C" void* mgs_create(const char* json,const char* raw,size_t ib,size_t ob) {
    try {return new MgsContext(json,raw,ib,ob);} catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" void mgs_destroy(void* ctx) {delete static_cast<MgsContext*>(ctx);}
extern "C" int mgs_bindings(void* ctx) {return static_cast<MgsContext*>(ctx)->bindings;}
static int forward_impl(void* ctx,const float* in,size_t ib,float* out,size_t ob,
                        double* detailed) {
    try {
        if(!ctx || !in || !out || !detailed) throw std::runtime_error("Null forward argument");
        const auto& checked=*static_cast<MgsContext*>(ctx);
        if(ib!=checked.in_bytes || ob!=checked.out_bytes) throw std::runtime_error("Forward buffer length mismatch");
        auto& c=*static_cast<MgsContext*>(ctx);const auto start=Clock::now();
        c.input.write(0,const_cast<char*>(reinterpret_cast<const char*>(in)),c.in_bytes);
        const auto input_done=Clock::now();
        auto values=c.session.forward({c.input});
        const auto submitted=Clock::now();
        if(values.size()!=1 || !values[0].waitForReady(std::chrono::seconds(30))) throw std::runtime_error("NPU output timeout");
        const auto wait_done=Clock::now();
        std::ostringstream stream(std::ios::out|std::ios::binary);
        values[0].dump(stream,"SFB");const auto bytes=stream.str();
        if(bytes.size()!=c.out_bytes) throw std::runtime_error("SFB output length mismatch");
        std::memcpy(out,bytes.data(),bytes.size());const auto finished=Clock::now();
        detailed[0]=std::chrono::duration<double>(input_done-start).count();
        detailed[1]=std::chrono::duration<double>(submitted-input_done).count();
        detailed[2]=std::chrono::duration<double>(wait_done-submitted).count();
        detailed[3]=std::chrono::duration<double>(finished-wait_done).count();
        detailed[4]=std::chrono::duration<double>(finished-start).count();
        return 0;
    } catch(const std::exception& e) {error_text=e.what();return 1;}
}
// ABI 2 remains the production compatibility entry point. Its middle bucket
// preserves the historical execute+wait meaning used by old workers.
extern "C" int mgs_forward(void* ctx,const float* in,size_t ib,float* out,size_t ob,double* times) {
    double detailed[5]={0.,0.,0.,0.,0.};
    const int rc=forward_impl(ctx,in,ib,out,ob,detailed);
    if (!rc) {
        times[0]=detailed[0];
        times[1]=detailed[1]+detailed[2];
        times[2]=detailed[3];
    }
    return rc;
}
// Optional ABI-2 extension. This symbol is detected by the worker at runtime,
// so an older board library can still serve the three-bucket contract.
extern "C" int mgs_forward_detailed(void* ctx,const float* in,size_t ib,float* out,
                                     size_t ob,double* times) {
    return forward_impl(ctx,in,ib,out,ob,times);
}
