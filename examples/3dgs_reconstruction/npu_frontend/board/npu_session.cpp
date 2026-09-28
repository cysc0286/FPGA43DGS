// In-process dynamic Matmul bridge. No task files or per-block process launch.
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xir/core/network.h>
#include <icraft-xir/ops/hard_op.h>
#include <chrono>
#include <memory>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace icraft::xrt;
using namespace icraft::xir;
using Clock=std::chrono::steady_clock;
static thread_local std::string last_error;
static int matmuls(const HardOpNode* op) {
    int result=op->name=="icraft::xir::MatmulNode" ? 1 : 0;
    for (const auto& child:op->sub_hard_ops) result+=matmuls(child.as<HardOpNode>());
    return result;
}
struct HgsContext {
    HostDevice host;
    Device device;
    Network network;
    Session session;
    std::vector<Tensor> inputs;
    size_t in_bytes=0, out_bytes=0;
    int hard_ops=0, matmul_ops=0;
    HgsContext(const char* json, const char* raw, int tile) {
        if (tile!=32 && tile!=64 && tile!=128 && tile!=256) throw std::runtime_error("Invalid tile");
        network=Network::CreateFromJsonFile(std::filesystem::path(json));
        network.lazyLoadParamsFromFile(std::filesystem::path(raw));
        in_bytes=size_t(tile)*128*4; out_bytes=size_t(tile)*tile*4;
        if (network.inputs().size()!=2 || network.outputs().size()!=1) throw std::runtime_error("I/O count mismatch");
        for (int k=0;k<3;++k) {
            auto type=(k<2 ? network.inputs()[k] : network.outputs()[0]).tensorType();
            if (!type->element_dtype.isFP32() || type.bytes()!=(k<2 ? in_bytes : out_bytes))
                throw std::runtime_error("FP32 host ABI mismatch");
        }
        host=HostDevice::Default();
        device=Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
        session=Session::Create<zg330::ZG330Backend,HostBackend>(network,{device,host});
        session.apply();
        for (const auto& entry:session.getForwards()) {
            const auto& op=std::get<0>(entry);const auto& bound=std::get<1>(entry);
            if (op->typeKey()=="icraft::xir::HardOpNode") {
                if (!bound.is<zg330::ZG330Backend>()) throw std::runtime_error("Host fallback");
                ++hard_ops; matmul_ops+=matmuls(op.as<HardOpNode>());
            } else if (op->typeKey()!="icraft::xir::InputNode" && op->typeKey()!="icraft::xir::OutputNode") {
                throw std::runtime_error("Unexpected runtime compute");
            }
        }
        if (hard_ops<1 || matmul_ops!=1) throw std::runtime_error("NPU Matmul binding missing");
        for (const auto& value:network.inputs()) {
            Tensor t(value);t.mallocOn(host.defaultMemRegion());inputs.push_back(t);
        }
    }
};
extern "C" const char* hgs_error() { return last_error.c_str(); }
extern "C" void* hgs_create(const char* json,const char* raw,int tile) {
    try { return new HgsContext(json,raw,tile); }
    catch(const std::exception& e) { last_error=e.what();return nullptr; }
}
extern "C" void hgs_destroy(void* ptr) { delete static_cast<HgsContext*>(ptr); }
extern "C" int hgs_binding(void* ptr) { return static_cast<HgsContext*>(ptr)->matmul_ops; }
extern "C" int hgs_dot(void* ptr,float* left,float* right,float* scores,double* timing) {
    try {
        auto& c=*static_cast<HgsContext*>(ptr);
        const auto t0=Clock::now();
        c.inputs[0].write(0,reinterpret_cast<char*>(left),c.in_bytes);
        c.inputs[1].write(0,reinterpret_cast<char*>(right),c.in_bytes);
        const auto t1=Clock::now();auto output=c.session.forward(c.inputs);const auto t2=Clock::now();
        if (output.size()!=1 || !output[0].waitForReady(std::chrono::seconds(10))) throw std::runtime_error("Output timeout");
        // Even nominal FP32 graph outputs can retain SDK-managed conversion.
        // Raw Tensor::read failed the real score oracle; use the verified SFB path.
        std::ostringstream buffer(std::ios::out|std::ios::binary);
        output[0].dump(buffer,"SFB");
        const auto bytes=buffer.str();
        if (bytes.size()!=c.out_bytes) throw std::runtime_error("Converted output size mismatch");
        std::memcpy(scores,bytes.data(),c.out_bytes);
        const auto t3=Clock::now();
        timing[0]=std::chrono::duration<double>(t1-t0).count();
        timing[1]=std::chrono::duration<double>(t2-t1).count();
        timing[2]=std::chrono::duration<double>(t3-t2).count();
        return 0;
    } catch(const std::exception& e) { last_error=e.what();return 1; }
}
