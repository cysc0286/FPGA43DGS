// Persistent-session bring-up harness. CPU emulation is a separate build.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifndef HGS_CPU_REFERENCE
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xir/core/network.h>
#include <icraft-xir/ops/hard_op.h>
using namespace icraft::xrt;
using namespace icraft::xir;
#endif
using Clock = std::chrono::steady_clock;
double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b-a).count();
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void read_exact(std::istream& f, void* dst, size_t bytes) {
    f.read(static_cast<char*>(dst), bytes);
    require(bool(f), "Truncated input file");
}
uint32_t read_u32(std::istream& f) {
    unsigned char x[4]; read_exact(f, x, 4);
    return uint32_t(x[0]) | (uint32_t(x[1])<<8) | (uint32_t(x[2])<<16) | (uint32_t(x[3])<<24);
}
void write_u32(std::ostream& f, uint32_t v) {
    unsigned char x[4]; for (int k=0;k<4;++k) x[k]=static_cast<unsigned char>(v>>(8*k));
    f.write(reinterpret_cast<char*>(x), 4);
}

int main(int argc, char** argv) try {
    require(argc==5, "Usage: block_runner model.json model.raw jobs.bin NEW_OUTPUT_DIRECTORY");
    const auto begin=Clock::now();
    const std::filesystem::path outdir(argv[4]);
    require(!std::filesystem::exists(outdir), "Output directory exists; preserve earlier evidence");
    const uint16_t endian=1;
    require(*reinterpret_cast<const uint8_t*>(&endian)==1 && sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
            "ABI requires little-endian IEEE FP32");
    std::ifstream jobs(argv[3], std::ios::binary);
    char magic[8]; read_exact(jobs, magic, 8);
    require(std::memcmp(magic,"HGSJOB01",8)==0, "Job magic/version mismatch");
    const auto tile=read_u32(jobs), count=read_u32(jobs), n=read_u32(jobs), m=read_u32(jobs);
    require(tile==32 || tile==64 || tile==128 || tile==256, "Unsupported tile");
    require(n>0 && m>0 && n<=100000 && m<=100000, "Bring-up job requires 1..100000 descriptors per image");
    const uint64_t expected_count=((uint64_t(n)+tile-1)/tile)*((uint64_t(m)+tile-1)/tile);
    require(count==expected_count && count<=1000000, "Block count mismatch or excessive job");
    const size_t input_bytes=size_t(tile)*128*sizeof(float), output_bytes=size_t(tile)*tile*sizeof(float);
    require(std::filesystem::file_size(argv[3])==24+uint64_t(count)*(16+2*input_bytes), "Job file size mismatch");
    std::filesystem::create_directories(outdir);
    std::ofstream profile(outdir/"placement.tsv");
    profile<<"op_id\top_name\top_type\tbackend\n";
    int hard_ops=0, matmul_ops=0;
    std::vector<float> left(tile*128), right(128*tile), scores(tile*tile);
#ifdef HGS_CPU_REFERENCE
    profile<<"-1\tscalar_reference\tCPU\tCPU\n";
    const bool npu=false;
    auto execute=[&]() {
        std::fill(scores.begin(), scores.end(), 0.f);
        for (size_t i=0;i<tile;++i) for (size_t j=0;j<tile;++j) {
            float value=0;
            for (size_t k=0;k<128;++k) value+=left[i*128+k]*right[k*tile+j];
            scores[i*tile+j]=value;
        }
    };
#else
    const bool npu=true;
    auto net=Network::CreateFromJsonFile(std::filesystem::path(argv[1]));
    net.lazyLoadParamsFromFile(std::filesystem::path(argv[2]));
    require(net.inputs().size()==2 && net.outputs().size()==1, "Network I/O count mismatch");
    const std::array<std::array<int64_t,4>,3> shapes={{{1,tile,128,1},{1,128,tile,1},{1,1,tile,tile}}};
    for (int i=0;i<3;++i) {
        auto type=(i<2 ? net.inputs()[i] : net.outputs()[0]).tensorType();
        require(type->element_dtype.isFP32() && type->shape.size()==4, "Expected rank-four FP32 host tensor");
        for (int k=0;k<4;++k) require(type->shape[k]==shapes[i][k], "Network shape mismatch");
        require(type.bytes()==(i<2 ? input_bytes : output_bytes), "Network byte size mismatch");
    }
    auto host=HostDevice::Default();
    auto dev=Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
    auto session=Session::Create<zg330::ZG330Backend,HostBackend>(net,{dev,host});
    session.apply();
    // SpeedMode merges generated HardOps. Attribute Matmul through its preserved
    // children rather than requiring an unfused top-level operator name.
    std::function<int(const HardOpNode*)> count_matmul = [&](const HardOpNode* hard) {
        int found=hard->name=="icraft::xir::MatmulNode" ? 1 : 0;
        for (const auto& child:hard->sub_hard_ops) {
            profile<<child->op_id<<'\t'<<child->name<<'\t'<<"merged_child"<<'\t'<<hard->op_id<<'\n';
            found+=count_matmul(child.as<HardOpNode>());
        }
        return found;
    };
    for (const auto& entry:session.getForwards()) {
        const auto& op=std::get<0>(entry); const auto& bound=std::get<1>(entry);
        profile<<op->op_id<<'\t'<<op->name<<'\t'<<op->typeKey()<<'\t'<<bound->typeKey()<<'\n';
        if (op->typeKey()=="icraft::xir::HardOpNode") {
            require(bound.is<zg330::ZG330Backend>(), "Hard operator fell back to host");
            ++hard_ops;
            matmul_ops+=count_matmul(op.as<HardOpNode>());
        } else {
            require(op->typeKey()=="icraft::xir::InputNode" || op->typeKey()=="icraft::xir::OutputNode",
                    "Unexpected runtime computation outside HardOp");
        }
    }
    profile.flush();
    require(matmul_ops==1 && hard_ops>0, "No unique NPU Matmul binding; refusing execution attribution");
    std::vector<Tensor> inputs;
    for (int i=0;i<2;++i) { Tensor t(net.inputs()[i]); t.mallocOn(host.defaultMemRegion()); inputs.push_back(t); }
#endif
    const auto applied=Clock::now();
    std::ofstream output(outdir/"scores.bin",std::ios::binary);
    output.write("HGSOUT01",8);
    for (auto value:{tile,count,n,m}) write_u32(output,value);
    std::ofstream timing(outdir/"timing.csv");
    timing<<std::setprecision(12)<<"sample,file_read_us,input_write_us,forward_call_us,wait_read_convert_us,file_write_us,block_total_us\n";
    double warmup_us=0, block_total_us=0;
    for (uint32_t c=0;c<count;++c) {
        const auto t0=Clock::now();
        const uint32_t i=read_u32(jobs), j=read_u32(jobs), ni=read_u32(jobs), nj=read_u32(jobs);
        const auto columns=(m+tile-1)/tile;
        require(i==(c/columns)*tile && j==(c%columns)*tile && ni==std::min(tile,n-i) && nj==std::min(tile,m-j),
                "Reordered or invalid block coordinates");
        read_exact(jobs,left.data(),input_bytes); read_exact(jobs,right.data(),input_bytes);
        for (float v:left) require(std::isfinite(v) && v>=0 && v<=255.f/512, "Invalid left descriptor");
        for (float v:right) require(std::isfinite(v) && v>=0 && v<=255.f/512, "Invalid right descriptor");
        const auto t1=Clock::now();
        auto write_inputs=[&]() {
#ifndef HGS_CPU_REFERENCE
            inputs[0].write(0,reinterpret_cast<char*>(left.data()),input_bytes);
            inputs[1].write(0,reinterpret_cast<char*>(right.data()),input_bytes);
#endif
        };
        auto forward=[&]() {
#ifdef HGS_CPU_REFERENCE
            execute();
            return 0;
#else
            return session.forward(inputs);
#endif
        };
        auto read_scores=[&](auto& result) {
#ifndef HGS_CPU_REFERENCE
            require(result.size()==1, "Output count changed");
            require(result[0].waitForReady(std::chrono::seconds(10)), "NPU output timeout");
            std::ostringstream buffer(std::ios::out|std::ios::binary);
            result[0].dump(buffer,"SFB");
            auto bytes=buffer.str();
            require(bytes.size()==output_bytes, "SFB output byte size mismatch");
            std::memcpy(scores.data(),bytes.data(),output_bytes);
#else
            (void)result;
#endif
            for (float v:scores) require(std::isfinite(v) && v>=0, "Invalid score output");
        };
        if (c==0) {
            const auto w0=Clock::now();
            for (int w=0;w<3;++w) { write_inputs(); auto result=forward(); read_scores(result); }
            warmup_us=us(w0,Clock::now());
        }
        const auto t2=Clock::now(); write_inputs();
        const auto t3=Clock::now(); auto result=forward();
        const auto t4=Clock::now(); read_scores(result);
        const auto t5=Clock::now();
        for (auto v:{i,j,ni,nj}) write_u32(output,v);
        output.write(reinterpret_cast<const char*>(scores.data()),output_bytes);
        require(bool(output), "Score file write failed");
        const auto t6=Clock::now();
        const double total=us(t0,t1)+us(t2,t6); // excludes warmup
        block_total_us+=total;
        timing<<c<<','<<us(t0,t1)<<','<<us(t2,t3)<<','<<us(t3,t4)<<','<<us(t4,t5)<<','<<us(t5,t6)<<','<<total<<'\n';
    }
    require(jobs.peek()==std::char_traits<char>::eof(), "Trailing input data");
    output.flush(); timing.flush(); profile.flush();
    require(bool(output) && bool(timing) && bool(profile), "Evidence write failed");
    const auto end=Clock::now();
    std::ofstream status(outdir/"status.json");
    status<<std::setprecision(12)<<"{\n  \"completed\": true,\n  \"npu_executed\": "<<(npu?"true":"false")
          <<",\n  \"blocks\": "<<count<<",\n  \"hard_ops\": "<<hard_ops<<",\n  \"matmul_ops\": "<<matmul_ops
          <<",\n  \"setup_apply_us\": "<<us(begin,applied)<<",\n  \"warmup_calls\": 3,\n  \"warmup_us\": "<<warmup_us
          <<",\n  \"measured_blocks_total_us\": "<<block_total_us<<",\n  \"process_body_us\": "<<us(begin,end)
          <<",\n  \"logical_input_bytes\": "<<uint64_t(count)*input_bytes*2
          <<",\n  \"logical_output_bytes\": "<<uint64_t(count)*output_bytes
          <<",\n  \"actual_dma_bytes\": null,\n  \"precision_verified\": false,\n"
          <<"  \"timing_scope\": \"file streaming plus SDK input/forward/wait/SFB output; no TopTwo or SfM; DMA and compute may overlap within SDK calls\"\n}\n";
    status.flush(); require(bool(status), "Status write failed");
    std::cout<<"blocks="<<count<<" npu_executed="<<npu<<"\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
