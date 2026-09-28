// Real FPGA register test. Expected values come from the independent Python oracle.
#include <icraft-xrt/dev/zg330_device.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Vector { uint32_t op, a, b; uint64_t expected; uint32_t error; };

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: ./test_basic_alu vectors.txt");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open vector file");
        std::vector<Vector> vectors;
        std::string line, extra;
        while (std::getline(input, line)) {
            uint64_t op, a, b, expected, error;
            std::istringstream fields(line);
            if (!(fields >> std::hex >> op >> a >> b >> expected >> error) ||
                (fields >> extra) || op > UINT32_MAX || a > UINT32_MAX ||
                b > UINT32_MAX || error > 1) {
                throw std::runtime_error("Malformed vector at line " + std::to_string(vectors.size()+1));
            }
            vectors.push_back({uint32_t(op), uint32_t(a), uint32_t(b), expected, uint32_t(error)});
        }
        if (input.bad() || vectors.size() != 2800)
            throw std::runtime_error("Expected the complete 2800-vector regression corpus");

        auto device = icraft::xrt::Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
        try {
            auto fpai = device.cast<icraft::xrt::ZG330Device>();
            constexpr uint32_t base = 0x400c0000;
            auto read = [&](uint32_t offset) -> uint32_t {return fpai.defaultRegRegion().read(base + offset, false);};
            auto write = [&](uint32_t offset, uint32_t value) {fpai.defaultRegRegion().write(base + offset, value, false);};
            auto capability = read(0x94);
            auto version = read(0xc0);
            std::cout << "ALU_CAPABILITY=0x" << std::hex << capability
                      << " ADDER_VERSION=0x" << version << std::dec << '\n';
            if (capability != 0x414c5531 || version != 0x20230628)
                throw std::runtime_error("ALU1 test bitstream is not loaded; no commands were submitted");
            if (read(0x90) & 1) throw std::runtime_error("ALU busy before test; refusing concurrent access");
            uint32_t sequence = read(0x8c);
            std::array<unsigned, 13> passed{};
            const std::array<const char*, 13> names = {"ADD32", "SUB32", "MUL_U32", "MUL_S32", "AND", "OR", "XOR",
                "SHL32", "SHR32", "SAR32", "MIN_S32", "MAX_S32", "INVALID_OPCODE"};
            auto started = std::chrono::steady_clock::now();
            for (size_t i = 0; i < vectors.size(); ++i) {
                const auto& v = vectors[i];
                if (++sequence == 0) ++sequence;
                write(0x0c, v.a);
                write(0x10, v.b);
                write(0x14, v.op);
                write(0x20, sequence); // commit last; exactly one request outstanding
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                while (read(0x8c) != sequence) {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("FPGA completion timeout at vector " + std::to_string(i));
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
                uint64_t result = read(0x84);
                result |= uint64_t(read(0x88)) << 32;
                auto status = read(0x90);
                if (result != v.expected || status != (v.error << 1) || read(0x8c) != sequence) {
                    std::ostringstream message;
                    message << "Mismatch vector=" << i << std::hex << " op=" << v.op
                            << " a=" << v.a << " b=" << v.b << " result=" << result
                            << " expected=" << v.expected << " status=" << status
                            << " expected_error=" << v.error;
                    throw std::runtime_error(message.str());
                }
                ++passed[v.op < 12 ? v.op : 12];
            }
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            for (size_t i = 0; i < passed.size(); ++i)
                std::cout << names[i] << ": " << passed[i] << " passed\n";
            std::cout << "PASS: FPGA basic_alu 2800/2800 exact matches\n"
                      << "HOST_REGISTER_TEST_SECONDS=" << elapsed << '\n'
                      << "Timing includes SDK register IO and polling; this is not FPGA kernel latency.\n";
        } catch (...) {
            icraft::xrt::Device::Close(device);
            throw;
        }
        icraft::xrt::Device::Close(device);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
