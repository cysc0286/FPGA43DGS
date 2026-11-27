// Native ICraft register access for the isolated J17 pan23/tilt21 candidate.
// Default is a dry run. No Linux GPIO numbering or /dev/mem access is used.
#include <icraft-xrt/dev/zg330_device.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include "camera_servo_scan.h"
#include "camera_servo_jog.h"
#include "camera_servo_stop.h"

namespace {
constexpr uint32_t BASE = 0x400c0300, MAGIC = 0x53525631;
constexpr uint32_t CONTROL = 0x10, PAN = 0x14, TILT = 0x18, HEARTBEAT = 0x1c;
volatile std::sig_atomic_t interrupted = 0;
void stop_requested(int) { interrupted = 1; }

class ScanLog {
    FILE* file = nullptr;
public:
    explicit ScanLog(const std::string& path) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("Cannot create new scan log; existing evidence is not overwritten");
        file = ::fdopen(fd, "w");
        if (!file) { ::close(fd); throw std::runtime_error("Cannot open scan log stream"); }
        if (std::fprintf(file,"plan_ms,elapsed_ms,event,requested_pan_us,requested_tilt_us,active_pan_us,active_tilt_us,frame_counter,status\n") < 0 ||
            std::fflush(file)) {
            std::fclose(file); file = nullptr;
            throw std::runtime_error("Cannot initialize scan log");
        }
    }
    void record(const camera_scan::Point& p, double elapsed_ms, uint32_t pan,
                uint32_t tilt, uint32_t frame, uint32_t status) {
        if (std::fprintf(file,"%u,%.3f,%s,%u,%u,%u,%u,%u,%u\n",p.time_ms,elapsed_ms,
                         p.event.c_str(),p.pan_us,p.tilt_us,pan,tilt,frame,status) < 0 || std::fflush(file))
            throw std::runtime_error("Scan log write failed; disabling PWM");
    }
    ~ScanLog() { if (file) std::fclose(file); }
    ScanLog(const ScanLog&) = delete;
    ScanLog& operator=(const ScanLog&) = delete;
};

class Servo {
    int lock_fd = -1;
    icraft::xrt::Device device;
    icraft::xrt::ZG330Device native;
    bool opened = false, verified = false, may_be_enabled = false;
    uint32_t servo_abi = 0;
    void close_noexcept() noexcept {
        if (opened) {
            try { icraft::xrt::Device::Close(device); } catch (...) {}
            opened = false;
        }
        if (lock_fd >= 0) { ::close(lock_fd); lock_fd = -1; }
    }
public:
    explicit Servo(bool read_legacy_status=false) {
        lock_fd = ::open("/run/lock/fpga43dgs-gsc1.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0) throw std::runtime_error("Cannot open board ownership lock");
        if (::flock(lock_fd, LOCK_EX | LOCK_NB)) {
            close_noexcept();
            throw std::runtime_error("Board is in use; no register was written");
        }
        try {
            device = icraft::xrt::Device::Open("axi://zg330aiu?npu=0x40000000&dma=0x80000000");
            opened = true;
            native = device.cast<icraft::xrt::ZG330Device>();
            const auto renderer_id = native.defaultRegRegion().read(0x400c0200, false);
            const auto renderer_abi = native.defaultRegRegion().read(0x400c0204, false);
            std::cout << "Renderer ID=0x" << std::hex << renderer_id
                      << " ABI=0x" << renderer_abi << std::dec << '\n';
            if (renderer_id != 0x464c4b31 || renderer_abi != 0x20000)
                throw std::runtime_error("Frozen FLK1 renderer ABI is absent; no application register was written");
            const auto id = read(0);
            if (id != MAGIC) {
                std::cerr << "Observed servo-window ID 0x" << std::hex << id << std::dec << '\n';
                throw std::runtime_error("SRV1 is absent; no application register was written");
            }
            servo_abi = read(4);
            if ((servo_abi != 3 && !(read_legacy_status && servo_abi == 2)) || read(8) != 100000000 || read(0x0c) != 20000 ||
                read(0x30) != 500000 || read(0x34) != (500u | 2500u << 16) ||
                read(0x38) != (1000u | 2000u << 16)) {
                throw std::runtime_error("Servo ABI3/clock/period/watchdog/limits required; ABI2 is status-only after stop-pulse incident; no application register was written");
            }
            // Check that the real clock is running at the expected rate before
            // arming. ABI parameters alone cannot establish the physical rate.
            const auto before = read(0x24);
            const auto start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const auto after = read(0x24);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double frames_per_second = static_cast<uint32_t>(after - before) / elapsed;
            std::cout << "Observed frame-counter rate=" << frames_per_second << " Hz (software sampled)\n";
            if (frames_per_second < 40.0 || frames_per_second > 60.0)
                throw std::runtime_error("Frame-counter rate inconsistent with 50Hz; no application register was written");
            if (!read_legacy_status && (read(CONTROL) || (read(0x20) & 3)))
                throw std::runtime_error("Outputs are already enabled; no application register was written");
            verified = true;
        } catch (...) { close_noexcept(); throw; }
    }
    Servo(const Servo&) = delete;
    Servo& operator=(const Servo&) = delete;
    uint32_t read(uint32_t offset) { return native.defaultRegRegion().read(BASE + offset, false); }
    void write(uint32_t offset, uint32_t value) {
        if (!verified) throw std::runtime_error("Servo identity was not verified");
        if (servo_abi != 3) throw std::runtime_error("ABI2 permits status only; no application register was written");
        if (interrupted && !(offset == CONTROL && value == 0))
            throw std::runtime_error("Interrupted; disabling PWM");
        if ((offset == PAN && (value < 500 || value > 2500)) ||
            (offset == TILT && (value < 1000 || value > 2000)) ||
            (offset == CONTROL && value > 3) ||
            (offset == camera_stop::REQUEST && value != 1) ||
            (offset != PAN && offset != TILT && offset != CONTROL && offset != HEARTBEAT && offset != camera_stop::REQUEST)) {
            throw std::runtime_error("Rejected unsafe register request before write");
        }
        // Mark before the SDK call so a partially completed enable is cleaned up.
        if (offset == CONTROL && value) may_be_enabled = true;
        native.defaultRegRegion().write(BASE + offset, value, false);
        if (offset == CONTROL && !value) may_be_enabled = false;
    }
    void pause_frame() {
        if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
        write(HEARTBEAT, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
        if (read(0x20) & 0x100) throw std::runtime_error("Hardware heartbeat expired; disabling PWM");
    }
    void stop() {
        camera_stop::finish(*this, [&]() {
            if (interrupted) throw std::runtime_error("Interrupted during stop; disabling PWM");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (interrupted) throw std::runtime_error("Interrupted during stop; disabling PWM");
        });
        may_be_enabled = false;
    }
    void status() {
        std::cout << "SRV1 ABI" << servo_abi << ", 100MHz, 50Hz, pan J17/23, tilt J17/21\n";
        for (uint32_t offset : {0x10u, 0x14u, 0x18u, 0x20u, 0x24u, 0x28u, 0x2cu})
            std::cout << "offset=0x" << std::hex << offset << " value=0x" << read(offset) << std::dec << '\n';
        std::cout << "status read only; no application register was written\n";
    }
    void smoke(bool tilt) {
        const uint32_t pulse_register = tilt ? TILT : PAN;
        write(CONTROL, 0);
        write(PAN, 1500);
        write(TILT, 1500);
        write(HEARTBEAT, 1);
        write(CONTROL, tilt ? 2 : 1);
        int current = 1500;
        for (const int target : {1500, 1450, 1500, 1550, 1500}) {
            while (current != target) {
                const int difference = target - current;
                current += difference > 2 ? 2 : difference < -2 ? -2 : difference;
                write(pulse_register, static_cast<uint32_t>(current));
                pause_frame();
            }
            std::cout << "requested_pulse_us=" << current << std::endl;
            for (int frame = 0; frame < 15; ++frame) pause_frame();
        }
        stop();
        std::cout << "Command sequence complete; PWM disabled. Physical movement requires observation.\n";
    }
    void jog(bool tilt, uint32_t from, uint32_t to) {
        const auto wait = [&]() {
            if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
        };
        std::cout << "JOG_BEGIN axis=" << (tilt ? "tilt" : "pan")
                  << " from_us=" << from << " target_us=" << to
                  << " slew_us_per_20ms=2 hold_seconds=5" << std::endl;
        camera_jog::execute(tilt, from, to, *this, wait);
        std::cout << "JOG_END target_us=" << to
                  << "; PWM disabled. Installed angle requires observation." << std::endl;
    }
    void scan(const std::vector<camera_scan::Point>& points, ScanLog& log) {
        using Clock = std::chrono::steady_clock;
        if (interrupted) throw std::runtime_error("Interrupted before scan");
        const auto started = Clock::now();
        auto next_update_not_before = started;
        unsigned late_intervals = 0;
        const auto wait = [&]() {
            if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
            std::this_thread::sleep_until(next_update_not_before);
            const auto now = Clock::now();
            // Never chase a missed deadline with back-to-back position writes.
            if (now > next_update_not_before + std::chrono::milliseconds(camera_scan::PERIOD_MS)) {
                ++late_intervals;
            }
            if (interrupted) throw std::runtime_error("Interrupted; disabling PWM");
        };
        const auto trace = [&](const camera_scan::Point& p) {
            next_update_not_before = Clock::now() + std::chrono::milliseconds(camera_scan::PERIOD_MS);
            const auto pan = read(0x28), tilt = read(0x2c), frame = read(0x24), status = read(0x20);
            const double elapsed = std::chrono::duration<double,std::milli>(Clock::now()-started).count();
            log.record(p,elapsed,pan,tilt,frame,status);
            if (p.event == "capture")
                std::cout << "capture_marker plan_ms=" << p.time_ms << " requested_pan_us=" << p.pan_us
                          << " requested_tilt_us=" << p.tilt_us << std::endl;
        };
        camera_scan::execute(points,*this,wait,trace);
        std::cout << "Scan complete; PWM disabled. elapsed_seconds="
                  << std::chrono::duration<double>(Clock::now()-started).count()
                  << " late_intervals=" << late_intervals
                  << ". Requested positions are not measured mechanical angles.\n";
    }
    ~Servo() {
        if (may_be_enabled && verified) {
            try { write(CONTROL, 0); }
            catch (...) { std::cerr << "Disable failed; hardware heartbeat timeout remains the fallback\n"; }
        }
        close_noexcept();
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "jog") {
            const bool execute = argc == 7 && std::string(argv[5]) == "--execute" &&
                std::string(argv[6]) == "--wiring-j17-23-21";
            if ((argc != 5 && !execute) ||
                (std::string(argv[2]) != "pan" && std::string(argv[2]) != "tilt"))
                throw std::runtime_error("Usage: camera_servo_native jog pan|tilt FROM_US TO_US [--execute --wiring-j17-23-21]");
            const bool tilt = std::string(argv[2]) == "tilt";
            const auto from = camera_scan::unsigned_decimal(argv[3]);
            const auto to = camera_scan::unsigned_decimal(argv[4]);
            camera_jog::validate(tilt, from, to); // reject before opening hardware
            if (!execute) {
                std::cout << "DRY RUN; no device opened. Jog " << (tilt ? "tilt" : "pan")
                          << " from_us=" << from << " target_us=" << to
                          << "; slew <=2us/20ms, hold 5s, then disable PWM.\n";
                return 0;
            }
            std::signal(SIGINT, stop_requested);
            std::signal(SIGTERM, stop_requested);
            Servo servo;
            servo.jog(tilt, from, to);
            return 0;
        }
        if (argc >= 2 && std::string(argv[1]) == "scan") {
            const bool execute = argc == 7 && std::string(argv[3]) == "--execute" &&
                std::string(argv[4]) == "--wiring-j17-23-21" && std::string(argv[5]) == "--log";
            if (argc != 3 && !execute)
                throw std::runtime_error("Usage: camera_servo_native scan PLAN.csv [--execute --wiring-j17-23-21 --log NEW.csv]");
            std::ifstream file(argv[2]);
            if (!file) throw std::runtime_error("Cannot open scan plan");
            const auto points = camera_scan::read(file); // whole plan checked BEFORE device access
            std::cout << "Scan samples=" << points.size() << " nominal_seconds="
                      << points.size()*camera_scan::PERIOD_MS/1000.0 << '\n';
            if (!execute) {
                std::cout << "DRY RUN; no device opened. Angles require position-servo calibration.\n";
                return 0;
            }
            std::signal(SIGINT, stop_requested);
            std::signal(SIGTERM, stop_requested);
            ScanLog log(argv[6]);
            Servo servo;
            servo.scan(points,log);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "camera_servo_native [dry-run|status]\n"
                      << "camera_servo_native smoke pan|tilt --execute --wiring-j17-23-21\n"
                      << "camera_servo_native jog pan|tilt FROM_US TO_US [--execute --wiring-j17-23-21]\n"
                      << "camera_servo_native scan PLAN.csv [--execute --wiring-j17-23-21 --log NEW.csv]\n";
            return 0;
        }
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "dry-run")) {
            std::cout << "DRY RUN; no device opened\n"
                      << "Wiring: pan J17/23; tilt J17/21; common ground J17/2\n"
                      << "Single axis: 1500 -> 1450 -> 1500 -> 1550 -> 1500 us\n"
                      << "Slew <=2us/20ms; nominal pitch limit +/-45 degrees; PWM stops at exit\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "status") { Servo servo(true); servo.status(); return 0; }
        if (argc != 5 || std::string(argv[1]) != "smoke" ||
            (std::string(argv[2]) != "pan" && std::string(argv[2]) != "tilt") ||
            std::string(argv[3]) != "--execute" || std::string(argv[4]) != "--wiring-j17-23-21")
            throw std::runtime_error("Usage: camera_servo_native [dry-run|status|smoke pan|tilt --execute --wiring-j17-23-21]");
        std::signal(SIGINT, stop_requested);
        std::signal(SIGTERM, stop_requested);
        Servo servo;
        servo.smoke(std::string(argv[2]) == "tilt");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
