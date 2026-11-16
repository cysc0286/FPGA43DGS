// Board-independent trajectory contract shared by the ARM runner and host tests.
#ifndef CAMERA_SERVO_SCAN_H
#define CAMERA_SERVO_SCAN_H
#include <cstdint>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace camera_scan {
constexpr uint32_t CONTROL = 0x10, PAN = 0x14, TILT = 0x18;
constexpr uint32_t HEARTBEAT = 0x1c, STATUS = 0x20;
constexpr uint32_t PERIOD_MS = 20, MAX_STEP_US = 10, MAX_ROWS = 180000;
struct Point {
    uint32_t time_ms, pan_us, tilt_us;
    std::string event;
};
inline void validate(const std::vector<Point>& points) {
    if (points.size() < 2 || points.size() > MAX_ROWS)
        throw std::runtime_error("Scan must contain 2..180000 samples (at most one hour)");
    if (points.front().pan_us != 1500 || points.front().tilt_us != 1500 ||
        points.front().event != "start" || points.back().event != "end")
        throw std::runtime_error("Scan must start at calibrated 1500us center and have start/end markers");
    for (size_t i = 0; i < points.size(); ++i) {
        const Point& p = points[i];
        if (p.time_ms != i * PERIOD_MS || p.pan_us < 500 || p.pan_us > 2500 ||
            p.tilt_us < 1000 || p.tilt_us > 2000)
            throw std::runtime_error("Invalid time grid or hardware pulse limit in scan");
        if (p.event != "start" && p.event != "move" && p.event != "settle" &&
            p.event != "capture" && p.event != "home" && p.event != "end")
            throw std::runtime_error("Unknown scan event");
        if (i) {
            const Point& last = points[i - 1];
            const uint32_t dp = p.pan_us > last.pan_us ? p.pan_us-last.pan_us : last.pan_us-p.pan_us;
            const uint32_t dt = p.tilt_us > last.tilt_us ? p.tilt_us-last.tilt_us : last.tilt_us-p.tilt_us;
            if (dp > MAX_STEP_US || dt > MAX_STEP_US)
                throw std::runtime_error("Scan exceeds 10us slew per 20ms sample");
        }
    }
}
inline uint32_t unsigned_decimal(const std::string& value) {
    if (value.empty() || value.size() > 10 || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Scan numbers must be unsigned decimal integers");
    const unsigned long long number = std::stoull(value);
    if (number > UINT32_MAX) throw std::runtime_error("Scan integer overflow");
    return static_cast<uint32_t>(number);
}
inline std::vector<Point> read(std::istream& input) {
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty scan file");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "time_ms,pan_us,tilt_us,event") throw std::runtime_error("Scan CSV header mismatch");
    std::vector<Point> points;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() > 100) throw std::runtime_error("Scan CSV line too long");
        std::stringstream row(line);
        std::string t, pan, tilt, event;
        if (!std::getline(row,t,',') || !std::getline(row,pan,',') ||
            !std::getline(row,tilt,',') || !std::getline(row,event) || event.find(',') != std::string::npos)
            throw std::runtime_error("Scan CSV requires exactly four fields");
        points.push_back(Point{unsigned_decimal(t),unsigned_decimal(pan),unsigned_decimal(tilt),event});
        if (points.size() > MAX_ROWS) throw std::runtime_error("Scan is longer than one hour");
    }
    if (input.bad()) throw std::runtime_error("Failed reading scan file");
    validate(points);
    return points;
}

// Device provides read/write. Wait supplies one paced 20ms interval and throws
// on cancellation. Trace can persist requested and observed register values.
// Validation precedes every application write. No catch-up bursts are used.
template<class Device, class Wait, class Trace>
void execute(const std::vector<Point>& points, Device& device, Wait wait, Trace trace) {
    validate(points);
    try {
        device.write(CONTROL,0);
        device.write(PAN,points.front().pan_us);
        device.write(TILT,points.front().tilt_us);
        device.write(HEARTBEAT,1);
        device.write(CONTROL,3);
        for (const Point& point : points) {
            // Read BEFORE refreshing: a stalled CPU must not silently re-arm
            // a watchdog timeout and continue a physically interrupted scan.
            if (device.read(STATUS) & 0x100)
                throw std::runtime_error("Hardware watchdog expired; scan aborted");
            device.write(PAN,point.pan_us);
            device.write(TILT,point.tilt_us);
            device.write(HEARTBEAT,1);
            trace(point);
            wait();
            if (device.read(STATUS) & 0x100)
                throw std::runtime_error("Hardware watchdog expired during sample hold; scan aborted");
        }
        device.write(CONTROL,0);
        if (device.read(CONTROL) || (device.read(STATUS) & 3))
            throw std::runtime_error("PWM disable readback failed");
    } catch (...) {
        try { device.write(CONTROL,0); } catch (...) {}
        throw;
    }
}
} // namespace camera_scan
#endif
