// Bounded single-axis positioning for observing and calibrating installed neutral.
#ifndef CAMERA_SERVO_JOG_H
#define CAMERA_SERVO_JOG_H
#include "camera_servo_scan.h"

namespace camera_jog {
constexpr uint32_t MAX_DISTANCE_US = 100, STEP_US = 2, HOLD_FRAMES = 250;
inline void validate(bool tilt, uint32_t from, uint32_t to) {
    const uint32_t low = tilt ? 1000 : 500, high = tilt ? 2000 : 2500;
    if (from < low || from > high || to < low || to > high)
        throw std::runtime_error("Jog pulse exceeds hardware limits");
    const uint32_t distance = from > to ? from-to : to-from;
    if (distance > MAX_DISTANCE_US)
        throw std::runtime_error("Jog is limited to 100us per observed step");
}

// The caller supplies the last observed pulse. A mismatch or active output is
// rejected before writes. The target is held for 5s, then pulses stop without
// an implicit return to 1500us. Register values are not mechanical feedback.
template<class Device, class Wait>
void execute(bool tilt, uint32_t from, uint32_t to, Device& device, Wait wait) {
    using namespace camera_scan;
    validate(tilt, from, to);
    const uint32_t pulse_register = tilt ? TILT : PAN;
    const uint32_t active_register = tilt ? 0x2c : 0x28;
    if (device.read(CONTROL) || (device.read(STATUS) & 3))
        throw std::runtime_error("Jog requires disabled outputs before starting");
    if (device.read(active_register) != from)
        throw std::runtime_error("Jog starting pulse differs from active readback; inspect status first");
    try {
        device.write(CONTROL, 0);
        device.write(pulse_register, from);
        device.write(HEARTBEAT, 1);
        device.write(CONTROL, tilt ? 2 : 1);
        const auto tick = [&]() {
            if (device.read(STATUS) & 0x100)
                throw std::runtime_error("Hardware watchdog expired; jog aborted");
            device.write(HEARTBEAT, 1);
            wait();
            if (device.read(STATUS) & 0x100)
                throw std::runtime_error("Hardware watchdog expired during jog hold");
        };
        // Let the initial enable latch before adjusting the pulse.
        tick();
        uint32_t current = from;
        while (current != to) {
            if (device.read(STATUS) & 0x100)
                throw std::runtime_error("Hardware watchdog expired before jog step");
            const uint32_t distance = current > to ? current-to : to-current;
            const uint32_t step = distance > STEP_US ? STEP_US : distance;
            current = current < to ? current+step : current-step;
            device.write(pulse_register, current);
            tick();
        }
        for (uint32_t frame = 0; frame < HOLD_FRAMES; ++frame) tick();
        if (device.read(active_register) != to)
            throw std::runtime_error("Jog target pulse did not latch");
        device.stop(); // complete-pulse stop; exceptions below still abort now
    } catch (...) {
        try { device.write(CONTROL, 0); } catch (...) {}
        throw;
    }
}
} // namespace camera_jog
#endif
