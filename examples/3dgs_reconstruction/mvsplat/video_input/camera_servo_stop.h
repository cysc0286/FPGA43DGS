// ABI3 normal completion keeps the final PWM pulse intact. Faults still abort.
#ifndef CAMERA_SERVO_STOP_H
#define CAMERA_SERVO_STOP_H
#include <cstdint>
#include <stdexcept>
namespace camera_stop {
constexpr uint32_t REQUEST = 0x3c;
// Wait supplies a paced 1ms poll and must honor cancellation. Hardware applies
// REQUEST at a frame boundary; a bus failure must not become an unbounded hold.
template<class Device, class Wait>
void finish(Device& device, Wait wait) {
    device.write(REQUEST, 1);
    for (unsigned poll=0; poll<50; ++poll) {
        const auto status=device.read(0x20);
        if (status & 0x100)
            throw std::runtime_error("Watchdog expired while waiting for complete-pulse stop");
        if (!device.read(0x10) && !(status & 3)) return;
        wait();
    }
    throw std::runtime_error("Complete-pulse stop did not acknowledge within 50 polls");
}
} // namespace camera_stop
#endif
