#include "camera_servo_jog.h"
#include <iostream>
#include <utility>

struct FakeDevice {
    std::vector<std::pair<uint32_t,uint32_t>> writes;
    uint32_t control=0, status=0x100, pan=1500, tilt=1500;
    bool stop_fails=false;
    unsigned normal_stops=0;
    void write(uint32_t offset, uint32_t value) {
        writes.emplace_back(offset,value);
        if (offset==camera_scan::CONTROL) { control=value; status=(status&0x100)|value; }
        if (offset==camera_scan::HEARTBEAT) status &= ~0x100u;
        if (offset==camera_scan::PAN) pan=value;
        if (offset==camera_scan::TILT) tilt=value;
    }
    uint32_t read(uint32_t offset) {
        if (offset==camera_scan::CONTROL) return control;
        if (offset==0x28) return pan;
        if (offset==0x2c) return tilt;
        return status;
    }
    void stop() { ++normal_stops;if(stop_fails)throw std::runtime_error("stop ack missing");write(camera_scan::CONTROL,0); }
};
int checks=0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class Call> void rejects(Call call, const char* message) {
    bool failed=false;
    try { call(); } catch(const std::exception&) { failed=true; }
    check(failed,message);
}
int main() {
    try {
        FakeDevice device;
        unsigned waits=0;
        camera_jog::execute(true,1500,1600,device,[&](){++waits;});
        check(waits==301,"paced slew plus five-second hold");
        check(device.tilt==1600 && device.pan==1500 && device.control==0,"target retained and PWM disabled");
        check(device.normal_stops==1,"normal jog uses complete-pulse stop");
        uint32_t last=1500;
        for (const auto& item:device.writes) {
            check(item.first!=camera_scan::PAN,"other axis must not be written");
            if (item.first==camera_scan::CONTROL) check(item.second==0 || item.second==2,"tilt-only enable");
            if (item.first==camera_scan::TILT) {
                check(item.second>=last && item.second-last<=2,"slew bounded to 2us");
                last=item.second;
            }
        }
        FakeDevice reverse;
        reverse.pan=1600;
        camera_jog::execute(false,1600,1599,reverse,[](){});
        check(reverse.pan==1599 && reverse.tilt==1500 && !reverse.control,"odd reverse step and pan-only control");
        for (const auto& pair: {std::make_pair(1500u,1601u),std::make_pair(999u,1000u),
                                std::make_pair(2000u,2001u),std::make_pair(UINT32_MAX,1500u)}) {
            FakeDevice invalid;
            rejects([&](){camera_jog::execute(true,pair.first,pair.second,invalid,[](){});},"invalid jog rejected");
            check(invalid.writes.empty(),"invalid jog writes nothing");
        }
        FakeDevice mismatch;
        mismatch.tilt=1490;
        rejects([&](){camera_jog::execute(true,1500,1550,mismatch,[](){});},"starting pulse mismatch rejected");
        check(mismatch.writes.empty(),"mismatch writes nothing");
        FakeDevice busy;
        busy.control=1;
        rejects([&](){camera_jog::execute(true,1500,1550,busy,[](){});},"active outputs rejected");
        check(busy.writes.empty(),"active output check writes nothing");
        FakeDevice cancelled;
        rejects([&](){camera_jog::execute(true,1500,1550,cancelled,[](){throw std::runtime_error("cancelled");});},"cancellation propagated");
        check(cancelled.control==0,"cancellation disables PWM");
        FakeDevice expired;
        unsigned ticks=0;
        rejects([&](){camera_jog::execute(true,1500,1550,expired,[&](){++ticks;expired.status|=0x100;});},"watchdog expiry propagated");
        check(ticks==1 && expired.control==0,"watchdog not silently refreshed");
        FakeDevice last_expired;
        unsigned last_ticks=0;
        rejects([&](){camera_jog::execute(true,1500,1500,last_expired,[&](){if(++last_ticks==251)last_expired.status|=0x100;});},"final hold watchdog checked");
        check(last_expired.control==0,"final hold failure disables PWM");
        FakeDevice stop_failure;stop_failure.stop_fails=true;
        rejects([&](){camera_jog::execute(true,1500,1500,stop_failure,[](){});},"stop acknowledgement failure propagates");
        check(!stop_failure.control,"stop failure uses immediate abort");
        std::cout << "PASS: camera_servo_jog " << checks << " checks; mock device only\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
