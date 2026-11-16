#include "camera_servo_scan.h"
#include <fstream>
#include <iostream>
#include <utility>

struct FakeDevice {
    std::vector<std::pair<uint32_t,uint32_t>> writes;
    uint32_t control=0, status=0;
    void write(uint32_t offset, uint32_t value) {
        writes.push_back(std::make_pair(offset,value));
        if (offset==camera_scan::CONTROL) { control=value; status=(status&0x100)|value; }
        if (offset==camera_scan::HEARTBEAT) status &= ~0x100u;
    }
    uint32_t read(uint32_t offset) { return offset==camera_scan::CONTROL ? control : status; }
};
int checks=0;
void check(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class Call> void rejects(Call call, const std::string& name) {
    bool threw=false;
    try { call(); } catch(const std::exception&) { threw=true; }
    check(threw,name);
}
int main(int argc,char** argv) {
    try {
        const std::string good="time_ms,pan_us,tilt_us,event\n0,1500,1500,start\n20,1502,1501,move\n40,1500,1500,end\n";
        std::stringstream stream(good);
        auto plan=camera_scan::read(stream);
        check(plan.size()==3,"valid parser");
        std::stringstream crlf("time_ms,pan_us,tilt_us,event\r\n0,1500,1500,start\r\n20,1500,1500,end\r\n");
        check(camera_scan::read(crlf).size()==2,"CRLF parser");
        for (const auto& bad : {
            std::string("bad header\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,1500,1500,start\n20,1500,2001,end\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,1500,1500,start\n20,1520,1500,end\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,1500,1500,start\n21,1500,1500,end\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,-1,1500,start\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,1500,1500,start,extra\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,1500,1500,start\n20,1500,1500,unknown\n"),
            std::string("time_ms,pan_us,tilt_us,event\n0,4294967296,1500,start\n")}) {
            rejects([&](){std::stringstream input(bad);camera_scan::read(input);},"reject malformed plan");
        }
        FakeDevice device;
        int ticks=0,traces=0;
        camera_scan::execute(plan,device,[&](){++ticks;},[&](const camera_scan::Point&){++traces;});
        check(ticks==3 && traces==3,"one wait and trace per sample");
        check(device.writes.front()==std::make_pair(camera_scan::CONTROL,0u),"disable before arming");
        check(device.control==0 && device.writes.back()==std::make_pair(camera_scan::CONTROL,0u),"disable on completion");
        auto bad_plan=plan;bad_plan[1].tilt_us=2500;
        FakeDevice untouched;
        rejects([&](){camera_scan::execute(bad_plan,untouched,[](){},[](const camera_scan::Point&){});},"validate before write");
        check(untouched.writes.empty(),"invalid plan wrote nothing");
        FakeDevice cancelled;
        rejects([&](){camera_scan::execute(plan,cancelled,[](){throw std::runtime_error("signal");},[](const camera_scan::Point&){});},"cancellation propagates");
        check(cancelled.control==0,"disable on cancellation");
        FakeDevice log_failure;
        rejects([&](){camera_scan::execute(plan,log_failure,[](){},[](const camera_scan::Point&){throw std::runtime_error("disk full");});},"trace failure propagates");
        check(log_failure.control==0,"disable on disk failure");
        FakeDevice expired;
        int waits=0;
        rejects([&](){camera_scan::execute(plan,expired,[&](){++waits;expired.status|=0x100;},[](const camera_scan::Point&){});},"watchdog failure propagates");
        check(waits==1 && expired.control==0,"expired watchdog not silently refreshed");
        FakeDevice final_expiry;
        int final_waits=0;
        rejects([&](){camera_scan::execute(plan,final_expiry,[&](){if(++final_waits==3)final_expiry.status|=0x100;},[](const camera_scan::Point&){});},"watchdog checked after final hold");
        check(final_waits==3 && final_expiry.control==0,"last-interval timeout disables PWM");
        if(argc==2) {
            std::ifstream generated(argv[1]);
            check(bool(generated),"generated Python plan exists");
            auto actual=camera_scan::read(generated);
            FakeDevice all;
            camera_scan::execute(actual,all,[](){},[](const camera_scan::Point&){});
            check(all.control==0,"Python plan executes through shared C++ engine");
        }
        std::cout<<"PASS: camera_servo_scan "<<checks<<" checks; host mock only, no hardware\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
