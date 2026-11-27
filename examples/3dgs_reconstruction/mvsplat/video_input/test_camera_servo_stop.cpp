#include "camera_servo_stop.h"
#include <iostream>
struct FakeDevice {
    uint32_t status=2,control=2;
    unsigned writes=0;
    void write(uint32_t offset,uint32_t value) {
        if(offset!=camera_stop::REQUEST || value!=1) throw std::runtime_error("unexpected write");
        ++writes;
    }
    uint32_t read(uint32_t offset) { return offset==0x10 ? control : status; }
};
int checks=0;
void check(bool value,const char* message) {++checks;if(!value)throw std::runtime_error(message);}
template<class Call>void rejects(Call call,const char* message) {
    bool failed=false;try{call();}catch(const std::exception&){failed=true;}check(failed,message);
}
int main() {
    try {
        FakeDevice normal;unsigned polls=0;
        camera_stop::finish(normal,[&](){if(++polls==20)normal.status=normal.control=0;});
        check(polls==20 && normal.writes==1,"wait for frame acknowledgement");
        FakeDevice one_sided;unsigned one_polls=0;
        camera_stop::finish(one_sided,[&](){
            if(++one_polls==1)one_sided.control=0;
            if(one_polls==4)one_sided.status=0;
        });
        check(one_polls==4,"requested disable alone is not enough");
        FakeDevice stuck;unsigned timeout_polls=0;
        rejects([&](){camera_stop::finish(stuck,[&](){++timeout_polls;});},"missing ack fails");
        check(timeout_polls==50,"stop wait bounded");
        FakeDevice expired;unsigned expiry_polls=0;
        rejects([&](){camera_stop::finish(expired,[&](){++expiry_polls;expired.status|=0x100;});},"watchdog aborts stop");
        check(expiry_polls==1,"expired watchdog not refreshed");
        FakeDevice cancelled;
        rejects([&](){camera_stop::finish(cancelled,[](){throw std::runtime_error("signal");});},"cancellation propagates");
        check(cancelled.writes==1,"no immediate disable hidden in normal-stop helper");
        std::cout<<"PASS: camera_servo_stop "<<checks<<" checks; mock device only\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
