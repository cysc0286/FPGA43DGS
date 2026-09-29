// Reuse the frozen projection arithmetic verbatim; retain the Gaussian rows.
// LOAD replaces a complete scene atomically. PROJECT changes only the camera.
#define main frozen_attributes_main
#include "attributes.cpp"
#undef main
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>

int main() try {
    std::vector<float> rows;
    std::vector<Attribute> attrs;
    // Reuse the frozen camera decoder against an anonymous memory file, so
    // its inversion and float conversion remain exactly the same arithmetic.
    const int camera_fd=memfd_create("gs-camera",MFD_CLOEXEC);
    if(camera_fd<0)throw std::runtime_error("Camera memfd creation failed");
    const std::string camera_path="/proc/self/fd/"+std::to_string(camera_fd);
    uint32_t count=0;
    std::cout<<"ATTRIBUTES_READY"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
        std::istringstream input(line);
        std::string action,path,output,extra;
        if(!(input>>action))throw std::runtime_error("Empty command");
        if(action=="QUIT")break;
        if(action=="LOAD") {
            if(!(input>>std::quoted(path)) || (input>>extra))throw std::runtime_error("LOAD path required");
            uint32_t next_count=0;
            auto next=load_model(path.c_str(),next_count);
            rows.swap(next);count=next_count;
            std::cout<<"SCENE_LOADED "<<count<<std::endl;
        } else if(action=="LOADROWS") {
            uint32_t next_count=0;
            if(!(input>>next_count) || (input>>extra) || !next_count || next_count>1000000)
                throw std::runtime_error("LOADROWS requires a valid Gaussian count");
            std::vector<float> next(size_t(next_count)*62);
            std::cin.read(reinterpret_cast<char*>(next.data()),next.size()*sizeof(float));
            if(!std::cin || std::cin.gcount()!=std::streamsize(next.size()*sizeof(float)))
                throw std::runtime_error("LOADROWS payload incomplete");
            for(float value:next)if(!std::isfinite(value))
                throw std::runtime_error("LOADROWS nonfinite Gaussian");
            rows.swap(next);count=next_count;
            std::cout<<"SCENE_LOADED "<<count<<std::endl;
        } else if(action=="PROJECT" || action=="PROJECTMEM") {
            if(!count)throw std::runtime_error("PROJECT needs loaded scene");
            if(action=="PROJECTMEM") {
                if(!(input>>std::quoted(output)) || (input>>extra))
                    throw std::runtime_error("PROJECTMEM needs output path");
                std::array<char,136> payload;
                std::cin.read(payload.data(),payload.size());
                if(!std::cin || std::cin.gcount()!=std::streamsize(payload.size()))
                    throw std::runtime_error("Camera payload incomplete");
                if(lseek(camera_fd,0,SEEK_SET)!=0 ||
                   write(camera_fd,payload.data(),payload.size())!=ssize_t(payload.size()))
                    throw std::runtime_error("Camera memory write failed");
                path=camera_path;
            } else if(!(input>>std::quoted(path)>>std::quoted(output)) || (input>>extra))
                throw std::runtime_error("PROJECT needs camera and output paths");
            Camera c=load_camera(path.c_str());
            attrs.resize(count);
            std::fill(attrs.begin(),attrs.end(),Attribute{});
            uint32_t valid=0;
            for(uint32_t i=0;i<count;++i)valid+=project(rows.data()+size_t(i)*62,c,attrs[i]);
            std::ofstream file(output,std::ios::binary);
            if(!file)throw std::runtime_error("Attribute output open");
            file.write("FLKATR01",8);uint32_t header[4]={count,c.w,c.h,valid};
            file.write(reinterpret_cast<const char*>(header),sizeof(header));
            file.write(reinterpret_cast<const char*>(attrs.data()),attrs.size()*sizeof(Attribute));
            file.close();if(!file)throw std::runtime_error("Attribute output write");
            std::cout<<"PROJECT_COMPLETE"<<std::endl;
        } else throw std::runtime_error("Unknown attributes command");
    }
    close(camera_fd);
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
