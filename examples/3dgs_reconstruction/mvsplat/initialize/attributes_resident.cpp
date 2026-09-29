// Reuse the frozen projection arithmetic verbatim; retain the Gaussian rows.
// LOAD replaces a complete scene atomically. PROJECT changes only the camera.
#define main frozen_attributes_main
#include "attributes.cpp"
#undef main
#include <sstream>

int main() try {
    std::vector<float> rows;
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
        } else if(action=="PROJECT") {
            if(!count || !(input>>std::quoted(path)>>std::quoted(output)) || (input>>extra))
                throw std::runtime_error("PROJECT needs loaded scene, camera and output");
            Camera c=load_camera(path.c_str());
            std::vector<Attribute> attrs(count);uint32_t valid=0;
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
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
