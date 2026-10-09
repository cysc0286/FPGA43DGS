// Board-side preprocessing of the official 3DGS model. Reference formulas:
// locked GraphDECO diff-gaussian-rasterization forward.cu/auxiliary.h.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using V3 = std::array<float, 3>;
using M3 = std::array<V3, 3>;

struct Attribute {
    float q[10]; // x, y, conic a/b/c, opacity, RGB, depth
    uint16_t min_x, min_y, max_x, max_y;
    uint32_t valid;
};
static_assert(sizeof(Attribute) == 52, "attribute ABI changed");

template <typename T> void read_exact(std::ifstream& in, T* dst, size_t n) {
    if (!in.read(reinterpret_cast<char*>(dst), n * sizeof(T)))
        throw std::runtime_error("truncated input");
}

struct Camera {
    uint32_t w, h, source_w, source_h;
    double r[3][3], p[3], fx, fy;
    float v[3][4], pos[3], focal_x, focal_y;
};

Camera load_camera(const char* path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8]; read_exact(f, magic, 8);
    if (std::memcmp(magic, "FLCAM001", 8)) throw std::runtime_error("camera ABI");
    Camera c{};
    uint32_t dims[4]; read_exact(f, dims, 4);
    c.w = dims[0]; c.h = dims[1]; c.source_w = dims[2]; c.source_h = dims[3];
    for (auto& row : c.r) read_exact(f, row, 3);
    read_exact(f, c.p, 3); read_exact(f, &c.fx, 1); read_exact(f, &c.fy, 1);
    if (f.peek() != EOF || !c.w || !c.h || !c.source_w || !c.source_h ||
        c.w > 2048 || c.h > 2048) throw std::runtime_error("invalid camera dimensions");
    // Match the official float32 cast after inverting camera-to-world in float64.
    double augmented[3][6]{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) augmented[i][j] = c.r[i][j];
        augmented[i][i + 3] = 1;
    }
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col])) pivot = row;
        if (std::abs(augmented[pivot][col]) < 1e-10) throw std::runtime_error("singular camera");
        for (int j = 0; j < 6; ++j) std::swap(augmented[col][j], augmented[pivot][j]);
        double factor = augmented[col][col];
        for (int j = 0; j < 6; ++j) augmented[col][j] /= factor;
        for (int row = 0; row < 3; ++row) if (row != col) {
            factor = augmented[row][col];
            for (int j = 0; j < 6; ++j) augmented[row][j] -= factor * augmented[col][j];
        }
    }
    for (int i = 0; i < 3; ++i) {
        double t = 0;
        for (int j = 0; j < 3; ++j) {
            c.v[i][j] = float(augmented[i][j + 3]);
            t -= augmented[i][j + 3] * c.p[j];
        }
        c.v[i][3] = float(t);
        c.pos[i] = float(c.p[i]);
    }
    c.focal_x = float(c.fx * c.w / c.source_w);
    c.focal_y = float(c.fy * c.h / c.source_h);
    return c;
}

std::vector<float> load_model(const char* path, uint32_t& count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("model open");
    auto bytes = f.tellg(); f.seekg(0);
    std::string line;
    if (!std::getline(f, line) || line != "ply" ||
        !std::getline(f, line) || line != "format binary_little_endian 1.0")
        throw std::runtime_error("model PLY format");
    const std::vector<std::string> prefix = {"x", "y", "z", "nx", "ny", "nz"};
    std::vector<std::string> names;
    bool got_count = false, ended = false;
    for (int n = 0; n < 256 && std::getline(f, line); ++n) {
        if (line == "end_header") { ended = true; break; }
        if (line.rfind("element vertex ", 0) == 0) {
            if (got_count) throw std::runtime_error("duplicate vertex count");
            count = std::stoul(line.substr(15)); got_count = true;
        } else if (line.rfind("property float ", 0) == 0) {
            names.push_back(line.substr(15));
        } else if (line.rfind("comment ", 0) != 0) {
            throw std::runtime_error("unsupported PLY field: " + line);
        }
    }
    if (!ended || !got_count || !count || count > 1000000 || names.size() != 62)
        throw std::runtime_error("unexpected model schema");
    std::vector<std::string> expected = prefix;
    for (int i = 0; i < 3; ++i) expected.push_back("f_dc_" + std::to_string(i));
    for (int i = 0; i < 45; ++i) expected.push_back("f_rest_" + std::to_string(i));
    expected.push_back("opacity");
    for (int i = 0; i < 3; ++i) expected.push_back("scale_" + std::to_string(i));
    for (int i = 0; i < 4; ++i) expected.push_back("rot_" + std::to_string(i));
    if (names != expected || int64_t(bytes) != int64_t(f.tellg()) + int64_t(count) * 62 * 4)
        throw std::runtime_error("model length or property order");
    std::vector<float> rows(size_t(count) * 62);
    read_exact(f, rows.data(), rows.size());
    return rows;
}

V3 multiply(const M3& a, V3 v) {
    return {a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
            a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
            a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2]};
}
float dot(V3 a, V3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

void color(const float* g, const Camera& c, float* rgb) {
    V3 d = {g[0]-c.pos[0], g[1]-c.pos[1], g[2]-c.pos[2]};
    float length = std::sqrt(dot(d,d));
    if (!(length > 0)) throw std::runtime_error("invalid SH direction");
    for (auto& x : d) x /= length;
    float x=d[0], y=d[1], z=d[2], xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
    constexpr float c0=.28209479177387814f, c1=.4886025119029199f;
    constexpr float c2[5]={1.0925484305920792f,-1.0925484305920792f,.31539156525252005f,-1.0925484305920792f,.5462742152960396f};
    constexpr float c3[7]={-.5900435899266435f,2.890611442640554f,-.4570457994644658f,.3731763325901154f,-.4570457994644658f,1.445305721320277f,-.5900435899266435f};
    for (int ch=0;ch<3;++ch) {
        auto sh=[&](int k)->float { return k ? g[9+ch*15+k-1] : g[6+ch]; };
        float value=c0*sh(0);
        value=value-c1*y*sh(1)+c1*z*sh(2)-c1*x*sh(3);
        value=value+c2[0]*xy*sh(4)+c2[1]*yz*sh(5)+c2[2]*(2*zz-xx-yy)*sh(6)
                   +c2[3]*xz*sh(7)+c2[4]*(xx-yy)*sh(8);
        value=value+c3[0]*y*(3*xx-yy)*sh(9)+c3[1]*xy*z*sh(10)
                   +c3[2]*y*(4*zz-xx-yy)*sh(11)
                   +c3[3]*z*(2*zz-3*xx-3*yy)*sh(12)
                   +c3[4]*x*(4*zz-xx-yy)*sh(13)
                   +c3[5]*z*(xx-yy)*sh(14)+c3[6]*x*(xx-3*yy)*sh(15);
        rgb[ch]=std::max(0.f,value+.5f);
    }
}

bool project(const float* g, const Camera& c, Attribute& out) {
    V3 p={g[0],g[1],g[2]}, t{};
    for (int i=0;i<3;++i)
        t[i]=c.v[i][0]*p[0]+c.v[i][1]*p[1]+c.v[i][2]*p[2]+c.v[i][3];
    if (!(t[2]>.2f)) return false;
    float px=c.focal_x*t[0]/(t[2]+1e-7f)+.5f*(c.w-1);
    float py=c.focal_y*t[1]/(t[2]+1e-7f)+.5f*(c.h-1);
    float scale[3]={std::exp(g[55]),std::exp(g[56]),std::exp(g[57])};
    float norm=std::sqrt(g[58]*g[58]+g[59]*g[59]+g[60]*g[60]+g[61]*g[61]);
    if (!(norm>0)) throw std::runtime_error("zero quaternion");
    float w=g[58]/norm,x=g[59]/norm,y=g[60]/norm,z=g[61]/norm;
    M3 rotation={V3{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)},
                 V3{2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)},
                 V3{2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}};
    M3 world{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        for(int k=0;k<3;++k)world[i][j]+=rotation[i][k]*scale[k]*scale[k]*rotation[j][k];
    float limx=1.3f*c.w/(2*c.focal_x),limy=1.3f*c.h/(2*c.focal_y);
    float tx=std::clamp(t[0]/t[2],-limx,limx)*t[2];
    float ty=std::clamp(t[1]/t[2],-limy,limy)*t[2];
    V3 du={c.focal_x/t[2],0,-c.focal_x*tx/(t[2]*t[2])};
    V3 dv={0,c.focal_y/t[2],-c.focal_y*ty/(t[2]*t[2])};
    V3 wu{},wv{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){wu[i]+=c.v[j][i]*du[j];wv[i]+=c.v[j][i]*dv[j];}
    float a=dot(wu,multiply(world,wu))+.3f;
    float b=dot(wu,multiply(world,wv));
    float cc=dot(wv,multiply(world,wv))+.3f;
    float det=a*cc-b*b;
    if (det==0 || !std::isfinite(det)) return false;
    float mid=.5f*(a+cc);
    float lambda1=mid+std::sqrt(std::max(.1f,mid*mid-det));
    float lambda2=mid-std::sqrt(std::max(.1f,mid*mid-det));
    int radius=int(std::ceil(3*std::sqrt(std::max(lambda1,lambda2))));
    int tiles_x=(c.w+15)/16,tiles_y=(c.h+15)/16;
    auto clip=[](float v,int bound){return std::clamp(int(v),0,bound);};
    int minx=clip((px-radius)/16,tiles_x),miny=clip((py-radius)/16,tiles_y);
    int maxx=clip((px+radius+15)/16,tiles_x),maxy=clip((py+radius+15)/16,tiles_y);
    if (minx==maxx || miny==maxy) return false;
    out.q[0]=px;out.q[1]=py;out.q[2]=cc/det;out.q[3]=-b/det;out.q[4]=a/det;
    out.q[5]=1.f/(1.f+std::exp(-g[54]));
    color(g,c,&out.q[6]);out.q[9]=t[2];
    out.min_x=minx;out.min_y=miny;out.max_x=maxx;out.max_y=maxy;out.valid=1;
    for(float q:out.q)if(!std::isfinite(q))throw std::runtime_error("nonfinite projected attribute");
    return true;
}

int main(int argc,char**argv)try {
    if(argc!=4)throw std::runtime_error("usage: attributes model.ply camera.bin output.attr");
    auto start=Clock::now();Camera c=load_camera(argv[2]);uint32_t count=0;
    auto rows=load_model(argv[1],count);auto loaded=Clock::now();
    std::vector<Attribute> attrs(count);uint32_t valid=0;
    for(uint32_t i=0;i<count;++i)valid+=project(rows.data()+size_t(i)*62,c,attrs[i]);
    auto projected=Clock::now();
    std::ofstream out(argv[3],std::ios::binary);
    if(!out)throw std::runtime_error("attribute output open");
    out.write("FLKATR01",8);uint32_t header[4]={count,c.w,c.h,valid};
    out.write(reinterpret_cast<const char*>(header),sizeof(header));
    out.write(reinterpret_cast<const char*>(attrs.data()),attrs.size()*sizeof(Attribute));
    if(!out)throw std::runtime_error("attribute output write");out.close();
    auto done=Clock::now();
    auto ms=[](Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();};
    std::cout<<std::fixed<<std::setprecision(3)<<"points="<<count<<" valid="<<valid
             <<" read_ms="<<ms(start,loaded)<<" projection_ms="<<ms(loaded,projected)
             <<" write_ms="<<ms(projected,done)<<" total_ms="<<ms(start,done)<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
