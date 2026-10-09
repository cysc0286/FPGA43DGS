// Scene-static arithmetic matches the frozen attributes.cpp operation order.
// Include after the frozen projection implementation in namespace projection.
struct PreparedGaussian {
    projection::M3 world{};
    float opacity=0;
    double importance=0;
};

inline PreparedGaussian prepare_gaussian(const float* g) {
    using namespace projection;
    PreparedGaussian p;
    float scale[3]={std::exp(g[55]),std::exp(g[56]),std::exp(g[57])};
    float norm=std::sqrt(g[58]*g[58]+g[59]*g[59]+g[60]*g[60]+g[61]*g[61]);
    if(!(norm>0))throw std::runtime_error("zero quaternion");
    float w=g[58]/norm,x=g[59]/norm,y=g[60]/norm,z=g[61]/norm;
    M3 rotation={V3{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)},
                 V3{2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)},
                 V3{2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        for(int k=0;k<3;++k)p.world[i][j]+=rotation[i][k]*scale[k]*scale[k]*rotation[j][k];
    p.opacity=1.f/(1.f+std::exp(-g[54]));
    p.importance=double(p.opacity)*(double(scale[0])*scale[1]+double(scale[1])*scale[2]+double(scale[2])*scale[0]);
    if(!std::isfinite(p.importance))throw std::runtime_error("invalid Gaussian scale");
    for(auto& row:p.world)for(float v:row)if(!std::isfinite(v))throw std::runtime_error("nonfinite covariance");
    return p;
}

inline bool project_prepared(const float* g,const PreparedGaussian& p,
                             const projection::Camera& c,projection::Attribute& out,
                             float support_power=-1.f,unsigned support_guard=0) {
    using namespace projection;
    V3 pos={g[0],g[1],g[2]},t{};
    for(int i=0;i<3;++i)t[i]=c.v[i][0]*pos[0]+c.v[i][1]*pos[1]+c.v[i][2]*pos[2]+c.v[i][3];
    if(!(t[2]>.2f))return false;
    float px=c.focal_x*t[0]/(t[2]+1e-7f)+.5f*(c.w-1);
    float py=c.focal_y*t[1]/(t[2]+1e-7f)+.5f*(c.h-1);
    float limx=1.3f*c.w/(2*c.focal_x),limy=1.3f*c.h/(2*c.focal_y);
    float tx=std::clamp(t[0]/t[2],-limx,limx)*t[2];
    float ty=std::clamp(t[1]/t[2],-limy,limy)*t[2];
    V3 du={c.focal_x/t[2],0,-c.focal_x*tx/(t[2]*t[2])};
    V3 dv={0,c.focal_y/t[2],-c.focal_y*ty/(t[2]*t[2])};
    V3 wu{},wv{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){wu[i]+=c.v[j][i]*du[j];wv[i]+=c.v[j][i]*dv[j];}
    float a=dot(wu,multiply(p.world,wu))+.3f;
    float b=dot(wu,multiply(p.world,wv));
    float cc=dot(wv,multiply(p.world,wv))+.3f;
    float det=a*cc-b*b;
    if(det==0||!std::isfinite(det))return false;
    float mid=.5f*(a+cc);
    float lambda1=mid+std::sqrt(std::max(.1f,mid*mid-det));
    float lambda2=mid-std::sqrt(std::max(.1f,mid*mid-det));
    int radius=int(std::ceil(3*std::sqrt(std::max(lambda1,lambda2))));
    int radius_x=radius,radius_y=radius;
    if(support_guard&&support_power>=0&&det>0&&a>0&&cc>0) {
        // Optional conservative candidate, validated against the installed FP16
        // renderer: alpha >= 1/255 has axis support sqrt(2*log(255*opacity)*cov).
        // Keep the legacy outer radius and pad for projection/half rounding.
        // This changes only Tile membership, never conic values or depth order.
        // Half-conic error can still grow for ill-conditioned ellipses; this is
        // opt-in, not a proof of lossless pruning for arbitrary scenes.
        radius_x=std::min(radius,int(std::ceil(std::sqrt(support_power*a)+support_guard)));
        radius_y=std::min(radius,int(std::ceil(std::sqrt(support_power*cc)+support_guard)));
    }
    int nx=(c.w+15)/16,ny=(c.h+15)/16;
    auto clip=[](float v,int bound){return std::clamp(int(v),0,bound);};
    int minx=clip((px-radius_x)/16,nx),miny=clip((py-radius_y)/16,ny);
    int maxx=clip((px+radius_x+15)/16,nx),maxy=clip((py+radius_y+15)/16,ny);
    if(minx==maxx||miny==maxy)return false;
    out.q[0]=px;out.q[1]=py;out.q[2]=cc/det;out.q[3]=-b/det;out.q[4]=a/det;
    out.q[5]=p.opacity;color(g,c,&out.q[6]);out.q[9]=t[2];
    out.min_x=minx;out.min_y=miny;out.max_x=maxx;out.max_y=maxy;out.valid=1;
    for(float v:out.q)if(!std::isfinite(v))throw std::runtime_error("nonfinite projection");
    return true;
}
