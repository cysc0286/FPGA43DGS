// Frozen pre-optimization split used only by the differential test.
static void baseline_split(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode){
 for(unsigned t=0;t<tiles;t++){
  Word header=input.read(),control=header;control.range(511,510)=1;output.write(control);
  unsigned n=header.range(31,0),ox=header.range(47,32),oy=header.range(63,48);
  unsigned width=header.range(68,64),height=header.range(73,69);
  for(unsigned j=0;j<n;j++){
   Word raw=input.read(),q=0;
   for(unsigned k=0;k<9;k++){
#pragma HLS UNROLL
    q.range(k*16+15,k*16)=pack_half(half(unpack_float(raw,k*32)));
   }
   float x=unpack_float(raw,0),y=unpack_float(raw,32);
   float a=unpack_float(raw,64),b=unpack_float(raw,96),c=unpack_float(raw,128);
   float det=a*c-b*b,trace=a+c,delta=hls::sqrtf((a-c)*(a-c)+4.f*b*b);
   bool spiky=(trace+delta)>=9.f*(trace-delta);
   // Original 3DGS uses a lower-bound discriminant of 0.1 in radius selection.
   // Invalid reconstruction conservatively keeps sub-tiles rather than inventing
   // a finite box. Validated scene inputs require positive definite conics.
   float radius=65504.f;
   if(det>0.f){
    float vx=c/det,vy=a/det,xy=-b/det,mid=.5f*(vx+vy);
    float d=mid*mid-(vx*vy-xy*xy);if(d<.1f)d=.1f;
    radius=hls::ceilf(3.f*hls::sqrtf(mid+hls::sqrtf(d)));
   }
   q[186]=spiky;q.range(223,192)=j+1;
   for(unsigned sub=0;sub<4;sub++){
    unsigned sx=(sub&1)*8,sy=(sub>>1)*8;
    if(sx<width&&sy<height){
     unsigned w=width-sx,h=height-sy;if(w>8)w=8;if(h>8)h=8;
     unsigned px=ox+sx,py=oy+sy;
     bool hit=x+radius>=float(px)&&y+radius>=float(py)&&x-radius<=float(px+w-1)&&y-radius<=float(py+h-1);
     if(mode==0||hit){
      Word p=q;p.range(159,144)=px;p.range(175,160)=py;p.range(180,176)=w;p.range(185,181)=h;p.range(229,228)=sub;
      output.write(p);
     }
    }
   }
  }
  control=header;control.range(511,510)=2;output.write(control);
 }
 Word end=0;end.range(511,510)=3;output.write(end);
}
