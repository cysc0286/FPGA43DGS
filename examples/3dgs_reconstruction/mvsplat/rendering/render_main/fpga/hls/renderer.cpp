// FLICKER full-FP16 rendering baseline. Input is Gaussian features, never alpha.
// Explicit half intermediates freeze operation rounding; no fused multiply-add.
// The surrounding transport/queue is a 30TAI platform adaptation.
#include "renderer.hpp"
static half power(half x,half y,half mx,half my,half a,half b,half c){
#pragma HLS INLINE
 half dx=half(mx-x),dy=half(my-y);
 half ax=half(a*dx),cy=half(c*dy),bx=half(b*dx);
 half xx=half(ax*dx),yy=half(cy*dy),xy=half(bx*dy);
 half sum=half(xx+yy),neg=half(half(-0.5f)*sum);
 return half(neg-xy);
}
void flicker_render(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode){
#pragma HLS INTERFACE ap_fifo port=input
#pragma HLS INTERFACE ap_fifo port=output
#pragma HLS INTERFACE ap_ctrl_hs port=return
 half rr[256],gg[256],bb[256],tt[256];
 unsigned last[256];bool done[256];
#pragma HLS RESOURCE variable=rr core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_2P_BRAM
 for(unsigned tile=0;tile<tiles;++tile){
  Word header=input.read();unsigned n=header.range(31,0);
  ap_uint<16> ox=header.range(47,32),oy=header.range(63,48);
  ap_uint<5> width=header.range(68,64),height=header.range(73,69);
  half br=unpack_half(header,80),bg=unpack_half(header,96),bc=unpack_half(header,112);
  for(unsigned p=0;p<256;p++){
#pragma HLS PIPELINE II=1
   rr[p]=gg[p]=bb[p]=half(0);tt[p]=half(1);last[p]=0;done[p]=false;
  }
  for(unsigned j=0;j<n;j++){
   Word q=input.read();
   half mx=unpack_half(q,0),my=unpack_half(q,16),a=unpack_half(q,32),b=unpack_half(q,48),c=unpack_half(q,64);
   half opacity=unpack_half(q,80),red=unpack_half(q,96),green=unpack_half(q,112),blue=unpack_half(q,128);
   // mode is reserved for subsequent CTU integration. Reject unsupported mode
   // in the host/controller; this baseline always evaluates every live pixel.
   for(unsigned p=0;p<256;p++){
#pragma HLS PIPELINE II=1
    unsigned x=p&15,y=p>>4;
    if(x<width&&y<height&&!done[p]){
     half pw=power(half(unsigned(ox)+x),half(unsigned(oy)+y),mx,my,a,b,c);
     if(pw<=half(0)){
      half ex=hls::half_exp(pw),alpha=half(opacity*ex);
      if(alpha>half(.99f))alpha=half(.99f);
      if(alpha>=half(1.f/255.f)){
       half one_minus=half(half(1)-alpha),next=half(tt[p]*one_minus);
       if(next<half(.0001f))done[p]=true;
       else{
        half weight=half(alpha*tt[p]);
        half dr=half(red*weight),dg=half(green*weight),db=half(blue*weight);
        rr[p]=half(rr[p]+dr);gg[p]=half(gg[p]+dg);bb[p]=half(bb[p]+db);
        tt[p]=next;last[p]=j+1;
       }
      }
     }
    }
   }
  }
  Word out=0;
  for(unsigned p=0;p<256;p++){
#pragma HLS PIPELINE II=1
   half r=half(rr[p]+half(br*tt[p])),g=half(gg[p]+half(bg*tt[p])),b=half(bb[p]+half(bc*tt[p]));
   ap_uint<128> pixel=0;
   pixel.range(15,0)=pack_half(r);pixel.range(31,16)=pack_half(g);
   pixel.range(47,32)=pack_half(b);pixel.range(63,48)=pack_half(tt[p]);
   pixel.range(95,64)=last[p];pixel.range(127,96)=tile*256+p;
   out.range((p%4)*128+127,(p%4)*128)=pixel;
   if((p&3)==3)output.write(out);
  }
 }
}
