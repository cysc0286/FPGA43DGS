// FLICKER III-B / IV-C full-FP16 CTU ablation. No FP8 format is inferred.
#include "ctu.hpp"
struct Prepared {Word q;half limit;bool dense;bool bypass;};
struct Request {Prepared p;bool second;bool end;};
struct Tested {Word q;ap_uint<4> mask;bool dense;bool second;bool end;};
struct Weights {half e0,e1,e2,e3;};

static void prepare(hls::stream<Word>& input,hls::stream<Prepared>& output,
                    unsigned count,unsigned mode){
 for(unsigned i=0;i<count;i++){
#pragma HLS PIPELINE II=1
  Prepared p;p.q=input.read();bool spiky=p.q[186];
  p.dense=mode==1||(mode==3&&!spiky)||(mode==4&&spiky);p.bypass=mode==0;
  half opacity=unpack_half(p.q,80);
  p.limit=opacity>half(0)?hls::half_log(half(half(255)*opacity)):half(-65504);
  output.write(p);
 }
}

// Dense: two requests, two rectangles per request. Sparse: one request,
// rectangles formed from each mini-tile's top-left and bottom-right leaders.
static void expand(hls::stream<Prepared>& input,hls::stream<Request>& output,unsigned count){
 Prepared held;bool second=false;unsigned index=0;
 while(index<count){
#pragma HLS PIPELINE II=1
  if(!second)held=input.read();
  Request r;r.p=held;r.second=second;r.end=false;output.write(r);
  if(held.dense&&!held.bypass&&!second)second=true;
  else {second=false;index++;}
 }
 Request end{};end.end=true;output.write(end);
}

// Algorithm 1: each rectangle computes four shared axis terms and four cross
// terms. The pair stage instantiates this computation twice, not eight ACUs.
static Weights rectangle(Word q,unsigned x0,unsigned y0,unsigned x1,unsigned y1){
#pragma HLS INLINE
 half mx=unpack_half(q,0),my=unpack_half(q,16);
 half a=unpack_half(q,32),b=unpack_half(q,48),c=unpack_half(q,64);
 half dx0=half(half(x0)-mx),dy0=half(half(y0)-my);
 half dx1=half(half(x1)-mx),dy1=half(half(y1)-my);
 half xx0=half(dx0*dx0),xx1=half(dx1*dx1),yy0=half(dy0*dy0),yy1=half(dy1*dy1);
 half sx0=half(half(half(.5f)*xx0)*a),sx1=half(half(half(.5f)*xx1)*a);
 half sy0=half(half(half(.5f)*yy0)*c),sy1=half(half(half(.5f)*yy1)*c);
 half t0=half(half(dx0*dy0)*b),t1=half(half(dx1*dy0)*b);
 half t2=half(half(dx0*dy1)*b),t3=half(half(dx1*dy1)*b);
 Weights w;w.e0=half(half(sx0+sy0)+t0);w.e1=half(half(sx1+sy0)+t1);
 w.e2=half(half(sx0+sy1)+t2);w.e3=half(half(sx1+sy1)+t3);return w;
}
static unsigned clip(unsigned x,unsigned extent){
#pragma HLS INLINE
 return x<extent?x:extent-1;
}
static ap_uint<4> hits(Weights w,half limit){
#pragma HLS INLINE
 ap_uint<4> mask=0;mask[0]=w.e0<=limit;mask[1]=w.e1<=limit;
 mask[2]=w.e2<=limit;mask[3]=w.e3<=limit;return mask;
}
static void pair_test(hls::stream<Request>& input,hls::stream<Tested>& output){
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Request r=input.read();Tested t;t.end=r.end;end=r.end;
  t.q=r.p.q;t.dense=r.p.dense&&!r.p.bypass;t.second=r.second;t.mask=0;
  if(!r.end){
   Word q=r.p.q;unsigned ox=q.range(159,144),oy=q.range(175,160);
   unsigned width=q.range(180,176),height=q.range(185,181);
   unsigned x00=0,x01=0,y00=0,y01=0,x10=0,x11=0,y10=0,y11=0;
   if(r.p.dense){
    unsigned top=r.second?4:0;
    x00=0;x01=3;y00=top;y01=top+3;
    x10=4;x11=7;y10=top;y11=top+3;
   }else{
    x00=0;x01=4;y00=0;y01=4;x10=3;x11=7;y10=3;y11=7;
   }
   Weights a=rectangle(q,ox+clip(x00,width),oy+clip(y00,height),ox+clip(x01,width),oy+clip(y01,height));
   Weights b=rectangle(q,ox+clip(x10,width),oy+clip(y10,height),ox+clip(x11,width),oy+clip(y11,height));
   ap_uint<4> ma=hits(a,r.p.limit),mb=hits(b,r.p.limit);
   if(r.p.bypass)t.mask=15;
   else if(r.p.dense){t.mask[r.second?2:0]=ma!=0;t.mask[r.second?3:1]=mb!=0;}
   else t.mask=ma|mb;
   // Invalid edge mini-tiles never receive work, including bypass mode.
   if(width<=4){t.mask[1]=0;t.mask[3]=0;}
   if(height<=4){t.mask[2]=0;t.mask[3]=0;}
  }
  output.write(t);
 }
}
static void merge(hls::stream<Tested>& input,hls::stream<Word>& output){
 ap_uint<4> first=0;bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Tested t=input.read();end=t.end;
  if(!t.end){
   if(t.dense&&!t.second)first=t.mask;
   else {t.q.range(227,224)=t.dense?ap_uint<4>(first|t.mask):t.mask;output.write(t.q);}
  }
 }
}
void flicker_ctu(hls::stream<Word>& input,hls::stream<Word>& output,unsigned count,unsigned mode){
#pragma HLS INTERFACE ap_fifo port=input
#pragma HLS INTERFACE ap_fifo port=output
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS DATAFLOW
 hls::stream<Prepared> prepared;
 hls::stream<Request> requests;
 hls::stream<Tested> tested;
#pragma HLS STREAM variable=prepared depth=16
#pragma HLS STREAM variable=requests depth=16
#pragma HLS STREAM variable=tested depth=16
#pragma HLS RESOURCE variable=prepared core=FIFO_SRL
#pragma HLS RESOURCE variable=requests core=FIFO_SRL
#pragma HLS RESOURCE variable=tested core=FIFO_SRL
 prepare(input,prepared,count,mode);expand(prepared,requests,count);
 pair_test(requests,tested);merge(tested,output);
}
