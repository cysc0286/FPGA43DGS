// Reuse the already verified PRTU arithmetic without altering the FLK0 or
// standalone CTU source. Framing adds ordered header/end tokens for DATAFLOW.
#include "pipeline.hpp"
#include "../ctu/ctu.cpp"
#if FLK_EXACT_LIMIT_ROM
#include "limit_rom/table.hpp"
#endif

static void prepare_frames(hls::stream<Word>& input,hls::stream<Prepared>& output,unsigned mode){
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Prepared p;p.q=input.read();unsigned type=kind(p.q);end=type==3;
  bool spiky=p.q[186];
  p.dense=type==0&&(mode==2||(mode==4&&!spiky)||(mode==5&&spiky));
  p.bypass=mode<=1||type!=0;
  half opacity=unpack_half(p.q,80);
#if FLK_EXACT_LIMIT_ROM
  // Exhaustively verified mapping of the entire old expression, including
  // signed zero, nonpositive values, infinities and NaNs. No interpolation.
  p.limit=exact_opacity_limit(opacity);
#else
  p.limit=opacity>half(0)?hls::half_log(half(half(255)*opacity)):half(-65504);
#endif
  output.write(p);
 }
}

static void expand_frames(hls::stream<Prepared>& input,hls::stream<Request>& output){
 Prepared held;bool second=false,end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  if(!second)held=input.read();
  Request r;r.p=held;r.second=second;r.end=kind(held.q)==3;end=r.end;output.write(r);
  second=held.dense&&!held.bypass&&!second;
 }
}

static void test_frames(hls::stream<Request>& input,hls::stream<Tested>& output){
 bool end=false;
 while(!end){
// The installed design deliberately shares the rectangle arithmetic at II=8.
// FLK_CTU_FAST is an opt-in high-resource candidate: each request is
// independent, so HLS may replicate the rectangle/hit arithmetic and issue a
// request every cycle. The default remains the installed resource-balanced
// implementation until the full-board timing report accepts this candidate.
#if FLK_CTU_FAST
#pragma HLS PIPELINE II=1
#elif FLK_CTU_MID
#pragma HLS PIPELINE II=4
#elif FLK_CTU_SIX
#pragma HLS PIPELINE II=6
#else
#pragma HLS PIPELINE II=8
#endif
  Request r=input.read();Tested t;t.end=r.end;end=r.end;
  t.q=r.p.q;t.dense=r.p.dense&&!r.p.bypass;t.second=r.second;t.mask=0;
  if(kind(t.q)==0){
   Word q=t.q;unsigned ox=q.range(159,144),oy=q.range(175,160);
   unsigned width=q.range(180,176),height=q.range(185,181);
   unsigned x00,x01,y00,y01,x10,x11,y10,y11;
   if(r.p.dense){
    unsigned top=r.second?4:0;
    x00=0;x01=3;y00=top;y01=top+3;x10=4;x11=7;y10=top;y11=top+3;
   }else{
    x00=0;x01=4;y00=0;y01=4;x10=3;x11=7;y10=3;y11=7;
   }
   Weights a=rectangle(q,ox+clip(x00,width),oy+clip(y00,height),ox+clip(x01,width),oy+clip(y01,height));
   Weights b=rectangle(q,ox+clip(x10,width),oy+clip(y10,height),ox+clip(x11,width),oy+clip(y11,height));
   ap_uint<4> ma=hits(a,r.p.limit),mb=hits(b,r.p.limit);
   if(r.p.bypass)t.mask=15;
   else if(r.p.dense){t.mask[r.second?2:0]=ma!=0;t.mask[r.second?3:1]=mb!=0;}
   else t.mask=ma|mb;
   if(width<=4){t.mask[1]=0;t.mask[3]=0;}
   if(height<=4){t.mask[2]=0;t.mask[3]=0;}
  }
  output.write(t);
 }
}

static void merge_frames(hls::stream<Tested>& input,hls::stream<Word>& output){
 ap_uint<4> first=0;bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Tested t=input.read();end=t.end;
  if(kind(t.q)!=0)output.write(t.q);
  else if(t.dense&&!t.second)first=t.mask;
  else{t.q.range(227,224)=t.dense?ap_uint<4>(first|t.mask):t.mask;output.write(t.q);}
 }
}

void framed_ctu(hls::stream<Word>& input,hls::stream<Word>& output,unsigned mode){
#pragma HLS INLINE off
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
 prepare_frames(input,prepared,mode);expand_frames(prepared,requests);
 test_frames(requests,tested);merge_frames(tested,output);
}
