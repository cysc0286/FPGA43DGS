// FLICKER IV-B: sub-tile AABB -> CTU -> masked mini-tile FIFOs -> VRUs.
// Four VRU channels are an explicit resource-scaled FPGA adaptation.
#include "pipeline.hpp"

// Keep the installed single-transfer design as the default. The dual-port variant
// is an opt-in experiment: it widens only state transfers, not Gaussian order.
#ifndef FLK_STATE_PORTS
#define FLK_STATE_PORTS 1
#endif
#if FLK_STATE_PORTS != 1 && FLK_STATE_PORTS != 2
#error "FLK_STATE_PORTS must be 1 or 2"
#endif
#ifndef FLK_GROUP_SUBTILES
#define FLK_GROUP_SUBTILES 0
#endif
#ifndef FLK_GROUP_TRIM_RANGE
#define FLK_GROUP_TRIM_RANGE 0
#endif
#if FLK_GROUP_TRIM_RANGE && !FLK_GROUP_SUBTILES
#error "Trimming requires grouped subtiles"
#endif
#ifndef FLK_LANE_FIFO_STORAGE
#define FLK_LANE_FIFO_STORAGE 0
#endif
#if FLK_LANE_FIFO_STORAGE < 0 || FLK_LANE_FIFO_STORAGE > 2
#error "FLK_LANE_FIFO_STORAGE must be 0, 1 or 2"
#endif
#if FLK_LANE_FIFO_STORAGE && FLK_GROUP_SUBTILES != 3
#error "Packed lane queues require the validated compact group cache"
#endif
#ifndef FLK_GROUP_SHARED_ATTR
#define FLK_GROUP_SHARED_ATTR 0
#endif
#if FLK_GROUP_SHARED_ATTR && FLK_GROUP_SUBTILES != 3
#error "Shared attributes require the compact grouped evaluator"
#endif
#ifndef FLK_GROUP_EARLY_DRAIN
#define FLK_GROUP_EARLY_DRAIN 0
#endif
#ifndef FLK_DPC
// PipeGS-inspired differential power computation candidate.  It is opt-in:
// the installed path keeps the original FP16 quadratic expression and bitwise
// output contract until this candidate passes C/RTL, P&R and image gates.
#define FLK_DPC 0
#endif
#ifndef FLK_DPC_INTERLEAVE
// A dependency-broken DPC candidate.  The pixel stream remains row-major,
// but the quadratic is advanced along y with four independent column
// contexts.  A context is revisited every four cycles, allowing the FP16
// add/sub latency to be hidden without changing output ordering.
#define FLK_DPC_INTERLEAVE 0
#endif
#ifndef FLK_DPC_BLOCK_TABLE
// A block-restarted DPC candidate.  One exact FP16 anchor is evaluated at the
// first row of a four-row block; the remaining rows read a small local table.
// This is opt-in and mutually exclusive with the feedback implementation.
#define FLK_DPC_BLOCK_TABLE 0
#endif
#ifndef FLK_DPC_ROW_TABLE
// Restart at every actual lane row, including masked anchor pixels. The
// producer carries the table index explicitly for both three/four cores.
#define FLK_DPC_ROW_TABLE 0
#endif
#if FLK_DPC_ROW_TABLE && (!FLK_DPC || FLK_DPC_BLOCK_TABLE || FLK_DPC_INTERLEAVE)
#error "Row-table DPC requires FLK_DPC and no other DPC variant"
#endif
#if FLK_DPC_INTERLEAVE && !FLK_DPC
#error "FLK_DPC_INTERLEAVE requires FLK_DPC"
#endif
#if FLK_DPC_BLOCK_TABLE && !FLK_DPC
#error "FLK_DPC_BLOCK_TABLE requires FLK_DPC"
#endif
#if FLK_DPC_INTERLEAVE && FLK_DPC_BLOCK_TABLE
#error "FLK_DPC_INTERLEAVE and FLK_DPC_BLOCK_TABLE are mutually exclusive"
#endif
#ifndef FLK_STREAM_SHARED_ATTR
// PipeGS-inspired stream metadata reuse. One group header carries invariant
// geometry/opacity; following pixel requests carry only coordinates/mask.
// The installed grouped evaluator remains unchanged unless explicitly enabled.
#define FLK_STREAM_SHARED_ATTR 0
#endif
#ifndef FLK_STREAM_SHARED_ATTR_FIFO
// Independent PipeGS-like attribute FIFO candidate. The producer emits one
// raw-attribute token per Gaussian group; a separate materializer expands it
// into the original request ABI before the numeric evaluator. This remains an
// opt-in experiment so the installed path is unchanged.
#define FLK_STREAM_SHARED_ATTR_FIFO 0
#endif
#ifndef FLK_STREAM_COMPACT_REQUEST
// The compact request ABI is an opt-in PipeGS metadata-reuse experiment.
// The default keeps the validated 133-bit request format unchanged.
#define FLK_STREAM_COMPACT_REQUEST 0
#endif
#ifndef FLK_STREAM_MATERIALIZED_SRL
// Keep the materializer FIFO mapping unchanged unless a candidate explicitly
// requests SRL storage and passes the same protocol/RTL checks.
#define FLK_STREAM_MATERIALIZED_SRL 0
#endif
#if FLK_STREAM_COMPACT_REQUEST && (!FLK_STREAM_SHARED_ATTR_FIFO || FLK_DPC_ROW_TABLE)
#error "Compact requests require shared attribute FIFO and no row-table DPC"
#endif
#if FLK_STREAM_MATERIALIZED_SRL && !FLK_STREAM_SHARED_ATTR_FIFO
#error "Materialized SRL requires shared attribute FIFO"
#endif
#if FLK_STREAM_SHARED_ATTR && FLK_STREAM_SHARED_ATTR_FIFO
#error "Inline and FIFO shared attributes are mutually exclusive"
#endif
#if FLK_STREAM_SHARED_ATTR_FIFO && FLK_STREAM_LANE != 2
#error "FLK_STREAM_SHARED_ATTR_FIFO requires FLK_STREAM_LANE=2"
#endif
#if FLK_STREAM_SHARED_ATTR && FLK_STREAM_LANE != 2
#error "FLK_STREAM_SHARED_ATTR requires FLK_STREAM_LANE=2"
#endif
#if FLK_GROUP_EARLY_DRAIN && !FLK_GROUP_SUBTILES
#error "Early drain requires grouped subtiles"
#endif
#ifndef FLK_SKIP_INVISIBLE
#define FLK_SKIP_INVISIBLE 0
#endif
#ifndef FLK_EXPAND_TRIM
#define FLK_EXPAND_TRIM 0
#endif
#ifndef FLK_STREAM_LANE
#define FLK_STREAM_LANE 0
#endif
#if FLK_STREAM_COMPACT_REQUEST && FLK_STREAM_LANE != 2
#error "Compact requests require the three-lane streamed evaluator (FLK_STREAM_LANE=2)"
#endif
#if FLK_STREAM_LANE < 0 || FLK_STREAM_LANE > 2
#error "FLK_STREAM_LANE must be 0, 1 or 2"
#endif
#if FLK_STREAM_LANE && (FLK_GROUP_SUBTILES != 3 || FLK_GROUP_EARLY_DRAIN)
#error "Stream lane requires compact grouped records and no early-drain shortcut"
#endif
#ifndef FLK_STREAM_CORES
#define FLK_STREAM_CORES 4
#endif
#if FLK_STREAM_CORES != 3 && FLK_STREAM_CORES != 4
#error "FLK_STREAM_CORES must be 3 or 4"
#endif
#if FLK_STREAM_CORES == 3 && FLK_STREAM_LANE != 2
#error "Balanced three-core mapping requires the compact streaming lane"
#endif
// External DMA and CTU words remain 512 bits. After dispatch, the lane only
// consumes parameters [223:0], subtile [229:228], and the control kind [511:510].
// Keep the low range contiguous; unused mask bits cost less than another ABI.
#if FLK_LANE_FIFO_STORAGE
typedef ap_uint<232> LaneWord;
static LaneWord pack_lane(Word q){
#pragma HLS INLINE
 LaneWord packed=0;packed.range(229,0)=q.range(229,0);
 packed.range(231,230)=q.range(511,510);return packed;
}
static Word unpack_lane(LaneWord packed){
#pragma HLS INLINE
 Word q=0;q.range(229,0)=packed.range(229,0);
 q.range(511,510)=packed.range(231,230);return q;
}
#else
typedef Word LaneWord;
static LaneWord pack_lane(Word q){
#pragma HLS INLINE
 return q;
}
static Word unpack_lane(LaneWord q){
#pragma HLS INLINE
 return q;
}
#endif
#if FLK_EXACT_EXP_ROM == 2
#include "exp_rom/table_compact.hpp"
#elif FLK_EXACT_EXP_ROM == 1
#include "exp_rom/table.hpp"
#endif

// Separate per-Gaussian geometry from sub-tile expansion. Control tokens use
// the same ordered FIFOs as data; downstream backpressure propagates upstream.
static void frame_gaussians(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles){
 for(unsigned t=0;t<tiles;t++){
  Word header=input.read(),control=header;control.range(511,510)=1;output.write(control);
  unsigned n=header.range(31,0);
  for(unsigned j=0;j<n;j++){
#pragma HLS PIPELINE II=1
   Word raw=input.read();raw.range(351,288)=header.range(63,0);
   raw.range(361,352)=header.range(73,64);raw.range(415,384)=j+1;
   raw.range(511,510)=0;output.write(raw);
  }
  control=header;control.range(511,510)=2;output.write(control);
 }
 Word end=0;end.range(511,510)=3;output.write(end);
}

static void prepare_geometry(hls::stream<Word>& input,hls::stream<Word>& output){
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=16
  Word raw=input.read(),q=raw;unsigned type=kind(raw);end=type==3;
  if(type==0){
   q=0;
   for(unsigned k=0;k<9;k++){
#pragma HLS UNROLL
    q.range(k*16+15,k*16)=pack_half(half(unpack_float(raw,k*32)));
   }
   float x=unpack_float(raw,0),y=unpack_float(raw,32);
   float a=unpack_float(raw,64),b=unpack_float(raw,96),c=unpack_float(raw,128);
   float det=a*c-b*b,trace=a+c,delta=hls::sqrtf((a-c)*(a-c)+4.f*b*b);
   bool spiky=(trace+delta)>=9.f*(trace-delta);
   float radius=65504.f;
   if(det>0.f){
    float vx=c/det,vy=a/det,xy=-b/det,mid=.5f*(vx+vy);
    float d=mid*mid-(vx*vy-xy*xy);if(d<.1f)d=.1f;
    radius=hls::ceilf(3.f*hls::sqrtf(mid+hls::sqrtf(d)));
   }
   q[186]=spiky;q.range(223,192)=raw.range(415,384);
   q.range(159,144)=raw.range(335,320);q.range(175,160)=raw.range(351,336);
   q.range(185,176)=raw.range(361,352);
   // Exact original FP32 additions/subtractions, once per Gaussian record.
   q.range(287,256)=fp_struct<float>(x+radius).data();
   q.range(319,288)=fp_struct<float>(y+radius).data();
   q.range(351,320)=fp_struct<float>(x-radius).data();
   q.range(383,352)=fp_struct<float>(y-radius).data();
  }
  output.write(q);
 }
}

// The FP32 projection/radius path is the only stage in split that is still
// intentionally serialized at II=16.  The dual candidate keeps the exact
// arithmetic and record order, but alternates records across two identical
// workers.  It is opt-in because it spends extra floating-point resources.
static void geometry_dispatch(hls::stream<Word>& input,
                              hls::stream<Word>& left,
                              hls::stream<Word>& right){
 bool end=false; unsigned lane=0;
 while(!end){
#pragma HLS PIPELINE II=1
  Word q=input.read(); end=kind(q)==3;
  if(end || lane==0) left.write(q);
  if(end || lane==1) right.write(q);
  if(!end) lane^=1;
 }
}

static void geometry_merge(hls::stream<Word>& left,
                           hls::stream<Word>& right,
                           hls::stream<Word>& output){
 bool left_end=false,right_end=false,emitted_end=false; unsigned lane=0;
 while(!(left_end&&right_end)){
#pragma HLS PIPELINE II=1
  Word q;
  if(lane==0){q=left.read();left_end=kind(q)==3;}
  else{q=right.read();right_end=kind(q)==3;}
  if(kind(q)!=3 || !emitted_end){output.write(q); if(kind(q)==3)emitted_end=true;}
  lane^=1;
 }
}

static void prepare_geometry_dual(hls::stream<Word>& input,hls::stream<Word>& output){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 hls::stream<Word> left,right,left_done,right_done;
 #pragma HLS STREAM variable=left depth=16
 #pragma HLS STREAM variable=right depth=16
 #pragma HLS STREAM variable=left_done depth=16
 #pragma HLS STREAM variable=right_done depth=16
 #pragma HLS RESOURCE variable=left core=FIFO_SRL
 #pragma HLS RESOURCE variable=right core=FIFO_SRL
 #pragma HLS RESOURCE variable=left_done core=FIFO_SRL
 #pragma HLS RESOURCE variable=right_done core=FIFO_SRL
 geometry_dispatch(input,left,right);
 prepare_geometry(left,left_done);
 prepare_geometry(right,right_done);
 geometry_merge(left_done,right_done,output);
}

static void expand_subtiles(hls::stream<Word>& input,hls::stream<Word>& output,unsigned mode){
 bool end=false;
 while(!end){
  Word q=input.read();unsigned type=kind(q);end=type==3;
  if(type)output.write(q);
  else{
#if FLK_SKIP_INVISIBLE
   // The renderer discards alpha below 1/255. Since exp(power) <= 1,
   // opacity below this threshold can never produce a visible contribution.
   if(unpack_half(q,80)<half(1.f/255.f)) continue;
#endif
   unsigned ox=q.range(159,144),oy=q.range(175,160),width=q.range(180,176),height=q.range(185,181);
   float xmax=unpack_float(q,256),ymax=unpack_float(q,288),xmin=unpack_float(q,320),ymin=unpack_float(q,352);
   // A projected Gaussian often covers one or two 8x8 subtiles.  The
   // default loop is retained for the installed bitstream; the opt-in
   // variant shortens the loop trip count from the integer footprint so no
   // empty subtiles enter the CTU.  The emitted order is still sub=0,1,2,3.
#if FLK_EXPAND_TRIM
   unsigned subtiles_x=width>8?2:1,subtiles_y=height>8?2:1;
   unsigned sub_limit=subtiles_x*subtiles_y;
#else
   unsigned sub_limit=4;
#endif
   for(unsigned sub=0;sub<sub_limit;sub++){
#pragma HLS PIPELINE II=4
#if FLK_EXPAND_TRIM
    unsigned sub_id=subtiles_x==1?sub*2:sub;
#else
    unsigned sub_id=sub;
#endif
    unsigned sx=(sub_id&1)*8,sy=(sub_id>>1)*8;
    if(sx<width&&sy<height){
     unsigned w=width-sx,h=height-sy;if(w>8)w=8;if(h>8)h=8;
     unsigned px=ox+sx,py=oy+sy;
     bool hit=xmax>=float(px)&&ymax>=float(py)&&xmin<=float(px+w-1)&&ymin<=float(py+h-1);
     if(mode==0||hit){
      Word p=q;p.range(383,256)=0;
      p.range(159,144)=px;p.range(175,160)=py;p.range(180,176)=w;p.range(185,181)=h;p.range(229,228)=sub_id;
      output.write(p);
     }
    }
   }
  }
 }
}

static void split(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 hls::stream<Word> framed,geometry;
#pragma HLS STREAM variable=framed depth=16
#pragma HLS STREAM variable=geometry depth=16
#pragma HLS RESOURCE variable=framed core=FIFO_BRAM
#pragma HLS RESOURCE variable=geometry core=FIFO_BRAM
 frame_gaussians(input,framed,tiles);
#if FLK_GEOM_DUAL
 prepare_geometry_dual(framed,geometry);
#else
 prepare_geometry(framed,geometry);
#endif
 expand_subtiles(geometry,output,mode);
}

#ifdef FLK_SPLIT_TEST
void flicker_split_probe(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode){
#pragma HLS INTERFACE ap_fifo port=input
#pragma HLS INTERFACE ap_fifo port=output
#pragma HLS INTERFACE ap_ctrl_hs port=return
 split(input,output,tiles,mode);
}
#endif

#if FLK_STREAM_CORES == 3
static void dispatch(hls::stream<Word>& input,hls::stream<LaneWord>& a,hls::stream<LaneWord>& b,hls::stream<LaneWord>& c){
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Word q=input.read();unsigned type=kind(q);end=type==3;
  // Keep the original CTU spatial masks and apply them per remapped pixel.
  if(type||q.range(227,224)!=0){LaneWord packed=pack_lane(q);a.write(packed);b.write(packed);c.write(packed);}
 }
}
#else
static void dispatch(hls::stream<Word>& input,hls::stream<LaneWord>& a,hls::stream<LaneWord>& b,hls::stream<LaneWord>& c,hls::stream<LaneWord>& d){
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  Word q=input.read();unsigned type=kind(q);end=type==3;
  LaneWord packed=pack_lane(q);
  if(type||q[224])a.write(packed);
  if(type||q[225])b.write(packed);
  if(type||q[226])c.write(packed);
  if(type||q[227])d.write(packed);
 }
}
#endif

static half power(half x,half y,half mx,half my,half a,half b,half c){
#pragma HLS INLINE
 half dx=half(mx-x),dy=half(my-y);
 half ax=half(a*dx),cy=half(c*dy),bx=half(b*dx);
 half xx=half(ax*dx),yy=half(cy*dy),xy=half(bx*dy);
 return half(half(half(-.5f)*half(xx+yy))-xy);
}

template<unsigned LANE> static void evaluate_mini(Word q,half rr[16],half gg[16],half bb[16],half tt[16],unsigned last[16],bool stopped[16]){
#pragma HLS INLINE off
 unsigned ox=q.range(159,144),oy=q.range(175,160),w=q.range(180,176),h=q.range(185,181),ordinal=q.range(223,192);
 half mx=unpack_half(q,0),my=unpack_half(q,16),a=unpack_half(q,32),b=unpack_half(q,48),c=unpack_half(q,64);
 half opacity=unpack_half(q,80),red=unpack_half(q,96),green=unpack_half(q,112),blue=unpack_half(q,128);
 for(unsigned p=0;p<16;p++){
#pragma HLS PIPELINE II=1
  unsigned x=(LANE&1)*4+p%4,y=(LANE>>1)*4+p/4;
  if(x<w&&y<h&&!stopped[p]){
   half pw=power(half(ox+x),half(oy+y),mx,my,a,b,c);
   if(pw<=half(0)){
#ifdef FLK_EXACT_EXP_ROM
    half exponential=exact_negative_exp<LANE>(pw);
#else
    half exponential=hls::half_exp(pw);
#endif
    half alpha=half(opacity*exponential);if(alpha>half(.99f))alpha=half(.99f);
    if(alpha>=half(1.f/255.f)){
     half next=half(tt[p]*half(half(1)-alpha));
     if(next<half(.0001f))stopped[p]=true;
     else{half weight=half(alpha*tt[p]);rr[p]=half(rr[p]+half(red*weight));gg[p]=half(gg[p]+half(green*weight));bb[p]=half(bb[p]+half(blue*weight));tt[p]=next;last[p]=ordinal;}
    }
   }
  }
 }
}

#if FLK_GROUP_SUBTILES
#if FLK_GROUP_SUBTILES == 3
// The evaluator reads only bits 0..223: half parameters, origin/extent, ordinal.
// Control/subtile tags are consumed by render_lane before storing the record.
#if FLK_STREAM_CORES == 3
// Three-core ownership needs the original four mask bits after dispatch.
typedef ap_uint<228> GroupRecord;
#else
typedef ap_uint<224> GroupRecord;
#endif
#else
typedef Word GroupRecord;
#endif
// Each lane owns 64 different pixels across four subtiles. Evaluate one
// Gaussian's selected subtiles in a single monotonic 64-pixel traversal.
// A complete call commits before the next Gaussian, so alpha order is intact.
template<unsigned LANE> static unsigned evaluate_group(GroupRecord records[4],ap_uint<4> valid,
 half rr[64],half gg[64],half bb[64],half tt[64],unsigned last[64],bool stopped[64]){
#pragma HLS INLINE off
 unsigned retired=0;
#if FLK_DPC
 // One recurrence state per subtile.  The four-row 4x4 lane footprint is
 // initialized once per row; pixels in that row advance with additions.
 half dpc_pw[4],dpc_dx[4];
#pragma HLS ARRAY_PARTITION variable=dpc_pw complete
#pragma HLS ARRAY_PARTITION variable=dpc_dx complete
#endif
#if FLK_GROUP_SHARED_ATTR
 // Dispatch groups records from one Gaussian ordinal. Its FP16 parameters and
 // ordinal are identical in every selected subtile; only origin/extent vary.
 // Load invariant attributes once, avoiding long per-pixel parameter pipelines.
 // The caller only enters with valid != 0, including sparse nonzero first slots.
 unsigned first=valid[0]?0:(valid[1]?1:(valid[2]?2:3));
 Word shared=records[first];
 unsigned ordinal=shared.range(223,192);
 half mx=unpack_half(shared,0),my=unpack_half(shared,16);
 half a=unpack_half(shared,32),b=unpack_half(shared,48),c=unpack_half(shared,64);
 half opacity=unpack_half(shared,80),red=unpack_half(shared,96);
 half green=unpack_half(shared,112),blue=unpack_half(shared,128);
#endif
#if FLK_GROUP_TRIM_RANGE
 // Skip wholly absent trailing subtiles. The constant zero start and explicit
 // bound retain HLS's proof that all feedback addresses are distinct within a
 // call. A variable start caused II=60 in the preserved rejected experiment.
 unsigned limit=valid[3]?64:(valid[2]?48:(valid[1]?32:16));
 for(unsigned p=0;p<64 && p<limit;p++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=64
#else
 for(unsigned p=0;p<64;p++){
#endif
#pragma HLS PIPELINE II=1
  unsigned sub=p/16,k=p%16;Word q=records[sub];
  unsigned ox=q.range(159,144),oy=q.range(175,160),w=q.range(180,176),h=q.range(185,181);
#if !FLK_GROUP_SHARED_ATTR
  unsigned ordinal=q.range(223,192);
#endif
  unsigned x=(LANE&1)*4+k%4,y=(LANE>>1)*4+k/4;
  if(valid[sub]&&x<w&&y<h&&!stopped[p]){
#if !FLK_GROUP_SHARED_ATTR
   half mx=unpack_half(q,0),my=unpack_half(q,16),a=unpack_half(q,32),b=unpack_half(q,48),c=unpack_half(q,64);
   half opacity=unpack_half(q,80),red=unpack_half(q,96),green=unpack_half(q,112),blue=unpack_half(q,128);
#endif
   half pw;
#if FLK_DPC
   // With dx=mx-x and dy=my-y, advancing x by one changes power by
   // a*dx - a/2 + b*dy.  Reusing that delta removes the per-pixel
   // quadratic products while preserving the original depth/alpha order.
   if((k&3)==0){
    half dx=half(mx-half(ox+x)),dy=half(my-half(oy+y));
    dpc_pw[sub]=power(half(ox+x),half(oy+y),mx,my,a,b,c);
    dpc_dx[sub]=half(half(a*dx)-half(half(.5f)*a)+half(b*dy));
   }
   pw=dpc_pw[sub];
   dpc_pw[sub]=half(dpc_pw[sub]+dpc_dx[sub]);
   dpc_dx[sub]=half(dpc_dx[sub]-a);
#else
   pw=power(half(ox+x),half(oy+y),mx,my,a,b,c);
#endif
   if(pw<=half(0)){
#ifdef FLK_EXACT_EXP_ROM
    half exponential=exact_negative_exp<LANE>(pw);
#else
    half exponential=hls::half_exp(pw);
#endif
    half alpha=half(opacity*exponential);if(alpha>half(.99f))alpha=half(.99f);
    if(alpha>=half(1.f/255.f)){
     half next=half(tt[p]*half(half(1)-alpha));
     if(next<half(.0001f)){
      stopped[p]=true;
#if FLK_GROUP_EARLY_DRAIN
      // This reduction is not read by the pixel arithmetic. Count only the
      // first transition; already-stopped pixels never enter this branch.
      retired++;
#endif
     }
     else{half weight=half(alpha*tt[p]);rr[p]=half(rr[p]+half(red*weight));gg[p]=half(gg[p]+half(green*weight));bb[p]=half(bb[p]+half(blue*weight));tt[p]=next;last[p]=ordinal;}
    }
   }
  }
 }
 return retired;
}
#endif

template<unsigned LANE> static void render_lane(hls::stream<LaneWord>& input,hls::stream<PixelBits>& output){
#pragma HLS INLINE off
 half rr[64],gg[64],bb[64],tt[64];unsigned last[64];bool stopped[64];
 half mr[16],mg[16],mb[16],mt[16];unsigned ml[16];bool ms[16];
#if FLK_STATE_PORTS == 2
// True dual-port memories accept two state writes per cycle without banking.
#pragma HLS RESOURCE variable=rr core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=last core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=stopped core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=mr core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=mg core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=mb core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=mt core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=ml core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=ms core=RAM_T2P_BRAM
#else
#pragma HLS RESOURCE variable=rr core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_2P_BRAM
#endif
#if FLK_GROUP_SUBTILES
 GroupRecord records[4];ap_uint<4> valid=0;unsigned group_ordinal=0;
#if FLK_GROUP_EARLY_DRAIN
 // Sixty-four includes invalid edge pixels. Partial tiles conservatively
 // retain work instead of requiring new header or active-mask semantics.
 unsigned remaining=64;
#endif
#if FLK_GROUP_SUBTILES == 2
 // Four wide records are too shallow for efficient BRAM mapping. Register
 // only this read-only evaluator input; never partition the pixel feedback.
#pragma HLS ARRAY_PARTITION variable=records complete dim=1
#elif FLK_GROUP_SUBTILES == 3
 // A shallow parameter cache does not need large BRAMs or fully partitioned
 // registers. Pixel-state memories and their feedback dependencies are intact.
#pragma HLS RESOURCE variable=records core=RAM_2P_LUTRAM
#endif
 // Initialize inactive slots so C simulation never reads uninitialized words.
 for(unsigned sub=0;sub<4;sub++)records[sub]=0;
#endif
 bool end=false;unsigned tile=0;
 while(!end){
  Word q=unpack_lane(input.read());unsigned type=kind(q);
#if FLK_GROUP_SUBTILES
  unsigned incoming_ordinal=q.range(223,192);
  if(valid!=0&&(type!=0||incoming_ordinal!=group_ordinal)){
#if FLK_GROUP_EARLY_DRAIN
   remaining-=evaluate_group<LANE>(records,valid,rr,gg,bb,tt,last,stopped);
#else
   evaluate_group<LANE>(records,valid,rr,gg,bb,tt,last,stopped);
#endif
   valid=0;
  }
#endif
  if(type==3)end=true;
  else if(type==1){
#if FLK_GROUP_EARLY_DRAIN
   remaining=64;
#endif
   for(unsigned p=0;p<64;p++){
#pragma HLS PIPELINE II=1
    rr[p]=gg[p]=bb[p]=half(0);tt[p]=half(1);last[p]=0;stopped[p]=false;
   }
  }else if(type==2){
   half br=unpack_half(q,80),bg=unpack_half(q,96),bc=unpack_half(q,112);
   for(unsigned p=0;p<64;p++){
#pragma HLS PIPELINE II=1
    unsigned sub=p/16,k=p%16,x=(sub&1)*8+(LANE&1)*4+k%4,y=(sub>>1)*8+(LANE>>1)*4+k/4;
    PixelBits out=0;out.range(15,0)=pack_half(half(rr[p]+half(br*tt[p])));
    out.range(31,16)=pack_half(half(gg[p]+half(bg*tt[p])));
    out.range(47,32)=pack_half(half(bb[p]+half(bc*tt[p])));out.range(63,48)=pack_half(tt[p]);
    out.range(95,64)=last[p];out.range(127,96)=tile*256+y*16+x;output.write(out);
   }
   tile++;
  }else{
   unsigned sub=q.range(229,228);
#if FLK_GROUP_SUBTILES
#if FLK_GROUP_EARLY_DRAIN
   // Drain all input/control records even when no pixel can change again;
   // otherwise upstream backpressure would deadlock completion and DMA.
   if(remaining!=0){records[sub]=q;valid[sub]=1;group_ordinal=incoming_ordinal;}
#else
   records[sub]=q;valid[sub]=1;group_ordinal=incoming_ordinal;
#endif
#else
   // A bounded 16-pixel working set makes both ownership and aliasing explicit.
   // Load, evaluate, and commit finish before the next ordered Gaussian starts.
   for(unsigned k=0;k<16;k++){
#pragma HLS PIPELINE II=1
#if FLK_STATE_PORTS == 2
#pragma HLS UNROLL factor=2
#endif
    unsigned p=sub*16+k;mr[k]=rr[p];mg[k]=gg[p];mb[k]=bb[p];mt[k]=tt[p];ml[k]=last[p];ms[k]=stopped[p];
   }
   evaluate_mini<LANE>(q,mr,mg,mb,mt,ml,ms);
   for(unsigned k=0;k<16;k++){
#pragma HLS PIPELINE II=1
#if FLK_STATE_PORTS == 2
#pragma HLS UNROLL factor=2
#endif
    unsigned p=sub*16+k;rr[p]=mr[k];gg[p]=mg[k];bb[p]=mb[k];tt[p]=mt[k];last[p]=ml[k];stopped[p]=ms[k];
   }
#endif
  }
 }
}

#if FLK_STREAM_LANE == 2
#include "streamed_lane_compact.hpp"
#elif FLK_STREAM_LANE == 1
#include "streamed_lane.hpp"
#endif

#if FLK_STREAM_CORES == 3
static void gather(hls::stream<PixelBits>& a,hls::stream<PixelBits>& b,hls::stream<PixelBits>& c,hls::stream<Word>& output,unsigned tiles){
 PixelBits pixels[3][86];
#pragma HLS ARRAY_PARTITION variable=pixels complete dim=1
 for(unsigned t=0;t<tiles;t++){
  unsigned na=0,nb=0,nc=0;
  while(na<86||nb<85||nc<85){
#pragma HLS PIPELINE II=1
   if(na<86&&!a.empty()){pixels[0][na]=a.read();na++;}
   if(nb<85&&!b.empty()){pixels[1][nb]=b.read();nb++;}
   if(nc<85&&!c.empty()){pixels[2][nc]=c.read();nc++;}
  }
  Word word=0;unsigned lane=0,index=0;
  for(unsigned p=0;p<256;p++){
#pragma HLS PIPELINE II=1
   word=word>>128;word.range(511,384)=pixels[lane][index];
   if(p%4==3)output.write(word);
   if(lane==2){lane=0;index++;}else lane++;
  }
 }
}
#else
static void gather(hls::stream<PixelBits>& a,hls::stream<PixelBits>& b,hls::stream<PixelBits>& c,hls::stream<PixelBits>& d,hls::stream<Word>& output,unsigned tiles){
 PixelBits pixels[4][64];
#pragma HLS ARRAY_PARTITION variable=pixels complete dim=1
 for(unsigned t=0;t<tiles;t++){
#if FLK_STREAM_LANE
  // A lane may finish earlier than another once evaluation overlaps blending.
  // Drain each lane independently into the existing one-Tile bank. A lockstep
  // a/b/c/d read can hold a fast lane's completed Tile behind a slow lane and
  // backpressure the shared dispatcher before it delivers that slow lane's end.
  // Counters are independent ownership; serialization below remains identical.
  unsigned na=0,nb=0,nc=0,nd=0;
  while(na<64||nb<64||nc<64||nd<64){
#pragma HLS PIPELINE II=1
   if(na<64&&!a.empty()){pixels[0][na]=a.read();na++;}
   if(nb<64&&!b.empty()){pixels[1][nb]=b.read();nb++;}
   if(nc<64&&!c.empty()){pixels[2][nc]=c.read();nc++;}
   if(nd<64&&!d.empty()){pixels[3][nd]=d.read();nd++;}
  }
#else
  for(unsigned p=0;p<64;p++){
#pragma HLS PIPELINE II=1
   pixels[0][p]=a.read();pixels[1][p]=b.read();pixels[2][p]=c.read();pixels[3][p]=d.read();
  }
#endif
  Word word=0;
  for(unsigned p=0;p<256;p++){
#pragma HLS PIPELINE II=1
   unsigned x=p&15,y=p>>4,lane=((x>>2)&1)+2*((y>>2)&1),sub=(x>>3)+2*(y>>3),k=(y%4)*4+x%4;
   // Fixed wiring: after four pixels the earliest pixel is in bits 127:0.
   // A variable .range() creates a large 512-bit barrel shifter in HLS 2018.3.
   word=word>>128;word.range(511,384)=pixels[lane][sub*16+k];
   if(p%4==3)output.write(word);
  }
 }
}

#endif

void flicker_render_pipeline(hls::stream<Word>& input,hls::stream<Word>& output,unsigned tiles,unsigned mode){
#pragma HLS INTERFACE ap_fifo port=input
#pragma HLS INTERFACE ap_fifo port=output
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS DATAFLOW
 hls::stream<Word> subtiles,masks;
 hls::stream<LaneWord> q0,q1,q2;
 hls::stream<PixelBits> p0,p1,p2;
#if FLK_STREAM_CORES == 4
 hls::stream<LaneWord> q3;hls::stream<PixelBits> p3;
#endif
#pragma HLS STREAM variable=subtiles depth=16
#pragma HLS STREAM variable=masks depth=16
#pragma HLS STREAM variable=q0 depth=8
#pragma HLS STREAM variable=q1 depth=8
#pragma HLS STREAM variable=q2 depth=8
#if FLK_STREAM_CORES == 4
#pragma HLS STREAM variable=q3 depth=8
#endif
#pragma HLS STREAM variable=p0 depth=8
#pragma HLS STREAM variable=p1 depth=8
#pragma HLS STREAM variable=p2 depth=8
#if FLK_STREAM_CORES == 4
#pragma HLS STREAM variable=p3 depth=8
#endif
#pragma HLS RESOURCE variable=subtiles core=FIFO_SRL
#pragma HLS RESOURCE variable=masks core=FIFO_SRL
#if FLK_LANE_FIFO_STORAGE == 2
// Use spare block RAM for queues, retaining one word/cycle and FIFO ordering.
// HLS 2018.3 estimates thirteen BRAM18 lanes per 232-bit FIFO; physical
// synthesis may trim more unused bits, so the whole-board report decides fit.
#pragma HLS RESOURCE variable=q0 core=FIFO_BRAM
#pragma HLS RESOURCE variable=q1 core=FIFO_BRAM
#pragma HLS RESOURCE variable=q2 core=FIFO_BRAM
#if FLK_STREAM_CORES == 4
#pragma HLS RESOURCE variable=q3 core=FIFO_BRAM
#endif
#else
#pragma HLS RESOURCE variable=q0 core=FIFO_SRL
#pragma HLS RESOURCE variable=q1 core=FIFO_SRL
#pragma HLS RESOURCE variable=q2 core=FIFO_SRL
#if FLK_STREAM_CORES == 4
#pragma HLS RESOURCE variable=q3 core=FIFO_SRL
#endif
#endif
#pragma HLS RESOURCE variable=p0 core=FIFO_SRL
#pragma HLS RESOURCE variable=p1 core=FIFO_SRL
#pragma HLS RESOURCE variable=p2 core=FIFO_SRL
#if FLK_STREAM_CORES == 4
#pragma HLS RESOURCE variable=p3 core=FIFO_SRL
#endif
 split(input,subtiles,tiles,mode);framed_ctu(subtiles,masks,mode);
#if FLK_STREAM_CORES == 3
 dispatch(masks,q0,q1,q2);
 render_lane_stream<0>(q0,p0);render_lane_stream<1>(q1,p1);render_lane_stream<2>(q2,p2);
 gather(p0,p1,p2,output,tiles);
#else
 dispatch(masks,q0,q1,q2,q3);
#if FLK_STREAM_LANE
 render_lane_stream<0>(q0,p0);render_lane_stream<1>(q1,p1);render_lane_stream<2>(q2,p2);render_lane_stream<3>(q3,p3);
#else
 render_lane<0>(q0,p0);render_lane<1>(q1,p1);render_lane<2>(q2,p2);render_lane<3>(q3,p3);
#endif
 gather(p0,p1,p2,p3,output,tiles);
#endif
}
