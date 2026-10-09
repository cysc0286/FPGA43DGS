#pragma once
#include "coord_half.hpp"

// Keep the frozen FP16 arithmetic while making the pixel partition explicit.
// Four lanes use a 4x4 quadrant per subtile; two lanes use a 4x8 strip.
#if FLK_STREAM_CORES == 2
#define HGR_PIXELS_PER_LANE 128
#define HGR_PIXELS_PER_SUBTILE 32
#elif FLK_STREAM_CORES == 4
#define HGR_PIXELS_PER_LANE 64
#define HGR_PIXELS_PER_SUBTILE 16
#else
#error "HGR supports only two or four physical lanes"
#endif
#define HGR_STOP_WIDTH (32 + HGR_PIXELS_PER_LANE)
#define HGR_TILE_MSB (HGR_STOP_WIDTH - 1)
#define HGR_TILE_LSB HGR_PIXELS_PER_LANE
#define HGR_STOP_MSB (HGR_PIXELS_PER_LANE - 1)

struct HgrMeta {
 ap_uint<2> type;
 ap_uint<16> red,green,blue;
 ap_uint<32> ordinal;
 ap_uint<8> limit;
};
typedef ap_uint<90> HgrMetaWord;
typedef ap_uint<17> HgrAlpha;
typedef ap_uint<HGR_STOP_WIDTH> HgrStopWord;

static HgrMetaWord pack_hgr_meta(const HgrMeta& meta){
#pragma HLS INLINE
 HgrMetaWord bits=0;
 bits.range(15,0)=meta.red;bits.range(31,16)=meta.green;
 bits.range(47,32)=meta.blue;bits.range(79,48)=meta.ordinal;
 bits.range(81,80)=meta.type;bits.range(89,82)=meta.limit;
 return bits;
}
static HgrMeta unpack_hgr_meta(HgrMetaWord bits){
#pragma HLS INLINE
 HgrMeta meta;
 meta.red=bits.range(15,0);meta.green=bits.range(31,16);
 meta.blue=bits.range(47,32);meta.ordinal=bits.range(79,48);
 meta.type=bits.range(81,80);meta.limit=bits.range(89,82);
 return meta;
}

template<unsigned LANE> static void hgr_emit_group(
 GroupRecord records[4],ap_uint<4> valid,unsigned tile,ap_uint<HGR_PIXELS_PER_LANE>& stopped,
 hls::stream<HgrStopWord>& feedback,hls::stream<HgrMetaWord>& metadata,
 hls::stream<HgrAlpha>& alpha_out){
#pragma HLS INLINE off
 // Nonblocking in both directions breaks the feedback wait cycle. Epochs
 // prevent the tail of one Tile from affecting the next Tile's live pixels.
 for(unsigned n=0;n<2;n++){
  HgrStopWord status;
  if(feedback.read_nb(status)&&status.range(HGR_TILE_MSB,HGR_TILE_LSB)==tile)
   stopped|=status.range(HGR_STOP_MSB,0);
 }
 unsigned first=valid[0]?0:(valid[1]?1:(valid[2]?2:3));
 GroupRecord shared=records[first];
 half mx=unpack_half(shared,0),my=unpack_half(shared,16);
 half a=unpack_half(shared,32),b=unpack_half(shared,48),c=unpack_half(shared,64);
 half opacity=unpack_half(shared,80);
 unsigned limit=valid[3]?4*HGR_PIXELS_PER_SUBTILE:
                (valid[2]?3*HGR_PIXELS_PER_SUBTILE:
                (valid[1]?2*HGR_PIXELS_PER_SUBTILE:HGR_PIXELS_PER_SUBTILE));
 HgrMeta meta;
 meta.type=0;meta.red=shared.range(111,96);meta.green=shared.range(127,112);
 meta.blue=shared.range(143,128);meta.ordinal=shared.range(223,192);
 meta.limit=limit;metadata.write(pack_hgr_meta(meta));
 for(unsigned p=0;p<HGR_PIXELS_PER_LANE&&p<limit;p++){
#pragma HLS LOOP_TRIPCOUNT min=32 max=128
#pragma HLS PIPELINE II=1
  unsigned sub=p/HGR_PIXELS_PER_SUBTILE,k=p%HGR_PIXELS_PER_SUBTILE;
  GroupRecord q=records[sub];
  unsigned ox=q.range(159,144),oy=q.range(175,160);
  unsigned w=q.range(180,176),h=q.range(185,181);
  unsigned x=(LANE&1)*4+k%4;
#if FLK_STREAM_CORES == 2
  unsigned y=k/4;
#else
  unsigned y=(LANE>>1)*4+k/4;
#endif
  HgrAlpha token=0;
  if(valid[sub]&&x<w&&y<h&&!stopped[p]){
   half pw=power(coord_half(ox+x),coord_half(oy+y),mx,my,a,b,c);
   if(pw<=half(0)){
    half alpha=half(opacity*exact_negative_exp<LANE>(pw));
    if(alpha>half(.99f))alpha=half(.99f);
    if(alpha>=half(1.f/255.f)){token.range(15,0)=pack_half(alpha);token[16]=1;}
   }
  }
  alpha_out.write(token);
 }
}

template<unsigned LANE> static void hgr_packetize_precompute(
 hls::stream<LaneWord>& input,hls::stream<HgrStopWord>& feedback,
 hls::stream<HgrMetaWord>& metadata,hls::stream<HgrAlpha>& alpha_out){
#pragma HLS INLINE off
 GroupRecord records[4];
#pragma HLS RESOURCE variable=records core=RAM_2P_LUTRAM
 for(unsigned sub=0;sub<4;sub++)records[sub]=0;
 ap_uint<4> valid=0;ap_uint<HGR_PIXELS_PER_LANE> stopped=0;
 unsigned group_ordinal=0,tile=0;
 bool have_group=false,end=false,pending=false;
 LaneWord held=0;
 while(!end){
  LaneWord raw=pending?held:input.read();
  Word q=unpack_lane(raw);
  pending=false;
  unsigned type=kind(q),incoming_ordinal=q.range(223,192);
  if(type==0){
   if(have_group&&incoming_ordinal!=group_ordinal){
    hgr_emit_group<LANE>(records,valid,tile,stopped,feedback,metadata,alpha_out);
    valid=0;have_group=false;held=raw;pending=true;
   }else{
    if(!have_group){group_ordinal=incoming_ordinal;have_group=true;}
    unsigned sub=q.range(229,228);records[sub]=q.range(223,0);valid[sub]=1;
   }
  }else if(have_group){
   hgr_emit_group<LANE>(records,valid,tile,stopped,feedback,metadata,alpha_out);
   valid=0;have_group=false;held=raw;pending=true;
  }else{
   HgrMeta meta;
   meta.type=type;meta.red=0;meta.green=0;meta.blue=0;meta.ordinal=0;meta.limit=0;
   if(type==1)stopped=0;
   if(type==2){
    meta.red=q.range(95,80);meta.green=q.range(111,96);meta.blue=q.range(127,112);
    tile++;
   }
   metadata.write(pack_hgr_meta(meta));
   if(type==3)end=true;
  }
 }
 // Hardware processes overlap; consume the final completion token before
 // returning so feedback cannot survive into the next top-level transaction.
 // Sequential C simulation has no inter-process feedback and destroys this
 // local channel after the call. RTL co-simulation verifies the actual cycle.
#ifdef __SYNTHESIS__
 HgrStopWord status;
   do {status=feedback.read();} while(status.range(HGR_TILE_MSB,HGR_TILE_LSB)!=ap_uint<32>(0xffffffffu));
#endif
}

template<unsigned LANE> static void hgr_composite(
 hls::stream<HgrMetaWord>& metadata,hls::stream<HgrAlpha>& alpha_in,
 hls::stream<HgrStopWord>& feedback,hls::stream<PixelBits>& output){
#pragma HLS INLINE off
 half rr[HGR_PIXELS_PER_LANE],gg[HGR_PIXELS_PER_LANE],bb[HGR_PIXELS_PER_LANE],tt[HGR_PIXELS_PER_LANE];
 unsigned last[HGR_PIXELS_PER_LANE];
#pragma HLS RESOURCE variable=rr core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_2P_BRAM
 bool stopped[HGR_PIXELS_PER_LANE];
 bool end=false;unsigned tile=0;
 while(!end){
  HgrMeta meta=unpack_hgr_meta(metadata.read());
  if(meta.type==3){
   HgrStopWord done=0;done.range(HGR_TILE_MSB,HGR_TILE_LSB)=0xffffffffu;
   feedback.write(done);end=true;
  }
  else if(meta.type==1){
   for(unsigned p=0;p<HGR_PIXELS_PER_LANE;p++){
#pragma HLS PIPELINE II=1
    rr[p]=half(0);gg[p]=half(0);bb[p]=half(0);tt[p]=half(1);last[p]=0;stopped[p]=false;
   }
  }else if(meta.type==2){
   half br=unpack_half16(meta.red),bg=unpack_half16(meta.green),bc=unpack_half16(meta.blue);
   for(unsigned p=0;p<HGR_PIXELS_PER_LANE;p++){
#pragma HLS PIPELINE II=1
    unsigned sub=p/HGR_PIXELS_PER_SUBTILE,k=p%HGR_PIXELS_PER_SUBTILE;
    unsigned x=(sub&1)*8+(LANE&1)*4+k%4;
#if FLK_STREAM_CORES == 2
    unsigned y=(sub>>1)*8+k/4;
#else
    unsigned y=(sub>>1)*8+(LANE>>1)*4+k/4;
#endif
    PixelBits out=0;
    out.range(15,0)=pack_half(half(rr[p]+half(br*tt[p])));
    out.range(31,16)=pack_half(half(gg[p]+half(bg*tt[p])));
    out.range(47,32)=pack_half(half(bb[p]+half(bc*tt[p])));
    out.range(63,48)=pack_half(tt[p]);out.range(95,64)=last[p];
    out.range(127,96)=tile*256+y*16+x;output.write(out);
   }
   tile++;
  }else{
   half red=unpack_half16(meta.red),green=unpack_half16(meta.green),blue=unpack_half16(meta.blue);
   unsigned limit=meta.limit;
   ap_uint<HGR_PIXELS_PER_LANE> snapshot=0;
   for(unsigned p=0;p<HGR_PIXELS_PER_LANE&&p<limit;p++){
#pragma HLS LOOP_TRIPCOUNT min=32 max=128
#pragma HLS PIPELINE II=1
    HgrAlpha token=alpha_in.read();
    bool stop=stopped[p];
    if(token[16]&&!stop){
     half alpha=unpack_half16(token.range(15,0));
     half next=half(tt[p]*half(half(1)-alpha));
     if(next<half(.0001f))stop=true;
     else{
      half weight=half(alpha*tt[p]);
      rr[p]=half(rr[p]+half(red*weight));gg[p]=half(gg[p]+half(green*weight));
      bb[p]=half(bb[p]+half(blue*weight));tt[p]=next;last[p]=meta.ordinal;
     }
    }
    stopped[p]=stop;
    // Packing is outside the state read path; a packed stop-state update
    // would serialize every pixel behind the floating-point decision.
    // Keep the newest stop bit at the top of the physical lane mask.  The
    // four-lane implementation used bit 63; two lanes own 128 pixels each.
    // Hard-coding 63 silently corrupts the feedback mask only in RTL, because
    // C simulation executes the producer and consumer sequentially.
    snapshot>>=1;snapshot[HGR_PIXELS_PER_LANE-1]=stop;
   }
   HgrStopWord status=0;status.range(HGR_TILE_MSB,HGR_TILE_LSB)=tile;
   status.range(HGR_STOP_MSB,0)=snapshot>>(HGR_PIXELS_PER_LANE-limit);
   feedback.write_nb(status);
  }
 }
}

template<unsigned LANE> static void render_lane_hgr(hls::stream<LaneWord>& input,
                                                  hls::stream<PixelBits>& output){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 hls::stream<HgrMetaWord> metadata;
 hls::stream<HgrAlpha> alpha;
 hls::stream<HgrStopWord> feedback;
#pragma HLS STREAM variable=metadata depth=2
#pragma HLS STREAM variable=alpha depth=128
#pragma HLS STREAM variable=feedback depth=2
#pragma HLS RESOURCE variable=metadata core=FIFO_SRL
#pragma HLS RESOURCE variable=alpha core=FIFO_SRL
#pragma HLS RESOURCE variable=feedback core=FIFO_SRL
 hgr_packetize_precompute<LANE>(input,feedback,metadata,alpha);
 hgr_composite<LANE>(metadata,alpha,feedback,output);
}
