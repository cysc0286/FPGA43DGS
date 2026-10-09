// Opt-in lane: overlap stateless Gaussian evaluation with ordered accumulation.
// Included by pipeline.cpp after LaneWord, GroupRecord and power are defined.
// No false-dependence pragma is used on per-pixel feedback state. Accumulation
// finishes each bounded group before reading the next group's header.
typedef ap_uint<256> EvalWord;
typedef ap_uint<128> BlendWord;

static EvalWord lane_command(Word q,unsigned type,unsigned count=0){
#pragma HLS INLINE
 EvalWord packet=0;packet.range(223,0)=q.range(223,0);
 packet.range(232,230)=type;packet.range(240,234)=count;return packet;
}

template<unsigned LANE> static void emit_group(GroupRecord records[4],ap_uint<4> valid,
 hls::stream<EvalWord>& output){
#pragma HLS INLINE off
 unsigned first=valid[0]?0:(valid[1]?1:(valid[2]?2:3));
 unsigned limit=valid[3]?64:(valid[2]?48:(valid[1]?32:16));
 output.write(lane_command(records[first],4,limit));
 for(unsigned p=0;p<64&&p<limit;p++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=64
  unsigned sub=p/16,k=p%16;Word q=records[sub];
  unsigned w=q.range(180,176),h=q.range(185,181);
  unsigned x=(LANE&1)*4+k%4,y=(LANE>>1)*4+k/4;
  EvalWord packet=lane_command(q,0);packet.range(229,224)=p;
  packet[233]=valid[sub]&&x<w&&y<h;output.write(packet);
 }
}

template<unsigned LANE> static void produce_groups(hls::stream<LaneWord>& input,
 hls::stream<EvalWord>& output){
#pragma HLS INLINE off
 GroupRecord records[4];ap_uint<4> valid=0;unsigned ordinal=0;bool end=false;
#pragma HLS RESOURCE variable=records core=RAM_2P_LUTRAM
 for(unsigned i=0;i<4;i++)records[i]=0;
 while(!end){
  Word q=unpack_lane(input.read());unsigned type=kind(q);
  unsigned incoming=q.range(223,192);
  if(valid!=0&&(type!=0||incoming!=ordinal)){
   emit_group<LANE>(records,valid,output);valid=0;
  }
  if(type==0){unsigned sub=q.range(229,228);records[sub]=q;valid[sub]=1;ordinal=incoming;}
  else{output.write(lane_command(q,type));end=type==3;}
 }
}

template<unsigned LANE> static void evaluate_pixels(hls::stream<EvalWord>& input,
 hls::stream<BlendWord>& output){
#pragma HLS INLINE off
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  EvalWord packet=input.read();unsigned type=packet.range(232,230);end=type==3;
  Word q=0;q.range(223,0)=packet.range(223,0);
  BlendWord result=0;result.range(90,88)=type;
  if(type==4){
   result.range(47,0)=q.range(143,96);
   result.range(79,48)=q.range(223,192);result.range(86,80)=packet.range(240,234);
  }else if(type==2){result.range(47,0)=q.range(127,80);}
  else if(type==0&&packet[233]){
   unsigned p=packet.range(229,224),k=p%16;
   unsigned x=(LANE&1)*4+k%4,y=(LANE>>1)*4+k/4;
   unsigned ox=q.range(159,144),oy=q.range(175,160);
   half pw=power(half(ox+x),half(oy+y),unpack_half(q,0),unpack_half(q,16),
                 unpack_half(q,32),unpack_half(q,48),unpack_half(q,64));
   if(pw<=half(0)){
#ifdef FLK_EXACT_EXP_ROM
    half exponential=exact_negative_exp<LANE>(pw);
#else
    half exponential=hls::half_exp(pw);
#endif
    half alpha=half(unpack_half(q,80)*exponential);
    if(alpha>half(.99f))alpha=half(.99f);
    result.range(15,0)=pack_half(alpha);result[87]=alpha>=half(1.f/255.f);
   }
  }
  output.write(result);
 }
}

template<unsigned LANE> static void accumulate_groups(hls::stream<BlendWord>& input,
 hls::stream<PixelBits>& output){
#pragma HLS INLINE off
 half rr[64],gg[64],bb[64],tt[64];unsigned last[64];bool stopped[64];
#pragma HLS RESOURCE variable=rr core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_2P_BRAM
 bool end=false;unsigned tile=0;
 while(!end){
  BlendWord command=input.read();unsigned type=command.range(90,88);
  Word fields=0;fields.range(127,0)=command;
  if(type==3)end=true;
  else if(type==1){
   for(unsigned p=0;p<64;p++){
#pragma HLS PIPELINE II=1
    rr[p]=gg[p]=bb[p]=half(0);tt[p]=half(1);last[p]=0;stopped[p]=false;
   }
  }else if(type==2){
   half br=unpack_half(fields,0),bg=unpack_half(fields,16),bc=unpack_half(fields,32);
   for(unsigned p=0;p<64;p++){
#pragma HLS PIPELINE II=1
    unsigned sub=p/16,k=p%16,x=(sub&1)*8+(LANE&1)*4+k%4,y=(sub>>1)*8+(LANE>>1)*4+k/4;
    PixelBits out=0;out.range(15,0)=pack_half(half(rr[p]+half(br*tt[p])));
    out.range(31,16)=pack_half(half(gg[p]+half(bg*tt[p])));
    out.range(47,32)=pack_half(half(bb[p]+half(bc*tt[p])));out.range(63,48)=pack_half(tt[p]);
    out.range(95,64)=last[p];out.range(127,96)=tile*256+y*16+x;output.write(out);
   }
   tile++;
  }else if(type==4){
   unsigned limit=command.range(86,80),ordinal=command.range(79,48);
   half red=unpack_half(fields,0),green=unpack_half(fields,16),blue=unpack_half(fields,32);
   // Addresses are unique inside this bounded traversal. Let HLS preserve
   // the complete read/modify/write barrier at its end; never suppress RAW.
   for(unsigned p=0;p<64&&p<limit;p++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=64
    BlendWord pixel=input.read();Word alpha_bits=0;alpha_bits.range(15,0)=pixel.range(15,0);
    half alpha=unpack_half(alpha_bits,0);
    if(pixel[87]&&!stopped[p]){
     half next=half(tt[p]*half(half(1)-alpha));
     if(next<half(.0001f))stopped[p]=true;
     else{
      half weight=half(alpha*tt[p]);
      rr[p]=half(rr[p]+half(red*weight));gg[p]=half(gg[p]+half(green*weight));
      bb[p]=half(bb[p]+half(blue*weight));tt[p]=next;last[p]=ordinal;
     }
    }
   }
  }
 }
}

template<unsigned LANE> static void render_lane_stream(hls::stream<LaneWord>& input,
 hls::stream<PixelBits>& output){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 hls::stream<EvalWord> requests;hls::stream<BlendWord> results;
#pragma HLS STREAM variable=requests depth=16
#pragma HLS STREAM variable=results depth=16
#pragma HLS RESOURCE variable=requests core=FIFO_SRL
#pragma HLS RESOURCE variable=results core=FIFO_SRL
 produce_groups<LANE>(input,requests);
 evaluate_pixels<LANE>(requests,results);
 accumulate_groups<LANE>(results,output);
}
