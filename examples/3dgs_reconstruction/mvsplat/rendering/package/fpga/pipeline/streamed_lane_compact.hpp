// Streamed lane with separate group metadata. Color, ordinal and control tokens
// bypass the 75-cycle power/alpha pipeline instead of occupying every stage.
// Data and metadata remain FIFO ordered; group length binds their consumption.
// Bit 131 is the end token; bit 132 marks the first request of a Gaussian
// group when stream-side attribute reuse is enabled.
#if FLK_DPC_ROW_TABLE
// Explicit lane-local row index (135:133) and anchor (136). Coordinates of
// absent subtiles may be zero, so they cannot define the recurrence schedule.
typedef ap_uint<137> EvalWord;
#else
typedef ap_uint<133> EvalWord;
#endif
#if FLK_STREAM_COMPACT_REQUEST && (!FLK_STREAM_SHARED_ATTR_FIFO || FLK_DPC_ROW_TABLE)
#error "Compact requests require attribute FIFO without row-table metadata"
#endif
#if FLK_STREAM_MATERIALIZED_SRL && !FLK_STREAM_SHARED_ATTR_FIFO
#error "Materialized FIFO selection requires the shared attribute FIFO"
#endif
#if FLK_STREAM_SHARED_ATTR && FLK_STREAM_SHARED_ATTR_FIFO
#error "Inline and FIFO shared attributes are mutually exclusive"
#endif
#if FLK_STREAM_COMPACT_REQUEST && FLK_STREAM_LANE != 2
#error "Compact requests require FLK_STREAM_LANE=2"
#endif
// Geometry/opacity travel once through the attribute FIFO. The request FIFO
// carries only two 17-bit coordinates and three control bits in this mode.
#if FLK_STREAM_COMPACT_REQUEST
typedef ap_uint<37> RequestWord;
#else
typedef EvalWord RequestWord;
#endif
static RequestWord pack_request(EvalWord packet){
#pragma HLS INLINE
#if FLK_STREAM_COMPACT_REQUEST
 return packet.range(132,96);
#else
 return packet;
#endif
}
static EvalWord expand_request(RequestWord request){
#pragma HLS INLINE
#if FLK_STREAM_COMPACT_REQUEST
 EvalWord packet=0;packet.range(132,96)=request;return packet;
#else
 return request;
#endif
}
typedef ap_uint<17> BlendWord;
typedef ap_uint<89> GroupCommand;
#if FLK_STREAM_SHARED_ATTR_FIFO
typedef ap_uint<96> AttrWord;
#endif

static GroupCommand group_command(Word q,unsigned type,unsigned count=0){
#pragma HLS INLINE
 GroupCommand cmd=0;cmd.range(88,87)=type;cmd.range(86,80)=count;
 cmd.range(79,48)=q.range(223,192);
 cmd.range(47,0)=type==2?q.range(127,80):q.range(143,96);return cmd;
}

template<unsigned LANE> static void emit_group(GroupRecord records[4],ap_uint<4> valid,
 hls::stream<RequestWord>& output,hls::stream<GroupCommand>& commands
#if FLK_STREAM_SHARED_ATTR_FIFO
 ,hls::stream<AttrWord>& attributes
#endif
 ){
#pragma HLS INLINE off
 unsigned first=valid[0]?0:(valid[1]?1:(valid[2]?2:3));
 const unsigned pixels=FLK_STREAM_CORES==3?(258-LANE)/3:64;
#if FLK_STREAM_CORES == 3
 unsigned limit=pixels;
#else
 unsigned limit=valid[3]?64:(valid[2]?48:(valid[1]?32:16));
#endif
 commands.write(group_command(records[first],0,limit));
 for(unsigned p=0;p<pixels&&p<limit;p++){
#pragma HLS PIPELINE II=1
#if FLK_STREAM_CORES == 3
#pragma HLS LOOP_TRIPCOUNT min=85 max=86
  unsigned index=p*3+LANE,tx=index&15,ty=index>>4;
  unsigned sub=(tx>>3)+2*(ty>>3),x=tx&7,y=ty&7;
#else
#pragma HLS LOOP_TRIPCOUNT min=16 max=64
  unsigned sub=p/16,k=p%16;
  unsigned x=(LANE&1)*4+k%4,y=(LANE>>1)*4+k/4;
#endif
  Word q=records[sub];
  unsigned ox=q.range(159,144),oy=q.range(175,160),w=q.range(180,176),h=q.range(185,181);
  EvalWord packet=0;
#if FLK_STREAM_SHARED_ATTR_FIFO
  // The FIFO carries one immutable raw-attribute token per group. The
  // materializer below expands it before the numeric evaluator, so this
  // producer never feeds a floating-point state register back into II=1.
  if(p==0){
   // `p` is the request position, not necessarily the first valid sub-tile.
   // A sparse group may start at sub-tile 1/2/3, so source the shared fields
   // from the command's actual first valid record rather than from `q`.
   AttrWord attr=0;attr.range(95,0)=records[first].range(95,0);attributes.write(attr);
   packet[132]=1;
  }
#elif FLK_STREAM_SHARED_ATTR
  // Only the first request carries the invariant FP16 attributes. The group
  // header already carries color/ordinal on the independent command FIFO.
  if(p==0)packet.range(95,0)=q.range(95,0);
#else
  packet.range(95,0)=q.range(95,0);
#endif
  // 17 bits retain the original unsigned 16-bit origin plus pixel offset.
  packet.range(112,96)=ox+x;packet.range(129,113)=oy+y;
#if FLK_STREAM_SHARED_ATTR
  if(p==0)packet[132]=1;
#endif
  bool enabled=valid[sub]&&x<w&&y<h;
#if FLK_STREAM_CORES == 3
  enabled=enabled&&q[224+(x>>2)+2*(y>>2)];
#endif
  packet[130]=enabled;
#if FLK_DPC_ROW_TABLE
  // Reconstruct the tile origin from a known-valid record. A masked anchor
  // still initializes the row using the current Gaussian, never stale RAM.
  Word shared=records[first];
  unsigned origin_x=unsigned(shared.range(159,144))-(first&1)*8;
  unsigned origin_y=unsigned(shared.range(175,160))-(first>>1)*8;
  packet.range(95,0)=shared.range(95,0);
#if FLK_STREAM_CORES == 3
  packet.range(112,96)=origin_x+tx;packet.range(129,113)=origin_y+ty;
  packet.range(135,133)=tx/3;packet[136]=tx<3;
#else
  packet.range(112,96)=origin_x+(sub&1)*8+x;
  packet.range(129,113)=origin_y+(sub>>1)*8+y;
  packet.range(135,133)=k%4;packet[136]=(k%4)==0;
#endif
#endif
  output.write(pack_request(packet));
 }
}

template<unsigned LANE> static void produce_groups(hls::stream<LaneWord>& input,
 hls::stream<RequestWord>& output,hls::stream<GroupCommand>& commands
#if FLK_STREAM_SHARED_ATTR_FIFO
 ,hls::stream<AttrWord>& attributes
#endif
 ){
#pragma HLS INLINE off
 GroupRecord records[4];ap_uint<4> valid=0;unsigned ordinal=0;bool end=false;
#pragma HLS RESOURCE variable=records core=RAM_2P_LUTRAM
 for(unsigned i=0;i<4;i++)records[i]=0;
 while(!end){
  Word q=unpack_lane(input.read());unsigned type=kind(q),incoming=q.range(223,192);
  if(valid!=0&&(type!=0||incoming!=ordinal)){
   emit_group<LANE>(records,valid,output,commands
#if FLK_STREAM_SHARED_ATTR_FIFO
    ,attributes
#endif
   );valid=0;
  }
  if(type==0){unsigned sub=q.range(229,228);records[sub]=q;valid[sub]=1;ordinal=incoming;}
  else{
   commands.write(group_command(q,type));end=type==3;
   if(end){EvalWord stop=0;stop[131]=1;output.write(pack_request(stop));}
  }
 }
}

#if FLK_STREAM_SHARED_ATTR_FIFO
// Expand a group token into the original full-request ABI before the numeric
// evaluator. The stage carries only 96 raw bits and a marker; it performs no
// half conversion or arithmetic, avoiding the feedback dependence that made
// the previous in-loop cache reach II=30.
template<unsigned LANE> static void materialize_attrs(hls::stream<RequestWord>& input,
 hls::stream<AttrWord>& attributes,hls::stream<EvalWord>& output){
#pragma HLS INLINE off
 AttrWord current=0;bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  EvalWord packet=expand_request(input.read());end=packet[131];
  if(end){output.write(packet);}
  else{
   if(packet[132])current=attributes.read();
   packet.range(95,0)=current;packet[132]=0;output.write(packet);
  }
 }
}
#endif

#if FLK_DPC_ROW_TABLE
typedef ap_uint<96> DpcRowWord;

// Numeric initialization has no loop-carried state. Every actual row anchor
// produces one raw table token, even when that anchor is masked out.
template<unsigned LANE> static void prepare_dpc_rows(hls::stream<EvalWord>& input,
 hls::stream<EvalWord>& requests,hls::stream<DpcRowWord>& tables){
#pragma HLS INLINE off
 const unsigned columns=FLK_STREAM_CORES==3?6:4;bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  EvalWord packet=0;bool available=input.read_nb(packet);
  if(available){
   end=packet[131];
   if(!end&&packet[136]){
    Word q=0;q.range(95,0)=packet.range(95,0);
    half x=half(unsigned(packet.range(112,96))),y=half(unsigned(packet.range(129,113)));
    half mx=unpack_half(q,0),my=unpack_half(q,16),a=unpack_half(q,32);
    half b=unpack_half(q,48),c=unpack_half(q,64);
    half dx=half(mx-x),dy=half(my-y);
    const half stride=half(FLK_STREAM_CORES==3?3.f:1.f);
    const half stride_squared=half(FLK_STREAM_CORES==3?9.f:1.f);
    half value=power(x,y,mx,my,a,b,c);
    half step=half(half(stride*half(half(a*dx)+half(b*dy)))-
                   half(half(.5f)*half(a*stride_squared)));
    half curvature=half(a*stride_squared);DpcRowWord table=0;
    for(unsigned col=0;col<columns;col++){
#pragma HLS UNROLL
     table.range(16*col+15,16*col)=pack_half(value);
     value=half(value+step);step=half(step-curvature);
    }
    tables.write(table);
   }
   requests.write(packet);
  }
 }
}

template<unsigned LANE> static void materialize_dpc_rows(hls::stream<EvalWord>& input,
 hls::stream<DpcRowWord>& tables,hls::stream<EvalWord>& output){
#pragma HLS INLINE off
 DpcRowWord current=0;bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  EvalWord packet=input.read();end=packet[131];
  if(!end){
   if(packet[136])current=tables.read();
   unsigned col=packet.range(135,133);
   packet.range(15,0)=current.range(col*16+15,col*16);
  }
  output.write(packet);
 }
}

template<unsigned LANE> static void evaluate_dpc_alpha(hls::stream<EvalWord>& input,
 hls::stream<BlendWord>& output){
#pragma HLS INLINE off
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  EvalWord packet=0;bool available=input.read_nb(packet);
  if(available){
   end=packet[131];BlendWord result=0;
   if(!end){
    Word q=0;q.range(95,0)=packet.range(95,0);half pw=unpack_half(q,0);
    if(packet[130]&&pw<=half(0)){
#ifdef FLK_EXACT_EXP_ROM
     half exponential=exact_negative_exp<LANE>(pw);
#else
     half exponential=hls::half_exp(pw);
#endif
     half alpha=half(unpack_half(q,80)*exponential);
     if(alpha>half(.99f))alpha=half(.99f);
     result.range(15,0)=pack_half(alpha);result[16]=alpha>=half(1.f/255.f);
    }
    output.write(result);
   }
  }
 }
}

template<unsigned LANE> static void evaluate_pixels(hls::stream<EvalWord>& input,
 hls::stream<BlendWord>& output){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
 hls::stream<EvalWord> requests,powers;hls::stream<DpcRowWord> tables;
#pragma HLS STREAM variable=requests depth=16
#pragma HLS STREAM variable=powers depth=16
#pragma HLS STREAM variable=tables depth=4
 prepare_dpc_rows<LANE>(input,requests,tables);
 materialize_dpc_rows<LANE>(requests,tables,powers);
 evaluate_dpc_alpha<LANE>(powers,output);
}
#else
template<unsigned LANE> static void evaluate_pixels(hls::stream<EvalWord>& input,
 hls::stream<BlendWord>& output){
#pragma HLS INLINE off
#if FLK_STREAM_SHARED_ATTR
  // Keep the reuse state in raw FP16 bits.  Updating half objects inside the
  // II=1 loop creates a long carried conversion dependency in Vivado HLS;
  // bit registers preserve the exact FP16 contract without that dependency.
  ap_uint<96> shared_attr_bits=0;
#endif
#if FLK_DPC
 // PipeGS DPC state for the current 4-pixel row.  Requests are emitted in
 // row-major order inside each 4x4 lane footprint, so a row boundary is
 // visible from the low two coordinate bits without changing the stream ABI.
#if FLK_DPC_ROW_TABLE
 const unsigned dpc_columns=FLK_STREAM_CORES==3?6:4;
 half dpc_row[dpc_columns];
#pragma HLS ARRAY_PARTITION variable=dpc_row complete
#elif FLK_DPC_BLOCK_TABLE
 half dpc_block[4][4];
#pragma HLS ARRAY_PARTITION variable=dpc_block complete dim=1
#pragma HLS ARRAY_PARTITION variable=dpc_block complete dim=2
#elif FLK_DPC_INTERLEAVE
 half dpc_pw[4],dpc_dy[4];
#pragma HLS ARRAY_PARTITION variable=dpc_pw complete
#pragma HLS ARRAY_PARTITION variable=dpc_dy complete
#else
 half dpc_pw=0,dpc_dx=0;
#endif
#endif
 bool end=false;
 while(!end){
#pragma HLS PIPELINE II=1
  // A blocking read stalls all 75 stages when requests are empty in HLS
  // 2018.3. The metadata consumer can then wait for an unflushed alpha while
  // the producer waits for metadata space. Inject an invalid bubble instead;
  // only accepted packets produce results, preserving FIFO order and count.
  EvalWord packet=0;bool available=input.read_nb(packet);
  if(available)end=packet[131];BlendWord result=0;
  if(available&&!end){
   if(packet[130]
#if FLK_DPC_ROW_TABLE
      ||packet[136]
#endif
   ){
    Word q=0;q.range(95,0)=packet.range(95,0);
    unsigned x=packet.range(112,96),y=packet.range(129,113);
#if FLK_STREAM_SHARED_ATTR
     if(packet[132]){
      shared_attr_bits=packet.range(95,0);
     }
     half mx=unpack_half16(shared_attr_bits.range(15,0));
     half my=unpack_half16(shared_attr_bits.range(31,16));
     half a=unpack_half16(shared_attr_bits.range(47,32));
     half b=unpack_half16(shared_attr_bits.range(63,48));
     half c=unpack_half16(shared_attr_bits.range(79,64));
#else
    half mx=unpack_half(q,0),my=unpack_half(q,16),a=unpack_half(q,32),
         b=unpack_half(q,48),c=unpack_half(q,64);
#endif
    half pw;
#if FLK_DPC
#if FLK_DPC_ROW_TABLE
    const half stride=half(FLK_STREAM_CORES==3?3.f:1.f);
    const half stride_squared=half(FLK_STREAM_CORES==3?9.f:1.f);
    if(packet[136]){
     half dx=half(mx-half(x)),dy=half(my-half(y));
     half value=power(half(x),half(y),mx,my,a,b,c);
     half step=half(half(stride*half(half(a*dx)+half(b*dy)))-
                    half(half(.5f)*half(a*stride_squared)));
     half curvature=half(a*stride_squared);
     for(unsigned column=0;column<dpc_columns;column++){
#pragma HLS UNROLL
      dpc_row[column]=value;
      value=half(value+step);step=half(step-curvature);
     }
    }
    pw=dpc_row[unsigned(packet.range(135,133))];
#elif FLK_DPC_BLOCK_TABLE
    // Block-restarted DPC: for each local column, create four FP16 values at
    // the first row of a 4-row block.  Later rows only select the table entry;
    // no FP16 value is fed back into the next loop iteration.  The table is
    // intentionally rebuilt at every row with y&3==0, so a new Gaussian or a
    // new subtile cannot inherit state from the previous group.
    unsigned col=x&3,row=y&3;
    if(row==0){
     half dx=half(mx-half(x)),dy=half(my-half(y));
     half f0=power(half(x),half(y),mx,my,a,b,c);
     half d0=half(half(c*dy)-half(half(.5f)*c)+half(b*dx));
     half d1=half(d0+d0),d2=half(d1+d0);
     half c3=half(half(c+c)+c);
     dpc_block[col][0]=f0;
     dpc_block[col][1]=half(f0+d0);
     dpc_block[col][2]=half(half(f0+d1)-c);
     dpc_block[col][3]=half(half(f0+d2)-c3);
     pw=f0;
    }else{
     pw=dpc_block[col][row];
    }
#elif FLK_DPC_INTERLEAVE
    // The stream is row-major: for one fixed local column, successive
    // requests advance y and are four cycles apart.  Advancing y uses
    //   f(y+1)-f(y) = c*dy - c/2 + b*dx,
    //   delta(y+1) = delta(y) - c.
    // Four independent column contexts therefore remove the single-state
    // loop-carried dependency that made the original DPC reach II=6.
    unsigned col=x&3,row=y&3;
    if(row==0){
     half dx=half(mx-half(x)),dy=half(my-half(y));
     dpc_pw[col]=power(half(x),half(y),mx,my,a,b,c);
     dpc_dy[col]=half(half(c*dy)-half(half(.5f)*c)+half(b*dx));
    }
    pw=dpc_pw[col];
    dpc_pw[col]=half(dpc_pw[col]+dpc_dy[col]);
    dpc_dy[col]=half(dpc_dy[col]-c);
#else
    // power() stores dx=mx-x, dy=my-y.  For x+1, the exact FP16-form
    // difference is a*dx - a/2 + b*dy; advance the difference after use.
    if((x&3)==0){
     half dx=half(mx-half(x)),dy=half(my-half(y));
     dpc_pw=power(half(x),half(y),mx,my,a,b,c);
     dpc_dx=half(half(a*dx)-half(half(.5f)*a)+half(b*dy));
    }
    pw=dpc_pw;
    dpc_pw=half(dpc_pw+dpc_dx);
    dpc_dx=half(dpc_dx-a);
#endif
#else
    pw=power(half(x),half(y),mx,my,a,b,c);
#endif
    if(packet[130]&&pw<=half(0)){
#ifdef FLK_EXACT_EXP_ROM
     half exponential=exact_negative_exp<LANE>(pw);
#else
     half exponential=hls::half_exp(pw);
#endif
    half alpha=half(
#if FLK_STREAM_SHARED_ATTR
      unpack_half16(shared_attr_bits.range(95,80))
#else
      unpack_half(q,80)
#endif
      *exponential);
     if(alpha>half(.99f))alpha=half(.99f);
     result.range(15,0)=pack_half(alpha);result[16]=alpha>=half(1.f/255.f);
    }
   }
   output.write(result);
  }
 }
}

#endif
template<unsigned LANE> static void accumulate_groups(hls::stream<GroupCommand>& commands,
 hls::stream<BlendWord>& input,hls::stream<PixelBits>& output){
#pragma HLS INLINE off
 const unsigned pixels=FLK_STREAM_CORES==3?(258-LANE)/3:64;
 half rr[pixels],gg[pixels],bb[pixels],tt[pixels];unsigned last[pixels];bool stopped[pixels];
#pragma HLS RESOURCE variable=rr core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=gg core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=bb core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=tt core=RAM_2P_BRAM
 bool end=false;unsigned tile=0;
 while(!end){
  GroupCommand command=commands.read();unsigned type=command.range(88,87);
  Word fields=0;fields.range(88,0)=command;
  if(type==3)end=true;
  else if(type==1){
   for(unsigned p=0;p<pixels;p++){
#pragma HLS PIPELINE II=1
    rr[p]=gg[p]=bb[p]=half(0);tt[p]=half(1);last[p]=0;stopped[p]=false;
   }
  }else if(type==2){
   half br=unpack_half(fields,0),bg=unpack_half(fields,16),bc=unpack_half(fields,32);
   for(unsigned p=0;p<pixels;p++){
#pragma HLS PIPELINE II=1
#if FLK_STREAM_CORES == 3
    unsigned index=p*3+LANE,x=index&15,y=index>>4;
#else
    unsigned sub=p/16,k=p%16,x=(sub&1)*8+(LANE&1)*4+k%4,y=(sub>>1)*8+(LANE>>1)*4+k/4;
#endif
    PixelBits out=0;out.range(15,0)=pack_half(half(rr[p]+half(br*tt[p])));
    out.range(31,16)=pack_half(half(gg[p]+half(bg*tt[p])));
    out.range(47,32)=pack_half(half(bb[p]+half(bc*tt[p])));out.range(63,48)=pack_half(tt[p]);
    out.range(95,64)=last[p];out.range(127,96)=tile*256+y*16+x;output.write(out);
   }
   tile++;
  }else{
   unsigned limit=command.range(86,80),ordinal=command.range(79,48);
   half red=unpack_half(fields,0),green=unpack_half(fields,16),blue=unpack_half(fields,32);
   // Unique addresses within a group; retain the real writeback barrier.
   for(unsigned p=0;p<pixels&&p<limit;p++){
#pragma HLS PIPELINE II=1
#if FLK_STREAM_CORES == 3
#pragma HLS LOOP_TRIPCOUNT min=85 max=86
#else
#pragma HLS LOOP_TRIPCOUNT min=16 max=64
#endif
    BlendWord pixel=input.read();Word alpha_bits=0;alpha_bits.range(15,0)=pixel.range(15,0);
    half alpha=unpack_half(alpha_bits,0);
    if(pixel[16]&&!stopped[p]){
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
 hls::stream<RequestWord> requests;hls::stream<BlendWord> results;
#if FLK_STREAM_SHARED_ATTR_FIFO
 hls::stream<EvalWord> materialized;hls::stream<AttrWord> attributes;
#endif
 hls::stream<GroupCommand> commands;
#pragma HLS STREAM variable=requests depth=16
#if FLK_STREAM_SHARED_ATTR_FIFO
#pragma HLS STREAM variable=materialized depth=16
#if FLK_STREAM_MATERIALIZED_SRL
#pragma HLS RESOURCE variable=materialized core=FIFO_SRL
#endif
#pragma HLS STREAM variable=attributes depth=4
#pragma HLS RESOURCE variable=attributes core=FIFO_SRL
#endif
#pragma HLS STREAM variable=results depth=16
#pragma HLS STREAM variable=commands depth=4
#pragma HLS RESOURCE variable=results core=FIFO_SRL
#if FLK_STREAM_FIFO_BRAM
 // Spend spare block RAM on the two wide queues. Their contents, depths and
 // FIFO order are unchanged; synthesis may add an internal read latency.
#pragma HLS RESOURCE variable=requests core=FIFO_BRAM
#pragma HLS RESOURCE variable=commands core=FIFO_BRAM
#else
#pragma HLS RESOURCE variable=requests core=FIFO_SRL
#pragma HLS RESOURCE variable=commands core=FIFO_SRL
#endif
#if FLK_STREAM_SHARED_ATTR_FIFO
 produce_groups<LANE>(input,requests,commands,attributes);
 materialize_attrs<LANE>(requests,attributes,materialized);
 evaluate_pixels<LANE>(materialized,results);
#else
 produce_groups<LANE>(input,requests,commands);
 evaluate_pixels<LANE>(requests,results);
#endif
 accumulate_groups<LANE>(commands,results,output);
}
