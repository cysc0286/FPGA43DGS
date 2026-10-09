<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="hls_pipegs_hgr_v4_coord_2lane_fix2" top="flicker_render_pipeline">
    <files>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipegs_hgr_v4_coord_2lane/tb.cpp" sc="0" tb="1" cflags=" -std=c++0x -DFLK_GROUP_SUBTILES=3 -DFLK_GROUP_TRIM_RANGE=1 -DFLK_LANE_FIFO_STORAGE=1 -DFLK_GROUP_SHARED_ATTR=1 -DFLK_EXACT_EXP_ROM=2 -DFLK_STREAM_CORES=2 -DFLK_STREAM_LANE=0 -DFLK_PIPEGS_HGR=1 -DFLK_HGR_DPC=0 -Wno-unknown-pragmas" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipegs_hgr_v4_coord_2lane/../hls/renderer.cpp" sc="0" tb="1" cflags=" -std=c++0x -Wno-unknown-pragmas" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipegs_hgr_v4_coord_2lane/framed_ctu.cpp" sc="0" tb="false" cflags="-std=c++0x -DFLK_GROUP_SUBTILES=3 -DFLK_GROUP_TRIM_RANGE=1 -DFLK_LANE_FIFO_STORAGE=1 -DFLK_GROUP_SHARED_ATTR=1 -DFLK_EXACT_EXP_ROM=2 -DFLK_STREAM_CORES=2 -DFLK_STREAM_LANE=0 -DFLK_PIPEGS_HGR=1 -DFLK_HGR_DPC=0" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipegs_hgr_v4_coord_2lane/pipeline.cpp" sc="0" tb="false" cflags="-std=c++0x -DFLK_GROUP_SUBTILES=3 -DFLK_GROUP_TRIM_RANGE=1 -DFLK_LANE_FIFO_STORAGE=1 -DFLK_GROUP_SHARED_ATTR=1 -DFLK_EXACT_EXP_ROM=2 -DFLK_STREAM_CORES=2 -DFLK_STREAM_LANE=0 -DFLK_PIPEGS_HGR=1 -DFLK_HGR_DPC=0" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="true" clean="true" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

