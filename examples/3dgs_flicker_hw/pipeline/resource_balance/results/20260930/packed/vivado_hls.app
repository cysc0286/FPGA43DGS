<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="hls_pipeline_grouppacked_20260930" top="flicker_render_pipeline">
    <includePaths/>
    <libraryFlag/>
    <files>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipeline/tb.cpp" sc="0" tb="1" cflags=" -std=c++0x -Wno-unknown-pragmas" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipeline/../hls/renderer.cpp" sc="0" tb="1" cflags=" -std=c++0x -Wno-unknown-pragmas" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipeline/framed_ctu.cpp" sc="0" tb="false" cflags="-std=c++0x" blackbox="false"/>
        <file name="D:/ADProjects/FPGA43DGS/examples/3dgs_flicker_hw/pipeline/pipeline.cpp" sc="0" tb="false" cflags="-std=c++0x -DFLK_EXACT_EXP_ROM=2 -DFLK_STATE_PORTS=1 -DFLK_GROUP_SUBTILES=3 -DFLK_GROUP_TRIM_RANGE=1 -DFLK_LANE_FIFO_STORAGE=1" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="true" clean="true" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

