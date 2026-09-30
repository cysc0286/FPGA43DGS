# FLK1 复跑入口

2026-09-30：针对换视角渲染的 FPGA 主耗时，新增独立候选与负结果记录：
[像素工作集搬运及跨子块合并](lane_workset/README.md)。保持原位流和默认
单子块路径；HLS/RTL 周期改善不能当作已上板的整帧加速。

这是 FLICKER AABB→CTU→FIFO→VRU 集成版本，和上一级 FLK0 分开保存。状态、数值范围和未完成项见 [VALIDATION.md](VALIDATION.md)，数据格式见 [CONTRACT.md](CONTRACT.md)。工作目录为仓库根目录 `D:\ADProjects\FPGA43DGS`。

2026-09-28 已补板端CPU前处理与分组排序；从PLY＋相机开始复跑的入口为 [frontend/README.md](../frontend/README.md)，完整功能/性能边界为 [frontend/VALIDATION.md](../frontend/VALIDATION.md)。下文主要针对已准备有序列表的硬件构建与渲染测试，不能把其计时当作整链时延。

2026-09-27 当前已安装 FLK1 split II16，构建 `board_cat_20260927_210933`，程序仍为 `stagecat_20260927T122707`；两视角六模式、串行/流水和连续十帧实板结果见 [split 对照](../evidence/split_board_comparison_20260927/VALIDATION.md)。先前 FLK1 lean 及其 [冻结汇总](../evidence/board_cat_summary_20260927/REPORT.md) 继续作为历史基线。无需重新安装或构建即可复测。下文构建命令仅用于产生新的独立候选，不应覆盖已冻结的 HLS 项目、平台和 BOOT。

## 编译与仿真

```powershell
$env:FLK_HLS_PROJECT='hls_pipeline_ctu8'
& examples/3dgs_flicker_hw/pipeline/build.ps1
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/pipeline/run_integration.py --project hls_pipeline_ctu8
```

`FLK_HLS_PROJECT` 指定独立候选目录，不能重用需保留的历史目录。依次执行 C 参考、综合和 C/RTL；当前候选 CTU II=8，四个 VRU II=1。源码调度依赖不能靠 pragma 豁免；必须检查 cosim 结论与依赖诊断。`run_integration.py` 默认使用已冻结的 `pipeline_real_protocol`，六模式对比原始字节，并测试 DDR 停顿和输出所有权；它不是完整场景的速度测试。`freeze_hls.py --project <名称> --console <日志>` 保存通过版本的源码、RTL、报告和哈希。

完整场景 `pipeline_full_v0` / `pipeline_full_v10` 已冻结，不直接覆盖。需要新数据时给 `prepare.py --output` 和 `reference.py` 传新的目录；记录 case、输入 SHA256、所用 HLS project、编译产物 SHA256 与源快照。

## 板级实现与安装

`prepare_project.py --project hls_pipeline_ctu8 --console <日志> --platform flicker_cat_compact_fpga` 只允许新建独立平台，不覆盖已失败的 `platform/flicker_cat_fpga`。`build_board.ps1 -PlatformName flicker_cat_compact_fpga` 保持原厂 Hook 和 Lite 管脚，按时间戳保存报告。打包、审查和冻结从 `status.txt` 的 PROJECT 字段定位此次实际平台；安装还核对集成测试和平台所用的 HLS RTL 哈希一致。

针对 compact 布局后 Slice 占用 99.55% 的拥塞，另有独立 `flicker_cat_lean_fpga` 候选：仅移除不用的 GSC1/GSB1 历史计算核，保留原厂 adder/SDK DMA。`prepare_legacy_trim.py --output <新目录>` 生成候选，`run_legacy_trim.py --candidate <该目录>` 对实际原厂 RTL 做差分回归；将通过的证据目录传给 `prepare_project.py --legacy-trim-evidence <证据> --platform <新平台>`。不能把未经回归的精简文件直接复制进平台。该候选的旧 GSC1/GSB1 capability 应明确缺失，不能假冒兼容；早期实验仍用各自冻结平台。

布线后使用 `audit.tcl` 检查新渲染、DMA 和寄存器路径的实际时钟、setup/hold 以及约束覆盖；新路径失败必须修复。不能用 HLS 估计代替布线报告，也不能增加组合环路或时序豁免。`package.ps1 -ReportDir <本次报告目录>` 验证 BOOT 非 PL 分区不变、PL payload 精确匹配。

`install_boot.py` 需要 `--boot-dir`、`--integration`，在整板有原厂历史违例时还需要经过审查的 `--timing-review`。它只接受当前 FLK0 的已知 BOOT 哈希，检查源码和集成证据，保存旧 BOOT 与回滚命令，**不会自动重启**。网络重启后先确认新 boot_id、FLK1 capability/ABI 和 idle，再运行原 SDK 与块传输回归。JTAG 保持断开，不使用旧 `/dev/mem` 写实验。

若实际选择 lean 平台，SDK 回归改用 `pipeline/probe_block.py`：只改变平台能力识别条件，保留旧探针三种加法长度、保护区及十五组块传输比较。标准 compact 则仍用 `board/probe_block.py`。安装器依据所选平台清单核对额外的 legacy 差分验证与源码哈希。

## 实板验证与对照

由 `stage_board.py` 创建独立程序和数据目录，不复用无法核对哈希的旧二进制。原生 ARM 编译结果记录于 `stagecat_*`。SSH 使用现有固定主机公钥和仅在进程环境提供的 `FPGA_BOARD_PASSWORD`，不在源码或报告保存口令。

```powershell
# $stage 是已编译且源码/输入哈希匹配的 stagecat 证据目录。
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/pipeline/validate_raw.py --stage $stage --reference examples/3dgs_flicker_hw/evidence/pipeline_real_protocol --mode 0
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/pipeline/measure.py --stage $stage --backends cpu1 cpu4 cpu_dense4 mode0 mode1 mode2 mode3 mode4 mode5 --repeats 3 --warmup 1
# 对完整、实际成功的测量登记；硬件必须 bit-exact 匹配所选 FP16 模式。
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/register_measurement.py $evidence --version FLK1_BOARD_CAT
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/pipeline/analyze_measurement.py $evidence
```

本次复测使用 `$stage='examples/3dgs_flicker_hw/evidence/stagecat_20260927T122707'`。串行消融使用 `--backends mode0 mode1 mode2 --schedule serial --repeats 3 --warmup 1`；连续性使用 `--backends mode2 --schedule pipeline --repeats 10 --warmup 1`。串行/稳定性只登记 summary，不调用要求全组 CPU 图像的 `analyze_measurement.py`。物理来源另存 `physical_provenance.json`，必须核对实际 boot_id、BOOT、平台、stage 哈希，不能盲目复制旧来源。

`summarize_board.py --matrix <九后端完整目录> --serial <三模式串行目录> --stability <十次目录> --output <新目录>` 核对测量完成、同 BOOT/程序、Golden 与计时/回读哈希，生成包含所有样本统计的 `comparison.json` 和 `REPORT.md`。输出目录必须不存在；失败记录和历史数据不被覆盖。当前每组 3/10 次的 P95/P99 仅是描述性分位数。

首次部署还要对两个完整视角执行 `validate_raw.py`，至少核对 mode 0 与 Dense mode 2，再开始计时。性能默认使用两个完整模型视角，包含准备、上传、提交、等待、回读和像素重组；CPU FP32 与 FPGA FP16 的差别保留在报告中。原 CPU Dense 没有新 sub-tile AABB，不能冒充完全同算法的硬件比较。`--schedule serial` 是同硬件串行对照；不能把硬件周期与已包含它的主机等待再次相加。

任何硬件超时均先保留日志、确认引擎状态并复位，不立即重新分配可能仍被 DMA 使用的内存。失败、慢版本和近似质量误差都保留，不能用后一次成功覆盖。
