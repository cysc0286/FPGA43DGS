# split 前端流水优化

2026-09-27。用户目标是 FPGA 平台完整处理流程与实际速度优先，不要求复制 ASIC 配置。当前仅改变 split 前端调度，不改变画质；NPU 延后，下游定位/检测任务不在本工程范围。

## 比较合同

同一 train/iteration_7000 全模型 559,263 Gaussian、视角 0/10、320×178、同一 ARM 可执行文件、相同六模式、同一 64 B 输入 ABI、200 MHz。输入仍为已投影排序列表。延迟包括打包、SDK 搬运、提交、等待、回读和重组，排除离线 GPU 前处理。不得写成完整 Pose→Framebuffer。

硬件必须逐位匹配原来的各模式 FP16 Golden；本候选不使用任何放宽后的质量门槛。与官方 FP32 的 PSNR/SSIM 和局部差异继续记录，历史严格检查仍为失败。本增量不证明实拍照片重建质量。

模式选择另列速度/画质折中，不能与调度优化混算。仍以 Dense 前后逐位一致为主要结论；已知旧版 Smooth-Focused 两视角 PSNR 43.642/44.393 dB、SSIM 0.996545/0.997844，可作为现有模式候选。Sparse/Spiky-Focused 的 PSNR 约 31–33 dB、SSIM 最低 0.962459，不能仅因为可能较快就默认替代 Dense。质量数据来源 `cat_measure_20260927T194410/analysis.json`，新版尚未上板时已明确这一选择顺序。

## 诊断与实现

按优先级检查三个假设：split 非流水的几何计算限制供数；VRU 每次调用的装载/排空限制吞吐；CPU 打包及交接造成空闲。本增量只验证第一个，后两项保持不变。旧版与新版 CTU/dispatch/render/gather 源码已逐段核对一致。

原 split 每条 Gaussian–Tile 记录依次计算半径/形状，再展开四个 sub-tile，几何与展开不能重叠。新 split 拆为有序 framing→geometry→sub-tile expansion，使用有限深度 BRAM FIFO；geometry 目标 II=16，展开内层 II=4。保持原 FP32 运算顺序，提前共享同一记录的四个包围盒边界。不增加跨 Tile 缓存，不向 CPU 转移几何成本，不改变排序。

头、尾和任务结束标记随同一有序通道传递。FIFO 背压限制在途数据，不靠消除依赖 pragma 获取吞吐。

## 验证来源

- 修改前快照：`../evidence/split_baseline_20260927T205810`。
- 正式旧板对照：`../evidence/cat_measure_20260927T211227`，预热1/测量5，两视角、CPU4/CPU Dense4/FPGA mode0/1/2。
- HLS：`../build/hls_pipeline_splitopt`；冻结 `../evidence/hls_pipeline_splitopt_20260927T210845`。
- 前端原实现差分：`../evidence/split_differential_20260927T211400`，六模式、68 Gaussian、5 Tile，465 输出记录，C/RTL PASS。
- 新版真实 RTL 集成：`../evidence/pipeline_integration_20260927T210941`，六模式、4,608 输出像素，DDR 停顿/背压/保护区/所有权 PASS。
- 旧版对应集成：`../evidence/pipeline_integration_20260927T211443`。
- 独立物理候选：`platform/flicker_splitopt_fpga`，`../build/board_cat_20260927_210933`。

`run_integration.py --profile-split` 只增加 testbench 层次探针，计数实际 HLS input/output blocking condition 和握手次数。两个阻塞计数可能重叠，不能相加得到总闲置率；小规模仿真不能替代完整场景板端忙碌率。

## 实板验收入口

物理实现、现有约束下的新路径时序、BOOT 分区一致性和实际源哈希全部核对后，使用 `install_candidate.py` 显式指定已核实的旧 BOOT SHA256。自动保存备份与回滚命令，不重复安装旧包。网络重启后先验收 SDK、状态和原始回读，再运行完整对照。

后续正式指标从新旧 `summary.json`、`analysis.json`、`physical_provenance.json` 和最终 routed 报告汇总；构建中不填入预计收益或虚构 FPS。功耗未测，LPIPS 未测。
