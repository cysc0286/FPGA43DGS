# FLICKER 方法复现与 30TAI Lite 移植

日期：2026-09-26。FAMERS 全文及作者工程暂不可得，现选 FLICKER 的贡献感知筛选机制作为下一项可核对的论文加速目标。论文：*FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting*，DATE 2026；[作者全文](https://arxiv.org/abs/2603.01158)，本地全文 `papers/pdf/06_FLICKER_DATE2026.pdf`。

## 当前实测进度

2026-09-27 最新状态：NPU 暂停，唯一主线为 FLICKER。工作区 [3dgs_flicker_hw](examples/3dgs_flicker_hw/README.md) 已完成 FLK0 的完整求值、块传输、双缓冲，以及 FLK1 的 AABB→CTU→FIFO→四 VRU 实际上板。FLK1 两完整模型视角 Dense 平均 1.419/1.328 s，同轮四线程 CPU 5.787/4.688 s、CPU Dense 2.098/1.640 s；同位流 AABB 后再启用 Dense CAT 加速 1.302/1.370 倍。六模式输出逐位匹配各自 FP16 golden，但相对官方 FP32 严格精度失败。详见 [FLK1 实板报告](examples/3dgs_flicker_hw/pipeline/VALIDATION.md)。新路径 200 MHz 时序通过，整板原厂 AI 脉宽违例仍在，Slice 99.80%；不能称整板全时序通过。

旧 [FLK0 实板报告](examples/3dgs_flicker_hw/VALIDATION.md) 的 2.094/1.937 s 与全部失败候选保留。新 FLK1 关闭筛选反而慢到 3.937/3.555 s，不能只报相对这个慢基线的 CAT 加速。最终选择四 VRU、II=8 时间复用 CTU 和全 FP16 是 FPGA 适配，不等于作者每拍两矩形、32 VRU 和混合精度 ASIC 配置。这些进展不等于完整论文已完成。

### 2026-09-26 CPU 阶段的历史结果

已独立编写并在 30TAI Lite 四核 ARM 上运行 [mini-tile CAT 软件参考](examples/3dgs_flicker_cat/README.md)。两个 559,263-Gaussian 完整模型视角的 Dense CAT 平均 2.74x / 2.84x 快于同路径无筛选 CPU，输出与该 CPU 基线的 PSNR 48.25 / 50.03 dB、SSIM 0.99926 / 0.99947；但被拒绝的有效贡献分别有 123,784 / 64,374 次，两个视角均**未通过**原有官方逐像素严格门槛。[详细性能、误差、哈希和限制](examples/3dgs_flicker_cat/VALIDATION.md)及 [版本 CSV](examples/3dgs_flicker_cat/results_summary.csv)已经保存。此处的 2.74x / 2.84x 是**ARM 软件机制收益**，FPGA CTU、整帧 FPGA 加速与 NPU 本轮尚无新增实测。

论文中的像素矩形共享计算已在 leader CAT 中单独验证：`dense_pr` 与 `dense` 两视角逐字节相同，但在 ARM 上慢约 2.5% / 2.7%；其适用目标是降低硬件算术成本，不能将 CPU 负收益写成硬件加速。软件当前沿用已准备的 16x16 有序 Tile，没有论文 8x8 sub-tile AABB、CTU/FIFO、混合精度或完整论文的模型准备流程。

## 复现边界

- 论文 §II–IV 描述原始按深度排序的 3DGS、8x8 sub-tile AABB、4x4 mini-tile Contribution-Aware Test (CAT)、自适应 leader pixel 选择、Algorithm 1 像素矩形共享计算，以及 CTU/FIFO/渲染流水。CAT 的测试阈值为 alpha < 1/255。当前第一目标是完整渲染路径中 CAT 相对同路径无 CAT 的收益；不能只统计理论跳过项。
- 论文 §V 的完整系统另包含 30K 轮原始训练、贡献剪枝、3K 轮微调、聚类、FP16 渲染和混合精度 CTU。现有未经这套处理的预训练模型可用于 CTU 单项消融，不等价于论文完整模型设置。完整论文复现须另做模型准备、画质和各项消融。
- 论文硬件用 Verilog、TSMC 28 nm 综合和含 LPDDR4 的周期仿真。作者 RTL、仿真器和 FPGA 工程本次未核实公开。因此目标是依论文公开规格在 30TAI Lite 上独立实现并实测机制，不是直接移植作者工程，也不能沿用其 FPS、面积或能效数字。
- NPU 不是 FLICKER 已核实的硬件路径。后续接入 NPU 必须列为本项目异构扩展，和论文机制加速分别计量。

## 可验收顺序

1. **已完成首轮：**冻结同一完整场景、相机、分辨率、模型哈希和无筛选的有序 CPU Golden；建立逐 Gaussian–mini-tile 掩码参考，对照当前 CAT 访问轨迹的逐像素 alpha >= 1/255 真值，统计漏筛和候选 Gaussian–pixel 数；在完整图像上测 PSNR/SSIM、最大/平均像素差和最后贡献项差异。仍需更多场景/视角验证近似质量。
2. **已完成软件消融：**按论文 §III 单独启用 4x4 CAT、Dense/Sparse/按长短轴比 3 分类的自适应 leader pixel，再启用 Algorithm 1 的像素矩形共享计算。现有软件结果计入 CAT 准备、筛选和合成，不含 GPU 准备输入。独立 CAT 微阶段时延未单列；不能据此推断硬件 CTU 周期收益。
3. **FLK0 实板已完成：**FPGA 从 Gaussian 几何参数及有序 Tile 列表实际计算 power/exp/alpha、按序合成并回读完整图像，两个完整视角逐位匹配 FP16 参考。旧 GSB1 仅接收 CPU 已算好的 alpha，保留为历史对照。硬件正确性与相对官方 FP32 的近似误差分别登记。
4. **FLK1 实板已完成首轮：**相同 FPGA 位流加入论文 §IV 的 sub-tile AABB、CAT/CTU、mini-tile 分发和具备背压的 FIFO，两完整视角六模式对照、串行/流水和连续十帧通过各自参考。计入主机参数准备、搬运、同步及回读，离线 GPU 前处理仍不在计时范围。后续先修复持续流水调度的结构开销，再补混合精度及论文其余模块，不假定 ASIC 配置可直接装入 30TAI Lite。
5. 若要宣称完整论文方法级复现，补齐论文 §V 的剪枝、微调、聚类、量化与相同口径的质量实验，逐项记录偏差。未完成时仅称 CAT/CTU 机制复现与 FPGA 移植。

每次工程交付必须复述同输入的 CPU、FPGA 无 CTU、FPGA 有 CTU 的完整帧时延/FPS，工作量与传输量、画质、LUT/FF/DSP/BRAM、整板时序，以及功耗实测状态；明确收益、退化和未完成项。旧 CPU 四核 7.614 s、CPU+GSB1 FPGA 38.840 s 仅作历史，不混入本轮基线。现已取得 CTU 实测收益，数据见 [分版本表](examples/3dgs_flicker_hw/versions.csv)与 [FLK1 完整对照](examples/3dgs_flicker_hw/evidence/board_cat_summary_20260927/REPORT.md)。

备选路线：GEMM-GS（DAC 2026）的[全文](https://arxiv.org/abs/2604.02120)及[作者 CUDA 实现](https://github.com/shieldforever/GEMM-GS)均可获得，适合直接复核六维点积/GEMM 的 GPU 加速；其 Tensor Core 结果不能算 FPGA 论文复现。若选择该路线，应先在 GPU 复核作者 baseline/开关，再把矩阵化求值当作本板 NPU/FPGA 移植实验，分别报告。
