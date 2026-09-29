# 2026-09-30 板端优化验收索引

固定 NYU 视频、两个 128×128 上下文、32768 Gaussian、目标视角 7/15/22。
场景时间按 VIDEO_COMPLETE→FRAME_COMPLETE；预热不纳入性能指标。

|候选|两次视频结束到首帧|均值|说明|
|---|---|---:|---|
|rows|26.933 / 26.994 s|26.964 s|内存 Gaussian 行交接，首目标优先|
|frame|26.162 / 26.447 s|26.305 s|相机内存接口、批量 PPM 转换、去重复模型哈希|
|archive|26.980 / 26.707 s|26.843 s|NPZ 写盘/重读移到首帧之后；未证明更快|

三个候选均是板端 CPU 网络 + FPGA 渲染，不是 NPU 加速结果。都未到 20 s。
原历史四线程串行均值 27.876 s，配置和采样不同，只作历史参考。

换视角新接口 8 次：均值 0.18459 s，中位数 0.16403 s，范围 0.15598–0.33192 s。
旧接口 4 次均值约 0.823 s；观测整次调用提速约 4.46 倍，主要来自软件封装。
每次浮点帧缓冲与 PPM 都与冻结 FPGA 输出逐字节一致；不是新增 FPGA 算力。
第 7 帧质量沿用一致图像的 PSNR 19.4519 dB、SSIM 0.80446，仍未达原 20 dB。

## 代码增量：Gaussian rows 转换

`export.factor_covariances` 增加了批量矩阵到四元数实现，并保留
`quaternion_backend="scipy"` 作为回退对照。32768 个 Gaussian 的 Windows 主机
烟测中，完整转换从 SciPy 的 35.66 ms 降到 29.03 ms；旋转矩阵逐元素误差低于
`1e-12`，序列化后协方差相对 Frobenius 误差最大 `4.804e-7`，导出合同通过。
这只是主机结果，尚未把它写成 30TAI Lite 的新速度；板端复测后才能更新场景准备
指标。新实现不会跳过任何高斯、协方差或图像校验。

## 结果入口

- `warm_rows_overlap3_v1/v2`：第一阶段整链记录。
- `frame_multiview_board_v1/v2`：四视角帧及 PPM 与冻结输出的实板对照。
- `warm_frame_overlap3_v1/v2`：渲染封装优化后的整链记录。
- `warm_archive_overlap3_v1/v2`：归档延后后的整链与内存记录。
- `source_snapshots/manifest.json`：上述三阶段从板上取回的源码版本与逐文件哈希。
- `npu_partition_cost_single_session/summary.json`：七分区打包、SDK、等待、解包。
  七个真实分区均未通过原数值门槛，不能以此计算加速比；79.5 MiB 是逻辑 I/O。
- `first_conv_factorial_board`：已知输入/权重与真实输入/权重分离控制。
- `first_conv_fp16_board`：真实首层 FP16 连续两次失败，最大误差 0.0898752。
- `first_conv_icraft_host_tf32/comparison.json`：ICraft 逐层诊断参考。
  板端相对 ICraft 参考最大差 0.09375、RMSE 0.00767243，未解释为正常量化误差。
- `first_conv_fp32_sfb_synthetic_board`：FP32 已知卷积预检通过但输出失败。
  该路径仍为隔离诊断，未放宽生产 ABI。
- `depth_coverage*` / `depth_conv*`：depth 编译与真实执行的失败证据。

## 当前边界和下一步

NPU 数值问题没有修复，多会话和完整前向尚未通过。设备缓冲跨图直连、连续 depth
覆盖未实现；完整流水不能标为 CPU+NPU+FPGA 已验收。通过最小首层的 SDK 层输出
和权重检查是后续前提，之后再恢复七分区与整网验收。错误输出的短耗时不作成绩。

网络前向约 16.3–16.8 s，Gaussian 行转换约 2.54–2.56 s，前者仍是首要加速项。
后者可评估直接协方差接口，避免先分解再重建，但须另行比较精度与图像。
本轮没有更改 FPGA 内核或重测资源、时序、功耗；保留冻结包回退。

详细说明见 [主工作文档](../../BOARD_WARM_NPU_VALIDATION.md)。
