# FLICKER mini-tile CAT 软件参考

本目录按 [FLICKER, DATE 2026](https://arxiv.org/abs/2603.01158) 论文 II-III 节独立编写，没有使用作者代码。它在悟净 30TAI Lite 的四核 ARM 上，对真实预训练场景的**已投影、有序完整帧**执行 4x4 mini-tile Contribution-Aware Test (CAT) 和有序 3DGS 合成。这是论文筛选机制的可运行参考与 FPGA 设计前的质量/性能门槛；本目录尚无 FPGA 或 NPU 执行。

`cat_reference.cpp` 与 [场景 ABI](../3dgs_scene/CONTRACT.md) 对接。`base` 不使用 CAT；`dense` 测四个角；`sparse` 测两个对角；`adaptive` 按长短轴比 3 选择 Dense/Sparse；`dense_pr` 在 Dense 上加论文 Algorithm 1 的像素矩形共享二次型计算。各模式使用同一有序 Tile 列表、alpha 阈值 1/255、提前终止和背景合成。CAT 比较的是二次型 `E <= ln(255*opacity)`；从 `alpha=opacity*exp(-E) >= 1/255` 可直接得到这一形式。

在仓库根目录运行，板卡 SSH 和已有完整场景数据需可用。密码只放进当前进程的 `FPGA_BOARD_PASSWORD` 环境变量，不写入文件：

```powershell
python examples/3dgs_flicker_cat/run_board.py n559263_v0_w320 --threads 4 --repeats 5 --warmup 1 --audit
python examples/3dgs_flicker_cat/run_board.py n559263_v10_w320 --threads 4 --repeats 5 --warmup 1 --audit
python examples/3dgs_flicker_cat/analyze_results.py examples/3dgs_flicker_cat/evidence/<view0-run> examples/3dgs_flicker_cat/evidence/<view10-run>
```

`run_board.py` 上传实际输入并校验哈希，在板上用 g++ 编译、运行单元检查及各模式，下载完整帧和逐次计时，再在 PC 上与官方 CUDA Golden 对照。`--audit` 额外逐项检查被 CAT 筛掉、但 alpha 本应达到阈值的候选；该审计时间**不计入**性能值。结果哈希、编译参数和板卡信息随每次运行保存在 `evidence/<时间戳>/result.json`，原始输出和计时也在同目录。较大的 `evidence/` 默认不进 Git，必须与摘要的哈希对应保存。正式结果与优劣分析见 [板端验证](VALIDATION.md)、[机器可读摘要](results_summary.csv)和[图像对照](comparison.png)。

复现范围不包括论文的 8x8 sub-tile AABB、CTU/FIFO 硬件流水、混合精度、模型剪枝/微调/聚类，也不包括 GPU 投影、SH 和排序的板端迁移。因此当前 FPS 仅为**已准备帧的 ARM 光栅化 FPS**，不是 Camera Pose 到屏幕的端到端 FPS，更不是 FPGA 加速数字。
