# HeteroGS：板端视频重建与 FPGA 渲染

当前主线：**视频选帧 → 位姿估计 → MVSplat 固定权重生成高斯 → CPU＋FPGA 渲染**。
使用悟净 30TAI Lite；当前高斯生成由 ARM CPU 执行，NPU 仍是隔离候选。

先读 [代码结构](docs/CODE_STRUCTURE.md) 和 [四模块接口](examples/3dgs_reconstruction/modules/INTERFACES.md)。
性能口径及当前版本选择集中在 [MAINLINE.md](MAINLINE.md)。

```text
examples/3dgs_reconstruction/
├── pipeline.py                  唯一整链 CLI：initialize / reconstruct
├── mvsplat/
│   ├── initialize/              视频前预热、资源所有权、四个状态事件
│   ├── video_input/             视频接收、解码、选帧
│   ├── pose_estimation/         两视图几何、留出目标 PnP
│   ├── gaussian_generation/     常驻 MVSplat、Gaussian → 渲染 ABI
│   ├── rendering/               最佳已验收 CPU＋FPGA 后端
│   │   ├── render_main/         四路主线冻结包，保留源码/硬件/证据
│   │   └── render_branch/       PipeGS HGR v4 独立候选，默认不调用
│   ├── scene_preparation.py     组织视频和位姿交接，不藏数值算法
│   ├── warm_pipeline.py         组织生成高斯与首帧，控制重叠及归档
│   └── npu/                    MVSplat 高斯生成阶段的 NPU 候选
└── modules/                     旧入口兼容与共用文件校验；不是第二套新主线
archive/history_20261009/         历史实验 ZIP、原始路径与哈希索引
releases/                        较早冻结后端回退包
```

## 使用当前主线

在板端已配置的环境，从重建目录运行：

```sh
python pipeline.py initialize --video INPUT.mp4 --out NEW_RUN \
  --weights weights/re10k.ckpt --vendor vendor/MVSplat_reference \
  --renderer /path/to/3dgs_renderer_v1_20260928 \
  --live-renderer /path/to/accepted/live_exact --render-profile mainline
```

这些路径需要指向板端已有依赖和匹配硬件的原生程序；命令不会安装或烧录固件。
`--renderer` 提供沿用的 CPU 源码/契约依赖，`--live-renderer` 才是当前常驻 C++ 后端。
遗漏 `--live-renderer` 会进入历史兼容后端，不能用于报告当前主线性能。
只做后续换视角时，使用 `rendering.runtime.LiveRenderer.mainline(...)`：场景加载一次，反复提交相机。

## 目前能说明的结果

- 2026-10-08，32768 高斯、128×128、三个固定视角、60 帧：四路主线换视角平均 **43.575 ms**，P95 **52.893 ms**。
- 同轮 HGR v4 两路候选 **53.302 ms**，所以继续选择四路主线。
- 画质对留出实拍图：PSNR **19.45–21.35 dB**、SSIM **0.784–0.804**；重建模糊仍在。
- 上述是已加载场景的相机到完整 RGB 时间，**不是视频结束到首帧时间**。
- 整理后的 Python 调用链尚未重新运行；本轮只核对文件、源码搬迁和冻结资产，没有板测或加速新成绩。

完整证据保留在 [主线包](examples/3dgs_reconstruction/mvsplat/rendering/render_main/README.md)
和 [同轮比较](examples/3dgs_reconstruction/mvsplat/rendering/render_branch/validation/RESULTS.md)。
历史成功与负结果已 [单独归档](archive/history_20261009/README.md)，无需把它们展开到主线。
