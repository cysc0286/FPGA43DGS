# HeteroGS：板端视频重建与 NPU＋FPGA 渲染

平台为悟净 30TAI Lite。主线：视频输入 → CPU 位姿估计 → MVSplat 生成 Gaussian → CPU＋FPGA 渲染 → 网页显示。
固定预训练权重在视频前加载；新场景执行前向推理，换视角复用已生成的场景。

## 六部分代码

所有业务代码位于 `examples/3dgs_reconstruction/mvsplat/`：

```text
initialize/           预热、生命周期、场景准备协调、计时和部署
video_input/          USB 录像／文件接收、解码、选帧和标定
pose_estimation/      特征、匹配、两视图几何和目标视角 PnP
gaussian_generation/  常驻模型、NPU 子图、CPU 剩余计算与 Gaussian 适配
rendering/            常驻 CPU＋FPGA 渲染、相机 ABI、冻结 render_main
web/                  采集／上传页面、视角控制、HTTP 和静态资源
```

[代码与接口](docs/CODE_STRUCTURE.md) · [项目约束](docs/PROJECT_REQUIREMENTS.md) · [当前主线与实测边界](MAINLINE.md) · [整理核验](docs/CORE_VALIDATION.md)

## 唯一整链入口

从 `examples/3dgs_reconstruction/` 运行：

```sh
python pipeline.py initialize --help
python pipeline.py web --help
python pipeline.py view --help
```

`initialize` 管理视频前预热及视频后首帧；`web` 提供 `/capture`、`/upload` 和交互查看器；`view` 只恢复已有场景。
原 OpenSplat／整段 SfM 冷链、旧发布包、独立渲染候选和重复兼容入口已移出当前代码树。

推荐 NPU 配置显式选择已采纳的十个子图：

```sh
python pipeline.py initialize --video INPUT.mp4 --out NEW_RUN \
  --weights /path/to/re10k.ckpt --vendor /path/to/MVSplat_reference \
  --renderer /path/to/matching/renderer --live-renderer /path/to/live_exact \
  --render-profile mainline --backend npu \
  --partitions backbone_cnn depth_head to_gaussians to_disparity \
    transformer_ffn_0 transformer_ffn_1 transformer_ffn_2 \
    transformer_ffn_3 transformer_ffn_4 transformer_ffn_5 \
  --partition-bundle /path/to/compiled --oracle-graphs /path/to/graphs \
  --npu-library /path/to/libmgs_npu.so --worker-python /path/to/arm/python \
  --npu-acceptance /path/to/current/acceptance.json --interactive
```

这些路径为占位，实际权重、SDK、图和硬件需另行配置。NPU 失败不静默回退。
CPU 对照显式选择 `--backend cpu` 并去掉 NPU 专用参数。
注意力、代价体等部分仍在 CPU；不能将十子图方案称为全网 NPU。
严格 TF32 张量门槛仍未关闭；图像资格、输入切换、资产身份、设备隔离和内存检查继续保留。
目录变化会使旧源码身份失效，重新部署需重新生成匹配资格，不能复用旧 receipt 充当新验收。

源码不包含私人录像、场景、权重、SDK 或 Python 环境。当前整理仅做本机软件核验，未新增板端性能数据。
