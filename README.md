# FPGA43DGS

面向悟净 30TAI Lite 的视频重建与 CPU＋FPGA 3DGS 渲染工程。当前研究目标是在画质合格的前提下缩短总耗时，充分使用板上资源。

## 主线结构

```text
视频输入 → 位姿与稀疏重建 → 高斯生成/优化 → CPU＋FPGA 渲染
```

| 模块 | 核心目录 | 当前状态 |
|---|---|---|
| 视频输入 | `examples/3dgs_reconstruction/modules/video_input` | 已有视频抽帧；实时摄像头尚未接入 |
| 位姿估计 | `examples/3dgs_reconstruction/modules/pose_estimation` | COLMAP CPU，已有30帧 ARM 验证 |
| 高斯生成 | `examples/3dgs_reconstruction/modules/gaussian_generation` | OpenSplat CPU 优化与导出；电脑参考已跑通，完整 ARM 训练尚未验收 |
| 渲染调用 | `examples/3dgs_reconstruction/modules/rendering` | 校验接口并调用已有冻结渲染器 |
| 实际渲染后端 | `examples/3dgs_flicker_hw` | CPU 投影/SH/分组排序，FPGA 筛选/求值/合成；四单元基线已留存 |
| NPU 候选 | `examples/3dgs_reconstruction/npu_frontend` | 匹配正确性已做局部实测，尚未形成默认加速收益 |

统一入口为 [pipeline.py](examples/3dgs_reconstruction/pipeline.py)。详细阅读：[目录与计划](MAINLINE.md)、[数据接口](examples/3dgs_reconstruction/modules/INTERFACES.md)、[仓库复用与反向传播](examples/3dgs_reconstruction/REPOSITORIES_AND_BACKPROP.md)、[NPU 加速分析](examples/3dgs_reconstruction/NPU_ACCELERATION_ANALYSIS_20260928.md)。

## 下载后检查

本次 GitHub 交付是核心源码包，包含模块、接口、CPU/C++、HLS/RTL、测试、补丁和构建脚本。完整运行环境、视频/场景模型、SDK、厂家平台、BOOT/位流和大体积实验原始文件需另外准备。冻结包目录在 GitHub 上只含源码子集，不能直接当作完整板端运行包。

在 Python 3.10+ 的独立环境中执行：

```text
python -m pip install -r requirements-core.txt
python tools/check_core.py
python examples/3dgs_reconstruction/pipeline.py --help
```

默认检查使用自动生成的合成接口文件，验证16项接口和8项资源控制行为；不连接板卡、不训练、不烧录。完整安装边界、固定第三方版本和重新打包命令见 [源码交付说明](docs/GITHUB_PACKAGE.md)，本次核验见 [CORE_VALIDATION.md](docs/CORE_VALIDATION.md)。

## 结果与边界

- [CPU 视频重建参考](examples/3dgs_reconstruction/VALIDATION.md)：有电脑训练与早期模型板端 CPU 渲染记录，不能称视频重建已全部迁移到 ARM。
- [CPU＋FPGA 整链验证](examples/3dgs_flicker_hw/frontend/VALIDATION.md)：历史两个固定视角、559263点、320×178，整链相对 CPU Dense 为1.444×/1.322×；FPGA FP16与CPU FP32，非同精度纯硬件收益，也非实时。
- [NPU 匹配记录](examples/3dgs_reconstruction/npu_frontend/BOARD_VALIDATION.md)：三对局部结果经修正一致，但整体慢于CPU；新前馈高斯网络仍是候选。

以上是已有历史实验，本次上传不产生新的速度、画质或硬件结果。文档中的 `evidence/`、`runs/` 和本机绝对路径多为本地实验引用，未全部上传；以每份报告的日期、输入和测量范围为准。

## 冻结基线与依赖

四单元回退源码和接口见 [releases](releases/README.md)。后续候选应另建版本，不覆盖原 BOOT、位流或测量结果。FLICKER是当前渲染方法主线，整篇论文的所有机制尚未复现。

仍被主线引用的 `examples/3dgs_compositor/board`、`3dgs_scene` 和 `3dgs_flicker_cat` 保留原路径。基础 ALU 源码在 [archive](archive/basic_demos_20260928/README.md)。第三方来源、版本及许可证见 [third_party/sources.json](third_party/sources.json)；仓库不包含设备密码或私钥。
