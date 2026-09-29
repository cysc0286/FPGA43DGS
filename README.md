# FPGA43DGS

面向悟净 30TAI Lite 的视频重建与 CPU＋FPGA 3DGS 渲染工程。当前研究目标是在画质合格的前提下缩短总耗时，充分使用板上资源。

## 主线结构

```text
视频输入 → 位姿与稀疏重建 → 高斯生成/优化 → CPU＋FPGA 渲染
```

| 模块 | 核心目录 | 当前状态 |
|---|---|---|
| 视频输入 | `examples/3dgs_reconstruction/modules/video_input` | 已有视频抽帧；实时摄像头尚未接入 |
| 位姿估计 | `examples/3dgs_reconstruction/modules/pose_estimation`、`examples/3dgs_reconstruction/mvsplat` | COLMAP CPU 完整路径和两视图快速路径均已在 ARM 上验证；快速路径仅重建局部共同视野 |
| 高斯生成 | `examples/3dgs_reconstruction/modules/gaussian_generation`、`examples/3dgs_reconstruction/mvsplat` | 固定权重 MVSplat 已在 ARM CPU 前馈生成高斯；OpenSplat 的完整 ARM 逐场景训练仍未验收 |
| 渲染调用 | `examples/3dgs_reconstruction/modules/rendering` | 校验接口并调用已有冻结渲染器 |
| 实际渲染后端 | `examples/3dgs_flicker_hw` | CPU 投影/SH/分组排序，FPGA 筛选/求值/合成；四单元基线已留存 |
| NPU 候选 | `examples/3dgs_reconstruction/npu_frontend` | 匹配正确性已做局部实测；当前 MVSplat 链路未使用 NPU |

统一入口为 [pipeline.py](examples/3dgs_reconstruction/pipeline.py)。详细阅读：[目录与计划](MAINLINE.md)、[数据接口](examples/3dgs_reconstruction/modules/INTERFACES.md)、[仓库复用与反向传播](examples/3dgs_reconstruction/REPOSITORIES_AND_BACKPROP.md)、[NPU 加速分析](examples/3dgs_reconstruction/NPU_ACCELERATION_ANALYSIS_20260928.md)。

新的固定权重板端候选使用独立入口 [MVSplat 板端链路](examples/3dgs_reconstruction/mvsplat/README.md)，不改变原 `pipeline.py` 的 OpenSplat 路径。输入板上视频，ARM 完成位姿和高斯生成，随后由 ARM CPU 预处理、FPGA 渲染；不需要针对新场景反向训练，也没有使用 NPU。

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
- [NPU 匹配记录](examples/3dgs_reconstruction/npu_frontend/BOARD_VALIDATION.md)：三对局部结果经修正一致，但整体慢于CPU；未接入当前 MVSplat 链路。
- [MVSplat 板端实测](examples/3dgs_reconstruction/mvsplat/FAST_VALIDATION.md)：同一段 3 秒视频的完整 SfM 路径首图 249.43 秒、三目标画质达标；两视图单进程候选首图 66.68 秒、峰值 592.04 MiB、32,768 个高斯，但首个目标只有 19.45 dB，低于预设 20 dB 门槛。两路径均完成板端视频到 FPGA 图像；目前没有同时满足约一分钟和三目标画质的版本。[精简结果与效果图](examples/3dgs_reconstruction/mvsplat/results/20260929/README.md)可随仓库核对。

前三条为已有历史实验，MVSplat 条目为 2026-09-29 新增实板结果。其计时为单次端到端样本，不能称稳定一分钟达标；视频只用首尾两帧生成局部高斯，尚未融合整段视频。文档中的 `evidence/`、`runs/` 和本机绝对路径多为本地实验引用，未全部上传；以每份报告的日期、输入和测量范围为准。

## 冻结基线与依赖

四单元回退源码和接口见 [releases](releases/README.md)。后续候选应另建版本，不覆盖原 BOOT、位流或测量结果。FLICKER是当前渲染方法主线，整篇论文的所有机制尚未复现。

仍被主线引用的 `examples/3dgs_compositor/board`、`3dgs_scene` 和 `3dgs_flicker_cat` 保留原路径。基础 ALU 源码在 [archive](archive/basic_demos_20260928/README.md)。第三方来源、版本及许可证见 [third_party/sources.json](third_party/sources.json)；仓库不包含设备密码或私钥。
