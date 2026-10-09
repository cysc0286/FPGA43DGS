# FPGA43DGS

**2026-10-09 渲染发布：旧版四路是默认，新版独立保存且不调用。**

| 目录 | 用途 |
|---|---|
| [rendering](examples/3dgs_reconstruction/mvsplat/rendering/README.md) | 主线常驻 CPU＋FPGA 渲染，`LiveRenderer.mainline(...)` |
| [rendering/package](examples/3dgs_reconstruction/mvsplat/rendering/package/README.md) | 旧版四路源码、固件、精确 RTL、哈希和验证冻结包 |
| [render_branch](examples/3dgs_reconstruction/mvsplat/render_branch/README.md) | 新版两路 PipeGS HGR v4，独立实验目录，不进入默认链 |

10 月 8 日同场景同轮各 60 帧：旧版 **43.575 ms**，新版 **53.302 ms**；
32768 高斯、128×128、场景已加载后的新相机到完整 RGB。
[完整对照与资源口径](examples/3dgs_reconstruction/mvsplat/render_branch/validation/RESULTS.md)。
本次是封装发布，不是新增速度测试。仓库根目录运行 `python tools/check_render_packages.py`。


面向悟净 30TAI Lite 的视频重建与 CPU＋FPGA 3DGS 渲染工程。当前研究目标是在画质合格的前提下缩短总耗时，充分使用板上资源。

**渲染主线已确定（2026-10-06）：四路 grouped-shared / 200 MHz + 常驻 C++ + 完整点集 Dense。**
推荐入口 `LiveRenderer.mainline(...)` 固定四线程、batch=2、稳定排序、direct_collect 和 NEON；
参数与产物身份集中在 `rendering/mainline.json`，交互基准默认测试 `mainline`。
32,768 高斯、128×128 的最新回退复测平均 **42.9170 ms**，P95 **44.0654 ms**；
同日原版两轮合计 120 帧平均 43.9201 ms。计时为已加载场景的新相机到完整 RGB，
画面保持原版逐位一致。PipeGS 候选更慢、16k 预览有明显质量代价，均不作为默认。
[主线配置、完整指标、资源、回退和版本选择理由](examples/3dgs_reconstruction/mvsplat/rendering/MAINLINE.md)。
本次固定入口和文档，未新增板端性能测量；以下是按日期保存的历史结果。

**历史换视角实测（2026-09-30）：45.212 ms，约 22.12 帧/秒，P95 46.263 ms。**
场景已加载、128×128、32,768 点；同轮旧程序为 48.632 ms，延迟减少 7.03%。
C++ 三趟排序与四核 Tile 列表生成已设为新接口默认，输出逐位不变。
本轮只优化相机到完整 RGB 帧返回，没有重测视频准备或改变 FPGA 位流。
[数据、源码、画质与后续硬件方向](examples/3dgs_reconstruction/mvsplat/rendering/VIEW_SWITCH_VALIDATION.md)。
以下是按版本保存的历史结果。

2026-09-30 新查表硬件已上板：同一个 A53 原生程序、32,768 高斯、128×128，
换视角平均 **49.059 ms**（60 帧，P95 50.204 ms），原 FPGA 为 50.294 ms。
硬件周期减少 **6.37%**，完整帧耗时减少 2.46%，三视角像素逐位不变。
完整视频首帧本轮复测 **24.694 s**，未显示整条准备链加速、未达 20 s。
源码、时序/资源、上板数据与回退入口见
[精确查表硬件报告](examples/3dgs_flicker_hw/pipeline/exp_rom/BOARD_VALIDATION.md)。
原四路冻结包仍保留；以下编译对照和原生接口记录属于前序版本。

前序编译器配对实测：完整点集 CPU＋FPGA **约 50 ms/帧**，优化后的纯 CPU
**约 161 ms/帧**；视频结束至首帧新测 **24.337 s**。编译配置、完整数据与
无收益的布局实验见 [编译优化报告](examples/3dgs_reconstruction/mvsplat/rendering/COMPILER_VALIDATION.md)。
这些编译器对照使用原 FPGA，与上面的新查表硬件结果分开记录。

2026-09-30：新增单进程原生 [rendering](examples/3dgs_reconstruction/mvsplat/rendering/README.md)。
128×128、32,768 高斯的完整点集换视角实板平均 **50.64 ms**；16,384 点近似预览 **31.93 ms**。
预览牺牲画质，两者不可混报。视频完整链、对照和效果图见
[本轮报告](examples/3dgs_reconstruction/mvsplat/rendering/VALIDATION.md)。

## 主线结构

```text
视频前 initialize 预热 → READY
  → video_input 接收完整视频 → VIDEO_COMPLETE
  → 抽帧/位姿 → MVSplat 前馈（已验证 ARM CPU；NPU 数值问题待修复）
  → Gaussian 导出与装入 → SCENE_READY
  → CPU 投影/排序＋FPGA 渲染 → FRAME_COMPLETE
```

源码包含预热、视频准备、常驻场景和 MVSplat NPU 分区候选。CPU＋FPGA 的预热和新原生渲染已有实板验证；NPU 数值问题仍待解决，不能把 ICraft 编译成功当作板端 NPU 加速成功。旧离线阶段见 [历史报告](examples/3dgs_reconstruction/mvsplat/INITIALIZE_NPU_VALIDATION.md)。

```text
examples/3dgs_reconstruction/
├── pipeline.py                         # 统一 CLI：reconstruct / initialize
├── modules/                            # 四模块接口与既有分阶段入口
└── mvsplat/
    ├── initialize/                     # 视频前权重/设备预热、常驻渲染候选
    ├── video_input/                    # 文件接收、抽帧、两视图与目标位姿
    ├── rendering/                      # 常驻原生换视角、帧交付、预览预算
    ├── warm_pipeline.py                # 视频后推理/PnP 调度、导出、首帧
    ├── npu/                            # 导出、编译、运行、核验与离线打包
    ├── infer.py / export.py            # 冷推理参考与 Gaussian 格式转换
    └── results/20260929/               # 小型指标、构建记录与效果图
```

预热时间和视频输入时间仅记录。**场景准备时间**从 `VIDEO_COMPLETE` 到首张指定视角的 `FRAME_COMPLETE`，包含首次渲染；`SCENE_READY` 是中间点。

| 模块 | 核心目录 | 当前状态 |
|---|---|---|
| 视频输入 | `examples/3dgs_reconstruction/modules/video_input` | 已有视频抽帧；实时摄像头尚未接入 |
| 位姿估计 | `examples/3dgs_reconstruction/modules/pose_estimation`、`examples/3dgs_reconstruction/mvsplat` | COLMAP CPU 完整路径和两视图快速路径均已在 ARM 上验证；快速路径仅重建局部共同视野 |
| 高斯生成 | `examples/3dgs_reconstruction/mvsplat` | 主线使用固定权重 MVSplat，在 ARM CPU 前馈生成高斯；OpenSplat 训练仅保留历史对照 |
| 渲染调用 | `examples/3dgs_reconstruction/modules/rendering` | 校验接口并调用已有冻结渲染器 |
| 实际渲染后端 | `examples/3dgs_flicker_hw` | CPU 投影/SH/分组排序，FPGA 筛选/求值/合成；四单元基线已留存 |
| 原生交互调用 | `examples/3dgs_reconstruction/mvsplat/rendering` | 单进程 CPU＋FPGA、常驻缓冲、内存帧交付；完整点集和近似预览已实板测试 |
| MVSplat NPU 候选 | `examples/3dgs_reconstruction/mvsplat/npu` | 7 个真实子图已有实板执行，但数值/权重预检问题未解决，未纳入可用整链 |
| 旧匹配 NPU 对照 | `examples/3dgs_reconstruction/npu_frontend` | 匹配正确性已有局部实测，不是当前 MVSplat 网络加速实现 |

统一入口为 [pipeline.py](examples/3dgs_reconstruction/pipeline.py)。板端默认主线：

```text
python examples/3dgs_reconstruction/pipeline.py reconstruct --video /path/to/video.mp4 --out NEW_RUN
```

这条命令调用 [MVSplat 板端链路](examples/3dgs_reconstruction/mvsplat/README.md)，默认使用画质已通过的完整 SfM 位姿路径；约一分钟的快速两视图路径需显式指定 `--pose-mode fast_pair --fused`，其首个视角未过预设画质门槛。两条路径都只在板端计算，不做逐场景反向训练，也未使用 NPU。原 `gaussian --opensplat` 仍可用于复核历史 CPU 训练结果，但不是当前主线。详细阅读：[目录与计划](MAINLINE.md)、[数据接口](examples/3dgs_reconstruction/modules/INTERFACES.md)、[仓库复用与反向传播](examples/3dgs_reconstruction/REPOSITORIES_AND_BACKPROP.md)。

## 下载后检查

本次 GitHub 交付是核心源码包，包含模块、接口、CPU/C++、HLS/RTL、测试、补丁和构建脚本。完整运行环境、视频/场景模型、SDK、厂家平台、BOOT/位流和大体积实验原始文件需另外准备。冻结包目录在 GitHub 上只含源码子集，不能直接当作完整板端运行包。

在 Python 3.10+ 的独立环境中执行：

```text
python -m pip install -r requirements-core.txt
python tools/check_core.py
python examples/3dgs_reconstruction/pipeline.py --help
```

默认检查使用自动生成的合成接口文件，验证16项接口、8项资源控制和3项主入口兼容行为；不连接板卡、不训练、不烧录。完整安装边界、固定第三方版本和重新打包命令见 [源码交付说明](docs/GITHUB_PACKAGE.md)，本次核验见 [CORE_VALIDATION.md](docs/CORE_VALIDATION.md)。

检查预热、视频、原生帧协议与 NPU 数据合同（本轮共 71 项软件测试，无需权重或 SDK）：

```text
python -m pip install -r requirements-mvsplat-checks.txt
python tools/check_core.py --mvsplat
```

实际模型编译/运行另需作者权重与匹配 ICraft SDK。Windows C++ 检查可在 x64 Visual Studio Developer shell 中执行；无需本仓库作者的私有 `vendor/export_env.bat`。Linux/ARM 构建入口及运行命令见 [预热](examples/3dgs_reconstruction/mvsplat/initialize/README.md) 和 [NPU](examples/3dgs_reconstruction/mvsplat/npu/README.md)。

## 结果与边界

- [CPU 视频重建参考](examples/3dgs_reconstruction/VALIDATION.md)：有电脑训练与早期模型板端 CPU 渲染记录，不能称视频重建已全部迁移到 ARM。
- [CPU＋FPGA 整链验证](examples/3dgs_flicker_hw/frontend/VALIDATION.md)：历史两个固定视角、559263点、320×178，整链相对 CPU Dense 为1.444×/1.322×；FPGA FP16与CPU FP32，非同精度纯硬件收益，也非实时。
- [NPU 匹配记录](examples/3dgs_reconstruction/npu_frontend/BOARD_VALIDATION.md)：三对局部结果经修正一致，但整体慢于CPU；未接入当前 MVSplat 链路。
- [MVSplat 板端实测](examples/3dgs_reconstruction/mvsplat/FAST_VALIDATION.md)：同一段 3 秒视频的完整 SfM 路径首图 249.43 秒、三目标画质达标；两视图单进程候选首图 66.68 秒、峰值 592.04 MiB、32,768 个高斯，但首个目标只有 19.45 dB，低于预设 20 dB 门槛。两路径均完成板端视频到 FPGA 图像；目前没有同时满足约一分钟和三目标画质的版本。[精简结果与效果图](examples/3dgs_reconstruction/mvsplat/results/20260929/README.md)可随仓库核对。

前三条为已有历史实验，MVSplat 条目为 2026-09-29 新增实板结果。其计时为单次端到端样本，不能称稳定一分钟达标；其中快速路径只用首尾两帧生成局部高斯，尚未融合整段视频。文档中的 `evidence/`、`runs/` 和本机绝对路径多为本地实验引用，未全部上传；以每份报告的日期、输入和测量范围为准。

## 冻结基线与依赖

四单元回退源码和接口见 [releases](releases/README.md)。后续候选应另建版本，不覆盖原 BOOT、位流或测量结果。FLICKER是当前渲染方法主线，整篇论文的所有机制尚未复现。

仍被主线引用的 `examples/3dgs_compositor/board`、`3dgs_scene` 和 `3dgs_flicker_cat` 保留原路径。基础 ALU 源码在 [archive](archive/basic_demos_20260928/README.md)。第三方来源、版本及许可证见 [third_party/sources.json](third_party/sources.json)；仓库不包含设备密码或私钥。
