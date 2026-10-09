# 3DGS 主线与目录入口

## 2026-10-09 发布入口

旧版四路 `grouped-shared` 固定在
[rendering/package](examples/3dgs_reconstruction/mvsplat/rendering/package/README.md)，
默认调用保持 `LiveRenderer.mainline(...)`。
新版在并列目录 [render_branch](examples/3dgs_reconstruction/mvsplat/render_branch/README.md)，
没有自动导入、构建或烧录。两者使用不同固件哈希；正常视频/预热/渲染均不选择新版。
10 月 8 日最新配对：旧版 43.575 ms、新版 53.302 ms（32768 点、128×128），
本次只封装既有结果。历史版本与证据继续保留。


2026-10-06 已确定渲染主线：**四路 grouped-shared + 常驻 C++ + 完整点集 Dense**。
交互调用统一用 `LiveRenderer.mainline(...)`；唯一配置清单为
[mainline.json](examples/3dgs_reconstruction/mvsplat/rendering/mainline.json)，
决策、职责、资源、产物身份及回退见
[渲染主线](examples/3dgs_reconstruction/mvsplat/rendering/MAINLINE.md)。
32,768 点、128×128，10 月 6 日主线两轮 120 帧平均 **43.9201 ms**；
最后回退复测 60 帧 **42.9170 ms**，P95 **44.0654 ms**。
软件最终配对曾测得 40.9671 ms；分别保留，不能拼成新收益。
PipeGS/DPC/CTU 候选继续独立实验；有损抽点预览不代表完整点集主线速度。
此次仅固定渲染选择和入口，未重测视频准备、NPU 或板端性能。

## 前序版本记录（不是当前默认选择）

2026-09-30 换视角主线：`mvsplat/rendering` 的 C++ 三趟排序与四核 Tile 列表生成，
同轮旧程序 48.632→**45.212 ms**，P95 46.263 ms，三视角输出逐位不变。
128×128、32,768 点、60 帧；已加载场景的新相机到完整 RGB 返回，不是视频
准备时间。默认 `radix_bits=11, parallel_tiles=True`；旧程序显式用 8/False。
本轮 FPGA 位流、前端与 NPU 未修改。
[完整换视角验证](examples/3dgs_reconstruction/mvsplat/rendering/VIEW_SWITCH_VALIDATION.md)。
下文保留前序结果与路径，不将历史数据当作当前速度。

2026-09-30 查表硬件增量已完成上板：同一 A53 原生二进制，完整点集换视角
50.294→**49.059 ms**（P95 50.204 ms）；硬件周期 5.570→5.215 百万，
减少 6.37%，三视角 raw/RGB 不变。200 MHz 未改变，四路并行仍保留。
全板 LUT/FF/BRAM/DSP 为 76.84%/62.04%/85.47%/63%，Slice 为 99.16%。
视频首帧复测 24.694 s，不能把毫秒级后端收益称为秒级准备链加速。
[源码、完整证据、质量和回退](examples/3dgs_flicker_hw/pipeline/exp_rom/BOARD_VALIDATION.md)。
硬件算子在 `pipeline/exp_rom`，布线实验在 `physical_opt`；ARM 原生链继续位于
`mvsplat/rendering`，没有混入 NPU 或覆盖原冻结包。

2026-09-30 编译器增量：同源码 A53 定向编译让纯 CPU 渲染
177.553→161.341 ms，CPU＋原 FPGA 50.604→49.995 ms；当前新源码默认
复测 50.294 ms。完整视频场景准备新测 **24.337347 s**，图像哈希不变。
详细配对数据、负结果、内存和画质见
[编译与布局验证](examples/3dgs_reconstruction/mvsplat/rendering/COMPILER_VALIDATION.md)。
软件结果使用原冻结 FPGA；新硬件候选不能混入这些对照。

2026-09-30 最新：单进程原生渲染位于
[mvsplat/rendering](examples/3dgs_reconstruction/mvsplat/rendering/README.md)。
同场景完整点集换视角 50.64 ms；可选 16k 近似预览 31.93 ms。
CPU 同调用范围对照 178.95 ms（既有 CPU FP32 / FPGA 精度不同），
全链实测、画质、负结果和计时边界见
[原生渲染验证](examples/3dgs_reconstruction/mvsplat/rendering/VALIDATION.md)。
下文 0.83 秒是旧版本历史，不代表当前原生接口。

2026-09-29 实板更新：视频前预热 CPU＋FPGA 路径已验证，四线程串行三次场景准备平均 **27.88 秒**（27.21–28.35 秒），预热 43.69–49.12 秒仅记录。默认 `initialize` 采用四线程串行，重叠候选保留；原冷 `reconstruct` 和完整 SfM 质量回退保持。常驻模型连续换视角约 0.83 秒，帧哈希匹配冻结 FPGA。首目标质量仍 19.45 dB、未过 20 dB；NPU 七分区真实执行但数值验收失败，不能标为可用整链后端。[本轮数据、效果图、失败分析及下一步](examples/3dgs_reconstruction/mvsplat/BOARD_WARM_NPU_VALIDATION.md)。下文是此前各阶段历史记录，以本条和实板报告为准。

2026-09-29 NPU 部署 v2 离线验证完成：默认顺序共享 IPC 缓冲由 79.5 降到 36.375 MiB，真实输入新旧模式 Gaussian 哈希一致；新增包完整性预检、ABI 2 和逐分区 CPU/候选测速，实际 NPU/整链收益仍待回板。逻辑 I/O 仍为 79.5 MiB，不能将容量下降当成搬运量或端到端时延下降。详见 [v2 报告与效果图](examples/3dgs_reconstruction/mvsplat/NPU_DEPLOYMENT_V2.md)。

2026-09-29 新增**离线候选**：视频前预热在 `mvsplat/initialize/`，视频接收及快速位姿在 `mvsplat/video_input/`，视频后的调度在 `mvsplat/warm_pipeline.py`；冷路径仍为 `reconstruct`，新候选入口是 `pipeline.py initialize`。NPU 子图导出、编译、常驻运行和核验位于 `mvsplat/npu/`。本轮用户在外，未访问板卡；7 子图编译及电脑参考回归通过不代表实板 NPU 部署成功。详见 [本轮实现与验收边界](examples/3dgs_reconstruction/mvsplat/INITIALIZE_NPU_VALIDATION.md)。

术语统一：**预热时间**（initialize，视频前）和**视频输入时间**只记录；**场景准备时间**从 `VIDEO_COMPLETE` 到指定视角的首帧 `FRAME_COMPLETE`，其中 `SCENE_READY` 为中间事件；**后续渲染时间**独立统计。新的闭文件交付适配器不包含真实摄像头或上传的采集计时。上述候选的新板端场景准备时间、常驻内存和 NPU 收益均待测。

更新：2026-09-29。当前主线是板端离线视频 → 位姿 → 固定权重 MVSplat 前馈生成高斯 → CPU＋FPGA 渲染。电脑用于开发、构建和参考验证，不参与新场景的板端计算。旧 OpenSplat 训练链仅作为可回放的历史对照。主命令：`python examples/3dgs_reconstruction/pipeline.py reconstruct --video INPUT.mp4 --out NEW_RUN`。

当前验收计时（2026-09-29 更新）：从**视频输入结束**开始，到板端完成高斯生成并用该模型完成**预先指定视角的一帧渲染**结束。以帧输出完成并通过哈希/完整性检查为终点，不等待人工打开图像。约一分钟是这个计时的期望目标。最新已加入起止事件：同一视频先复制并同步到板端，记录接收完成事件后启动，两线程 74.04 秒、同环境四线程 71.74 秒，均结束于第 7 帧 FPGA 图像哈希核验；每组仅一次完整运行，不含视频上传。第 7 帧仍未达到 20 dB，约一分钟与三视角画质同时通过尚未实现。场景覆盖与跨帧组融合独立验收，当前两视图模型只覆盖局部共同视野。[快速候选](examples/3dgs_reconstruction/mvsplat/FAST_VALIDATION.md) 的历史 66.68 秒和完整 SfM 位姿基线的 249.43 秒未记录接收完成事件，不能与新口径混报。NPU、GPU、逐场景训练没有进入当前生产链。

2026-09-29 主线切换：固定权重 [MVSplat 板端闭环](examples/3dgs_reconstruction/mvsplat/README.md) 已在板端完成视频文件→CPU SfM→CPU 高斯前馈→冻结 CPU+FPGA 渲染，并验证三张目标视角。它替代 OpenSplat 作为默认高斯生成方法，但不等于板端训练或 NPU 推理。完整 SfM **位姿**基线生成 32,768 高斯、恢复 30/30 帧位姿，三视角 PSNR 20.90–23.56 dB；首张 FPGA 图像 249.43 s，非实时。网络仅使用两张参考帧预测局部高斯，30/30 位姿注册不等于整段场景重建完成。完整指标、失败配置和计时口径见 [验证记录](examples/3dgs_reconstruction/mvsplat/VALIDATION.md)。

## 每轮结束时的指标交代

每次完成工程工作后，结尾都要明确交代当前视频→位姿→高斯→渲染链路，而不只报告所改模块：

- **主计时**：记录视频最后一帧输入/文件接收完成的时间戳作为起点；在板端完成高斯生成，并用所得模型渲染完成预先指定相机的一帧后停止。终点需有完整帧和输出哈希/状态校验，不要求人工显示；相机/视角编号须在运行前固定。包含解码、位姿、生成、导出和最终渲染；若有融合，也纳入。注明视频长度/帧数、设备、版本、实际场景覆盖与质量门槛。缺少起止时间戳时主指标写“未测”，不能用别的起点混报；出图成功与全场景覆盖合格分开判定。
- **辅助计时**：另报高斯场景模型就绪时间、各阶段、重建后单视角渲染的壁钟时间。模型就绪可作内部阶段边界，不把预览帧另设为主计时中点。重复渲染的验收总时间单列；网络纯推理、FPGA 内核和验收总时间都不能冒充主计时。
- **最终质量**：对同一目标视角报告相对原视频/留出图像的 PSNR、SSIM、局部误差和可见缺陷，逐视角核对预设门槛；另列 FPGA 相对 CPU/独立渲染参考的数值一致性，不能把硬件一致性误称重建画质。
- **代价与对照**：报告高斯数量、位姿覆盖、峰值进程内存，以及同输入同口径 CPU/FPGA 速度对照；涉及硬件变更时附资源、时序和功耗的实测状态，并分析收益和不足。
- **证据边界**：区分本轮新实测与引用的历史结果，给出记录位置、样本和重复次数；未测项目写“未测”，失败版本照实列出。仅改代码或文档时，明确说没有新增板端性能/画质数据，并复述当前基线与下一步验证。

同一段 3 秒/30 帧视频的历史完整 SfM 位姿路径从开始处理已存储视频到首图 249.43 s，三视角 PSNR 20.90/21.56/23.56 dB、SSIM 0.8323/0.8557/0.8750，峰值进程树 RSS 529.09 MiB；三视角达到 PSNR 20 dB、SSIM 0.7 的局部最低门槛。历史三视角 CPU/FPGA 预热和重复验证耗时 342.69 s，不能称为单帧出图时间。最新四线程快速候选按接收结束事件计时 71.74 s、峰值进程树 RSS 582.05 MiB、32,768 高斯，三视角 19.45/20.22/21.35 dB、SSIM 0.8045/0.7851/0.7842，质量与两线程逐字节一致，首视角门槛未过。完整 SfM 本轮未重跑；两条路径仍没有整段场景覆盖与融合验收。原始复测 JSON 与效果图见 `examples/3dgs_reconstruction/mvsplat/results/20260929/speed_threads/`。

```text
视频
  ↓ video_input
FrameSet：帧图像 + 帧号/时间戳清单
  ↓ pose_estimation
PoseSet：去畸变图像 + 内参/位姿 + 稀疏几何
  ↓ gaussian_generation
GaussianScene：优化后的高斯参数
  ↓ export（高斯模块内部格式适配）
RenderInput：model.ply + FLCAM001 相机
  ↓ rendering（调用已有冻结后端）
RenderResult：frame.bin + frame.ppm + result.json
```

## 四个模块

| 模块 | 当前实现路径 | 职责与状态 |
|---|---|---|
| 视频输入 | [video_input](examples/3dgs_reconstruction/modules/video_input/) | 从已有视频均匀抽帧；实时摄像头不是当前接口 |
| 位姿估计 | [pose_estimation](examples/3dgs_reconstruction/modules/pose_estimation/) | COLMAP CPU 特征/匹配/位姿/三角化、去畸变、内参适配；输出包含稀疏点，不仅是位姿 |
| 高斯生成 | [mvsplat](examples/3dgs_reconstruction/mvsplat/) | 主线：MVSplat 固定权重板端前馈；[gaussian_generation](examples/3dgs_reconstruction/modules/gaussian_generation/) 仅保留旧 OpenSplat 训练对照，二者不能混称训练 |
| 高斯渲染 | [rendering](examples/3dgs_reconstruction/modules/rendering/) | 校验模型/相机接口并调用已完成的 CPU＋FPGA 渲染包；不重新实现光栅化 |

统一入口：[pipeline.py](examples/3dgs_reconstruction/pipeline.py) 的 `reconstruct` 命令。详细字段与命令：[接口文档](examples/3dgs_reconstruction/modules/INTERFACES.md)。`video`/`pose` 分阶段命令仍由 MVSplat 完整路径使用；`gaussian`/`export`、`stages.py`、`run_cpu.py` 和 `bounded/run.py` 保留旧 OpenSplat 对照。

模块独立交付使用 `pipeline.py validate --only`，整段检查保留 `--through`。

2026-09-28 新分析目标：以质量合格后的总耗时最短为优先，允许充分使用板上资源；NPU 的前馈高斯、特征和批量属性候选见 [加速分析](examples/3dgs_reconstruction/NPU_ACCELERATION_ANALYSIS_20260928.md)。这些是候选研究，尚未切换默认重建方法或重新上板验证。

四模块采用运行目录中的文件交接，可单独运行和替换实现。它们不是四个必须同时常驻的服务；板端优先顺序执行，避免叠加训练、SfM 和渲染内存。这个整理没有新增实时建图、NPU 加速或两单元硬件。

## 已完成渲染后端

- 冻结运行包与回退基线：`releases/3dgs_renderer_v1_20260928/`，保留程序、源码、模型、相机、位流与清单。
- 后端研发源码：`examples/3dgs_flicker_hw/`；其中 `frontend/` 是**渲染前处理**，不是视频重建前端。
- CAT/PR 软件参考：`examples/3dgs_flicker_cat/`。
- `platform/` 和 `build/` 保持原路径，避免破坏已有厂商工程和绝对路径。
- 后续两单元轻量版应独立构建；本轮只整理模块，没有改变四单元硬件或数值规则。

## 实验与共享依赖

基础 ALU 源码与历史清单在 [archive/basic_demos_20260928](archive/basic_demos_20260928/README.md)；原始压缩包与实验输出仅保留在本地。

以下目录不作为主线入口，但暂不整目录移动：

| 路径 | 保留原因 |
|---|---|
| `examples/3dgs_compositor/board` | 当前 FLICKER 构建、上传、测量脚本仍导入其板卡工具；整目录搬走会破坏主线 |
| `examples/3dgs_reference`、`3dgs_tile_cpu`、`3dgs_scene`、`3dgs_batch`、`3dgs_fpga_eval` | 历史对照、模型/验证数据和可能的构建引用；下一次迁移需按依赖拆出共享资产，不能按“旧目录”删除 |
| `npu_3dgs` | 历史 NPU/异构实验和对照记录；不加入默认执行路径 |
| `examples/3dgs_reconstruction/npu_frontend` | 位姿估计模块的候选匹配实现，单独开发/验证；默认 CPU 链未切换 |

历史 evidence、runs 和 release 均保持可追溯；新目录整理的验证在 [本轮记录](examples/3dgs_reconstruction/evidence/module_refactor_20260928/VALIDATION.md)。

## 后续工作顺序

1. 优化板端位姿和高斯前馈的耗时，在保持三目标画质门槛的前提下缩短首图时间；当前没有稳定一分钟达标证据。
2. 扩展视频多组融合和场景覆盖，继续通过现有模型/相机 ABI 比较 CPU/FPGA。
3. 后端轻量化另建候选，保留四单元冻结版；减少逻辑资源不能等同于增加 Linux 内存。
4. 根据板端实测再选择前端 FPGA/NPU 候选。每轮保留正确性、画质、时延、内存和资源证据。
