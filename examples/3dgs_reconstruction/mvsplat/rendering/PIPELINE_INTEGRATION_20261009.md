# 预热链统一渲染主线 — 2026-10-09

## 已修复的边界

用户将前端优化交给队友，本次只修复已有视频入口与渲染主线的配置断点，
并评估后端 NPU 候选。不修改前端算法、冻结发布包或 FPGA 位流。

原 `initialize/session.py → PipelineRenderer` 调用通用构造函数，未启用
`direct_collect/neon_pack`，因此不能继承独立主线的实板成绩。修复后：

```text
initialize/session.py
  → PipelineRenderer(profile="mainline")
  → LiveRenderer.mainline(...)
  → rendering/mainline.json
  → 常驻原生程序：投影 → Tile 分组排序 → FPGA → RGB
```

- `mainline.json` 是完整配置的唯一来源；CLI 未指定的渲染参数不再用一套重复默认值覆盖它。
- 主线同时启用 direct/parallel/fused collect 和 NEON pack，保留完整高斯与 Dense mode 2。
- `--render-profile custom` 才能显式改线程、批次或使用抽点预览；主线下不相同的覆盖值会在设备访问前报错。
- `WarmSession.record.renderer_configuration` 和每帧 `configuration` 记录实际参数、profile、预期 BOOT 哈希。
- 首帧仍返回 raw/RGB 并保留原归档终点；交互 `LiveRenderer` 的完整 RGB 收到时间与归档耗时保持分开。
- `initialize/board_sync.py` 补齐 `run.py`、`session.py`、`rendering/runtime.py`、适配器和 `mainline.json` 的传输，创建缺失目录。
- `rendering/verify.py` 是参数化实验回归，显式选择 custom，避免旧实验开关被误认为主线。
- 不传 `--live-renderer` 仍保留旧 RendererRuntime；本修复不自动寻找原生程序、不更新 BOOT。

前端交接仍使用已有高斯行 `float32[N,62]` 或 PLY，以及 136 字节 `FLCAM001` 相机；
换场景装入一次，换视角重复传相机并接收完整 RGB。队友无需跟踪渲染实验开关。

## 本轮核验与计时边界

原生子进程、模型加载和 SSH 在集成测试中使用替身；测试覆盖真实 Python 路由、
原生启动参数、场景/帧协议、归档内容、配置记录、资源关闭及同步文件列表。
这些测试不代表板端 NPU/FPGA 已执行。本轮 192.168.126.49:22 的 3 秒 TCP 探测超时，
未上传、烧录、重启或重跑视频；新整链验收仍待板卡可达。
随后用户明确板卡离线并要求停止测试；此后仅完成代码/文档收口，不继续板卡连接或测试。

复现本地检查：

```powershell
& examples/3dgs_reconstruction/.venv_mvsplat/Scripts/python.exe tools/check_core.py --mvsplat
& examples/3dgs_reconstruction/.venv_mvsplat/Scripts/python.exe tools/check_render_packages.py
```

本轮结果和日志保存在 [results/20261009/mainline_integration](../results/20261009/mainline_integration/)。
停止测试前，343 个 Python 文件语法检查、86 项本地测试通过（含新增 11 项整合回归）。
冻结主线 151 个文件和独立候选 197 个文件的哈希校验通过；默认路径隔离检查通过。

| 指标 | 保存的主线基准 | 本轮状态 |
|---|---|---|
| 输入 | 32768 高斯、128×128、视角 7/15/22 | 未重做真实视频 |
| 后续相机到完整 RGB | 10 月 8 日 60 帧：均值 43.575 ms，中位数 42.183 ms，P95 52.893 ms，P99 62.080 ms | 未新增板端性能数据 |
| 场景准备时间 | VIDEO_COMPLETE → 首个 FRAME_COMPLETE，包含首帧归档合同 | 本次未测，不能拿 43.575 ms 代替 |
| 图像质量 | 历史留出实拍三视角 PSNR 19.4519/20.2223/21.3486 dB，SSIM 0.804464/0.785135/0.784232 | 模拟帧 RGB/raw 归档保持，真实图像未重测 |
| 全板资源 | LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63.00% | 位流未改，引用既有布线报告，未新增综合 |
| 时序 | setup/hold +0.030/+0.029 ns；渲染/DMA +0.316/+0.054 ns | 原厂 AI 脉宽及 GT/CDC 限制仍保留，不称全板签核 |
| 功耗、DDR 实际流量 | 无本轮实测 | 未测 |

时间来源：[10 月 8 日配对原始汇总](package/validation/board_20261008/paired_summary.json)；
质量、资源和时序来源：[主线合同](MAINLINE.md)。本轮优点是避免入口配置漂移；
局限是代码合同通过不等于整个视频链已按新参数上板验收，不宣称新缩时或画质改善。

## NPU 能否用于后端

可以作为候选，但当前正式渲染仍为 CPU＋FPGA。最直接的候选是把大批高斯的
视角相关预处理融合为少量固定形状的 ICraft 子图：相机变换、投影、二维协方差
及 SH 颜色求值。不随相机改变的世界协方差和 opacity 已在场景装载时缓存，
不应把再次消除这些计算计作新收益。

建议先按同一批 32768 点评估以下分工，而不是每点发起一次 NPU 调用：

| 单元 | 候选职责 | 必须核对 |
|---|---|---|
| NPU | 批量视角相关几何/颜色子图 | 真实设备执行、数值、padding/布局、动态相机输入及含交接的总时间 |
| CPU | 可见性/边界收尾、Tile 列表、稳定深度排序、提交 | 不规则索引与排序耗时不能算作已被 NPU 加速 |
| FPGA | 现有筛选、高斯求值、有序透明度及颜色累积 | 同输入图像质量、与 NPU 共存时的服务耗时 |

固定场景数据尽量设备侧复用，帧间仅更新相机，输出紧凑的高斯属性；是否能实现
设备驻留/零拷贝须以当前 SDK 的真实地址、布局与同步能力验收。不能根据芯片
共享内存的介绍就声称本工程已有零拷贝。

10 月 8 日各阶段均值：投影相关 8.610 ms，分组排序 8.844 ms（其中 radix
仅 2.449 ms），FPGA 服务 24.417 ms，RGB 打包 0.649 ms。投影计时还包含
部分收集开销。因此即使整个投影阶段零成本，43.575 ms 也只能理想降到
34.965 ms，降幅约 19.8%；这只是固定其余阶段的乐观上限，不是 NPU 实测。

已有本地 `npu_3dgs/README.md` 的 2026-09-26 NPU 几何记录中，512 点
compact19 为 1.085201 ms，而优化 CPU 为 0.093002 ms；说明旧的小批拆法无优势。
那次仅含位置/深度等有限几何，协方差与颜色沿用冻结输入，不能外推到本次完整预处理。
它与 MVSplat 七个子图的数值未过问题属于不同实验；不能混成“NPU 一直完全不可用”。
该历史 NPU 工作区未纳入本次 GitHub 发布，上述数值是历史摘录，并非此次测试。

更深入的方案可尝试 NPU 批量 Tile 求值、FPGA 保留有序合成，但会改变现在最重的
服务阶段，并可能重新展开大量 Gaussian–pixel 中间量、增加 DDR 搬运和浪费
提前终止后不需要的计算。应作为独立架构实验，不直接替换当前四路主线。

设计依据：官方 [3DGS forward 实现](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)
明确区分投影/二维协方差/SH 颜色预处理与有序像素合成；厂商
[FPAI 说明](https://www.fmsh.com/fpga/fpai/index.html) 描述 NPU 的矩阵乘加职责。
以上仅支持候选方向，不证明当前 ICraft 3.36.1 能完整编译这些算子或保证性能。

## 是否有必要让 NPU 进入渲染主线

当前判断：没有必要把 NPU 参与后端当作交付前提。现有 CPU＋FPGA 已能完成后端，
而 NPU 新候选尚未证明包含传输、布局转换、同步和 CPU 收尾后的整帧收益。
建议保持 CPU＋FPGA 渲染主线，把 NPU 批量预处理作为可选独立实验；只有同场景
整帧确实缩时、画质满足约定且共存稳定，才值得接入。即使最终后端不使用 NPU，
也不影响渲染链路完整性。

前端交给队友后，NPU 可优先由前端评估用于 MVSplat 等网络推理；这只是职责建议，
不代表此前真实 NPU 数值、布局和会话问题已经修复，也不承诺前端加速结果。

## 板卡恢复后的验收顺序

1. 用 `initialize.board_sync` 同步隔离目录，核对匹配的原生二进制及已验收 BOOT；不因本修复自动重烧。
2. `initialize.board_accept --live-renderer <匹配程序> --render-profile mainline ...` 跑同一视频。
3. 核对预热和首帧配置身份、完整高斯数、相机、raw/RGB 及画质，并重新测 VIDEO_COMPLETE → FRAME_COMPLETE。
4. 同场景测连续换视角并与独立主线对照；首帧归档和交互计时分列。
5. NPU 后端先做完整大批子图的独立正确性与总成本试验，通过后才接入主线配对。
