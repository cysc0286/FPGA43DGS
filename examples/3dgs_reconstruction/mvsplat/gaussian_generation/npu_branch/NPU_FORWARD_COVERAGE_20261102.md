# 11.532 秒网络前向：剩余 NPU 覆盖与迁移顺序

> 本文是六 FFN 迁移前的剖析快照，不是当前十子图的重新分项计时。后续六 FFN 已在真实 NPU 完成测试并获用户采纳；两个完整 U-Net 已完成导出/编译推进，但真实 SDK 输入切换失败；融合双头完整前向更慢。当前采用状态和正负结果以 [扩展记录](NPU_EXPANSION_20261102.md) 为准，下文旧问题和计划保留为当时证据。

日期：2026-11-02。父版本：`9efdd52`。

## 1. 本轮做了什么，哪些计时不能混用

本轮在实际开发板上，对现有四子图 NPU 路径增加外部计时 hooks；没有修改生产模型、权重、子图、SDK 桥接、FPGA、BOOT 或渲染场景。

- 上轮完整视频链路的网络前向：**11.532041 秒**；`VIDEO_COMPLETE → FRAME_COMPLETE`：**20.075221 秒**。
- 本轮重放同一份准备好的上下文，不执行视频解码／位姿准备：无 hooks **10.281976 秒**，有 hooks 两次 **10.283162／10.253939 秒**。
- 以下模块数据是两次带 hooks 重放的均值，不能冒充原来 11.532 秒那次缺失的细分记录。
- 原完整链包含准备阶段重叠，本轮是独立前向重放。调度、并发及运行背景不同；**不能把两者差值报告为优化加速，也不能据此更新视频首帧指标**。
- 父模块时间包含其子模块，不能把 Transformer 总时间与其 FFN 时间再次相加。

三次输出的 means、covariances、harmonics、opacities 均有限，并与当前保留的合格 NPU 场景逐字段精确一致。只保存计时、形状、哈希、资格日志，没有新增完整高斯结果副本。

本轮 PSS 峰值 **522.33 MiB**，系统最低可用内存 **239.29 MiB**；700 MiB PSS、96 MiB 系统余量及 64 MiB 设备地址隔离保持。没有实测 NPU 硬件利用率、功耗或 DDR 总线流量。

配套机器可读记录：`NPU_FORWARD_COVERAGE_20261102.json`。完整诊断在工作区外 `outputs/npu_forward_audit_20261102/`，不是新增业务模块或正式入口。

## 2. 真正仍留在 CPU 的热点

| 模块 | 本轮板端均值 | 当前部署／判断 |
|---|---:|---|
| 多视图 Transformer | 3.077 s | CPU；其中 FFN 是优先迁移候选 |
| 代价体细化 corr_refine_net | 1.794 s | CPU；主要是卷积，整块导出尚未完成 |
| 深度细化 refine_unet | 2.002 s | CPU；主要是卷积，整块导出尚未完成 |

| 模块 | 本轮板端均值 | 当前部署／判断 |
|---|---:|---|
| upsampler | 0.316 s | CPU；优先作为后深度连续图的一部分 |
| proj_feature | 0.402 s | CPU；单独迁移有过负收益 |
| regressor_residual | 0.032 s | CPU；小，不值得单独增加交接 |

细化网络内部的 Conv2d 独占时间分别约 **1.648／1.706 秒**，约占各自网络 **91.8%／85.2%**。因此是有实际计算依据的 NPU 候选，不能只因网络中含归一化和少量注意力就全部留在 CPU。

Transformer 的六个 FFN 合计 **1.822 秒**，约占 Transformer 的 **59.2%**。Q/K/V 与 merge 的 Linear 合计约 **0.453 秒**，所有 Linear 合计约 **2.169 秒**。窗口注意力及其整理、归一化等是剩余部分；不能把整个 3.077 秒都当作 MatMul。

DepthPredictor 的独占时间约 **1.015 秒**，合并包含代价体几何重采样与相关性、softmax、插值、拼接等函数式工作。外部函数 hooks 的命名筛选未匹配运行时隔离包名，本轮没有把它们进一步分开，**不能将这 1.015 秒全部写成 grid_sample 时间**。

编码器外层独占时间约 **0.251 秒**，还包括世界坐标高斯适配与整理等，非独立测得的 GaussianAdapter 完整时间。

## 3. 首选迁移：六个 FFN 连续块

现有每层 cross_attn_ffn.mlp 是：`Linear(256,1024) → GELU → Linear(1024,128)`。六层共约 **4.832G MAC**，是静态算术量，不是性能测量。

建议保留权重和 GELU，将逐 token 的两层 Linear 等价封装为两个 1×1 Conv：

1. 输入 `[2,1024,256]` token 改排为 `[2,256,32,32]`。
2. 原权重重排为 1×1 卷积权重，执行完整 FFN。
3. 输出 `[2,128,32,32]` 还原 token，交回现有 CPU LayerNorm／残差／窗口注意力。

这样适配已有单输入、单输出、4D 合同，不必先修改整个注意力协议。全前向六次边界逻辑 I/O 约 **18 MiB**；若设备可靠性要求单视图执行，则为十二次，逻辑字节总量不变。模型权重不同，六层不能共用同一组权重冒充六层计算。

这一步的当前时间预算是 1.822 秒；它不是可保证节省的秒数。真实收益必须扣除重排、IPC、SDK、同步和输出转换，并继续计入 fresh-input／重复执行规避的全部成本。新增图及多会话组合须重新检查有限值、输入切换、逐层误差和三留出视角；不能继承四子图的旧 receipt。

不建议先独立迁 Q、K、V、merge：会制造更多往返和大中间结果。后续目标是一个 attention block，再到完整六层 Transformer。整 Transformer 边界可以只有约 **2 MiB** 逻辑输入输出，但当前尚未证明完整图可编译或上板。

## 4. 大收益方向：两级细化网络，接成连续深度图

两个细化网络合计约 **3.796 秒**；再计 upsampler、proj_feature、residual，约 **4.547 秒**。这才是深度阶段的主要迁移预算。

### 已有证据纠正

历史 corr_refine_net／refine_unet 整块报 `RuntimeError: _Map_base::at`，发生在 **PyTorch→ONNX 导出**，尚未到 ICraft；不能写成 NPU 硬件不支持。

它们的 `Conv + GroupNorm + GELU` 前缀已经通过 ICraft parse／optimize／quantize／adapt／generate，非 I/O 顶层节点落到 ZG330 HardOp。但产物输出为未审定的 `@layout(***C)`，被本项目 ABI 审核拒绝；这不是完整 UNet 部署通过，也不允许把布局名称直接改成 NCHW 放行。

源码中的 AttentionBlock.forward 强制调用自定义 `checkpoint(..., True)`。该机制服务于反向重算，eval 并不自动关闭；是当前导出失败的强嫌疑。可以在导出专用实例中直接调用同一 `_forward`，保留 QKV、softmax、残差和权重，然后先与原模块 A/B 输出核对。**此根因和修复尚未实验证实**。

### 具体迁移边界

1. 先解决导出和输出布局，优先整块低分辨率 corr_refine_net；可把 residual 合入末端，避免为 32 ms 小层单独交接。
2. 再处理 refine_unet，尽量连上 upsampler、proj_feature 与末端预测头，减少高分辨率特征返回 CPU。
3. 支持的最大连续图要以实际编译 placement／布局审核和内存峰值为准；不能为凑整图跳过 LayerNorm、Attention 或更改模型。

单独加入 upsampler／proj_feature 不是稳妥加速方案：已有板测中，二者独立 NPU 边界约 575／711 ms，CPU 对照约 210／275 ms。它们属于历史不同轮次，不能拿来与本轮 316／402 ms 拼成严格加速比。其价值在于**和相邻深度计算连图，消除中间回传**。

特别注意，upsampler 的 16 MiB 输出同时供 projection 和高斯头使用；只迁移到 projection 时仍要处理这个分支，不能假设这 16 MiB 已消失。

## 5. 已迁移部分也能改：融合两个预测头

原来 11.532 秒那次四模块日志汇总：

| NPU 边界 | 完整调用耗时 | 调用数 |
|---|---:|---:|
| backbone_cnn + depth_head | 0.082 s | 2 |
| to_gaussians | 0.997 s | 32 |
| to_disparity | 0.195 s | 32 |

合计 **1.274648 秒**，包含布局、IPC、SDK、等待、输出转换；它不是 NPU 硬件忙碌时间。四个边界累计逻辑 I/O **47.83 MiB**，不是实测 DDR 流量。

可以把高斯头与 disparity 头合成一个分块图：高斯头输入的前 32 通道就是 refine_out，图内 Slice 后给 disparity；输出拼成 84+2=86 通道，再回到原字段。

- 两头 64 次 tile 提交可变成 32 次。
- 根据现有日志，可减少 **5.0625 MiB** 重复逻辑输入。
- 输出总数据不变；86 通道对齐、图内 Slice 和权重／会话稳定性仍须审核。
- 这只是优化目前约 1.192 秒两头调用的一部分，不能替代对剩余大网络的迁移。

## 6. 暂留 CPU 与实施顺序

代价体相机几何包含动态采样坐标、矩阵逆与 grid_sample；当前未找到已审定的 ICraft GridSample/ZG330 实现证据，不等于已经试编译证实不支持。保持 128 深度候选、chunk16 及现有临时缓冲复用，先不强行拆成许多 NPU 小算子。

世界坐标高斯适配和输入整理也先保留 CPU。它们还不是本轮确认的最大热点，贸然加边界未必划算。

**下一步建议：FFN 连续块 → 两级 UNet 的导出／布局及连续深度图 → 双预测头融合 → attention block／整 Transformer。** FFN 是较清楚的第一增量；UNet 是并行准备的更大收益方向。每一步都须实际编译、板端数值／输入切换、全前向含交接计时、原生图像及内存验收。

当前严格 TF32 逐元素门槛仍失败，SDK 权重预检／陈旧输出根因未关闭。当前路径只获得有限固定资产的原生图像资格；迁移分析没有扩大该资格范围。不得使用错误输出耗时宣称 NPU 加速。

网页没有更换场景：剖析前后及后续三次原生返回帧哈希一致。刚结束 NPU 后的第一请求为 112.46 ms，随后三次约 50.84–50.90 ms；这是共存后的功能检查，不替换历史 40–46 ms 稳态统计，也不是本轮新的渲染优化。

## 证据索引

- 当前生产模块：`catalog.py`、`runtime.py`、`export_graphs.py`、`compile_graphs.py`。
- 已有负对照：`NPU_DEPTH_VALIDATION_20261102.md`，独立 residual／upsampler／projection。
- 历史整块导出失败：`D:/ADProjects/FPGA43DGS/archive/local_workspace_20261009/examples/3dgs_reconstruction/mvsplat/results/20260930/depth_coverage_prefix_v1/result.json`。
- 历史前缀编译与 ABI 拒绝：同级 `depth_prefix_icraft_v1/manifest.json`。
- 官方本地 SDK 的 ZG330 Matmul／Softmax／LayerNorm 适配限制：`D:/ADProjects/FPGA43DGS/npu_3dgs/.vendor/icraft-3.36.1/docs/adapt/index_zhuge.html`。SDK 有算子文档不代表完整 MVSplat Transformer 已编译／数值通过。
- 原始模块／调用计时、保护测量与字段哈希：工作区外 `outputs/npu_forward_audit_20261102/profile.json`、`measurement.json`、`summary.json`。
