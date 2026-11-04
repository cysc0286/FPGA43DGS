# NPU 覆盖扩展：六个 FFN、完整细化网络与双预测头融合

日期：2026-11-02。状态：**用户已接受本轮六 FFN 的画质/精度，采纳十子图方案作为后续优化基线；原四子图保留回退，CLI 仍显式选择分区及匹配资产。**

用户采纳的是已展示结果的精度取舍；第 7 节原始严格逐元素检查与新输入一致性门槛失败仍按原值记录，不倒改历史。有限值、输入切换、重复一致性、资产身份和内存保护继续要求。完整 U-Net 的陈旧/无响应输出属于尚未解决的功能问题，不因接受 FFN 精度而自动放行。

本文记录本轮的实现边界、真实板端数据、已发现的失败与资格要求。第 7 节区分旧准备输入的正式图像资格、新输入的诊断筛选和完整视频首帧，不能将 ONNX 验证、编译或错误输出的短耗时写成已实现加速。

## 1. 本轮解决什么问题

此前板端剖析显示，六个 Transformer FFN 合计约 1.822 秒，两个完整细化网络合计约 3.796 秒。这些是已有同轮剖析的 CPU 时间预算，不是本轮候选已经节省的时间。剖析口径与 11.532 秒完整前向的区别，见 [剩余覆盖与迁移顺序](NPU_FORWARD_COVERAGE_20261102.md)。

本轮按三条路线扩展已有四子图部署：

1. 将六个逐 token FFN 各自作为完整计算块送入 NPU，保留原权重与非 FFN 路径。
2. 修复两个完整 U-Net 的导出障碍，验证整块部署，而非只迁移第一层卷积。
3. 将 `to_gaussians` 与 `to_disparity` 合入一次边界调用，减少两头重复交接。

这些变化只影响“上下文图像与相机参数 → Gaussian”的场景准备阶段。场景已生成后的换视角渲染仍由已有 CPU＋FPGA 后端完成，不能把本轮 NPU 扩展写成换视角渲染器加速。

## 2. 代码结构与四模块边界

四个业务模块保持：

```text
video_input
    ↓
pose_estimation
    ↓
gaussian_generation
    └─ npu_branch：本轮可选子图、导出、运行与验证
    ↓
rendering：已有 CPU＋FPGA 渲染后端
```

`initialize` 仍负责视频输入前的预热与常驻准备，不把视频后的计算搬到计时区间之外。CPU 回退入口、冻结渲染包及 BOOT 保留。本轮没有更改 FPGA RTL、位流或启动镜像。

本轮相关代码集中在 `gaussian_generation/npu_branch/`：

```text
ffn.py                  原 MLP 的 1×1 Conv 包装、token 边界代理、显式 attach
ffn_contract.py         不导入 Torch 的静态形状、层身份与 NumPy 分块合同
ffn_export.py           六个真实 FFN 的导出及输入/输出 oracle
unet_export.py          两个完整 U-Net 的临时 checkpoint 绕过与导出验证
onnx_output_control.py  真实输出尾算子候选，不重标 ICraft 张量布局
fused_heads.py          双头融合的原前向接口适配与帧内结果生命周期
fused_heads_export.py   原两头 ONNX 分支组合、完整图像 oracle 核对
assemble_bundle.py      已审计子图组合、来源与资格上下文绑定
catalog.py              分区名与原模型路径的明确对应
artifacts.py            图、权重、oracle、布局及清单的验证
runtime.py / worker.py  子图调用、输入生命周期、执行计费与运行身份
```

对应小型回归包括 `test_npu_ffn.py`、`test_npu_candidate_contracts.py`、`test_npu_assemble_bundle.py`。这些测试验证接口、等价变换和拒绝行为；不替代实板 SDK、整网图像或端到端性能验收。

## 3. 六个 FFN：保留权重的 Linear → 1×1 Conv

### 3.1 精确迁移范围

六个候选为 `transformer_ffn_0` 至 `transformer_ffn_5`，分别对应：

```text
backbone.transformer.layers.0.cross_attn_ffn.mlp
...
backbone.transformer.layers.5.cross_attn_ffn.mlp
```

每层仍执行原来的 `Linear(256,1024) → GELU → Linear(1024,128)`。窗口注意力、LayerNorm、残差连接及层间更新继续由原 CPU 前向执行。六层各自绑定自己的训练权重，不共用第一层权重冒充六层。

### 3.2 等价成立的条件

Linear 对每个 token 的通道做同一组线性变换；1×1 Conv 对每个空间位置的通道做同一组线性变换。因此可把 `[2,1024,256]` 的 token 输入整理成 `[2,256,32,32]`，将原二维权重增加两个长度为 1 的维度，执行完整两层卷积后还原 `[2,1024,128]`。

这一变换必须满足：

- 原 MLP 结构恰为两层 Linear 与中间 GELU；不漏 bias，不替换激活函数，保留 GELU 原有的 `approximate` 设置。
- 不跨 token 混合，不改变 token 排列，不插入归一化或残差，不删除任何 token。
- 不训练、压缩或量化原权重来伪造等价；导出使用原 CPU FP32 权重。SDK 内部精度转换仍须独立验收。
- 输出逻辑仍为 NCHW；若 ONNX 使用显式 NHWC 尾端，清单必须记录实际轴顺序，由 wire 转换恢复，不能根据字节数猜布局。

数学操作等价不意味着所有后端逐比特一致。Conv 与 Linear 的累加顺序、ICraft 内部精度可能带来误差，须分别检查小型 Torch 等价、真实输入 ONNX、SDK 输入切换与最终图像。

默认候选是完整双视图、32×32 空间边界：六层对应六次逻辑边界调用，约 18 MiB 逻辑输入输出。MLP 逐 token 独立，因此支持必要时的 batch-one 或无 halo 分块；分块不得删 token、缩图或改变原算子。SDK 若每请求执行两次，实际 SDK forward 次数与全部代价还要另外计入，不能把“六次边界”误写成只执行六次硬件 forward。

### 3.3 当前状态

六个 FFN 的真实输入 ONNX 检查及 ICraft 编译均已通过。原四子图加六 FFN 的十子图组合已通过旧 `e81a9a…` 准备输入的真实 NPU＋原生 FPGA 三视角图像资格、输入切换及重复检查，并完成新视频输入的首帧执行。**严格逐元素门槛仍未通过；新 `8f0146…` 输入也没有通过相同的绝对图像门槛，不能泛化为任意场景已合格。** 分级结果见第 7 节。

## 4. 两个完整 U-Net：先修导出，再查真实 SDK 行为

### 4.1 checkpoint 绕过的实际含义

原 `AttentionBlock.forward` 中的自定义 checkpoint 服务于反向传播时的重算；它不会因模型调用 `eval()` 就必然退出导出路径。原完整网络的失败发生于 Torch → ONNX，不应据此认定 NPU 不支持整个网络。

`unet_export.py` 仅在导出专用的 eval 实例上，临时令已识别的 `AttentionBlock.forward` 直接调用该实例原有 `_forward`。QKV、注意力、softmax、归一化、残差和参数均保留；不是删除注意力或用卷积替代注意力。作用域退出后恢复实例的原 `forward`，不修改类方法或生产模型的默认前向。

导出前先比较捕获 oracle 与原模块，再比较绕过前后真实输入 A/B 输出，之后才进行 ONNX 检查。这里去掉的是 autograd checkpoint 包装，**完整推理计算不变**；本候选也不用于板端训练。

### 4.2 `corr_refine_net`

完整网络已完成 ONNX 验证并通过 ICraft 编译，说明“只能部署前缀”的旧边界已经推进。但真实 SDK 输入切换未过：单次执行失败，启用 fresh-input 与每请求两次 SDK forward 后仍失败。

因此该候选不能接入正式链路，也不能以某次固定输入输出或编译成功宣称迁移完成。下一步需定位输入切换后的首次错误、输出陈旧及 SDK 缓冲/会话行为，而非继续增加重复执行次数掩盖错误。

### 4.3 `refine_unet` 与输出尾控制

原完整编译产物的输出布局为未审定的 `***C`，当前 ABI 审核拒绝。项目没有将该标签改成 NCHW/NHWC 强行放行。

为检验编译器边界，另构建了带真实恒等 1×1 Conv 尾端的候选：它执行对角单位权重与零 bias，并在 ARM ONNX Runtime 对照未变的真实 oracle，再重新编译和审计。此尾端按图中密集卷积算术量增加 **33,554,432 次乘法**，不是零代价重命名；实际设备是否优化掉其中零乘法不能仅凭图推断。

该候选通过编译，但板端 SDK 对真实 A/B 输入变化的输出没有相应变化，输入切换验收失败。它只是失败定位用的独立候选，未接入正式前向。布局审核通过与数据正确返回，是两道不同的门槛。

## 5. 两个预测头融合：减少交接，不更换模型

### 5.1 图内计算与调用数

融合图接收原 Gaussian head 的 163 通道输入；其前 32 通道与本次 `refine_unet` 输出一致，可同时供 disparity head 使用。图内保留两套原始节点与权重：一条执行原 `to_gaussians`，另一条从同一输入切出前 32 通道执行原 `to_disparity`，最后沿通道拼接为 84＋2 通道。

这里的“拼接原权重”应理解为**组合两条原权重计算分支**，不是相加、平均、重新训练或把两头数学上改成一个卷积。原卷积的 padding、激活、输出顺序均保留。

两个头仍采用单视图 36×36 执行块、32×32 有效区、两像素 halo，覆盖完整双视图。原两头合计 64 次分块边界调用，融合后为 32 次；约减少 5.06 MiB 重复逻辑输入。这是由当前形状推导的调用与边界字节，**不是实测 DDR 流量或保证的节省时间**。每块两次 SDK forward 的成本仍全部计入。

### 5.2 不改变原前向调用顺序

`FusedHeadAdapter` 保留原模型先调用 Gaussian head、后调用 disparity head 的结构。第一次调用执行融合图，只缓存本次的两个 disparity 输出通道；第二次调用取回该结果。

缓存只允许被同一帧、同一个细化 Tensor 消费一次。重入、未消费完成、换帧、Tensor 被原地修改或通道前缀不一致均拒绝。融合图与原两个独立 NPU 头互斥，不能同时 attach 后重复或跳过计算。

### 5.3 当前状态

完整双视图 host ONNX、ICraft 编译和 SDK 输入切换已通过。但新 `8f014…` 上下文下整网图像门槛未过，因此**尚不作为新的正式配置**。即使调用数减少，也必须以合格图像下的完整前向计时判断是否采用。

## 6. 运行可靠性、资产来源与新资格

### 6.1 规避策略也属于被测实现

FFN、两个完整细化网络和分块头要求 fresh SDK input Tensor。上述候选相应的两次 SDK forward 必须按真实次数计入 `sdk_passes`；`depth_head` 保持一次，不能因为扩展了候选就给所有子图一律增加或遗漏计费。

运行身份记录精确的 `ffn_partitions`、`refinement_partitions` 与 `spatial_heads` 选择，以及相应 settle 次数、fresh-input 和已有安全开关。双执行只是受限规避，不能写成 SDK 陈旧输出、权重预检或数值根因已修复。若输入切换仍失败，继续拒绝 READY。

### 6.2 源码与资产必须一同绑定

资格记录不仅绑定 checkpoint、ONNX、编译图与参数，还绑定实际 generated `src` 及运行代码。新增 `ffn.py`、`ffn_contract.py`、`fused_heads.py`、catalog、布局转换、分块和 worker/runtime 变化均参与执行源码身份；改变分区或执行开关后，不能沿用旧 receipt。

`ffn_contract.py` 刻意保持不导入 Torch：SDK worker 的轻量验证只读取 NumPy 与静态合同，不因验证 FFN 元数据再加载一套模型。

### 6.3 不重标旧 oracle 的来源

已有生产子图 oracle 与本轮新子图 fixtures 可来自不同真实上下文。`assemble_bundle.py` 默认拒绝混合；只有显式 `--allow-mixed-oracle-contexts` 并同时给出目标 `--context-sha256` 时，才允许组成待重新资格的组合。

组合清单分别记录：

- 各分区 oracle 原上下文、原 graph/compiled manifest 哈希和 source provenance；
- 新的全网资格目标 `qualification_context_sha256`；
- `oracle_contexts_mixed` 及其绑定范围。

目标上下文不表示所有 oracle 原本来自该输入。原分区 metadata、oracle、源清单不能被覆盖或重新贴标签；所有源必须仍绑定相同 checkpoint 与根输入形状，编译清单还须绑定准确导出清单。显式允许混合来源并不豁免全网重新资格。

## 7. 验证结果、时间与采用边界

### 7.1 三种实验口径必须分开

共同配置为固定官方权重、两张 128×128 上下文图、32768 个 Gaussian、`depth_chunk=16`，无逐场景训练。真实板端使用 NPU 与原生 FPGA 渲染，没有以本机 CPU 或 NumPy 渲染代替。

- **同输入配对诊断**：新 `8f01469e…` 上下文、CPU 三线程，各候选重复两次；只重放准备好的模型输入，不含视频解码和位姿阶段。`diagnostic_only=true`，不是新的资格凭证。
- **正式固定输入图像资格**：旧 `e81a9a7b…` 上下文、CPU 四线程，对十子图组合执行真实 NPU、原生 FPGA 三视角与重复检查；产出完整 receipt，但适用范围是该固定输入和绑定资产。
- **完整视频首帧**：同一公开视频生成新 `8f01469e…` 上下文，CPU 三线程、准备线程一条、启用准备重叠和 kernel warmup。每个配置各一次，以 `VIDEO_COMPLETE → FRAME_COMPLETE` 计时；只验收首个目标视角，不冒充全视频覆盖完成。

预热在输入视频前完成，其时间只记录。下面的网络前向计时不含视频解码/位姿；视频首帧时间包含视频结束后的必要准备及首帧渲染。独立诊断前向与存在准备重叠的完整视频前向有不同运行背景，不能相减拼接成额外收益。

### 7.2 同输入诊断：六 FFN 是本轮有效的缩时机制

| 配置 | 前向均值，n=2 | 逻辑请求 / SDK forward | 重复结果精确一致 |
|---|---:|---:|---|
| 原四子图 | 10.479287 s | 66 / 131 | 是 |
| 四子图＋全部六 FFN | 9.046889 s | 72 / 143 | 是 |
| 四子图＋前三 FFN | 9.973101 s | 69 / 137 | 是 |
| backbone＋depth＋融合双头 | 10.724823 s | 34 / 67 | 是 |

原四子图两次为 10.569590、10.388983 秒，六 FFN 两次为 9.038344、9.055434 秒。全六层在这组配对诊断中平均少 **1.432398 秒，13.669%**。这是包含接口代价的实际完整前向下降，不是单个 Conv 的设备周期；但 n=2 不足以声称稳定跨场景加速。

仅迁后三个 FFN 的均值为 9.837757 秒，但 `repeat_exact=false`，不采用。仅迁前三层虽然重复一致，节省小于全部六层，也不作为本轮首选。

融合双头确实把完整前向逻辑请求从 66 减至 34，SDK forward 从 131 减至 67，逻辑 I/O 从 47.830078 减至 42.767578 MiB；但均值反而增加 0.245537 秒，约 2.34%，因此**不因调用数更少就采用**。该组融合输出与四子图基线的图像指标一致，说明此次拒绝主要依据整体时间，而非虚构画质损失。

证据：[同输入筛选 JSON](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/paired_screen_v2/screen.json>)；各候选的 `screen.json` 与两次 `infer_*/inference.json` 位于同目录下。

### 7.3 旧 e81 输入：十子图正式图像资格通过，但严格张量仍失败

`qualify_ffn_reference/acceptance.json` 的 `complete=true`、`diagnostic_only=false`、`npu_executed=true`、`fpga_executed=true`、`gaussian_repeat_exact=true`。三视角 NPU 与同模型 CPU 的图像一致性为：

| 留出视角 | NPU 对 CPU PSNR | NPU 对 CPU SSIM | FPGA 实时帧计算 |
|---|---:|---:|---:|
| 第 7 帧 | 44.041821 dB | 0.995504 | 50.090 ms |
| 第 15 帧 | 43.662588 dB | 0.995357 | 50.010 ms |
| 第 22 帧 | 42.736135 dB | 0.995521 | 51.751 ms |

该输入通过最低 40 dB、SSIM 0.99，以及相对真值 PSNR 最多下降 0.2 dB、SSIM 最多下降 0.005 的原有图像政策。相对真值的最大 PSNR 下降为 0.031044 dB，最大 SSIM 下降为 0.000241。

**42.7–44.0 dB 是 NPU 相对 CPU 的一致性，不是重建图像相对拍摄真值的绝对画质。** 该组 NPU 对真值 PSNR 为 20.072264、20.219746、24.197748 dB。`strict_tensor_passed=false`，旧严格 TF32 逐元素数值问题和 SDK 根因没有关闭；不能将图像资格重新表述成全部张量通过。

表中的约 50 ms 是本轮固定场景三个原生 FPGA 帧的计算结果，未调整 FPGA 内核；它不是新的网页端到端交互延迟统计。

证据：[固定 e81 输入资格 JSON](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/qualify_ffn_reference/acceptance.json>)。

### 7.4 新 8f 输入：基线也未过绝对门槛，六 FFN 略改善但不能宣称新场景合格

同输入筛选中，原四子图与六 FFN 的 `image_gate_passed` 均为 false：

| 留出视角 | 四子图 PSNR / SSIM | 六 FFN PSNR / SSIM |
|---|---|---|
| 第 7 帧 | 37.600657 dB / 0.987611 | 37.715977 dB / 0.987760 |
| 第 15 帧 | 37.463048 dB / 0.988029 | 37.530338 dB / 0.988127 |
| 第 22 帧 | 36.306335 dB / 0.986061 | 36.460194 dB / 0.986268 |

这些也是 NPU 对 CPU 的一致性指标。六 FFN 在三张图上均有小幅改善，但仍低于既定的 40 dB / 0.99 门槛。不能把原四子图自身在该输入上的精度问题归因于新 FFN，也不能因新 FFN 没有加剧差异就宣称该输入已经通过旧门槛。

六 FFN 对真值的最大 PSNR 下降为 0.030855 dB、最大 SSIM 下降为 0.000575，未超过相对真值限制；绝对 NPU/CPU 一致性门槛仍未过，保留失败状态，不放宽阈值补签 receipt。

### 7.5 完整视频：一次执行缩短约 1.02 秒，不能与上述诊断均值混用

视频 SHA256 为 `0a7e3c684e917dd937ae1ec6c64f770c111b939f08f020b66965c1fccf9ae86a`，两次模型输入 SHA256 均为 `8f01469ed479d249550121b6bb12b920b7e1af8063a2eaf25ccf2c721df21d87`。

| 完整视频步骤 | 原四子图，n=1 | 四子图＋六 FFN，n=1 |
|---|---:|---:|
| 网络完整前向 | 11.381824 s | 10.221115 s |
| 视频结束 → Gaussian 场景 | 19.698824 s | 18.685203 s |
| 首次渲染 | 0.141506 s | 0.130596 s |
| 视频结束 → 首帧 | 19.840330 s | 18.815799 s |

本轮首帧单次少 **1.024531 秒，5.164%**；网络前向单次少 1.160709 秒。首帧执行结果表明该候选能够在现有 NPU→FPGA 流程中运行，**不代表新 8f 输入已通过第 7.4 节失败的图像资格，也不代表多轮稳定低于 20 秒**。不要将单次 18.815799 秒作为全部场景通过验收的结论。

实际十子图完整前向记录了 **72 次逻辑请求、143 次 SDK forward、65.830078 MiB 逻辑输入输出**：两头各 32 请求，backbone、depth head 与六个 FFN 各一请求；除 depth head 一次外，其余各请求都计入两次真实 SDK 执行。这些字节不是实测 DDR 流量，更不是 NPU 利用率。

证据：[四子图完整视频结果](<D:/ADProjects/FPGA for 3DGS/outputs/npu_memory_storage_20261102/expansion_baseline_video/warm_result.json>)、[四子图前向 JSON](<D:/ADProjects/FPGA for 3DGS/outputs/npu_memory_storage_20261102/expansion_baseline_video/inference/inference.json>)、[六 FFN 完整视频结果](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/video_ffn_reference/warm_result.json>)、[六 FFN 前向 JSON](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/video_ffn_reference/inference/inference.json>)。

### 7.6 内存、资源与拒绝的候选

完整视频测量覆盖包括预热在内的子进程树：四子图 PSS 峰值 579.118 MiB、最低 MemAvailable 269.371 MiB；六 FFN 为 568.155 MiB、257.410 MiB。两次都低于 700 MiB PSS 上限并保留 96 MiB 系统余量；不能凭 PSS 峰值略降就声称所有系统内存均减少，候选最低可用内存反而更低。

本轮没有改 FPGA RTL、BOOT 或冻结位流，没有重新综合布局布线，因此 **FPGA 资源配置不变**；NPU 硬件利用率、功耗、真实 DDR 流量均未测。SDK 总时间或 HardOp 数不能替代这些指标。

- 完整 `corr_refine_net`：ONNX/编译通过，单次和 fresh-input＋两次执行均报 `NPU stale output after input switch`，不采用。
- 完整 `refine_unet`：原 `***C` 输出被 ABI 拒绝；真实恒等 Conv 尾编译候选仍报 `NPU unchanged output after input switch`，不采用；额外 33.55M 乘法也未带来已验证收益。
- `fused_heads`：host、编译、输入切换通过，但减少请求后完整前向更慢，不采用；未将其与全六 FFN 合成新的正式默认。
- 仅后三 FFN：重复输出不完全相同，不因较短时间而采用。

完整拒绝证据保存在 `outputs/npu_expansion_20261102/` 下对应 `corr_probe*`、`refine_probe`、`paired_screen_v2`，失败输出耗时不进入有效加速结论。

### 7.7 当前配置、回退与回归

用户后续明确接受已展示的六 FFN 精度，因此当前采用原四子图加 `transformer_ffn_0` 至 `transformer_ffn_5` 的十子图方案作为后续工作基线，配套本轮准确 bundle、bridge、oracle 与 e81 固定输入的 qualification receipt。入口通过 `--partitions` 明确选择，不创建第五业务模块，不自动给所有 CPU/旧资产入口填入新分区。原四子图可回退；新上下文的历史门槛失败仍保留，不宣称全网 NPU。

本机最终全量 Python 回归日志记录 **277 项，6 项跳过，整体 OK**，对应 [回归日志](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/local_regression_final_v2.log>)。另外重跑源码身份专项，1/1 通过：本轮 receipt 的全部 10 个 `runtime/` 源文件与当前工作树 SHA256 一致，明确包含 `ffn.py`、`ffn_contract.py`、`fused_heads.py`。这不是重新执行了一轮板测；如果随后改运行代码、图、SDK 或开关，还须重新生成相应资格。

后续优先处理完整细化网络的真实 SDK 输入切换/输出问题，并提高新输入的一致性；只有连续图正确返回后，才有条件继续扩大深度阶段 NPU 覆盖。保留受限通过的六 FFN 候选与原四子图回退，不将减少调用数本身当作优化目标。

### 7.8 网页恢复、画面与额外诊断

已在 18080 恢复本轮六 FFN 生成的场景，网页 `/scene` 确认生成端为真实 NPU、渲染端为 FPGA。依次请求 yaw=0°、−5°、＋5°、0°，三个视角图像不同，返回原视角后 PNG SHA256 完全相同。四次原生渲染为 51.58、40.72、42.49、41.17 ms；本机 HTTP 全往返为 132.56、76.24、54.26、68.15 ms。这是恢复服务的冒烟验证，不是新一轮受控交互延迟分布，也不是视频首帧时间。

网页会话地址保存在本地私有 `outputs/npu_expansion_20261102/web/local_url.txt`，不入库。验证记录：[web/verification.json](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/web/verification.json>)；首帧与原四子图的直观对比：[first_frame_comparison.png](<D:/ADProjects/FPGA for 3DGS/outputs/npu_expansion_20261102/first_frame_comparison.png>)。128×128 图像放大仅用于观察，未增加实际分辨率。

另尝试隔离的 FP32 FFN0 诊断编译图。其输出实际为 Quantized FP32 / NCHWc16，不能当普通 FP32 / NHWC 重标；正式桥接 ABI 未放宽。诊断桥接在板上以 O3、O1 两次编译，均因进程 PSS 无法持续核算而被内存监控终止；采样峰值分别 641.17、588.89 MiB，但这不能追认全程内存验收通过。没有得到可执行的完整板端 FP32 对照，记录为诊断未完成，停止重复编译，不作为有效加速或精度修复。对应 `fp32_build*.measurement.json` 保留原失败状态。

监控仅在进程状态再次确认为 Z/X、PID 消失时跳过；短暂读取失败最多重试三次、间隔 5 ms，仍然存活且不可读就拒绝。未下调 PSS/可用内存阈值，也未用零 PSS 掩盖未知占用。

本轮收尾仅删除自建候选目录中已保存指标/哈希的失败大输出，以及本机存在且逐文件 SHA256 相同的编译/安装包副本，共 28 个文件、391,660,492 字节（373.52 MiB）；先保存 `reclaim_plan.json`，板上再次核对真实绝对路径、尺寸和哈希后才 unlink。当前展示场景、十子图组合及 receipt、原四子图、正式环境和权重均保留。根分区可用空间恢复至 1,116,260 KiB。没有删除未知目录或用户视频。

## 8. 用户采纳六 FFN 后的迁移顺序

后续按剩余工作量与连续计算边界安排，而不是追求分区数：先 `corr_refine_net`，再 `refine_unet`；在正确性/会话问题定位的同时准备剩余注意力投影，随后扩大连续图。下面的 CPU 时间仅取自迁移前剖析，不是当前 10.221 秒前向的重新分项结果，也不是可以直接兑现的节省时间。

- `corr_refine_net`：原约 1.794 秒，其中卷积约 91.8%。完整图可导出/编译，但实际 A/B 切换后输出不可靠。先在图内部定位首次错误；如果错误集中于注意力/归一化或边界布局，则按已验证块重新划分，保留完整模型计算，不能再以两次 forward 一律掩盖。
- `refine_unet`：原约 2.002 秒，卷积约 85.2%。完整图的输出布局、真实输入切换仍是阻碍；其后再联合 upsampler / proj_feature / 两预测头，考虑特征分支及实际 PSS 限制。
- `upsampler` 与 `proj_feature`：原同轮约 0.316 / 0.402 秒，单独 NPU 曾有负收益。只作为相邻大图融合候选，避免大特征往返 CPU；不能假设其共同输入分支已消失。
- 剩余 Transformer 投影：每层 self-attention 与 cross-attention 各有 Q/K/V/merge，六层共 48 个 Linear，原剖析约 0.453 秒。可以继续等价转 1×1 Conv，但不能制造 48 次小调用。Self QKV 同源可拼为 3C 输出；cross 的 Q 与 KV 来自不同 Tensor，宜 Q 单路、KV 同源融合，再评估整体注意力子层。
- 注意力矩阵乘、softmax、LayerNorm：本地 ICraft 有相关适配文档，但整块窗口划分、掩码、形状、轴与布局尚未验证。固定权重 Linear→Conv 的等价性不覆盖输入相关的 QKᵀ / attention×V；这些需要动态 MatMul 路径。归一化依赖当前输入，不能按固定仿射直接折叠进卷积。

代价体几何重采样、深度回归及世界坐标高斯适配继续保留 CPU。原 depth predictor 独占约 1.015 秒并不全是 grid_sample；尚无已验证的设备连续图，需先进一步拆分实测。视频解码、位姿和格式调度暂不因扩 NPU 覆盖而更换算法。这些候选优化的是视频结束后的场景准备，已有场景切换视角仍走 CPU＋FPGA。
