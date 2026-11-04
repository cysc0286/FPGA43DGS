# 2026-11-03 后续：运行时已接入，真实板端诊断失败，不采纳

用户后续明确要求继续上板，每1–2个模块就验证。因此本轮在下文离线准备基础上，已完成 catalog、配置、可逆方法适配、worker、资格工具和整合包的显式候选支持；已采纳十图及四模块默认架构不变。

真实NPU A→B→B→A中，输出有限且B重复精确一致，但返回A不同于首次A；严格参考93.6594%–94.6585%元素越界。四次完整逻辑调用1.317186/1.045598/1.037299/1.032018s，全部8次SDK执行计入，不能作为有效加速。PSS355.593MiB、最低可用529.539MiB，未触发内存/超时门槛。

本机组合链等价并不能解决真实SDK可靠性。候选未进入正式路径，九图整合包亦未获得完整资格；继续使用十图。后续仍应定位第一次输入切换偏差，不能单纯增大图来绕过问题。完整数值、原始证据位置、十图重核与断连待办见 [本轮验收](../../video_input/DUAL_INPUT_VALIDATION_20261103.md)。

以下为此前离线准备阶段的原始边界与结果，不把当时“未接运行时/未板测”作为当前状态。

# 完整细化与双预测头连续图：离线准备（2026-11-03）

本轮继续用户“先进一步尝试迁移 NPU，实验后续再做”的安排，只进行源码、真实源 ONNX、接口合同和 ICraft 编译检查。没有发起 SSH、板端 SDK、摄像头或完整视频实验。下述候选没有替换已采纳的十图正式部署，也没有新增图像资格；上一轮严格 NPU 数值问题仍然开放。

## 这次迁移的边界

准备的新图名是 `refine_heads_pipeline`，连续包含四个原模块：

```text
CPU upsampler ── up128 ──┬─ proj_feature ── proj32 ──┐
                        │                           │
CPU RGB3 / disparity1 / pdf_max1 ────────────────────┴─ refine_unet ── refined32
                        │                                              │
                        └──────── up128 + RGB3 ──────────────────────────┼─ to_gaussians → raw84
                                                                       └─ to_disparity → raw2
```

图输入是两视图 FP32/NCHW `[2,133,128,128]`，通道顺序为 up128、RGB3、粗视差1、置信度1；输出是 `[2,86,128,128]`，先 raw Gaussian84，再 raw disparity/density2。实际 ONNX 和编译器的输出经过真正的 NCHW→NHWC 转置；没有依靠总元素数重解释布局。

`proj_feature` 的卷积、完整 `refine_unet` 的归一化与跨视图注意力、两个头的原权重及运算全部保留。up128 在图内同时供投影与 Gaussian 头使用，不再从两个独立入口重复提交。完整细化网不按小块或单视图切开；不减少像素、高斯或深度候选。

CPU 仍负责 upsampler、先前的粗深度 softmax/期望、最终视差/密度处理、近远平面约束和世界坐标高斯转换。原有的消融 CPU 路径、state_dict 键和后续 reshape/clamp/sigmoid 保留。

没有先把 upsampler 强塞进该图：它使用低分辨率输入，而其结果还供 Gaussian 头复用。只融合 upsampler/proj/refine、随后又完整回传 up128，可能增加边界数据。当前选择通过单个同分辨率打包入口解决跨模块复用，避免新增多输入 ABI；并不表示 NPU 不支持上采样或原生多输入。

## 能减少什么交接

下面以“独立完整 refine NPU＋现有两头分块 NPU”为对照，不是把现有十图中的 CPU refine 改写成已部署 NPU。统计只含静态 FP32 逻辑输入输出，每个逻辑请求计一次，不含 SDK 双执行、布局临时副本，也不是 DDR 实测。

| 这一段的接口 | 分离候选 | 新连续候选 |
|---|---:|---:|
| 逻辑请求数 | 65 | 1 |
| 逻辑 I/O | 53.080078 MiB | 27.375 MiB |
| 输入/输出 | refine 与两头反复交接 | 16.625 / 10.75 MiB |

即减少 25.705078 MiB（约48.43%）逻辑边界，同时把原 CPU 投影卷积纳入。65→1 不是完整网络只剩一次调用，也不是 SDK 实际调用次数；其他模块仍有原接口。后续需要按实测输入切换策略计入全部 SDK forward，不能预先假定本图一次执行就可靠。

融合可能增加设备侧中间张量的存活时间。Gaussian 头隐藏通道数较大，原细化特征、up128 和两个输出也有各自生命周期；这些静态数组不能直接相加冒充活跃内存峰值。减少接口不保证执行时间或 PSS 下降。

## 本机数值与编译结果

原旧七图和新 refine oracle 来自不同上下文，不能直接混拼成“同一真实视频输入”。本轮利用已经保存的真实 ARM A/B 上下文：从同组 refine 输入取 RGB、粗视差与置信度，从同组 corr 输入取 feat01；RGB 按原 ImageNet 常量归一化，经原 backbone ONNX 生成 CNN 特征，与 feat01 拼接后经原 upsampler ONNX 派生 up128。

派生 up128 经原 proj_feature 后，与同组 ARM 捕获的 projected32 比较，A/B 最大误差分别为 **7.033348e-6 / 4.023314e-6**，全部通过 `rtol=0.002, atol=0.0002`。两组原 context 身份和各源哈希均保留；这是**同上下文源 ONNX 派生输入**，不是 ARM 直接 hook 的完整 up128 或独立 ARM 双头参考。

连续图的参考输出由四张原始源 ONNX 独立依次执行产生，不从新组合图自产 oracle。真实完整图的 A→B→B→A 四次对比均逐位一致，最大误差和 RMSE 均为0；A/B 不同，同输入重复精确一致。此结果只证明本机组合、通道、复用与布局没有改变原源图语义，不能关闭 SDK 的实际数值问题。

| 核查 | 本轮结果 | 范围 |
|---|---|---|
| 原源 initializer 元素 | 975,990；分支重命名后内容保留 | 图结构，不等于设备分配 |
| ICraft 3.36.1 / TF32 | parse/optimize/quantize/adapt/generate 全过 | 本机编译 |
| 最终图 | 1,189 个 ZG330 HardOp；无顶层 Host 计算回退 | 编译审查，不是利用率 |
| 主机 ABI / raw | FP32/NHWC；raw 111,805,152字节（约106.63 MiB） | 文件体积，不是PSS/PL-DDR峰值 |

没有测量新增板端耗时、PSNR/SSIM、NPU 利用率、功耗、DDR 流量或资源峰值。FPGA RTL、频率、资源和 BOOT 没有变化；本轮不影响已准备场景之后的换视角渲染路径。

## 代码接口与捕获准备

所有新增代码留在 `gaussian_generation/npu_branch`，四业务模块不变：

- `refine_heads_export.py`：真实源图组合、来源/权重/图哈希检查、原源链派生参考与 A/B 切换核查；生成待验 ONNX 和 manifest。
- `refine_heads.py`：严格匹配原源码的可逆方法 seam，以及 `attach_refine_heads(predictor, execute)`；单次接收133通道、返回两个独立拥有存储的原始头。只覆盖 `_refine_to_heads` 方法，不安装假模块、不依靠 fused-head forward hook。
- `refine_heads_capture.py`：为后续一次参考前向准备四个原 CPU 模块的同调用捕获；核对顺序、两视图、共享特征逐位一致、有限值与生命周期，限制额外 owned 数组预算。

本轮尚未把 `refine_heads_pipeline` 注册到正式 catalog/配置/worker，也没有同步板端。方法 seam 与之前的 corr-depth seam 两种补丁顺序均通过；真实完整源的两种结果具有相同方法 AST。未来接入需要与独立 `proj_feature/refine_unet/to_gaussians/to_disparity` 及 fused-head 互斥，并重新绑定资产、来源和资格，不能复用旧十图 receipt。

捕获器默认额外 owned 数组预算32 MiB，128² 时峰值31.375 MiB，最终保留27.375 MiB；这是捕获工具自己的数组字节，不含 Torch、模型、SDK 或 PSS。每次仅一次 predictor 调用，异常移除 hooks 并释放数组；入口与每阶段拒绝已知 corr/refine/fused adapter、Partition 与代理，并要求 predictor 全部参数为 CPU FP32。身份由调用者提供，工具只认证四个原目标模块的同调用数据，不凭一个 SHA 字符串认证上下文文件或整个上游编码器来源。

捕获器产生的是单上下文 `capture.json + oracle.npz`，原 CPU expected 与 exporter 的源 ONNX 派生参考**不是同一证据来源**。当前 exporter 接收 A/B 聚合的 input-only 包，或明确标记 source-derived 的 expected；原 CPU hook expected 不可为了通过入口而改成 source-derived。后续应另外保留原 CPU expected 做独立比较，再为新图签发板端资格。

本轮新增三组39项本机合同测试；合并前轮相关回归共28组 **311项通过，无跳过**。覆盖真实小 Torch 原/改 CPU 等价、参数不变、消融路径、恢复和失败处理、共享输出隔离、两种 seam 共存、非正方形 ORT 布局、通道分支、来源拒绝以及捕获的同调用顺序。测试不是实板验收。

## 证据与后续顺序

私有证据留在仓库外 `outputs/npu_unet_diagnostics_20261103/`，不提交影片、输入张量、输出或令牌：

- `prepare_refine_heads_context_inputs.py`、`refine_heads_context_inputs_20261103/`：真实上下文派生输入、原来源哈希与 projected32 对照。
- `refine_heads_prepare_20261103/graphs/`、`compiled/`：新图、两个派生 oracle、五阶段日志和编译审查。
- `refine_heads_local_regression_final_20261103.log`：本机完整回归；`refine_heads_prepare_20261103/preparation_receipt.json`：离线证据身份，明确不是验收 receipt。

重建命令：

```powershell
python -m gaussian_generation.npu_branch.refine_heads_export `
  --linear-export <原七图导出目录> --refine-export <完整refine导出目录> `
  --fixtures <明确来源的A_B打包输入目录> --out <新的导出目录>
python -m gaussian_generation.npu_branch.compile_graphs `
  --graphs <新的导出目录> --out <新的编译目录> `
  --compiler <ICraft_3.36.1_bin> --partitions refine_heads_pipeline --qdtype tf32
```

后续实验先处理已有 SDK 数值偏差，再分别核验新图 A/B 响应、重复一致性、有限值与真实设备内存；按完整交接成本比较分离和连续方案。通过后再接候选运行时、重做完整图像与视频计时。仍保持 PSS700 MiB、MemAvailable96 MiB、设备隔离64 MiB；不能靠关闭保护或把未响应输出的短耗时当成收益。
