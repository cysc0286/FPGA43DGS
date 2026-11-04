# 连续深度 NPU 图与逐层定位入口（2026-11-03）

用户最新安排为“先进一步尝试迁移 NPU，实验后续再做”。本次只完成代码、本机真实源 ONNX 等价检查及 ICraft 编译，没有 SSH、板端 SDK 调用、完整视频或摄像头测试。已采纳的十子图、CPU 回退、四业务模块和冻结 FPGA/BOOT 未被本次候选替换。上一轮完整 corr/refine 的严格板端数值失败仍未关闭。

## 新连续图

输入是完整两视图 `[2,256,32,32]`，输出是 `[2,128,32,32]` 的低分辨率深度 logits，随后才在 CPU 执行原 softmax、深度候选加权及后续几何处理：

```text
raw_correlation_in ── corr_refine_net ───┐
                   └ regressor_residual ┴ Add → depth_head_lowres → logits
```

`corr_depth_export.py` 组合三张原始 ONNX 图。原 corr 的实际末端 NCHW→NHWC Transpose 经结构检查后移除，三条分支在真实 NCHW 轴顺序连接，最终 logits 再执行 NHWC 转换供编译器审查。没有按元素数量重解释布局。完整 U-Net 的 GroupNorm、跨视图 Attention、两视图顺序、残差与全部权重保留；没有把输入相关的归一化折成常量。

三张源图的 checkpoint SHA256 相同；各自旧 oracle 的来源场景继续独立记录，不把同权重误写成同输入。完整 corr 对本轮真实 ARM A/B hook expected 先做严格检查。连续深度参考再由原 corr、residual、Add、depth 三图按原顺序执行生成，明确是 **源 ONNX 派生深度参考**；不是独立 ARM 深度 hook，也不是新板端验收。

| 指标 | 本次结果 | 证据范围 |
|---|---|---|
| 原参数 | 6,485,248 个；按分支重命名后 initializer 内容精确不变 | 本机图结构核查 |
| A 最大误差 / RMSE | 1.525879e-5 / 6.305071e-7 | 连续 ORT 对独立源三图组合 |
| B 最大误差 / RMSE | 7.629395e-6 / 5.614234e-7 | 同上 |
| A→B→B→A | 严格门槛全过，返回同输入精确重复 | `rtol=0.002, atol=0.0002` |

ICraft 3.36.1 / TF32 的 parse、optimize、quantize、adapt、generate 均通过。现有审查确认输入输出都是 FP32/NHWC、869 个 ZG330 HardOp、无顶层 Host 计算回退；这不证明板端数值正确或性能收益。生成 raw 文件 54,199,120 字节（约 51.69 MiB），这是文件体积，不能代替 ARM PSS 或 PL-DDR 峰值。

## 减少的交接是什么

以下仅统计一轮逻辑调用的 FP32 输入输出边界，不含重复 SDK 执行，不是实际 DDR 流量：

| 对照分工 | 逻辑 I/O | 连续候选 |
|---|---:|---:|
| corr、residual、depth 三图均单独交 NPU | 8 MiB，3 次逻辑调用 | 3 MiB，1 次 |
| corr、depth 在 NPU，residual 留 CPU | 5 MiB，2 次逻辑调用 | 3 MiB，1 次 |

与第二行这个更接近当前扩展候选的对照相比，减少 2 MiB（40%）的逻辑边界流量和一次提交，同时把 residual 卷积与相加也放进图内。没有把 40% 写成整网或视频到首帧的提速。新图可能增加设备侧中间数据存活时间，收益仍取决于板端布局转换、执行、同步和设备内存。

## 运行时接口准备

`corr_depth.py` 提供两个显式工具；后续代码增量已将它们接入候选选择，默认不开启：

1. `patch_source(source)` 严格匹配生成副本中原来的 corr/residual/depth 块，抽成普通实例方法 `_corr_to_depth_logits`。校验 AST 与原语句位置，保留消融 CPU 分支和原 softmax 位置，返回补丁及源码哈希，不写原 vendor 文件、不改变 state_dict 键。
2. `attach_corr_depth(predictor, execute)` 只覆盖该完整方法。执行器每次收到完整两视图 FP32 输入并返回 logits；不缓存下次结果、不返回占位 corr、不把 residual/head 改成零模块。输入、输出、eval 模式、形状和有限值均检查，失败后关闭并恢复；返回值独立拥有内存，不能被下一次共享缓冲覆盖。

本机小 Torch 真实执行已覆盖原三步与单执行器的 A/A/B/B/A 输出、原 CPU 模块不重复运行、关闭恢复、异常恢复、共享输出复用及 state_dict 不变。真实上游 `DepthPredictorMultiView` 文件也通过严格补丁匹配，原文件未修改。

它已注册为 `corr_depth_pipeline`，正式配置、文件/视频/采集共用的 `GenerationSession`、`verify`、`qualify` 及配对工具都能显式选择。它仍是**待上板验收候选**，没有改变网页的已验十图配置，也没有签发图像 receipt。选择连续图会在副本中加方法 seam，原 vendor 文件和 state_dict 不改；候选必须使用 1..128 的整数 depth chunk，在创建 worker 前拒绝非法配置。

选择连续图与独立 `corr_refine_net / regressor_residual / depth_head` 互斥；融合双预测头与两个独立头同样互斥。校验同时用于配置、资产、组装、worker、方法挂载和资格 receipt，避免只在某一入口防重复执行。普通 CPU 剖析跳过方法型分区；单模块 benchmark 明确引导到配对工具，不会在加载大模型后才因缺少 seam 失败。

关闭 worker 或后续挂载失败时，方法适配器、六 FFN、普通模块与融合头依次恢复；小 Torch 联合执行验证原三模块没有被重复调用，恢复后 CPU 输出与参数一致。NPU 连续图要求 fresh SDK input，并按既有输入切换规避执行两次 forward，两次均计费。此处是代码合同，尚未证明新图实际需要的规避次数或实板收益。

代码身份也随 runtime、内存调度和 seam 文件一起绑定。即使默认计算路径不变，**同步本轮源码后旧 receipt 也会明确失效，必须重新资格验证**；未同步的冻结板端部署副本保持原样，不能删减哈希检查来复用旧资格。

## 十一图待验组合与调用边界

本机已从原十二图资产与新连续图组装出十一图候选，图、raw、ONNX、A/B oracle 和导出/编译哈希均重新审查。原 oracle 来自不同上下文，使用显式 mixed-oracle 选项保留各自来源；目标上下文只用于下一轮整网资格，不能把它写成所有旧 oracle 的输入来源。

| 组合 | 一次完整前向逻辑调用 / SDK forward | 状态 |
|---|---:|---|
| 已采纳十图 | 72 / 143 | 原冻结部署留作对照 |
| 独立 corr＋refine 十二图 | 74 / 147 | 前轮实板严格数值未过 |
| 连续 corr/residual/depth＋refine 十一图 | 73 / 146 | 本轮完成离线资产与接入，未上板 |

表中次数根据实际分块与两次执行合同核验，不是实板新计时。十一图保留独立 `to_gaussians / to_disparity`；融合双头也能与连续图共存，但未混入这个十一图比较组。配对工具按“覆盖的原操作”比较，允许融合前后语义覆盖相同而图边界不同，仍拒绝丢失覆盖或同一选择的重复比较。

新组合选择为：

```text
backbone_cnn transformer_ffn_0 transformer_ffn_1 transformer_ffn_2
transformer_ffn_3 transformer_ffn_4 transformer_ffn_5
corr_depth_pipeline refine_unet to_gaussians to_disparity
```

运行仍需新图的 readiness / 图像资格。不能给这组配置附上十图旧 receipt 后直接公开 READY。

## 单卷积到归一化的专用诊断入口

新增 `onnx_prefix_diagnostic.py`，用于上一轮已准备的 corr 首 Conv、corr 首 GN、refine 首 GN 三个候选。正式 `diagnostic_runner.py` 的独立 ARM hook 来源要求保持原样；新工具单独标记 `source_onnx_derived`、`independent_arm_intermediate=false`，拒绝注册过的生产分区名称，永不产生正式 receipt。

工具逐项绑定四个 manifest、原完整源图、编译图/参数、原始 ARM A/B fixture 与派生 oracle 的 SHA256；核对原节点/initializer 字节、真实恒等卷积尾、FP32 布局、根输入与阶段形状。三个候选的本机 `--preflight-only` 均通过。GN 的恒等卷积尾属于布局定位控制，额外计算已经单列，不能当作加速机制。

默认序列 `A,A,B,B,A,A`；显式 `--sdk-passes 1` 或 `2`，实际为 6 或 12 次 SDK 调用。每次执行的输出、首次偏差、陈旧结果、非有限值、重复性和全部调用耗时分别保留。双执行时不能隐藏第一个 pass 的错误或成本。将来运行仍要求独占设备锁、fresh input、64 MiB 设备隔离，并由 `measure.py` 保持 PSS 700 MiB / MemAvailable 96 MiB 保护。

## 本轮核验与证据

初版准备完成 14 组 158 项本机回归。接入增量后，完整本机依赖环境运行 25 组 **272 项全过，无跳过**；覆盖连续图配置/源码/方法生命周期、真实小 Torch 联合执行、worker 计费、配对覆盖、前缀诊断、分区组装、输出布局、内存和旧工具兼容。测试不能代替实际 NPU 或最终图像检查。

另将真实连续 ONNX 图通过现有 `PartitionRuntime` 子进程与共享缓冲执行 A→B→B→A：四次严格对比通过，A/B 输出不同，返回同一输入逐字节一致；最大误差仍分别为 1.525879e-5 / 7.629395e-6，worker 正常关闭。比较对象仍是源 ONNX 派生参考，`independent_arm_depth_reference=false`，`npu_executed=false`；不将本机 ORT 耗时当成板端成绩。

未变更 FPGA RTL、频率、资源或时序；本轮未测板端 PSS、DDR、利用率、功耗、PSNR/SSIM、视频结束到首帧或换视角时延。连续图减少交接有潜在收益，也可能增加设备中间张量存活与峰值，仍需实板确认。

私有证据目录（不进入 Git）：

- `outputs/npu_unet_diagnostics_20261103/corr_depth_prepare_20261103_v2/`：最终导出/编译 manifest、源图、原始日志、补丁副本和回归日志；较早无严格元数据检查的 `_20261103` 目录为历史候选，不是当前选择。
- `outputs/npu_unet_diagnostics_20261103/prefix_diagnostic_preflight_shape_contract_20261103/summary.json`：三项最终预检结果及完整重放 argv。
- `outputs/npu_unet_diagnostics_20261103/depth11_continuous_candidate_20261103/`：十一图导出及编译资产，标记未 NPU 执行、要求重新资格。
- `outputs/npu_unet_diagnostics_20261103/corr_depth_runtime_integration_20261103/local_transport_check.json`：真实图经过现有子进程接口的四次本机数值与布局检查。
- `outputs/npu_unet_diagnostics_20261103/corr_depth_runtime_regression_20261103.log`：272 项本机回归原始日志。

在 `examples/3dgs_reconstruction/mvsplat` 下重建连续图：

```powershell
python -m gaussian_generation.npu_branch.corr_depth_export `
  --corr-export <完整corr源图目录> --linear-export <原七图导出目录> `
  --fixtures <真实ARM_A_B捕获目录> --out <新的导出目录>
python -m gaussian_generation.npu_branch.compile_graphs `
  --graphs <新的导出目录> --out <新的编译目录> `
  --compiler <ICraft_3.36.1_bin> --partitions corr_depth_pipeline --qdtype tf32
```

下一轮先用专用前缀入口定位首个板端数值偏差，再进行连续图输入切换、十图/十二图/十一图的完整前向对照。只有新输入响应、完整图像与包括交接的净收益经过实板核验，才替换正式主线。USB 现场闭环及多组完整环视仍待后续实验，不将代码准备改写成验收完成。
