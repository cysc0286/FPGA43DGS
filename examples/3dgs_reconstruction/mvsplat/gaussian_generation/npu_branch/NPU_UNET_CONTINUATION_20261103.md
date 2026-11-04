# 两级细化网 NPU 迁移续作（2026-11-03 上午）

用户在夜间截止后明确恢复工作，随后指定“先进一步迁移 NPU，实验后续再做”。本轮完成隔离 NPU 诊断与迁移代码准备，没有新 USB 录像、完整视频重建或新图像资格；不覆盖夜间封存记录。

## 当前交付与采用边界

正式路径仍为用户已采纳的十子图：backbone、depth head、两个分块预测头和六个 FFN。CPU 回退、四业务模块、冻结 FPGA/BOOT、正式桥接库与现有环视网页保留。完整 `corr_refine_net`、`refine_unet` 已组装进十二图隔离候选，保留原权重、归一化、注意力、残差与两视图批次，并非仅迁移首层卷积。

候选位于私有输出 `outputs/npu_unet_diagnostics_20261103/depth12_candidate_20261103/`，已通过资产与 FP32/NHWC ABI 审计，`candidate_plan.json` 明确标记未执行整网 NPU、未获生产资格。编译 manifest SHA256：`e6af75935770babd1232067c3237c71c592753fa4c0e66120bac73ae90ad5f19`。

组装源为四图 `reproducible_candidate`、`ffn_export_v2/ffn_compiled`、`corr_export/corr_compiled` 和 `refine_identity/refine_identity_compiled`。不同 oracle 场景来源用显式 mixed-context provenance 保留；资格目标为原 e81a 场景，不能把目标标签冒充全部 oracle 来源，不能复用十图 receipt。

## 源 ONNX 与 SDK 问题分开核对

原 ARM CPU 捕获的两份非零 A/B stage fixture，在本机 ORT 单线程执行完整源 ONNX 的 A→B→A→B。NCHW 输入、NHWC 输出转回 NCHW 比较。所有调用有限、输入响应与重复一致，通过 `rtol=0.002, atol=0.0002`；原 ARM expected 不改。

| 完整源图 | A 最大绝对误差 / RMSE | B 最大绝对误差 / RMSE |
|---|---|---|
| corr_refine_net | 9.06e-6 / 5.66e-7 | 7.39e-6 / 4.89e-7 |
| refine_unet | 1.57e-5 / 7.02e-7 | 1.14e-5 / 6.24e-7 |

refine 的 identity 尾源图与原源图输出 hash 相同。更严格 1e-5/1e-6 门槛仍有少量失败，不声称不同实现逐位相同。这把问题缩小到 ICraft 编译或板端执行/回读，尚未证明具体根因。证据 `outputs/npu_unet_diagnostics_20261103/real_full_onnx_ort_real_ab_20261103/result.json`，这是正确性核对，不是电脑生产推理性能。

## 两级完整图的双执行诊断

板端单常驻 context、fresh input、真实 A×3/B×3/A×3：每图九个逻辑请求、十八次实际 SDK forward。两个网络均有限、A/B 可区分、重复与返回 A 精确一致；所有写入、执行、等待、SFB 转换成本保留。单执行对照仍会返回上一输入结果。

| 图 | SDK 总成本：首请求 / 后八次范围 | 稳定 A / B RMSE | 严格门槛 |
|---|---|---|---|
| corr_refine_net | 57.06 ms / 31.63–40.67 ms | 0.00966290 / 0.00908945 | 失败 |
| refine_unet | 344.67 ms / 216.14–217.62 ms | 0.00886358 / 0.00851313 | 失败 |

SDK 总成本包括两次执行的 input_write、submit、wait、output_convert；是完整请求和整网墙钟的子集。它不是高斯生成或视频到首帧耗时；不能只因响应正常就认定画质通过。

| 单图诊断进程 | 峰值进程树 PSS | 最低 MemAvailable |
|---|---|---|
| corr 九请求、十八 SDK | 127.30 MiB | 567.96 MiB |
| refine 九请求、十八 SDK | 239.92 MiB | 502.54 MiB |

这是分别运行的单图诊断，不能当作十二图整网峰值。证据 `outputs/ring_20261103/continuation/20261103T100315_probe_corr_refine_net_baseline_passes2/` 与 `20261103T100435_probe_refine_unet_baseline_passes2/`，各保存命令、上传哈希、逐次误差/成本和 measurement；私有库 SHA `dd9e22d5ff618efe107cb249b99617a88abb67e701aa0e44060a33a358bbfe99`。正式库未替换。

## 已拒绝的状态管理候选

- **每请求 `Session::sub(0)`、共享已部署后端**：按官方示例不重复 apply，保持子会话到 wait/SFB 完成，并核对 bindings。corr 九次输出全部保持最初 A，B 无响应，热调用约 244–249 ms。拒绝。证据 `20261103T101047_probe_corr_refine_net_sub_passes1/`。
- **显式 Host 新输出**：首次是本方诊断把 CPTR 调用 addr()，在 forward 前出错；修正为 cptr 后，SDK 后端内部仍在 CPTR 上调用 addr() 并报错。两个不同栈分别保留，不能混作 NPU 数值失败。证据 `20261103T095728_probe_corr_refine_net_owned/`、`20261103T101154_probe_corr_refine_net_owned_passes1/`。
- **显式 PL DDR 新输出**：按官方示例分配物理 chunk，九个输出全保留，检查地址不重叠/不被替换，仍用 SFB 回读。corr 首个 A 正常，之后重复输入输出改变，RMSE 最高约 1.228，拒绝，未继续 refine。额外占 9 MiB PL DDR，另有 64 MiB guard，不是 ARM PSS 或零拷贝。证据 `20261103T102032_probe_corr_refine_net_owned_passes1/`，库 SHA `4daf29d299b426996984fe37c2bff88d1ec9263f34f6653b7e038cde981ec74f`。

以上目录均在 `outputs/ring_20261103/continuation/`。这些诊断开关不进入生产库；未修改 ready、oracle 或设备隔离来换取通过。

sub、修正 CPTR 之后的 Host 输出、PL 输出三个负例的峰值 PSS/最低系统可用内存分别为 127.24/566.75、187.52/524.22、128.37/567.22 MiB。各次构建和正负控制的完整墙钟、SDK 成本、资源、门槛、输出误差及失败原因统一归档于该目录 `summary.json`；外层进程 `passed` 不等于内部 NPU 数值或图像通过。

## 逐层定位：Conv → GroupNorm → GELU

新增 `onnx_prefix_probe.py`，从核对过的完整源图截取前缀，保留原节点和 initializer Proto 字节、batch=2，以及实际 NHWC Transpose。中间参考明确是完整源 ONNX 派生，不冒充独立 ARM hook；原 ARM 完整阶段 fixture 保留。

| 边界 | 本机编译/ABI | 用途 |
|---|---|---|
| corr 首 Conv | FP32/NHWC、ZG330、无 Host 计算，通过 | 待逐层板测 |
| 两网络 GN / GN+GELU | `***C` 输出被审计拒绝 | 保留失败，不重标布局 |
| 两网络 GN + 实际 identity 1×1 Conv 尾 | FP32/NHWC、ZG330、无 Host 计算，通过 | 待逐层板测 |

identity 使用现有 `onnx_output_control.py`，A/B/A/B ORT 输出与原前缀精确一致，新增参数单列，每图额外约 3355 万次乘法；这是诊断仪器，不是加速机制。边界布局问题不能直接证明 GN 算错或不受支持，完整图本来能编译经过该层。

产物和最小传输清单见 `outputs/npu_unet_diagnostics_20261103/prefix_diagnostics_real_ab_20261103/{README.md,candidate_summary.json}`。后续板测要明确支持 ONNX-derived oracle 的来源合同，不能伪装成现有 ARM 边界捕获来绕过验证。

## 对比代码、资源与回归

`paired_inference.py` 新增显式 `--baseline-partitions`，比较同一个 ModelRuntime 的 10→12→12→10。每 case 只创建自己一套 worker，完全关闭/还原 CPU 模块后才进入下一套。加载、前后 oracle、attach/close 单列；完整前向计时包含剩余 CPU、布局、IPC 和每次 SDK。旧 CPU/backbone/expanded 模式保留。

修复“调用序列等于 CLI 分区列表”的错误：按实际模型顺序与 FFN/spatial 元数据展开，静态核对十图为 72 逻辑/143 SDK，十二图为 74/147；两个头仍各 32 块。漏项、多项、乱序、计费不足、非有限时间或前后 oracle 变化均拒绝。strict 失败继续保存，诊断不签 acceptance。`--worker-timeout` 显式可调，默认与公共 runtime 的 90 s 不改；十二图准备可指定 240 s，预热仍单列。

十二图共享 IPC 分配为 8.625 MiB；各静态边界 I/O 合计 34.389 MiB，未按分块/双执行展开，不是实测 DDR 流量。raw 文件合计 168.732 MiB，也不是实测 RAM/PL DDR。十二个 SDK context 同时常驻的实际峰值待测。

本轮 FPGA RTL、资源利用率与时钟配置不变；未新增综合/布局布线结果。NPU 瞬时利用率、功耗和真实 DDR 流量未测；没有用逻辑 I/O、SDK 调用占比或文件体积替代这些指标。

GCC GC 参数降低板端编译峰值；最新私有构建 99.316 s、峰值 PSS 446.14 MiB、最低 MemAvailable 192.71 MiB。`measure.py` 对 exec 期间 PSS 暂不可读增加有界重试（十次、九个 20 ms 间隔），接受值仍必须为实际 PSS 或退出证明；权限/持续不可读仍失败，不以旧值或零填补。700 MiB PSS、96 MiB 系统余量、64 MiB 设备隔离不变。

现有完整本机环境 `.venv_mvsplat`（Python 3.12.10，含 Torch/ONNX/ORT/OpenCV）跑 11 组 **115 项全部通过、无跳过**，日志 `outputs/ring_20261103/continuation/local_regression.log`。此前另一环境两项 OpenCV 缺包错误已在完整环境复核，无需安装依赖。

最后只读 `/scene` 仍报告 NPU 生成、FPGA 渲染、look 模式；原 viewer/renderer PID 30592/30597 保留，无诊断 worker/编译残留，可用内存约 627 MiB。正式库 SHA 为 `3d5c11b4249e142c8a816eb89c7fdf1bb501e4b18bdea4d51578a8055ccd6ffe`。证据 `outputs/ring_20261103/continuation/final_readonly_status.json`；这不是新画质或渲染延迟测试。

## 下一步

按用户安排后续实验：先测首 Conv/GN 定位数值，再跑十/十二图完整前向及峰值内存，之后进行原生 FPGA 三视角图像资格。只在输入响应、图像与扣除交接后的净收益通过时替换正式配置。摄像头现场与多组完整重建另行验收；本轮未更新视频结束到首帧、换视角或质量成绩，历史 20 s 未达事实保持。
