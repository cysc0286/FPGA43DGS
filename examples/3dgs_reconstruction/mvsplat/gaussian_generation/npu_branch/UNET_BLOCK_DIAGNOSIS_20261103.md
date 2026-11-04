# 完整深度网络的逐块输入切换诊断

本轮新增 `unet_probe_export.py`，只生成诊断资产，不接入正式分区、不操作板卡、不更改运行时、桥接库、权重或 FPGA。现有完整 corr/refine 图的输入切换问题仍未关闭。编译成功和 CPU ONNX 一致均不能代替真实 NPU 输入切换验收。

## 已有事实与本轮要区分的问题

2026-11-02 的完整图证据：corr 导出、编译及普通 FP32 NHWC 主机 ABI 审计通过，但固定输入重复仍漂移；refine 原输出 `***C` 被拒绝，在增加真实 identity 1×1 Conv 后通过布局审计，实际 A/B 仍无正确响应。fresh input、两次 SDK forward 和 reapply 没有关闭问题。不能据此确定是 GroupNorm、Attention 或布局中的哪一个造成。

本工具构造原模型的小范围差分环路：原 CPU 边界 A/B → 原块或合法前缀 → ONNX Runtime A/B → ICraft → 保留真实 SDK 会话的 A×3/B×3/A×3。最后一步由板端调度执行，必须同时看对 CPU oracle 的误差、同输入漂移、A/B 响应幅度和回到 A 的可恢复性，不能仅看 hash 或输出有限。

按优先级检验以下假设，而非认定根因：

1. **归一化的维度或数据排列出错。** 先比较 stem Conv、紧随其后的 GroupNorm、GELU 的独立图与累积前缀。如果首次异常在 GroupNorm，保留完整统计域再细查其编译图；不能把 GroupNorm 当作固定 BatchNorm 折叠。
2. **完整图中的临时缓冲、调度或结果读回状态有问题。** 如果每个原块都响应正确，合法累积前缀从某个长度开始失败，就集中检查这一边界前后的 scratch 分配、依赖和设备读回；多跑一次 forward 只能算成本内的对照。
3. **跨视图 Attention 的变形/归约轴处理有问题。** 若到包含 Attention 的 down/up 块才开始失败，以该块为最小图继续定位，始终保留两个视图及全部空间位置。不能按 batch-one 或小空间块拆开后冒充原网络。

## 边界与计算合同

- `stem_00/01/02`：原始 Conv、GroupNorm、GELU。
- `down_XX`：原 `UNetModel.input_blocks[XX]`；`middle_XX`：原 middle 子块。
- `up_XX`：原 output 块；独立图输入是原 forward 已拼接好 skip 的完整张量。
- `out_XX`：原 UNet 输出 Conv/归一化/激活；`unet_complete`：完整原 UNet。
- `tail_XX`（corr 存在）：原顶层尾卷积；`full`：完整原 corr/refine 网络。
- `--layers` 额外列出块内的 rank-four Conv2d/GroupNorm/GELU/SiLU。Attention 内 rank-three 的 qkv/proj/norm 不偷换成四维接口；额外层只支持独立块。

`--mode block` 消费捕获的原块输入，适合判断算子/块本身；`--mode prefix` 从原网络入口重放到指定节点，显式保留所有 skip 依赖，适合定位跨块调度。原有 Attention 实现、归一化、残差和权重都保留；只在导出上下文调用原 `AttentionBlock._forward` 以绕过 autograd checkpoint，离开后恢复。每个候选都要同时通过原 PyTorch 和 ORT A/B 检查。

输出固定为逻辑 NCHW oracle，ONNX 默认显式 NHWC 尾。遇到 `***C` 可以另外生成 `--identity-conv-tail` 对照，真实增加单位权重、零偏置的 1×1 Conv 并验证；不得改编译布局标签。额外乘法量按 N×H×W×C² 记录，后续计入耗时。

## 最短执行顺序

以下在板端正式 Python 环境、`mvsplat` 工作目录运行。路径由外层调度填入，所有输出为新目录。保留 `measure.py` 的 PSS 700 MiB / MemAvailable 96 MiB 保护；不通过放宽保护完成导出。

第一步用已有真实输入/期望 fixture 捕获边界。`--fixtures` 支持 `refine_unet.npz` 所在目录、已有导出根目录，或单个 `oracle.npz` 文件。实际第二份输入优先用 `--fixtures-b`；没有时显式 `--zero-b` 是合成诊断，不是新场景验收。

```sh
python measure.py --out capture_measurement.json --rss-mib 0 --pss-mib 700 --reserve-mib 96 --timeout 1800 -- \
  python -m gaussian_generation.npu_branch.unet_probe_export \
  --context CONTEXT --weights WEIGHTS --vendor VENDOR \
  --modules refine_unet --phase capture --fixtures REFINE_ORACLE \
  --zero-b --out CAPTURE
```

此步只用已缓存的 stage 输入运行原 CPU 子网，不再跑整个 MVSplat。hook 逐项落盘后释放临时数组，不同时保存所有边界 tensor；完整原输出须匹配既有 fixture。`capture_manifest.json` 记录路径、形状、调用顺序和逐文件 SHA256。失败或未完成的 capture 不允许作为导出依据。

第二步先导出三个 stem 点的独立图和前缀，避免一开始同时编译全部大图。

```sh
python measure.py --out export_measurement.json --rss-mib 0 --pss-mib 700 --reserve-mib 96 --timeout 1800 -- \
  python -m gaussian_generation.npu_branch.unet_probe_export \
  --context CONTEXT --weights WEIGHTS --vendor VENDOR \
  --modules refine_unet --phase export --capture-root CAPTURE \
  --nodes stem_00 stem_01 stem_02 --mode both --out EXPORT
```

第三步在电脑用既有 `compile_graphs` 编译候选：

```sh
python -m gaussian_generation.npu_branch.compile_graphs --graphs EXPORT --out COMPILED
```

候选名如 `refine_unet__stem_01__prefix`。编译器接受独立 manifest 中这些名字；它们未登记为正式 PartitionRuntime 分区，不能直接通过生产入口激活。板端最小桥接诊断应从编译 manifest 读取实际输入输出形状与图/raw 哈希，使用一个会话串行执行，不能绕过 ABI 审计。

新增隔离 runner `gaussian_generation.npu_branch.diagnostic_runner` 只接受一个未登记的、host-verified diagnostic block/prefix；它从源 capture/export/compiled manifest 绑定图、权重、oracle 和 ABI，默认要求真实 B fixture。它直接用同一 ABI-2 bridge、`SFB` 输出转换和显式 NHWC/NCHW 转换，单个 SDK 会话执行 A×3/B×3/A×3（9 次 forward）；每次回读前用 NaN 填充输出缓冲，未完整写入会直接失败。此 runner 不读写 catalog/runtime，不发布 READY，不做正式资格或加速声明。

先对 `stem_00` block 单图做板端诊断：

```sh
MGS_NPU_DDR_GUARD_MIB=64 python measure.py --out DIAG.measurement.json --rss-mib 0 --pss-mib 700 --reserve-mib 96 --timeout 240 -- \
  python -m gaussian_generation.npu_branch.diagnostic_runner \
  --graphs EXPORT --bundle COMPILED --partition refine_unet__stem_00__block \
  --capture-root CAPTURE --library ARM_BUILD/libmgs_npu.so --out DIAG
```

正式板端 Python/SDK 环境运行；A/B 必须来自两个真实 stage fixture，shape 相同，CPU 输出在 `rtol=2e-3, atol=2e-4` 下可区分。runner 自动启用当前 U-Net 诊断所需的 fresh SDK input Tensor；不要同时打开 `MGS_NPU_FORCE_WAIT`、`MGS_NPU_REAPPLY` 或修改 ETM/OCM 开关。需要单独测试等待语义时，用同一资产和输入重新运行一个新目录，只增加 `MGS_NPU_FORCE_WAIT=1`。9 次 forward 只用于定位输入响应、重复漂移、回到 A 的恢复性及 CPU 误差；即使全部通过，也不代表完整 U-Net、正式 NPU 分区、端到端性能或图像质量通过。

若同一 SDK 会话仍出现一调用延迟，再对完全相同的图和真实 A/B fixture 做“按输入段重建会话”诊断。它在 A 三次、B 三次、返回 A 三次之间分别重建并释放 SDK context，单段内仍重复三次；只改变 context 生命周期，不改图、输入、布局或输出转换。新增结果记录 `session_policy=one_context_per_input_segment` 和三组 context 信息。可在命令末尾增加 `--fresh-session-per-segment`，输出必须使用另一个新目录。若此模式消除旧输出而单会话模式复现，则问题指向会话保留状态；若仍延迟，session 生命周期假设不成立。它会重复网络/设备初始化，纯诊断，不进入生产热路径，也不代表已修复精度或证明净加速。先用 `stem_00` 这类小图做此对照，不要直接对完整 U-Net 反复重建以免增加内存和初始化负担。

stem 全过后，按 `down_*`、`middle_*`、`up_*`、`out_*` 缩小；可在新的 export 目录每次只选一两个点。需要块内层时，在首次 capture 加 `--layers`，用 `--phase list --layers` 查看准确名字，再以 `--mode block --nodes layer_*` 选少量相关层。仅 capture 的节点可导出。

## 本机验证与尚未实测的指标

新增 13 项 tiny 测试通过：全部边界与原输出一致、up 输入保留 skip 拼接和两视图、原路径前缀一致、A/B ONNX Runtime 一致、NHWC 通道顺序、真实 identity Conv、hook/checkpoint 异常恢复、capture 哈希和不覆盖保护。加上已有完整导出/输出尾测试共 28 项通过；`py_compile` 通过。只运行微型网络，未在 Windows 运行完整预训练网络。

本轮不报告新增 NPU 加速、完整视频时间或画质改善：真实 ICraft 编译和 NPU 输入切换待板端调度。FPGA RTL/BOOT/资源不变；NPU 利用率、真实 DDR 流量、功耗均未测。优点是把完整图的问题缩成可重复的原块/前缀对照，同时不改变模型；代价是诊断资产落盘和重复 CPU 参考运行，仅用于定位，不进入首帧性能计时。

## 2026-11-03 实板续测：会话残留与数值误差分开定位

以下结果更新上文“待板测”的状态，仅为隔离诊断，未替换正式十子图方案。使用两份真实 ARM CPU stage fixture、原模型权重、相同编译图和 bridge；每组执行 A×3、B×3、再 A×3，共 9 次真实 SDK forward。所有输出均有限。唯一控制变量为一个 SDK context 贯穿 9 次，或在三个输入段之间关闭并重建 context；fresh input Tensor、64 MiB 设备地址隔离、700 MiB PSS 与 96 MiB 系统余量保护不变。

| 图 | 单 context 输入切换 | 按段重建 context | 结论 |
|---|---|---|---|
| 原 `refine_unet` 的最小 `stem_00` Conv | A/B 立即响应，重复及返回 A 精确一致 | 同样正常，A/B 输出 hash 与单 context 相同 | 单 Conv 未复现整图的一调用落后；但已出现数值误差 |
| 完整 `corr_refine_net` | 首个 B 精确等于旧 A，第二个 B 才更新；返回 A 同样落后一调用 | 第一个 B 即更新，三段重复及返回 A 均精确一致 | 重建 context 可规避本轮输入切换残留，但未修复数值 |
| 完整 `refine_unet`（已有 identity Conv 尾） | 与 corr 相同的一调用落后 | 第一个 B 即更新，三段重复及返回 A 均精确一致 | 同样是诊断规避，不能作为热路径部署方案 |

重建后的 A/B 输出分别与单 context 下稳定后的 A/B 输出逐文件 hash 相同，误差也相同。这把两个问题区分开了：完整图存在与 context 保留状态相关的输入切换问题；数值误差不会因为重建 context 消失。它尚不能证明 SDK 内部哪一块缓存、权重读回或调度是根因，也不能把偏差直接归咎于某个算子。最小 Conv 已超出严格门槛，因此不能只追查 Attention/GroupNorm 后面的误差。

| 图 | 稳定 A / B 的 RMSE | 最大绝对误差 A / B | 越逐元素门槛比例 A / B |
|---|---|---|---|
| `stem_00` | 0.00069320 / 0.00069991 | 0.00968170 / 0.00934052 | 4.235% / 4.954% |
| `corr_refine_net` | 0.00966290 / 0.00908945 | 0.14521092 / 0.12668538 | 91.744% / 92.402% |
| `refine_unet` | 0.00886358 / 0.00851313 | 0.16599226 / 0.12307298 | 92.491% / 92.504% |

全部仍不通过 `rtol=0.002, atol=0.0002`，结果中的 `production_qualified=false`、`acceptance=false` 保持。原 float32 oracle 未改；最小 stem 的真实 A/B 参考由原 ONNX Conv 权重计算，并通过独立 NumPy 完整卷积核对，不冒充完整 U-Net hook 捕获或新视频图像资格。

### 资源与重建会话的真实代价

下表墙钟为外层 `measure.py` 的完整诊断进程时间，包含 Python/SDK 初始化和九调用；PSS 为进程树采样峰值，系统余量为采样最低 `MemAvailable`，不是 NPU 显存占用。箭头表示单 context → 三段重建 context，不是加速前后生产性能。

| 图 | 完整诊断墙钟 | 峰值 PSS | 最低系统可用内存 |
|---|---|---|---|
| `stem_00` | 6.924 → 13.344 s | 116.80 → 116.79 MiB | 519.07 → 519.24 MiB |
| `corr_refine_net` | 28.959 → 82.277 s | 121.79 → 125.10 MiB | 563.83 → 560.75 MiB |
| `refine_unet` | 39.097 → 102.999 s | 235.33 → 246.87 MiB | 498.36 → 486.17 MiB |

runner 自身墙钟（不含外层进程启动）分别为 stem 4.742 → 11.143 s、corr 26.062 → 79.411 s、refine 34.405 → 98.326 s。重建模式每个输入段首请求包含关闭/新建 context：stem 为 2.36–3.10 s，corr 为 25.56–25.78 s，refine 为 30.63–31.07 s；同一次请求的 bridge `sdk_total` 只统计输入写入、提交、等待及转换，不能掩盖 context 初始化代价。corr 热调用 `sdk_total` 约 16–17 ms，refine 约 86–102 ms，均不是可用加速成绩。重建 context 虽让响应正确，却增加几十秒初始化成本，且数值未过，不采纳到生产。

完整 refine 的首次三段重建测试在 90.137 s 超时（返回码 −15），对应 `.json` 为 0 字节，不解析或当作完整结果。随后只把诊断超时设为 150 s，在同图同输入和原内存保护下完整执行，返回码 0；这不是放松数值/内存门槛。外层 measurement 的 `status=passed` 仅表示进程正常结束，不能覆盖内部 `diagnostic_pass=false`。

原始证据位于工作区私有诊断根 `outputs/npu_unet_diagnostics_20261103/`：`real_stem_single_0743`、`real_stem_per_segment_0743`、`full_corr_refine_net_single_0750`、`full_corr_refine_net_per_segment_0750`、`full_refine_unet_single_0750`、`full_refine_unet_per_segment_0755_retry150`，各含 `.json`、`.measurement.json`、日志与命令；超时失败另保留 `full_refine_unet_per_segment_0750.measurement.json` 和日志。不得用旧超时空 JSON 覆盖重试证据。

另准备使用 SDK 官方 `forward(inputs, outputs)` 显式提供新输出 Tensor 的隔离 bridge 控制。板端确实尝试编译，但在 45.063 s 时系统可用内存降到 95.816 MiB，触发原有 96 MiB 余量保护（`host_reserve_limit`，返回码 −15）；采样峰值 PSS 542.808 MiB，未达到 700 MiB PSS 上限。没有产出已验证新库，也没有运行新输出 Tensor 的 NPU 对照，不能据此声称修复或无效。证据为 `outputs/ring_20261103/fresh_output_build.measurement.json` 及同前缀日志；不放松保护、不替换生产 bridge。

### 诊断后的展示完整性复核

现有游戏环拍场景在 0°、90°、180°、270°各返回一张 128×128 FPGA PNG，四张 SHA-256 全部与归档一致；FPGA 服务分别 23.42 / 27.73 / 31.52 / 26.12 ms，HTTP 往返 38.67 / 57.74 / 64.54 / 55.29 ms。`generation.backend=npu`、`npu_executed=true`、`renderer_backend=fpga` 不变。证据为私有 `private_media/rotation_ring_20261103/game_ring_8pair_20261103/angle_sweep_15deg_board_20261103/after_npu_probe_recheck_20261103.json`。这只是既有 Gaussian 场景的四角度回归，没有重新生成场景，也不是新 USB 录像；原有空洞、150.743 s 场景准备时间和未完成的真实摄像头闭环限制均保持。

随后隔离bridge编译造成系统余量守卫触发时，原viewer也由其外层保护有序停止（`VIEWER_COMPLETE` 后为 `host_reserve_limit`），因而页面短暂连接拒绝。已沿用同一场景恢复新viewer，并做最终实际浏览器0/90/180/270/0°确认式渲染：5/5显示128×128，回0°canvas PNG hash相同、无JS错误；FPGA22.93–29.47ms，浏览器自动化点击至新图50.10–107.67ms（含自动化检测，与页面自身请求计时分开）。证据在私有 `final_browser_20261103_retry4/browser.json` 和四张网页截图，详见 `initialize/NIGHT_PLAN_20261103.md` 最后验收段。当前没有NPU诊断或编译残留；仍未启动18080相机服务以避免双renderer，也未做新USB实录或新整链性能测量。

收尾板端快照 `final_browser_20261103_retry4/final_board_status.json` 确认唯一viewer30592与FPGA renderer30597、系统可用621.457MiB；本机最终8组83项回归通过，日志为 `outputs/ring_20261103/final_regression_0756.log`。这些不会改变本节完整细化图未获生产资格的结论。既有8组/63,788Gaussian场景的历史完整进程（含预热）峰值PSS565.129MiB、最低可用244.746MiB、视频交付到首帧150.742699s均未重测；可见条纹/底部缺口仍保留。详情及不同扫描口径见NIGHT_PLAN最后清单，之后不再测试以保护展示会话。
