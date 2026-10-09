# 渲染主线决定 — 2026-10-06

## 2026-10-09 封装补充

[四路冻结包](package/README.md) 是当前发布入口；[新版实验目录](../render_branch/README.md)
独立保存，主线不引用。10 月 8 日最新配对 60 帧主线平均 **43.575 ms**，
候选 **53.302 ms**；[完整对照](../render_branch/validation/RESULTS.md)。
本次只封装；以下 10 月 6 日数据是按日期保留的历史决定。


当前采用 **四路 grouped-shared 硬件 + 常驻原生 C++ + 完整点集有序 Dense 渲染**。
本次根据既有实板结果固定版本、配置和调用入口，没有新做板端测速或烧录。
硬件于 2026-10-06 候选对照结束时已回退至此版本；不把文档选择当作新的设备状态检查。

## 唯一推荐入口

软件配置与验收产物身份集中在 [mainline.json](mainline.json)。
新交互调用使用 `LiveRenderer.mainline`，不再手工拼接实验开关：

```python
from initialize.session import render_environment
from rendering.runtime import LiveRenderer

with LiveRenderer.mainline(binary, render_environment(), log_path) as renderer:
    renderer.load_scene(model_ply)  # 每次换场景一次；也可用 load_rows(rows)
    frame = renderer.render_camera(camera_bytes)  # FLCAM001，136 字节
    rgb = frame.rgb  # 完整 uint8 RGB；随后重复提交新的相机
    # 如需归档，调用 frame.archive(new_directory)，单列归档耗时。
```

`binary` 必须是支持新选项的 ARM 原生程序，FPGA 必须匹配下文验收硬件。
此入口只固定软件参数并在帧元数据记录 profile 和 expected_boot_sha256；
**它不会检查或更换正在运行的 BOOT**。兼容旧二进制的普通构造函数仍保留。
`rendering.benchmark` 默认只测 `mainline`；对照可显式选 `--profiles mainline,cpu4`。
CPU 对照采用原有 FP32 路径，不能称同精度纯硬件加速对照。

| 部分 | 固定配置与职责 |
|---|---|
| 场景 | 不抽点，`max_gaussians=0`，场景及不随视角变化的协方差/透明度准备常驻 |
| CPU | 4 线程投影及视角相关颜色、直接并行收集、11-bit 三趟完整深度稳定排序、并行 Tile 列表 |
| 提交 | `batch=2`，复用已有 DMA 缓冲及两银行队列 |
| FPGA | `grouped-shared`，4 路，200 MHz，mode 2 Dense；筛选/高斯求值/有序透明度颜色合成 |
| 复用 | 同一高斯最多四个子块共享属性，紧凑 LUTRAM 参数记录，精确指数 ROM |
| 帧交付 | 原生 FP16 来源像素 NEON RGB 封装，完整 RGB 内存返回 |
| 关闭项 | 抽点预览、Sparse、support_guard、额外 DMA 剖析、实验布局；NPU 不参与此渲染路径 |

`direct_collect` 的先决开关 `parallel_collect/fused_collect` 一并固定。
不改变原 FP16 运算、深度顺序、提前终止阈值及 512-bit DMA 接口。
这里的“不抽点”不取消正常视锥/贡献筛选；不是让每个高斯都对每个像素合成。

## 实板依据与适用范围

范围：同一个 **32,768 高斯、128×128、三个指定视角**的已加载场景。
首帧以后，从发送新相机前开始，到完整 RGB 收到并通过形状校验为止。
视频输入、位姿、生成高斯、场景加载、图像归档、远程通信及物理显示均不计入。

| 已保存的对照 | 结果 | 结论 |
|---|---|---|
| 最终软件同轮配对，每组 60 帧 | 42.9337 → **40.9671 ms**，减少 4.58% | 采用 direct_collect + NEON，三视角 raw/RGB 不变 |
| 10 月 6 日原四路 A1 + 回退 A2，共 120 帧 | 平均 **43.9201 ms**，FPGA 服务 **24.4381 ms** | 本次全部原版样本作为对照，不挑最快一轮 |
| 10 月 6 日回退后 A2，60 帧 | 平均 **42.9170 ms**，P95 **44.0654 ms**，P99 **44.1211 ms** | 最后一次已保存的主线实板复测 |
| 同轮 PipeGS compact/SRL 三路候选，120 帧 | 平均 44.8772 ms，FPGA 服务 27.1049 ms | 整帧慢 2.18%，服务慢 10.91%；输出相同但不采用 |
| 历史 16k 均匀预览，60 帧 | 27.337 ms；首视角 PSNR 19.45 → 16.99 dB | 约 27 ms 来自有损抽点，不能当完整点集主线成绩 |
| 历史 Sparse mode | 32.8712 ms；对 Dense 仅 17.68–17.82 dB | 明显缺失，不采用 |

当前对外可以表述为“**此测试场景后续换视角约 41–44 ms，约 23–24 帧/秒的串行吞吐**”。
这是多轮平均值的概括，不是每帧上限：A1 平均 44.9233 ms、P99 55.6828 ms 也保留。
不再用历史有损预览的约 30 ms 回答完整场景换视角耗时。

画质相对留出实拍图像：

| 视角 | PSNR dB | SSIM | 本次采用优化带来的额外损失 |
|---|---:|---:|---|
| 7 | 19.4519 | 0.804464 | 0，raw/RGB 逐位一致 |
| 15 | 20.2223 | 0.785135 | 0，raw/RGB 逐位一致 |
| 22 | 21.3486 | 0.784232 | 0，raw/RGB 逐位一致 |

保留原模型模糊、边缘缺失，首视角低于此前 20 dB 指引；采用此渲染器不等于
重建质量已理想，也不证明其他场景或分辨率相同性能。

## 固定硬件及回退资产

- 复建配置：`examples/3dgs_flicker_hw/pipeline/lane_workset/build_variant.ps1 -Variant grouped-shared -Project <新的构建目录>`。
- 独立平台：`platform/flicker_groupshared_fpga`。必须保留该变体宏；不能拿通用源码默认值替代。
- 已验收 BOOT SHA-256：`b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6`。
- 已验收 ARM 程序 SHA-256：`d1220d46d0cb6830077b973a2cfb2587664e36d28357bc79c18fb2eb6dfaa672`。
- 原四路冻结包 `releases/3dgs_renderer_v1_20260928` 与全部实验保留；这是旧回退包，不是当前主线的新产物。
- 主线硬件冻结产物在 `build/board_cat_20260930_213732/frozen_artifacts`，BOOT 在同构建目录的 `boot`。
- 哈希用于辨认已经测过的产物；重新编译后哈希可能不同，须重新做差分及板端核验，不能自动继承性能结论。

| 全板布线后资源 | 数量/容量 | 利用率 |
|---|---:|---:|
| LUT | 61,260 / 78,600 | 77.94% |
| FF | 98,377 / 157,200 | 62.58% |
| Slice | 19,618 / 19,650 | 99.84% |
| BRAM36 等效 | 218.5 / 265 | 82.45% |
| DSP | 252 / 400 | 63.00% |

全板 setup/hold +0.030/+0.029 ns，渲染/DMA +0.316/+0.054 ns。
仍继承原厂 AI 脉宽 -0.409 ns 及既有 GT/CDC 限制，不称完整全板签核；功耗未测。
Slice 余量很少，不能从尚余 DSP 推断还能直接加路数。

## 实验版本与后续工作

PipeGS 三路 compact/SRL 保留为对照；CTU 加速及四路 stream 的路由/时序问题、
DPC 的数值误差和 RTL 等待环均未解决，不能进入默认链。现有后端包含已验证的
FLICKER 启发筛选/求值以及参数复用优化，不称完整复现 PipeGS。

后续只从此主线派生候选，先解决约 24.4 ms 的 FPGA 服务，再看 CPU 投影/分组；
先分离有效工作、masked 请求和合成阻塞，证明能减周期后再进行完整布局布线。
保持完整点集基线，以同输入、同轮配对、全帧画质决定是否替换。用户允许小收益；
不再把旧记录中的“至少 10%”当硬性淘汰线，但低于运行波动的差异不能算已证明收益。
质量允许小范围变化：单视角 PSNR 下降 ≤0.5 dB、三视角平均 ≤0.3 dB、SSIM 下降 ≤0.01，
同时不能有明显孔洞、闪烁或轮廓破坏；当前选择比此更严格，输出保持一致。

## 本次配置固化的核验

现有 9 项渲染接口回归通过；Python 语法编译通过。以子进程替身核对了
实际传给原生程序的主线参数、帧配置身份、旧二进制参数兼容和配置副本隔离。
已验收原生程序哈希与保存的 provenance 一致，文档证据链接均存在。
两条独立 ARM 实验上传/重建脚本补入 `mainline.json`，重建同时携带新基准入口。
这些是主机配置/接口检查，不是新 ARM 编译、板测或画质测量。

证据入口：

- [软件最终验证](exact_workset/VALIDATION.md)
- [硬件原始验收与效果图](../../../3dgs_flicker_hw/pipeline/resource_balance/BOARD_VALIDATION.md)
- [10 月 6 日 A–B–B–A、资源和画面](../../../3dgs_flicker_hw/pipeline/resource_balance/results/20261006/pipegs_compact_request/BOARD_RESULTS.md)
- [全部近期版本和失败原因](../../../3dgs_flicker_hw/pipeline/resource_balance/results/20261006/RENDER_ATTEMPTS_REVIEW.md)
