# NPU Gaussian head 与输出拷贝候选（2026-11-02）

本轮继续缩短视频结束到首帧，保留四模块、正式双子图 NPU 和冻结 FPGA。
电脑以太网未连接，192.168.126.49:22 超时。本轮仅有真实权重图检查、ICraft 编译/Host
对照、SDK 头文件编译和原生 C++ 内存实验；没有 SSH 上传、ARM 链接、NPU 执行或固件操作。

## 选择 Gaussian head 的依据

上一轮 ARM CPU 稳定剖析中，to_gaussians 约 3.01 秒，corr_refine_net 约 1.38 秒、
refine_unet 约 1.55 秒。先选择结构简单且更重的 Conv→GELU→Conv 头，定位历史 NPU
非有限输出，而非继续加入已经比 CPU 慢的小投影分区。

使用原有固定权重导出的 ONNX 与真实两上下文输入，核验旧 manifest、ONNX 和 oracle
SHA256。原参考输出范围 -47.48195～42.17696，不能单凭这个范围将板端 Inf 归因于
FP16 输出溢出；设备内部或中间值仍需单独定位。

## 新候选与编译

head_probe.py 放在原 gaussian_generation/npu_branch 内，支持导出、单会话实板诊断、
ICraft Host 逐层检查。使用独立诊断 schema，正式入口拒绝这些图，不生成资格 receipt。

- full_b2：原两视图完整头，对照。
- conv_in_b2 / conv_gelu_b2 / conv_out_b2：第一卷积、第一卷积加激活、最终卷积，
  用真实输入及中间参考定位首次偏差；中间图不是最终加速方案。
- full_b1：同权重逐视图执行，必须处理两张图，不能靠少做一个视图省时。
- aligned_b1：通道 163→176、168→176、84→96，新增权重/偏置为零，保留原参数；
  两个真实视图分别核验，多余输出通道严格为零。

六个 TF32 图通过 parse/optimize/quantize/adapt/generate，审计无 Host 计算回退，
主机接口仍为 FP32 NHWC。full_b2、aligned_b1 另编译 FP16，两图也通过。
合计八个已编译、未实板执行的图。

| 配置 | 逻辑边界/调用 | HardOp | 说明 |
|---|---:|---:|---|
| full_b2 | 30.875 MiB | 11 | 一次处理两视图 |
| full_b1 | 15.4375 MiB | 11 | 两次才覆盖原工作量 |
| aligned_b1 | 17 MiB | 8 | 两次共 34 MiB |

补齐版本理论卷积 MAC 增约 15.37%，两视图边界由 30.875 增为 34 MiB。
HardOp 数减少只说明生成调度不同，不是周期/速度证明。规整布局可能有益，额外计算
与搬运也可能更慢；必须计入两次提交、布局、等待、拼接及通道裁剪。以上不是实测 DDR 流量。

## ICraft 逐层定位的新证据

同一真实 batch-two 输入，ICraft Host 执行 optimized / quantized 图并逐层导出 SFB。
输入逐位一致，未量化三层全部通过原严格门槛 rtol=2e-3、atol=2e-4。
TF32 首次严格失败在第一卷积，后续均有限；FP16 也从第一卷积开始失败。

| 最终层相对原 ONNX | 最大绝对误差 | RMSE | 非有限数 |
|---|---:|---:|---:|
| 未量化 ICraft Host | 0.00006866 | 0.000001421 | 0 |
| TF32 ICraft Host | 0.10094357 | 0.00722917 | 0 |
| FP16 ICraft Host | 0.10484982 | 0.00748930 | 0 |

TF32 第一卷积 max_abs=0.00529575、RMSE=0.00019956；GELU 后 RMSE=0.00176965。
这区分了可在 Host 重现的精度差异和未定位的板端 Inf。不能称严格数值已修复，也不能
用 Host 有限输出证明 ZG330 适配、生成、上传、执行与回读正确。
FP16 在此 Host 对照中没有精度优势，保留硬件对照，不替换 TF32 主线。

实板 run 每进程仅一个图/会话，A,A,B,B,A,A 切换输入，记录 SDK precheck、每步 hash、
非有限数、逐元素误差及五段 SDK 成本。batch-one 的 A/B 是两个真实视图。
预检失败/缺失、输入滞后或非有限输出均不能 pass；诊断保存失败以定位。
显式 --sdk-passes 2 完整累计两次执行，64 MiB PL DDR 隔离继续核验。

## 可选 C++ 输出拷贝候选

MGS_NPU_DIRECT_SFB=1 让 SDK SFB 转换写入有容量边界的 ostream，减少外围字符串
扩容、str() 拷贝和最后整块 memcpy。仍用 SDK 转换，不解释原始设备格式，不称零拷贝。
开关默认关闭并进入 runtime_identity，新库/策略必须重新资格验收。启用时先检查桥接库
能力符号，旧库不能静默忽略开关。原路径保留作对照。

output_sink_test.cpp 核验 write/put、局部覆盖、溢出、禁止空洞、负 seek、零容量、
短写和最终字节。下面是 Windows x64/O2、每种 20 次、交替顺序中位数，
仅 Host 字节序列化，不含 SDK、布局或网络推理：

| 载荷 | 原字符串路径 | 直接缓冲 | 结果 |
|---|---:|---:|---|
| 1 MiB | 0.6984 ms | 0.0228 ms | 字节完全相同 |
| 10.5 MiB | 7.4941 ms | 0.6955 ms | 字节完全相同 |

这是降低大边界临时内存/拷贝的候选，不是板端 NPU 收益。当前正式两子图输出较小，
它不能解释秒级提速；主要收益仍依赖把约 3 秒 Gaussian head 等计算真正放上 NPU。
生产桥接和既有 FP32 raw/SFB 诊断通过 SDK 头文件编译，ARM 链接未做。
同步/部署清单加入 .h，构建清单记录头文件 hash，两个诊断上传入口也带齐依赖。

## 软件核验与证据

- check_core.py --mvsplat：130 源码语法通过，16+3+156=175 测试，171 通过、4 环境跳过。
  新增 10 项全部在完整模型环境通过，其中 3 项使用真实 ONNX/ORT 验证图变换。
  另覆盖非有限值、shape、运行身份、诊断拒绝进入主线、部署头文件和 FP32 patch 兼容。
- render_main 151、render_branch 197 文件 hash 通过；93 调用源默认隔离通过。
- 六个真实权重/两视图导出图通过 Host oracle；八图编译/接口审计通过。
- C++ Host 契约/字节一致性、SDK 语法、git diff --check 通过。

原始图/权重/日志仅本机 outputs/npu_head_candidate_20261102，不提交。
graphs、compiled_tf32、compiled_fp16、icraft_host_layers、icraft_host_layers_fp16、native
分别保存来源、逐阶段编译和逐层误差。可移交摘要为同目录 summary.json。

## 沿用实板指标与下一步

本轮首帧、换视角、最终画质、板端内存、NPU 利用率、DDR 和功耗无新测。
2026-11-02 固定公开视频 128×128/32768 高斯历史：CPU 场景准备 24.512 秒，
修正双子图 NPU 24.353 秒，完整合成前向预热候选 23.933 秒，各一次，20 秒未达到。
修正候选首帧对 CPU：PSNR55.332 dB/SSIM0.999232；对实拍：19.450 dB/0.804061。
旧场景原生换视角40.63～45.95 ms；修正新场景网页和最后归档仍待恢复连接。
严格 TF32 tensor 门槛仍失败，图像资格及 backbone 两次执行规避不变。

FPGA/BOOT 未改，资源沿用历史：LUT77.94%、FF62.58%、Slice99.84%、
BRAM36 等效82.45%、DSP63%、200 MHz，既有 AI/GT/CDC 时序限制仍保留。

回板先保留旧库、receipt 和 BOOT，在独立目录构建新桥接库：

1. full_b2 控制后逐项跑 conv_in、conv_gelu、conv_out，每次独立进程/会话，定位首次非有限层。
2. full_b1 / aligned_b1 对比两视图完整成本，TF32/FP16 对照，保留所有失败。
3. 同图/输入/桥接比较 DIRECT_SFB=0/1，核验字节和输入切换后再测速。
4. 有效且稳定的候选才接完整模型，重做三视角原生 FPGA 和 CPU/NPU 配对整链计时，
   然后再更新正式配置与资格；现在不改变默认部署。

示例（现有 mvsplat 目录，每次新证据目录）：

```text
python -m gaussian_generation.npu_branch.head_probe export --graphs OLD_EXPORT --out NEW_GRAPHS
python -m gaussian_generation.npu_branch.compile_graphs --graphs NEW_GRAPHS --out NEW_COMPILED --compiler ICRAFT_BIN
python -m gaussian_generation.npu_branch.head_probe compare-icraft --graphs NEW_GRAPHS --bundle NEW_COMPILED --runner ICRAFT_RUN --out NEW_HOST_LAYERS
python -m gaussian_generation.npu_branch.build --sdk ARM_SDK --out NEW_ARM_BRIDGE
python -m gaussian_generation.npu_branch.head_probe run --graphs NEW_GRAPHS --bundle NEW_COMPILED --library NEW_ARM_BRIDGE/libmgs_npu.so --case conv_in_b2 --out NEW_BOARD_RESULT --save-outputs
```

实板命令使用现有独立 SDK 环境（不继承 Torch LD_PRELOAD），通过 measure.py 限制
超时、进程树 RSS 和可用内存。单图诊断通过不等于完整部署通过。
