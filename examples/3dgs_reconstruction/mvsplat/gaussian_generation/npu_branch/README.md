# NPU 候选：加速 MVSplat 高斯生成

`gaussian_generation/npu_branch/` 是 **gaussian_generation 的可选硬件后端**，不是第五个业务阶段，也不是 FPGA 渲染器。
它尝试把 MVSplat 里的图像特征提取、深度预测和高斯参数预测子图交给板上的 NPU。

```text
视频/位姿 → gaussian_generation（CPU，或验证通过后的 CPU＋NPU）
          → Gaussian 场景 → rendering（CPU＋FPGA）
```

高斯准备好之后切换相机，只调用后面的渲染器；当前这个 NPU 目录不参与换视角渲染。
保留它是为了后续缩短场景准备时间。默认 `--backend cpu` 不创建 PartitionRuntime；
只有显式选择 `--backend npu` 才尝试加载 NPU，并且必须先通过数值预检才能发布 READY。

## 当前状态

七个真实 MVSplat 分区已经在板上执行过，但尚未通过既定逐元素数值门槛；
还记录过权重预检错误和多会话非有限输出。因此当前可用主线仍为 CPU 生成高斯＋FPGA 渲染，
不能把候选短耗时当成已实现的 NPU 加速。详见 [历史板端结果](../../BOARD_WARM_NPU_VALIDATION.md)。
本次完成目录迁移、调用与部署路径同步；没有重新测试 NPU。

## 这里保存的内容

- `catalog.py`：七个子图与原 MVSplat 层名的对应关系。
- `export_graphs.py`、`compile_graphs.py`：真实权重子图导出和 ICraft 编译。
- `runtime.py`、`worker.py`、`bridge.cpp`：CPU Tensor/IPC/SDK 的调用、布局转换和会话管理。
- `readiness.py`、`verify.py`、诊断脚本：数值检查、权重/布局问题定位。
- `benchmark.py`：包含数据交接开销的分区对照计时。

它不是 YOLO 演示，也不是旧 `npu_frontend/` 的 SIFT 匹配试验。
下面保留原部署设计与复跑命令；其中编译/布局检查不等于真实 NPU 数值通过。
2026-10-09 已同步模块路径：历史报告中的 `npu/<文件>` 现位于
`gaussian_generation/npu_branch/<文件>`，旧实验数据、日志和哈希仍按原始路径保存。
后续实现和诊断代码继续放在本目录，不再创建同级 NPU 模块。

### 历史部署 v2 设计

顺序分区共用两块 IPC 映射，总容量 36.375 MiB；`per_partition` 保留 79.5 MiB 对照。
全部边界的逻辑 I/O 仍为 79.5 MiB/前向，不是实测 DDR 流量。输出复制到独立 Tensor，
避免下一任务覆盖。协议/桥接 ABI 为 2，旧 `.so` 不自动降级。
详情见 [v2 报告](../../NPU_DEPLOYMENT_V2.md)。这些设计的离线通过不能消除上述实板数值问题。

## 分工

|候选分区|原模型位置|编译中的卷积数|每次逻辑输入+输出|
|---|---|---:|---:|
|`backbone_cnn`|图像卷积特征提取|16|1.375 MiB|
|`regressor_residual`|代价体残差投影|1|3 MiB|
|`depth_head`|低分辨率深度预测头|2|2 MiB|
|`upsampler`|特征升采样|1|18 MiB|
|`proj_feature`|高分辨率特征投影|1|20 MiB|
|`to_gaussians`|高斯形状/颜色预测头|2|30.875 MiB|
|`to_disparity`|细深度和透明度预测头|2|4.25 MiB|

全部 7 分区共 25 个卷积、127 个 HardOp，ICraft 3.36.1、TF32 内部计算、FP32 主机 I/O、固定两视图 128×128。编译图内的计算操作都要求在 ZG330，无 Host 计算回退。`catalog.py` 锁定层名；真实 checkpoint 与真实视频张量来自官方模型前向 hook，没有替换网络权重。

CPU 仍执行图像/相机准备、跨视图注意力、代价体重采样、两个细化 UNet、Gaussian adapter、模型导出和渲染前处理。NPU 候选没有覆盖全网。按参数数或卷积数量不能推断加速率。

79.5 MiB 是全分区的逻辑边界字节，并非实测 DDR/DMA 流量；布局变换、IPC 缓冲拷贝和 SDK 转换还会增加开销。`upsampler` 的输出还供高斯预测头使用，不能简单删除其 CPU 可见结果。先按分区消融测量，再决定更大子图融合，不默认全开最快。

## 接口与失败行为

- `export_graphs.py`：导出 ONNX、真实输入/输出 oracle 和哈希；ONNXRuntime CPU 检查只证明导出数值。
- `compile_graphs.py`：parse/optimize/quantize/adapt/generate，逐阶段日志；审计实际 shape/layout/参数字节与算子位置。编译失败分区保留记录。
- `artifacts.py`：加载清单、重新审计图和参数，NCHW↔NHWC 显式转换；不允许仅凭字节数猜布局。
- `runtime.py`：把选中子模块替换为同签名调用；一个常驻 worker，多会话和可复用共享文件映射。其他模块明确在 CPU；错误不会静默回退。
- `worker.py` + `bridge.cpp`：独立 SDK 进程隔离 Torch 库环境；一个设备拥有者，保留会话。再次检查真实 `Session.getForwards()` 绑定。输出使用 SDK `SFB` 转换，不能直接读取内部张量冒充 FP32。
- `verify.py`：先逐个真实 oracle，再完整 MVSplat 推理、Gaussian 比较和重复推理哈希。板端只有通过这个入口后才考虑接整链。
- `check_images.py`：用同一个独立 CPU 渲染器比较 Gaussian 变化造成的图像影响；不会将 CPU 图标为 FPGA/NPU 实测。
- `package.py`：验证后制作可搬移的编译图、oracle 和相同上下文包，不上传。
- `preflight.py`：校验迁移后的全部文件、源图/编译图/oracle 关联、协议版本与缓冲预算；可只加载 ARM 动态库检查依赖，不打开 NPU 设备。
- `benchmark.py`：相同真实张量、相同 Torch 线程预算，对选定子模块做 CPU/候选交替顺序测量；保留逐样本时间与每次误差。NPU 候选时间含布局打包、IPC、SDK、同步与解包。
- `check_transport.py`：按实际 ICraft NHWC 接口，用真实张量比较旧临时数组路径与直接布局拷贝；是电脑内存测试，不是 NPU/DMA 吞吐。

请求/应答包含分区名、单调任务号和真实后端；结果返回前检查输出长度、shape、有限值。共享输出被复制成独立张量，下一次调用不会覆盖已有 Tensor。运行按顺序提交，没有新增并行双缓冲。文件映射用于避免逐子图创建进程与传输任务文件，不宣称零拷贝。会话建立在 READY 之前，但硬件第一次运行仍待计时。

## 复跑（在 `mvsplat` 目录）

电脑依赖：已有 MVSplat 环境，加 `onnx==1.17.0`、`onnxruntime==1.22.1`。模型固定为现有 `re10k.ckpt`。每次使用新的输出目录。

```text
python -m gaussian_generation.npu_branch.export_graphs --input INPUT --weights ../models/mvsplat/re10k.ckpt --vendor ../vendor/MVSplat_reference --out EXPORT
python -m gaussian_generation.npu_branch.compile_graphs --graphs EXPORT --out COMPILED
python -m gaussian_generation.npu_branch.build --sdk ../../../npu_3dgs/.vendor/icraft-3.36.1 --out SYNTAX_CHECK --syntax-only
python -m gaussian_generation.npu_branch.verify --backend onnx_reference --graphs EXPORT --bundle EXPORT --input INPUT --weights ../models/mvsplat/re10k.ckpt --vendor ../vendor/MVSplat_reference --partitions backbone_cnn regressor_residual depth_head upsampler proj_feature to_gaussians to_disparity --out HOST_VERIFY
python -m gaussian_generation.npu_branch.check_images --input INPUT --verification HOST_VERIFY --out HOST_IMAGES
python -m gaussian_generation.npu_branch.package --compiled COMPILED --graphs EXPORT --input INPUT --partitions backbone_cnn regressor_residual depth_head upsampler proj_feature to_gaussians to_disparity --out PACKAGE
```

恢复板端测试后，在 ARM 的 `mvsplat` 目录构建 `python -m gaussian_generation.npu_branch.build --sdk /root/heterogs_npu/sdk_3.36.1/usr --out ARM_BUILD`。把包解压到候选目录，先用包内相同上下文做数值验证，避免 PC/ARM 位姿差异污染 NPU 对照：

```text
python -m gaussian_generation.npu_branch.verify --backend npu --graphs candidate/graphs --bundle candidate/compiled --input candidate/input --weights ../weights/re10k.ckpt --vendor ../vendor/MVSplat_reference --library ARM_BUILD/libmgs_npu.so --partitions backbone_cnn --out NPU_BACKBONE_VERIFY
```

v2 新包包含 `deployment.json`，可重新拆成单分区包且保持源图关联。复制后先运行下面的预检；实际验证和测速通过已有 `measure.py` 的进程树 RSS/剩余内存/超时监控启动（环境仍按 `initialize/README.md` 配置 Torch 与 SDK）：

```text
python -m gaussian_generation.npu_branch.preflight --candidate candidate --out PREFLIGHT
python -m gaussian_generation.npu_branch.preflight --candidate candidate --require-board --library ARM_BUILD/libmgs_npu.so --out BOARD_PREFLIGHT
python measure.py --out NPU_VERIFY.measurement.json --rss-mib 650 --reserve-mib 128 --timeout 600 -- python -m gaussian_generation.npu_branch.verify --backend npu --graphs candidate/graphs --bundle candidate/compiled --input candidate/input --weights ../weights/re10k.ckpt --vendor ../vendor/MVSplat_reference --library ARM_BUILD/libmgs_npu.so --partitions backbone_cnn --out NPU_VERIFY
python measure.py --out NPU_BENCH.measurement.json --rss-mib 650 --reserve-mib 128 --timeout 600 -- python -m gaussian_generation.npu_branch.benchmark --backend npu --graphs candidate/graphs --bundle candidate/compiled --weights ../weights/re10k.ckpt --vendor ../vendor/MVSplat_reference --library ARM_BUILD/libmgs_npu.so --partitions backbone_cnn --repeats 20 --out NPU_BENCH
```

板端预检的动态库加载需要先设置 SDK 的 `LD_LIBRARY_PATH`，使用独立 SDK 环境，不能继承 Torch 的 `LD_PRELOAD`；例如 `env -u LD_PRELOAD LD_LIBRARY_PATH=/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu python -m gaussian_generation.npu_branch.preflight ...`。预检不会证明驱动可用或剩余内存足够；真实会话的内存尚未知，运行仍须监控。基准保留一个 CPU 模型用于逐分区对照，因此内存口径与替换后完整推理不同。

电脑回归将上述 `--backend` 改成 `onnx_reference`、`--bundle` 指向 `candidate/graphs` 并去掉 `--library`；结果必须标为主机 CPU。20 次重复输入的 P95/P99 只是经验分位数，不是实际视频分布的尾延迟。即便某分区 NPU 时间小于 CPU，也必须继续完整 Gaussian 数值、图像与视频到首帧验收，不能相加子模块时间宣称整链加速。

随后按新目录对 `to_gaussians` 及其他分区做独立与组合验证。门槛固定在 `verify.py`；不因硬件误差超限而自动放宽。`SFB` 布局、SDK 精度和实际运行绑定仍待实测。全网输出验收后，整链参数为 `pipeline.py initialize ... --backend npu --partition-bundle candidate/compiled --npu-library ARM_BUILD/libmgs_npu.so --partitions ...`，或交由 `initialize/board_accept.py` 包含资源监控后启动。

已有离线包路径及所有原始结果见 [本轮报告](../../INITIALIZE_NPU_VALIDATION.md)。回板必须记录布局打包、SDK 写入、forward+等待、SDK 输出转换、结果解包和外层总耗时；SDK forward 不等于纯硬件周期。运行模型、渲染进程、SDK 和共享映射的合计内存受 650 MiB/128 MiB 余量监控，不能只报 Torch RSS。

## 实板数值检查

```sh
python -m gaussian_generation.npu_branch.oracle_probe --bundle PACKAGE/compiled --graphs PACKAGE/graphs --library ARM/libmgs_npu.so --partitions backbone_cnn --out NEW_PROBE
```

该入口在加载完整 Torch 模型之前核验真实输入，默认两次执行并保存误差及输出。`--diagnostic-continue` 仅用于收集数值失败，仍以非零状态退出；非有限结果/设备错误继续立即停止。`initialize` 还会在同一常驻 worker 内按分区顺序执行两轮，校验失败不发布公共 READY，不静默回退。

`compile_graphs --qdtype fp32` 仅是精度诊断入口；本板实验产生的 FP32 图不符合当前已审计主机 ABI，且独立 SDK 诊断执行超时，未纳入可用部署包。不得修改清单或降低门槛绕过它。
