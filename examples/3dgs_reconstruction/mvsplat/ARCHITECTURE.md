# MVSplat 板端主线结构

本目录只描述“视频文件已经接收完成后，在悟净 30TAI Lite 上生成首帧并交给常驻渲染器”的主线。旧 OpenSplat、批接口、FAMERS 对照和一次性诊断结果留在同级历史目录，不作为默认入口。

## 数据流

```text
initialize/session.py
  预热固定权重、Torch、可选 NPU 会话和常驻 FPGA 进程
  -> READY
video_input/receipt.py
  只接收已经关闭的视频文件
  -> VIDEO_COMPLETE（场景准备计时起点）
video_input/prepare.py
  解码选帧 -> 两视图 SIFT/Essential -> 首个目标 PnP
  -> input.initial.json + context.npz
initialize/model_runtime.py
  MVSplat 前向推理 -> 内存 Gaussian；归档可延后
warm_pipeline.py
  高斯校验 -> 协方差分解 -> 62-float rows -> 常驻场景装入
initialize/renderer_runtime.py
  相机投影 -> Tile 分组/排序 -> 常驻 FPGA 合成 -> 帧校验
  -> SCENE_READY -> FRAME_COMPLETE（首帧计时终点）
```

## 模块边界和合同

|目录/文件|职责|输入/输出合同|板端状态|
|---|---|---|---|
|`initialize/session.py`|只做预热和资源所有权|权重、源码、设备路径 -> `READY` 资源|CPU；NPU 仅在数值预检通过时挂接|
|`initialize/run.py`|状态机与事件计时|视频路径/receipt -> 四事件 JSON|CPU 调度|
|`video_input/receipt.py`|接收已完成视频|文件 -> `VIDEO_COMPLETE`|CPU|
|`video_input/prepare.py`|首帧关键路径的解码、匹配、PnP|视频 -> `context.npz`、`input.initial.json`；其余目标随后补齐|CPU/OpenCV|
|`initialize/model_runtime.py`|常驻 MVSplat encoder|二视图张量 -> means、covariances、SH4、opacity|当前生产路径 CPU；NPU 七子图仍是隔离候选|
|`export.py`|世界协方差与渲染 ABI 转换|Gaussian 字典 -> 62-float rows、PLY、相机文件|CPU；使用批量四元数转换，保留 SciPy 回退|
|`initialize/renderer_runtime.py`|保持设备和场景进程常驻|rows 一次加载；136 字节相机重复提交 -> framebuffer|CPU 准备 + FPGA 合成|
|`initialize/attributes_resident.cpp`|驻留 Gaussian 和相机投影|`LOADROWS`、`PROJECTMEM` -> `attr.bin`|ARM CPU，复用 rows/属性缓冲|
|`initialize/group_sort_resident.cpp`|驻留排序进程|`SORT attr.bin scene.bin`|ARM CPU，进程只启动一次|
|`initialize/render_resident.cpp`|冻结 FPGA 渲染包的常驻封装|有序 Tile 命令 -> `frame.bin`|FPGA 计算；未改动冻结内核|
|`npu/`|编译、桥接、预检和诊断|ICraft 图、oracle、ABI、分区报告|当前不发布到生产链，数值门禁未通过|

## 时间口径

- `READY`、`VIDEO_COMPLETE`、`SCENE_READY`、`FRAME_COMPLETE` 是不可重复发布的状态事件。
- 用户要求的场景准备时间为 `VIDEO_COMPLETE -> FRAME_COMPLETE`。预热和视频接收时间只记录，不作为性能指标。
- 换视角使用已装入的 Gaussian rows，只测相机投影、排序、FPGA 合成和帧封装；不重新运行 MVSplat。
- `gaussians.npz`、PLY 和诊断文件必须继续保存并校验，但首帧可以先用经过哈希绑定的内存 Gaussian；归档在首帧后完成。

## 已验证的优化

1. 场景和排序器进程常驻；换视角不再重复加载模型、启动进程或读取 PLY。
2. 首帧优先发布 `input.initial.json`，其余目标 PnP 不阻塞首帧；`--overlap-prepare` 只在显式线程预算足够时允许。
3. 136 字节相机直接交给常驻投影进程，帧缓冲在内存中完成 RGB/PPM 转换。
4. 高斯 rows 的协方差特征分解后使用批量矩阵到四元数转换。`factor_covariances(..., quaternion_backend="scipy")` 保留为审计回退；新路径相对 SciPy 的旋转矩阵和协方差误差在回归中低于 `1e-12` / `5e-7`，但板端耗时必须单独复测后才能宣称收益。

## 当前实测边界

2026-09-30 固定 128×128、32768 Gaussian、NYU 视频的 CPU+FPGA 记录：

- 视频结束到首帧：rows 26.964 s、frame 26.305 s、archive 26.843 s；均未达到 20 s 目标。
- 常驻换视角 8 次：均值 0.18459 s，中位数 0.16403 s，范围 0.15598–0.33192 s；帧和 PPM 与冻结 FPGA 输出逐字节一致。
- 第 7 视角质量：PSNR 19.4519 dB、SSIM 0.80446；质量沿用同一输出，未因本轮代码改动而提升。
- 网络前向约 16.3–16.8 s，仍是首要瓶颈；Gaussian rows 转换约 2.54–2.56 s，四元数批量化是低风险增量，尚无新的板端计时。
- 七个 NPU 子图仍未通过逐元素数值门禁，且出现过 SDK 权重预检错误；错误输出的短时间不计入性能。

## 验证入口

```text
# 主线单元回归（在本目录）
python -m unittest discover -s . -p 'test*.py'

# 语法检查
python -m compileall -q .

# 常驻渲染复验（板端）
python -m initialize.validate_resident --scene RUN/renderer_input \
  --renderer FROZEN_PACKAGE --out NEW_VALIDATION
```

任何新后端必须先通过固定输入的逐字段/逐图像核验，再加入 `initialize/session.py` 的公开 READY 路径。NPU 当前只能作为隔离诊断后端；CPU+FPGA 是可回退的生产主线。
