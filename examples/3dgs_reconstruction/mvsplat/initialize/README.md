# 视频前预热与常驻渲染候选

2026-09-30：新增 [rendering 原生后端](../rendering/README.md)。
initialize 仍负责视频前权重和设备预热；rendering 独立负责场景装载后的换视角。
追加 --live-renderer 指定已构建的原生程序即可接回本视频链；首帧仍完成原归档核验。
追加 --render-max-gaussians 16384 --render-uniform-preview 可显式启用近似预览，
其画质和时间必须单列，不能冒充完整高斯结果。

状态：2026-09-29 已在 30TAI Lite 验证预热 CPU＋FPGA 首帧与连续换视角，详见 [实板结果](../BOARD_WARM_NPU_VALIDATION.md)。原离线证据见 [结果](../INITIALIZE_NPU_VALIDATION.md)。旧 `pipeline.py reconstruct` 保留冷启动对照；冻结 `releases/3dgs_renderer_v1_20260928` 未改。

## 结构与状态

```text
initialize/session.py
  固定权重 + 依赖导入 + 可选 NPU 会话 + FPGA 设备 + 投影服务
         ↓ READY
video_input/receipt.py：接收已经关闭的视频文件
         ↓ VIDEO_COMPLETE（场景准备计时起点）
video_input/prepare.py：解码 → 上下文对与基础位姿
         ├─ 主线程：已加载的 MVSplat 前向
         └─ 工作线程：其余目标相机 PnP
warm_pipeline.py：汇合 → 高斯导出/校验 → 常驻场景装入
         ↓ SCENE_READY（中间事件）
常驻投影 → Tile 分组/排序 → 常驻 FPGA 渲染 → 帧校验
         ↓ FRAME_COMPLETE（场景准备计时终点）
```

`run.py` 是应用入口；`session.py` 仅构建预热资源；`model_runtime.py` 保存模型并提供可重复调用的推理接口。`video_input` 不加载模型，`warm_pipeline` 不重新构造权重/设备。`prepare_fast.py` 是兼容入口，原命令和导入仍可使用。

预热、输入耗时只记录。`events.json` 使用同一进程的 monotonic 时钟相减，epoch 用于对照日志；严格按四事件顺序原子写入。`scene_preparation_seconds` 包含准备与首次渲染，`gaussian_scene_seconds` 和 `first_render_seconds` 是可相加的辅助项。`READY→VIDEO_COMPLETE` 可能包含等待用户交付文件，字段故意命名 `ready_to_receipt_seconds_record_only`。真实视频采集没有接入，输入耗时为未测；文件接收后的哈希核验已计入场景准备。

当前预热加载权重、依赖及会话，不执行虚构视频推理；后端算子第一次执行的惰性开销仍计入场景准备。CLI 验收一次视频；运行时类可复用，但多视频服务循环尚未做板端验收。

## 渲染驻留的确切范围

`attributes_resident.cpp` 引用冻结投影源文件，通过 `LOAD` 原子替换完整 Gaussian rows，通过 `PROJECT` 改相机。场景只加载一次，后续投影复用数据。`render_resident.cpp` 保留 FPGA SDK 设备句柄和互斥锁，按有序命令处理帧。

静态高斯驻留在 ARM 进程，**不是所有场景常驻 FPGA BRAM**。视角相关投影仍重算；分组/排序仍调用冻结程序；渲染每次的任务列表和工作区仍有准备/分配成本。没有新增双缓冲算法，原冻结渲染核内部调度保持原样。已验证 7→15→22→7 连续换视角与冻结 FPGA 哈希一致；实测命令到完整帧约 0.8 秒，详细数据见实板报告，尚非实时交互。

## 实板命令

先核实旧中断任务状态，不能因没有收到结果就再次同时启动。随后在隔离候选目录构建，不覆盖冻结包：

```sh
sh mvsplat/initialize/build_resident.sh /root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928
```

本机已准备 `board_accept.py` 包含既有 ARM 环境、进程树 RSS 650 MiB / 系统预留 128 MiB / 600 秒监控及取回证据。从 `mvsplat` 目录运行：

```text
python -m initialize.board_accept --board-root /root/fpga43dgs_reconstruction/NEW_CANDIDATE --run warm_cpu_serial --out ../runs/NEW_ACCEPTANCE --threads 4 --prepare-threads 1 --serial-prepare
```

当前默认四线程推理、串行准备；用新 run/out 和 `--threads 3 --overlap-prepare` 做重叠对照。原两线程串行可用 `--threads 2 --serial-prepare` 回退。主线程执行 Torch，单工作线程执行准备。重叠模式拒绝超出 CPU 核数的显式线程预算，不再让四线程 Torch 与四线程 OpenCV 叠加。独立 NPU worker 的线程/SDK 内存仍需额外实测。

省略 `--video` 的板端 `pipeline.py initialize` 会先打印 READY，再从 stdin 接收一个已完成的视频路径；预置文件模式在 READY 后立即交付文件。运行环境须沿用 `board_accept.py`，不要漏掉该板所需的私有库设置。

网络断开会保存原运行错误及“结果未知”，证据收集错误不会覆盖原错误；不会自动重跑。监控终止进程不等于已证明设备能无损恢复，后续先检查 SDK/设备状态。

## 验收顺序

1. 冷路径与预热串行：同线程、同输入、相同位姿与输出合同，核对哈希/画质；分别记录预热和场景准备。
2. 预热串行与 PnP/推理重叠：不能只看重叠秒数，要比较首帧总时延、峰值内存及最小系统余量。
3. 同一驻留场景连续改变相机，回到原相机检查帧哈希；排除旧场景、旧帧缓存和任务串号。
4. 再接入已通过独立数值验证的 NPU 子图。CPU/NPU/FPGA 共存、SDK 地址资源和真实 SFB 布局均需在板上检查。

NPU 预热现在强制先在真实保留会话上运行两轮 oracle 校验，全部通过才加载整网并发布公共 READY。默认 oracle 目录是编译包旁边的 `graphs/`，也可显式传 `--oracle-graphs`。部署 v2 的 TF32 图目前未通过此门槛，不能作为正式初始化后端。

连续换视角验收入口：

```sh
python -m initialize.validate_resident --scene RUN/renderer_input --renderer FROZEN_PACKAGE --out NEW_VALIDATION
```

它执行同一场景三目标视角及首视角回访，再分别运行冻结 CPU Dense/FPGA 对照，结果在 `validation.json`；常驻和独立进程的计时范围明确区分。
