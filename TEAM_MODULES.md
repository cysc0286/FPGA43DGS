# 三人模块分工与交接合同

更新：2026-09-28。当前代码分为四个模块；按三人组织时，将视频输入与位姿估计合为一组。最终部署目标仍是悟净 30TAI Lite，电脑仅承担开发、构建与参考验证。

## 当前接口做到了什么

主线：视频 → FrameSet → PoseSet → GaussianScene → RenderInput → RenderResult。

已有独立入口 `examples/3dgs_reconstruction/pipeline.py` 和文件校验器 `modules/contracts.py`，字段详见 [INTERFACES.md](examples/3dgs_reconstruction/modules/INTERFACES.md)。可复制文件交接，消费者不用重跑上游计算。本次补充 `validate --only`，使只有最小交接文件的接收人也能独立验收；原 `--through` 整段检查保留。

这是离线文件接口，适合当前阶段的独立开发和回归。它没有实现摄像头连续流、跨进程共享内存、三设备零拷贝、在线增量建图，或者完整的跨版本协议协商。校验通过说明当前文件结构和检查范围满足合同，不等价于重建准确、实板性能合格或所有文件已建立完整溯源。

## 三人职责

| 负责人 | 模块及代码范围 | 输入 → 交付物 | 首轮任务与验收 |
|---|---|---|---|
| A：视频与位姿 | `modules/video_input`、`modules/pose_estimation`；`npu_frontend` 为可选候选 | 视频 → `project/`、`project.json`、`sfm.json`，构成 PoseSet | 固定视频的 ARM 抽帧/SfM 可复跑；比较注册率、重投影误差、耗时和峰值内存；保留原 CPU 对照 |
| B：高斯重建与资源控制 | `modules/gaussian_generation`、`bounded` 及隔离的 OpenSplat 补丁 | PoseSet → 原始 `splat.ply`、`cameras.json`；再导出 `renderer_input/` | 首要补齐 ARM 训练构建、受限内存实际训练、模型与质量核验；电脑成功不能代替板端完成 |
| C：渲染与板级集成 | `modules/rendering`、`examples/3dgs_flicker_hw`，相关构建/传输代码 | RenderInput ＋目标相机 → `frame.bin`、`frame.ppm`、`result.json` | 固定模型/相机对比优化 CPU 与 FPGA；优化前处理、搬运和硬件；轻量候选独立构建，保留冻结四单元版本 |

B 同时负责训练结果到冻结渲染 ABI 的导出适配。C 不再要求 A/B 生成每个像素的 alpha/RGB 命令；模块交接是高斯模型和相机，渲染侧自己完成投影、分组及后续求值。C 内部的 CPU/FPGA 数据接口继续使用现有后端协议。

三人按功能负责，不按“一人一个芯片”分配。B 的 ARM 训练是当前最大交付风险，A/C 在各自模块稳定后可协助依赖、内存剖析和质量验证；不能因为三组人数相同就假定工作量相同。

## 设备放置建议

| 环节 | 默认部署 | 原因与后续候选 |
|---|---|---|
| 视频解码、抽帧、帧管理 | ARM CPU/现有视频软件栈 | 先支持已有视频；实时采集和硬件解码能力需另行核验，不承诺当前已有 |
| 特征提取/匹配、位姿求解、三角化及优化 | ARM CPU | 当前 COLMAP CPU 路线已有板端记录；动态匹配和求解保留 CPU。NPU 可研究规则矩阵计算或学习特征推理，但必须实测整段收益 |
| 高斯初始化、优化、增密/裁剪、导出 | ARM CPU 参考路线 | 现有训练代码使用 OpenSplat CPU；完整 ARM 训练尚未通过。控制点数、分辨率、缓存和线程是当前优先工作 |
| 模型读取、投影/协方差/SH、Tile 分组和深度排序、任务提交 | ARM CPU | 已有渲染前处理和调度实现；先剖析常驻缓存及静态量复用，再决定将哪个热点下沉 FPGA |
| 候选筛选、高斯求值、alpha、提前终止、有序颜色合成 | FPGA | 复用现有 FLICKER 路线。资源、频率、传输和完整帧时延一并比较，保留软件 Golden |
| NPU | 默认不插入主链 | 当前没有已验收的 NPU 高斯训练器；现有前端匹配候选暂未形成端到端收益 |

已有 FPGA 渲染器只做正向渲染；训练需要图像损失、梯度、参数更新及动态点管理，不能直接把它当作完整训练加速器。若 B 后续向 FPGA 迁移训练热点，需先明确前向/反向数值合同及缓存需求，再与 C 统一资源和构建，不能各自修改同一份平台工程。

NPU 的具体反例来自 [2026-09-28 板端记录](examples/3dgs_reconstruction/npu_frontend/BOARD_VALIDATION.md)：三对固定图像上 CPU 均值 225.514/231.207/320.303 ms，NPU 常驻加 CPU 精确重算 668.218/672.284/961.959 ms，耗时比 2.963/2.908/3.003。最终 2194 个匹配与整数 CPU 参考一致，但这只是三对局部接入；共享初始化单列，未运行完整 NPU SfM。这里引用历史数据，本次未重测，不能外推所有 NPU 方案均无收益。

后端两单元轻量化由 C 独立尝试，四单元冻结包不覆盖。减少渲染并行单元不等于整板资源减半，也不等于增加 Linux 内存。前端是否使用 FPGA，应由具体热点和资源报告决定；当前功能闭环不以增加前端 FPGA 核为前置条件。

## 最小交接内容与独立运行

A → B 的运行目录只需：

```text
POSE_RUN/
  project/images/*.png
  project/sparse/0/cameras.bin
  project/sparse/0/images.bin
  project/sparse/0/points3D.bin
  project.json
  sfm.json
```

B → C 的运行目录只需：

```text
RENDER_RUN/renderer_input/
  model.ply
  manifest.json
  novel_midpoint.bin      # 默认目标相机；也可交付其他显式指定的相机
```

不互传 Python 内存对象或上游工作目录绝对路径。相机坐标、矩阵方向、尺度、SH 排列、opacity/scale 激活语义及留出帧范围不允许各组私自改变。指定新相机时，使用 `render --camera NAME.bin --plan` 验证该相机。

从项目根目录执行；`python` 指对应开发机/板端已经配置的环境，运行目录应复制到自己的候选目录，禁止在冻结样例中直接覆盖结果：

```text
python examples/3dgs_reconstruction/pipeline.py validate --run POSE_RUN --only pose
python examples/3dgs_reconstruction/pipeline.py gaussian --run POSE_RUN --opensplat BOUNDED_OPENSPLAT --plan
python examples/3dgs_reconstruction/pipeline.py validate --run RENDER_RUN --only export
python examples/3dgs_reconstruction/pipeline.py render --run RENDER_RUN --renderer FROZEN_PACKAGE --out NEW_OUTPUT --backend fpga --plan
```

去掉 `--plan` 才执行计算；FPGA 渲染必须在已配置正确后端的平台执行，训练器必须匹配当前资源 ABI。`--plan` 不是编译、训练或上板成功证明。

固定测试输入可从 `examples/3dgs_reconstruction/runs/module_refactor_smoke_20260928` 复制对应产物；其视频、位姿、模型来源和电脑 CPU 验证见 [上一轮记录](examples/3dgs_reconstruction/evidence/module_refactor_20260928/VALIDATION.md)。B/C 不必等 A 提交新实现，可以先使用这份固定输入开展本模块优化。这不是完整数据集或所有场景的质量保证。

## 周期联合验收

建议每轮开始冻结输入和指标口径，每轮合并前做一次联合验证；默认每周至少联调一次，中途有可运行增量就提前接入。每人都提交实际代码、命令、原始结果和优劣分析，不能仅交方案。

1. **A 单独验收：**新 A ＋固定 B/C。检查帧标识、注册率、稀疏几何、重投影误差、完整模块时延和 RSS；登记失败帧。
2. **B 单独验收：**固定 PoseSet ＋新 B ＋固定 C。记录高斯数、迭代/总时间、RSS、对留出照片的 PSNR/SSIM；点数变少不自动等于质量合格。
3. **C 单独验收：**固定模型/相机 ＋新 C。检查对应精度软件参考、图像一致性和 Pose→Framebuffer 总时延；资源/时序与四单元冻结版同口径对照。
4. **联合验收：**新 A → 新 B → 新 C。视频、各边界文件、最终图像和日志保留同一 run 标识与哈希记录；区分电脑参考、ARM 执行和 FPGA 执行。

当前 B 的 ARM 训练仍未验收，因此电脑训练＋板端渲染只能叫分阶段联调，不能宣布视频到图像已全部板端完成。联调先串行顺序执行，避免 SfM、训练、渲染同时占用受限内存；需要流水重叠时另立共享内存和资源竞争验收。

公共 `contracts.py`、`pipeline.py` 和接口文档由 C 维护合并入口，变更需相关生产/消费两方一起复核。新增字段或数值变化要明确兼容策略，不私自改冻结二进制 ABI；所有权只是协调安排，不限制 A/B 修复各自发现的问题。个人优化分支独立保留，渲染 BOOT/位流构建和加载由 C 统一操作，避免同时写设备。

本地代码整理不等于已同步 GitHub；开始多人远程协作前需明确提交版本并提供所需输入包。大体积模型、环境和位流使用带 SHA256 的独立包，不假设 `git clone` 自动获得本地 `runs/` 或私有依赖。当前文档不声称已推送、已授予仓库权限或已生成新硬件。

## 本次增量验证

本次仅补独立交接验收及分工文档。实际回归结果见 [验证记录](examples/3dgs_reconstruction/evidence/team_interfaces_20260928/VALIDATION.md)。不重新计入历史训练画质或板端速度，不声称新增加速；性能、资源、时序、DMA 字节和功耗本轮未测。
