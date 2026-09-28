# 视频重建前端：NPU 特征匹配候选

## 2026-09-28 实板调试更新

已在 30TAI Lite 的 ZG330 实际运行矩阵乘法。修复 SDK 安装目录识别、GNU 宏兼容、融合 HardOp 的真实 Matmul 绑定检查，并新增常驻会话 `board/npu_session.cpp` / `native_backend.py`，避免每批启动进程和读写任务文件。

**原 v1 全分数误差门槛失败，不能把本轮称为 v1 数值验收通过。** TF32 输出在真实数据上的最大绝对误差约 0.0017–0.0020，超过原 0.0002。独立 v2 候选通过 `refine.py` 对有歧义的候选做 CPU 精确点积：三个图像对的 2,194 个最终匹配逐项一致，接回 COLMAP 后保留 2,152 个几何内点。0.003 误差预算仅在这批完整输入上验证，不能当作任意视频的硬件保证，默认 CPU 链保持不变。

常驻候选仍比同板 CPU 慢：三对均值分别为 668.2/672.3/962.0 ms，CPU 为 225.5/231.2/320.3 ms。它是已运行、可回溯的负结果，不是加速版完整 SfM。实测详情见 [BOARD_VALIDATION.md](BOARD_VALIDATION.md)，源数据在 `../evidence/board_debug_20260928`。

以下保留 v1 离线准备时的说明和复跑方法，其中“未上板”是当时的状态；当前以本节和实板报告为准。

2026-09-28。按用户要求，先准备后续迁移代码，板端测试等用户返回后进行。本目录独立于已经验证的 CPU 重建链和冻结渲染器。已完成软件数值检查、COLMAP 匹配接口验证及 ICraft 编译；**尚未在 NPU 上执行**。

选择的第一个模块是 **SIFT 描述子匹配中的矩阵点积**。同一个点从不同画面看过去，会产生特征描述子；这里批量计算两张图的特征相似度，辅助 CPU 找到对应点。它属于相机位姿恢复的前端，不是高斯训练，也不是高斯渲染。

```text
视频帧 → CPU 提取 SIFT
                 ↓ 两组 uint8 描述子，每个 128 维
           CPU 分块、转置、除以 512
                 ↓ 两个动态输入；每换一帧不需要重新编译
           NPU 候选：分块矩阵乘法
                 ↓ 每块 FP32 相似度矩阵
           CPU 合并双向 Top-2、角距离比值筛选、互相匹配
                 ↓ COLMAP 数据库独立副本
           CPU 几何验证 / SfM → CPU 高斯优化 → 冻结渲染后端
```

默认 CPU 参考链不变。这个候选遵循 COLMAP 4.2 **CPU brute-force** 匹配规则，现有主流程使用的默认索引匹配器是另一种搜索路径；两者不能直接混算纯硬件加速。软件测试对照的是官方同规则 brute-force 实现。

|代码|职责|
|---|---|
|`matcher.py` / `backends.py`|流式分块、双向 Top-2、NumPy/ONNX CPU 后端及独立整数参考|
|`prepare_cases.py`|只读提取真实 COLMAP 描述子、图像编号与输入哈希|
|`export_graph.py`|导出双动态输入 MatMul ONNX；没有场景权重|
|`compile_icraft.py` / `audit_graph.py`|五阶段 PC 编译及实际生成图的算子、布局、哈希检查|
|`block_io.py`|打包板端任务、逐块读取结果、验证数值和最终匹配|
|`board/block_runner.cpp`|单个 ICraft Session 连续处理所有块，记录实际绑定及分阶段时间|
|`build_runner.py` / `run_board.py`|本机构建、默认仅生成运行计划、显式 ARM 执行和证据采集|
|`colmap_bridge.py`|独立数据库中比对官方匹配或写入候选匹配，再做几何验证|
|`validate.py` / `test_*.py`|真实输入、尾块、重复匹配、截断、乱序、篡改等检查|

验收结果、资源代价和待测项见 [VALIDATION.md](VALIDATION.md)。已归档的输入、编译产物与日志见 `evidence/v1`。`runs` 中每次执行都使用新目录；失败日志也保留。

## 在电脑上复跑

以下 PowerShell 命令从本目录运行。现有 NPU 虚拟环境提供 ONNX/ORT，重建虚拟环境提供 pycolmap 4.2，两者无需合并。`prepare_cases.py` 可重新提取数据库，但下面直接使用已冻结的真实描述子。

```powershell
$py = 'D:\ADProjects\FPGA43DGS\npu_3dgs\.venv\Scripts\python.exe'
$sfmPy = 'D:\ADProjects\FPGA43DGS\examples\3dgs_reconstruction\.venv\Scripts\python.exe'
& $py export_graph.py --output generated/graph_new --tile 128
& $py validate.py --cases evidence/v1/cases --graph generated/graph_new --output runs/offline_new
& $py compile_icraft.py --graph generated/graph_new --output runs/compile_new --dtype tf32 --execute
& $py build_runner.py --kind cpu --output runs/build_cpu_new
$env:HGS_BLOCK_RUNNER = (Resolve-Path runs/build_cpu_new/block_runner_cpu.exe).Path
& $py -m unittest test_matcher test_block_io -v
& $py block_io.py pack --case evidence/v1/cases/pair_000_001.npz --output generated/jobs_new
& $sfmPy colmap_bridge.py --database ../runs/bounded_sfm_final_check/database.db --cases evidence/v1/cases --matches runs/offline_new --output runs/colmap_new --mode inject --verify-geometry
```

当前成功编译产物已在 `evidence/v1/compiled`，无需为了后续上板重复编译。128×128 是当前验证形状；代码还支持导出32/64/256，但这些形状未经过 ICraft 编译/上板验收。

## 回来后进行板端验证

这一节是待执行步骤。本轮未连接板卡。将本目录源文件、`evidence/v1/compiled`、上面的 `generated/jobs_new` 复制到板上一个新目录；不需要替换 BOOT。板端脚本仅依赖 Python 标准库，数值验证可以回到电脑执行。

在板端该目录运行；SDK 路径沿用历史安装位置，执行前需检查其仍存在：

```bash
python3 build_runner.py --kind npu --sdk /root/heterogs_npu/sdk_3.36.1 --output runs/build_arm
python3 run_board.py --runner runs/build_arm/block_runner_npu --model evidence/v1/compiled/sift_pair_dot_ZG.json --raw evidence/v1/compiled/sift_pair_dot_ZG.raw --bundle generated/jobs_new --output runs/board_first --execute --timeout 180
```

`run_board.py` 去掉 `--execute` 只生成计划；它没有 SSH、串口或远程更新功能。实际执行必须在 ARM64 Linux。程序拒绝覆盖旧目录，核验输入哈希、形状和实际算子后端，要求 Matmul 本身绑定 ZG330；不能仅凭存在搬运 HardOp 就认定计算在 NPU。

把 `runs/board_first` 取回电脑后：

```powershell
& $py block_io.py verify --bundle generated/jobs_new --output runs/board_first/runner/scores.bin --report runs/board_first/precision.json
```

正确性要求：所有有效/填充项的绝对误差 ≤ 0.0002，最终双向匹配索引完全相同。硬件结果过关后才在更多图像对上复跑，随后接独立 SfM 分支。`verify` 的 `npu_executed=false` 表示文件校验本身不能证明执行设备，须和 `run_board.py` 的运行记录、绑定表、同一输出 SHA 联合判断。

首轮只做功能与运行位置验收。正式速度对照还需同一 ARM、同一输入、同一匹配策略，记录原默认 CPU、官方 brute-force CPU、相同分块 CPU、NPU 四条路径；前两项区分算法代价，后两项定位后端收益。至少预热后重复测量，分别统计初始化、打包、SDK调用、等待/回读、Top-2和总耗时的均值/中位数/P95/P99、峰值RSS与CPU占用。当前 C++ 程序有三次预热、逐块计时和进程计时，尚未包含完整 Top-2/SfM 耗时，因此不能直接拿其数字宣称整链加速。

## 数据与数值合同

- 描述子：COLMAP 4.2 SIFT `uint8[N,128]`，不重新按向量范数归一化；输入为 `uint8/512`，右矩阵转置。
- ONNX：`[1,1,T,128] × [1,1,128,T] → [1,1,T,T]`。ICraft 输入为 NHWC `[1,T,128,1]` 和 `[1,128,T,1]`；单通道使平坦字节顺序相同。编译输出布局是 `***C`，不能擅自当成 NHWC 图像。SDK `SFB` 导出顺序仍需实板数值验收。
- 角距离 `acos(min(dot,1))`；最大距离0.7，严格 ratio `<0.8`；正反方向都做 ratio 再互检。相同分数按原全局下标处理；零分数无匹配。`scalar_reference` 是测试用全矩阵整数参考，运行候选不分配全相似度矩阵。
- 任务文件小端：24字节头 `<8s4I>`，magic=`HGSJOB01`，其余为 tile、块数、两图特征数。每块16字节 `<4I>` 为行起点、列起点、有效行数、有效列数，后跟两个连续 FP32 输入。结果同头布局、magic=`HGSOUT01`，每块索引后跟 `T*T` 个 FP32 分数。
- 输入/输出、编译图、运行程序均留哈希。数据文件是第一轮验证接口，含重复输入和文件读写；它尚不是帧间常驻缓存、零拷贝或 CPU/NPU/FPGA 三方流水。

## 方法来源

匹配规则依据 [COLMAP 4.2.0 的 SIFT 实现](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/feature/sift.cc)，自行编写分块和后端适配，并调用官方实现核对；矩阵图遵循 [ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)。这属于既有 COLMAP 前端的部署候选，不是新换了一篇 3DGS 论文，也未修改 FLICKER 渲染路线。
