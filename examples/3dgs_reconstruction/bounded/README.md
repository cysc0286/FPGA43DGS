# CPU 重建的受限资源执行版本

本增量仍沿用视频帧 → COLMAP CPU → OpenSplat CPU → PLY/FLCAM001，补齐资源控制、运行记录和 ARM 测试准备。新源码由 OpenSplat 提交 `62fd86fe3b62644b65890557ad9bf397abdc7cfb` 单独生成，原 `vendor/OpenSplat/build_cpu`、旧模型和历史指标保留。

2026-09-28 后续用户已授权实板调试。隔离 ARM 环境已安装，视频抽帧和 CPU SfM 已实跑通过（30/30 帧、1076 个初始点、峰值480.70MiB）；完整训练仍在核验。当前证据见 `../evidence/board_debug_20260928`，以下离线准备记录不再代表设备未连接。

ARM 环境使用 `arm_environment.sh`，仅对当前 shell 设置私有 Python/动态库路径。板端 glibc2.31 下 OpenCV 需预加载同环境的 `libgomp.so.1`，否则静态 TLS 导入失败。`setup_arm.sh` 消费已验证的离线包；不替换系统 Python。

`build.py` 通过短命子进程查询 LibTorch，避免构建协调器在整个编译期间持有约200MiB运行库；原生编译也受RSS/超时监控。更换编译器必须使用新的 `--build-dir`。GCC 首次构建触发560MiB上限，已保留失败记录，Clang独立构建在核验中。编译峰值不是训练峰值，不能混写。

## 代码与资源边界

|入口|作用|
|---|---|
|`prepare_source.py`|从锁定 Git 提交导出隔离源码，生成逐文件补丁和 SHA 清单|
|`resource_limits.hpp`|显式约束 LibTorch/OpenCV 线程、光栅线程、图像预处理并发、缓存与预取槽位|
|`build.py`|Windows/Linux CPU 构建，ARM 建议单编译任务、关闭 PCH；构建日志独立保存|
|`run.py`|配置驱动的整链入口，逐阶段执行与完整输出哈希验收|
|`monitor.py`|50 ms 采样进程树 RSS，超时、低可用内存、超限及取消时终止任务并保存日志|
|`preflight.py`|只读本机环境清单；`--probe` 才实际导入依赖并检查可执行程序|
|`test_controls.py` / `check_binary.py`|执行器回归和实际 C++ 训练限额功能检查|
|`package.py`|准备 Linux 源码/视频/测试输入包并逐文件核验压缩包|

上游默认压缩图像缓存可用物理内存的 90%，解码槽位会根据剩余内存增长。候选改成显式预算，且预处理最多同时处理一张图。压缩缓存允许当前正在使用的单个 blob 超过预算，梯度预算只约束光栅反向的工作线程副本；二者都不是整个训练进程的硬内存上限。RSS 监视器存在采样延迟，共享页也可能重复计数，不能替代内核内存限制或保证不会发生 OOM。

原上游 `--max-gaussians` 在剩余预算大于零时仍可能一次增密超量。启用 `strict_gaussians` 后，先保留原候选索引顺序中的 clone，再保留预算允许的 split，连删除父节点之前的临时参数数量也不超过上限。初始稀疏点超限会拒绝执行，不会悄悄删点。此策略在预算紧张时改变训练轨迹；线程数改变也可能改变梯度累加的浮点舍入，不能称为训练结果逐位等价。

## 两个待比较配置

|设置|`cpu_control`|`arm_candidate`|
|---|---:|---:|
|输入抽帧|30|30|
|SfM 最大宽度 / SIFT 特征上限|640 / 4096|480 / 2048|
|训练图像宽度|320|160|
|训练步数 / Gaussian 上限|1000 / 20000|1000 / 6000|
|LibTorch、光栅线程上限|8、8|2、2|
|解码线程 / 预取槽位|1 / 4|1 / 4|
|压缩缓存 / 光栅反向副本预算|16 / 16 MiB|8 / 8 MiB|
|RSS 停止阈值 / 系统剩余内存阈值|3072 / 128 MiB|560 / 128 MiB|
|单阶段超时|4 小时|4 小时|

这是预先固定的候选参数，不是新的实测性能。降低分辨率、特征数和点数会改变工作量，可能影响注册率、细节与画质；不能把两配置之比当作同质量硬件加速。`cpu_control` 保留旧工作负载尺寸，仍需在同一机器重新测量才能比较调度变化。

## 电脑上的入口

在项目 `examples/3dgs_reconstruction` 中执行：

```powershell
.\.venv\Scripts\python.exe bounded\prepare_source.py
.\.venv\Scripts\python.exe bounded\build.py --jobs 3
.\.venv\Scripts\python.exe bounded\test_controls.py
.\.venv\Scripts\python.exe bounded\run.py --profile arm_candidate --video data\nyu_snippet_curl.mp4 --opensplat vendor\OpenSplat_bounded_v1\build_bounded\opensplat.exe --run runs\arm_candidate_pc_01 --plan
```

源码目录已准备过时不重复执行 `prepare_source.py`，也不要覆盖它。`--plan` 只显示配置/命令，不创建运行目录。正式测试时去掉 `--plan`。先用 `--through sfm` 可仅验收位姿阶段；同配置、同路径补 `--resume` 可以核对已完成输出后继续训练。若此前阶段失败或中断，保留日志并使用新运行目录；目前不支持恢复 Adam 优化器状态。缺失、改动的输出或修改过的程序不能当作已完成阶段跳过。

完成整链时自动计算同显示域 PSNR/SSIM。`collect_evidence.py --runs <运行目录>` 可归档，CSV 追加新版本、拒绝重名，不覆盖旧两次基线。

## 回到板卡后按顺序测试

以下命令在解压后的候选包根目录执行，**本轮没有运行这些板端命令**。所有结果写新目录，不需要更改 BOOT 或 FPGA 设计。

1. 先用系统 Python 只读盘点，旧 Python 也可运行：

   ```bash
   python3 bounded/preflight.py --output evidence/arm_inventory_01
   ```

   上次盘点是 Python 3.8、Linux 可见约 993 MiB、可用约 750 MiB、无 swap，这些是历史状态，回去后需刷新。候选依赖要求独立 Python 3.10+ 环境、CPU Torch 2.7.1、pycolmap 4.2.0、NumPy、OpenCV Python、plyfile、psutil。不能把 Windows wheel 或其锁文件复制到 ARM 当作可安装依赖。AArch64 wheel、系统 glibc/libstdc++、OpenCV C++ 库和依赖是否适配，尚未验收；检查失败时先解决环境，不能跳过后称训练可运行。

2. 在适配好的独立 CPU 环境中构建。要求 CMake ≥3.21、Ninja、支持项目 C++20 的工具链及匹配的 OpenCV C++ 开发库。不要直接在仅有约 1 GiB RAM 的板上假定 LibTorch 编译会成功；可在兼容 AArch64 构建机单任务构建后复制，必须匹配板端 ABI。

   ```bash
   python bounded/build.py --source source/OpenSplat --jobs 1
   python bounded/preflight.py --probe --opensplat source/OpenSplat/build_bounded/opensplat --output evidence/arm_probe_01
   ```

   构建脚本不自动安装软件或改系统设置；无依赖时会明确停止。`ready_for_attempt=true` 只代表所检查依赖可用，仍不代表整链能装入内存。

3. 分阶段进行首次板端尝试，先看位姿和内存，再继续训练与导出：

   ```bash
   python bounded/run.py --profile arm_candidate --video data/video.mp4 --opensplat source/OpenSplat/build_bounded/opensplat --run runs/arm_candidate_01 --through sfm
   python bounded/run.py --profile arm_candidate --video data/video.mp4 --opensplat source/OpenSplat/build_bounded/opensplat --run runs/arm_candidate_01 --resume
   ```

   任一资源门槛触发就保留失败证据，不自动加 swap、提高上限或重启。SfM 至少注册 80% 输入帧且不少于 5 帧才继续。训练完成后导出 `renderer_input/model.ply`、各输入相机和 `novel_midpoint.bin`，继续调用冻结渲染后端。中点视角无真实 GT；留出帧仍参与 SfM，不是严格独立重建评估。

## 验收时必须补的指标

记录同一输入 SHA、源码/程序 SHA、配置与执行平台；逐阶段报告时间、RSS、注册率、重投影误差、初始点与最终 Gaussian 数；报告留出帧 PSNR/SSIM、导出相机与 PLY 一致性。整链至少多次重跑后才汇总均值和波动，少量样本不输出无意义的 P95/P99。功耗、LPIPS、ARM 性能、新 FPGA/NPU 性能目前均未测。小规模功能检查与 1000 步正式实验分开保存，不把不同尺寸、点数或步数直接换算成加速比。
