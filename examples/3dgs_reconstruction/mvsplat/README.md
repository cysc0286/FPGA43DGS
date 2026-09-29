# MVSplat 板端重建主线

本目录将固定权重 MVSplat 作为当前主线的高斯生成实现。完整路径使用真实 COLMAP 产物；快速候选直接从视频估计两视图几何。两者输出相同的 `model.ply`、`FLCAM001` 相机和 `manifest.json`，接入冻结 CPU+FPGA 渲染包。原 OpenSplat 分阶段训练代码保留为历史对照。

当前两种板端路径均已执行成功。完整路径三目标视角质量过门槛，历史入口到首图为 249.43 秒；快速单进程候选历史入口到首图为 66.68 秒，但第一视角 19.45 dB，未过 20 dB 门槛。新增的板端“输入结束到首张图哈希核验”同环境对照为两线程 74.04 秒、四线程 71.74 秒（短 3.1%）；四线程另一次探索记录为 69.88 秒。每配置样本很少，且视频已预置于板端。**一分钟与三视角质量同时达标尚未实现**。详见 [快速候选验证](FAST_VALIDATION.md) 与 [完整基线验证](VALIDATION.md)。

板端目标链：

```text
SD 卡内视频
  → ARM 抽帧
  → ARM COLMAP 位姿和稀疏几何
  → ARM MVSplat 预训练推理
  → ARM 世界坐标高斯参数导出
  → ARM 投影、Tile 分组与排序
  → FPGA 求值、筛选与有序合成
  → 指定相机图像
```

此路线没有逐场景反向训练。MVSplat 的已有权重仍来自作者预训练；本目录当前 CPU 实现不等于 NPU 部署。视频是存储的视频文件，不是实时摄像头/在线 SLAM。

## 固定来源

- MVSplat：`donydchen/mvsplat`，提交 `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`，MIT，许可保留在第三方源码中。
- 官方 `re10k.ckpt`：48,019,233 字节，SHA256 `83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a`；来自作者 README 所列 Google Drive 目录。权重保存在被忽略的 `../models/mvsplat`，不提交到代码仓库。
- `infer.py` 严格加载 471 个 encoder 状态张量。`runtime.py` 只跳过训练、可视化和数据集注册表的包初始化，实际编码器层来自锁定源码，没有用假层代替网络。
- `memory_runtime.py` 在运行目录生成独立源码副本和补丁：把 128 个深度候选按 16 个分块投影、相关性求和。不减少候选、不改权重/层/数值精度。原源码保持不动。

## 文件和接口

|入口|作用|
|---|---|
|`prepare.py`|读取板端或电脑 SfM；去畸变图像中心裁剪，变换内参；输出 `context.npz`、相机及哈希清单|
|`prepare_fast.py`|视频首尾帧 SIFT/Essential 几何与目标 PnP；相机内参当前为显式近似，输出相同上下文接口|
|`reconstruct_fast.py`|单进程调用快速准备、相同网络和导出，减少重复导入；不是常驻模型服务|
|`infer.py`|固定权重前馈；输出世界坐标 means/covariances/SH4/opacity、阶段时间和 RSS|
|`export.py`|世界协方差分解为 scale/quaternion，逆激活 opacity，输出既有 62-float PLY 和相机合同|
|`reference.py`|独立 CPU 世界协方差渲染参考；用于定位转换和 SH 截断误差，不是作者 CUDA 渲染器|
|`board_run.py`|对已有模型进行实板 CPU Dense/FPGA 对照，预热和测量分别保留|
|`deploy.py`|校验后上传隔离代码、权重、ARM 离线依赖；不修改系统 Python/旧环境|
|`board_pipeline.py`|在 ARM 从视频重新执行整个流程并测量；电脑仅用于调度、取回结果|
|`measure.py`|进程树 RSS、系统余量、超时监控；每个阶段结束退出以释放内存|
|`evaluate.py`|分开报告重建质量、硬件数值差别、SH4→SH3 损失及同范围速度|
|`evaluate_fast.py`|快速板链目标画质、CPU/FPGA 一致性、局部误差和效果图；质量失败仍保留报告|
|`test_bridge.py`|世界协方差、坐标半像素、SH 顺序和提前终止规则测试|
|`test_fast.py`|自动帧选择及非法索引拒绝；30 帧配置保持原索引|

训练数据归一化与相机约定：SfM 的单目尺度不固定。输入以第一参考相机作为坐标原点和朝向，将可见稀疏点的中位深度缩放至 10，保留作者近远界 1/100；精确相似变换写入 `input.json`。所有相机和高斯使用同一变换。三张目标图不输入 MVSplat，但已参与此前的 SfM，因此不是完全未见的几何测试集。

作者输出的协方差已在世界坐标；原导出可视化旋转仍在局部相机坐标。本接口从世界协方差求特征分解，序列化后再重建协方差检查。MVSplat 像素中心 `(i+0.5)/size` 对应冻结渲染器整数像素坐标与主点 `(size-1)/2`。

冻结包只支持 SH3：保存完整 SH4 `.npz`，仅在 PLY 导出截断四阶项；必须用独立参考量化损失，不能默认为无损。`acceptance.json` 保存本轮工程门槛，失败配置保留。

## 运行

电脑端使用隔离的 `.venv_mvsplat`；板端复用已配置的 ARM CPU Torch/pycolmap/OpenCV，其他轻量依赖安装至该候选的 `deps/`。板端执行前应设置私有库环境和 `PYTHONPATH`，同时保留至少 128 MiB 系统余量。不要对系统 Python 安装 CPU PyTorch，也不要启动未经测量的 256 输入配置。

```text
python mvsplat/prepare.py --pose-run POSE_RUN --out INPUT --size 128
python mvsplat/infer.py --input INPUT --weights weights/re10k.ckpt --vendor vendor/MVSplat_reference --out INFERENCE --threads 2 --depth-chunk 16
python mvsplat/export.py --input INPUT --inference INFERENCE --out RENDER_INPUT
```

板端整链统一入口（在仓库根目录执行）：

```text
python examples/3dgs_reconstruction/pipeline.py reconstruct --video /path/to/video.mp4 --out NEW_RUN --size 128 --threads 2 --repeats 3
```

快速候选使用同一入口，追加以下参数：

```text
python examples/3dgs_reconstruction/pipeline.py reconstruct --video /path/to/video.mp4 --out NEW_FAST_RUN --size 128 --threads 4 --pose-mode fast_pair --fused --focal-ratio 0.9 --repeats 1
```

`--repeats` 控制渲染验证次数，不改变首次出图的计时边界。快速路径默认按视频长度选首尾参考帧与四分之一、中点、四分之三目标帧；需要至少五个不重叠索引。它仅从一对图像生成局部高斯，不包含全视频多组融合、覆盖率估计或任意长视频的完整场景建图。第一/最后帧无共同视野、纯旋转、运动物体或弱纹理可能导致失败；不要因此降低几何内点门槛。

如需采用用户约定的计时口径，在板端收到视频最后一帧、文件写入完成时记录板端 Unix 秒时间戳，并给上述入口追加 `--input-ended-at-epoch TIMESTAMP`。`pipeline_result.json` 的 `input_end_to_first_verified_fpga_frame_seconds` 以该事件为起点，以指定目标视角第一张 FPGA 帧生成并通过 SHA256 校验为终点。未传时间戳时此值缺失，历史 `video_to_first_fpga_image_seconds` 只表示旧版入口到首图，不能当作输入结束时间。新测试用预置视频模拟接收完成事件，不计视频复制/上传；当前 2 视图局部模型也不代表整段视频覆盖完成。

板端已部署目录为 `/root/fpga43dgs_reconstruction/mvsplat_arm_20260929`。进入该目录后设置现有私有 CPU 环境：

```sh
export PATH=/root/fpga43dgs_reconstruction/arm_env/bin:$PATH
export LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib
export LD_PRELOAD="/root/fpga43dgs_reconstruction/arm_env/lib/libgomp.so.1 /root/fpga43dgs_reconstruction/arm_env/lib/python3.11/site-packages/torch.libs/libgomp-58a43326.so.1.0.0"
export PYTHONPATH="$PWD/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

复跑四线程快速候选时，同步设置 `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`，并传 `--threads 4`。最新运行记录保存解释器、显式线程数和这些白名单环境变量；预加载两个 OpenMP 库是本轮受控对照使用的设置。库路径对应当前锁定的 Python 3.11/Torch 2.7.1 环境，升级依赖后需重新核验。

此配置依赖当前板上已验证的 ARM CPU Torch 2.7.1/OpenCV/pycolmap 环境和冻结渲染包。仓库中的 Python 源码不能替代 ARM 二进制依赖、FPGA BOOT 或权重。重新部署使用 `deploy.py`，先准备其要求的官方源码、权重和 ARM 离线 wheels；不要复制电脑的虚拟环境到板子。

第三方源码获取与锁定：

```sh
git clone https://github.com/donydchen/mvsplat.git vendor/MVSplat_reference
git -C vendor/MVSplat_reference checkout 01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01
```

以上命令在 `examples/3dgs_reconstruction` 下执行。官方权重下载入口在锁定源码的 README；放到 `models/mvsplat/re10k.ckpt` 后核对本页 SHA256。上游 MIT 许可随第三方源码保留，本仓库未将第三方源码、权重、系统依赖或测试视频纳入本次提交准备。

运行目录必须全新，历史失败/画质不足/CPU 与 FPGA 输出均不覆盖。私有连接口令仅在环境中提供；不得写入代码或报告。

## 验收边界

128 和 256 是两个分辨率配置，PSNR、点数、内存与时间分开报告。电脑与 ARM 指标分开；三次重复只报告均值、中位数、范围，不据此推断稳定 P99。后端性能包括模型读取、CPU 前处理、排序、SDK/进程启动、搬运和渲染输出；整链另计视频/位姿/推理、导出和首图壁钟时间。

内存数字为采样进程 RSS 或进程树 RSS，不能当作 NPU 设备内存或 FPGA BRAM。单帧画质合格不证明大场景建图、实时运行、多窗口融合或导航准确度。当前新增模块没有改变 FPGA 逻辑，不能据此报告新的 LUT/DSP/BRAM、布线时序或功耗结果。实测完成与失败记录见 `VALIDATION.md`。
