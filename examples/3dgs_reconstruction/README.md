# 视频到 3DGS：板端 MVSplat 主线

当前默认链路使用 [MVSplat 板端前馈](mvsplat/README.md)：从离线视频到 FPGA 图像均已在板端执行。主入口为 `python pipeline.py reconstruct --video INPUT.mp4 --out NEW_RUN`，默认完整 SfM 路径。该配置三视角画质达标，但首图 249.43 秒；快速单进程首图 66.68 秒，首个视角画质未过门槛，需显式选择。原 OpenSplat 训练路径保留为历史对照，详见 [快速版本对照](mvsplat/FAST_VALIDATION.md)。

当前代码结构和 `READY → VIDEO_COMPLETE → SCENE_READY → FRAME_COMPLETE` 状态合同见 [MVSplat 主线结构](mvsplat/ARCHITECTURE.md)。主线的后端仍是经过数值核验的 CPU＋FPGA；NPU 诊断代码只在独立 `mvsplat/npu/` 范围内运行，未通过数值门禁时不会进入生产链。

视频文件、COLMAP 位姿、MVSplat 预训练网络高斯生成、冻结 FPGA 渲染均在悟净 30TAI Lite 上实测，见 [验收数据和效果图](mvsplat/VALIDATION.md)。下文从“2026-09-28 模块整理”起记录原 OpenSplat CPU 优化基线，不应把其中“板端训练未移植”的结论套到 MVSplat 前馈上。

## 2026-09-28 模块整理

当前入口：[四模块接口](modules/INTERFACES.md)、[分阶段执行器](pipeline.py)、[项目主线](../../MAINLINE.md)。

- `modules/video_input`：视频到图像帧。
- `modules/pose_estimation`：图像帧到位姿、内参与稀疏几何。
- `modules/gaussian_generation`：高斯优化、模型和相机导出。
- `modules/rendering`：调用已完成的冻结渲染器。

旧 `stages.py` 保留兼容命令，计算代码已迁入模块。当前产物、精度和冻结后端均沿用原定义；详细核验见 `evidence/module_refactor_20260928/VALIDATION.md`。


后续 NPU 迁移代码已在独立的 [npu_frontend](npu_frontend/README.md) 准备：先迁移 SIFT 描述子匹配中的矩阵点积，软件正确性和 ICraft 编译通过，板端执行待测。它尚未替换下述默认 CPU 链，也不承担高斯训练；[候选指标与边界](npu_frontend/VALIDATION.md)单独记录。

最新代码增量：已补齐[受限资源执行版本](bounded/README.md)，包括线程/缓存/高斯数约束、超时与内存退出、哈希续跑和 ARM 测试准备。Windows 编译及本地功能检查已通过，[本轮指标与待测项](bounded/VALIDATION.md)单独记录；用户本轮在外，板端测试延后。以下 1000 步和 ARM 渲染数字仍是原基线。

2026-09-28，已完成初版实际验证。输入是视频中的 RGB 帧，采用 **COLMAP CPU → OpenSplat CPU → 冻结渲染后端**。NPU 暂不接入。后续自摄视频继续使用同一条链，替换输入文件即可；视频采集硬件不作为当前前置条件。

```text
视频文件
  → 均匀抽帧（默认30帧）
  → COLMAP CPU：特征/匹配 → 位姿/稀疏点 → 去畸变
  → 相机与图像适配：320像素宽，主点对齐冻结渲染接口
  → OpenSplat CPU：29帧优化，1帧留出，1000步，SH0，20k数量上限
  → 高斯PLY + 相机FLCAM001
  → 已有板端CPU渲染器：指定相机或中点新视角
```

当前电脑 Intel i9-14900HX 上，两次完整参考运行累计约54.91/55.78秒，训练进程峰值1757/1824MiB。板端CPU已渲染首轮新模型，但 **板端位姿/训练尚未移植**，不能把电脑CPU时延写成开发板性能。

- [完整指标、正确性与限制](VALIDATION.md)
- [环境、运行命令、接口约定](SETUP.md)
- [两次运行对照](evidence/versions.csv)
- [源码、工具和数据来源](evidence/sources.json)
- [历史资源预算分析](RESOURCE_STUDY_20260928.md)
- [冻结渲染器接口](../../releases/3dgs_renderer_v1_20260928/INTERFACE.md)

## 当前实现

|文件|职责|
|---|---|
|`fetch_video.py`|下载并按SHA256校验公开室内RGB片段|
|`run_cpu.ps1` / `run_cpu.py`|完整电脑CPU入口、独立运行目录和每阶段耗时/RSS采样|
|`stages.py`|抽帧、COLMAP、相机/图像变换、PLY与目标相机导出|
|`build_opensplat.py`|构建锁定版本的纯CPU OpenSplat，不修改作者训练算法|
|`board_render.py`|上传新场景并调用冻结ARM CPU后端，回收两视角输出|
|`evaluate.py`|同显示域PSNR、SSIM及原始数值诊断|
|`collect_evidence.py`|归档原始日志、版本指标、图像和哈希|

## 方法来源与边界

- 几何恢复依据 [COLMAP](https://github.com/colmap/colmap) / Structure-from-Motion Revisited（CVPR 2016）；这里使用pycolmap调用相同CPU算法。
- 高斯优化采用 [OpenSplat](https://github.com/WebODM/OpenSplat) 1.2.2，提交 `62fd86fe3b62644b65890557ad9bf397abdc7cfb`。它是3DGS开源工程实现，不是新增FPGA论文，也不是原始GraphDECO代码的逐行复现。
- 渲染继续复用项目已冻结的FLICKER路线实现；本轮只调用其CPU对照后端，没有改RTL、BOOT或NPU图。

一个室内角落的3秒片段只验证基本链路，不能证明整间房、长视频、动态场景或实时SLAM。下一增量应先控制运行时内存、固定数据接口并评估ARM移植，避免再次更换方法主线。
