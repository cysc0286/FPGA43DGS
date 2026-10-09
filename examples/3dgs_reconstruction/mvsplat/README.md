# 当前 MVSplat 前馈主线

四个业务模块：`video_input/`、`pose_estimation/`、`gaussian_generation/`、`rendering/`。
视频前预热在 `initialize/`；视频后的阶段组织在 `scene_preparation.py` 和 `warm_pipeline.py`。
先读 [架构](ARCHITECTURE.md)、[接口](../modules/INTERFACES.md) 和 [仓库当前主线](../../../MAINLINE.md)。

当前采用作者固定预训练权重，不逐场景反向训练。两张上下文及其相机信息生成 Gaussian；
留出目标用 PnP 定位；这是已接收视频文件的两视图前馈重建，不是完整在线 SLAM。
CPU 执行网络；NPU 仍是待数值验收的候选。

## 运行入口

从重建目录运行 `pipeline.py initialize`，显式传 `--live-renderer` 与 `--render-profile mainline`，
复用当前四路后端。详细命令见 [根 README](../../../README.md)。
从本目录使用 `rendering.runtime.LiveRenderer.mainline(...)` 可独立做已加载场景的换视角。
`pipeline.py reconstruct` 保留旧冷启动/完整 SfM 质量回退，不能混作预热主线计时。

固定依赖：MVSplat 提交 `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`，
`re10k.ckpt` SHA256 `83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a`。
权重、vendor、ARM 环境与 SDK 不纳入源码包；本次没有重新安装这些依赖。

## 当前结果怎么读

2026-10-08 四路独立后端：32768 高斯、128×128、60 帧，换视角平均 43.575 ms。
这与历史预热整链约 26–28 s 的场景准备时间是两个口径；统一最新后端的完整视频链尚未复验。
完整后端 [冻结包](rendering/package/README.md)、[同轮画面与指标](render_branch/validation/RESULTS.md)。

历史报告 `VALIDATION.md`、`FAST_VALIDATION.md`、`BOARD_WARM_NPU_VALIDATION.md`
按各自日期阅读；旧原始结果通过仓库根 [归档索引](../../../archive/history_20261009/README.md) 恢复，
不是缺失或被新成绩覆盖。`INITIALIZE_NPU_VALIDATION.md` 和 `NPU_DEPLOYMENT_V2.md`
同样不构成新的 NPU 数值通过证明。

本轮只整理代码、接口和归档，未运行功能测试或上板测速。
