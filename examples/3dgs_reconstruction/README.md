# 视频重建主入口

统一 CLI 是 `pipeline.py`。当前推荐 `initialize`：视频前预热，然后使用已有视频完成场景准备与首帧。
必须显式传入 `--live-renderer` 才会选择当前常驻原生后端；该后端默认使用 mainline 配置。

当前四模块源码均在 `mvsplat/` 下：

| 模块 | 位置 |
|---|---|
| 视频接收、解码、选帧 | `mvsplat/video_input/` |
| 两视图位姿与目标 PnP | `mvsplat/pose_estimation/` |
| 固定权重高斯生成及 ABI 转换 | `mvsplat/gaussian_generation/` |
| 常驻 CPU＋FPGA 渲染 | `mvsplat/rendering/` |

参见 [模块接口](modules/INTERFACES.md)、[具体架构](mvsplat/ARCHITECTURE.md)、
[仓库当前结构](../../docs/CODE_STRUCTURE.md) 和 [当前主线指标](../../MAINLINE.md)。

`reconstruct` 是保留的冷启动/完整 SfM 对照；旧 `modules/`、`bounded/` 和 `npu_frontend/`
分别提供兼容合同、历史 OpenSplat 训练、旧特征 NPU 试验，均不代表当前推荐链路。
旧说明、源码试验和原始结果按日期保存到 [归档](../../archive/history_20261009/README.md)。
当前接口重整尚未进行新功能或实板回归，不能把旧测量当作本轮测试。
