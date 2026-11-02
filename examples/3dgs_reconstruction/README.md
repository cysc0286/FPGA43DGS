# 视频重建主入口

推荐入口是 pipeline.py initialize：视频前预热，随后运行视频→位姿→NPU 高斯生成→FPGA→网页。
推荐显式 --backend npu --partitions backbone_cnn depth_head，并提供匹配的资产和图像验收记录；
显式 --live-renderer 才调用当前常驻原生后端。完整命令见仓库 README。

四个业务模块保留在 mvsplat/：

| 模块 | 位置 |
|---|---|
| 接收、解码、选帧、标定 | mvsplat/video_input/ |
| 两视图位姿与目标 PnP | mvsplat/pose_estimation/ |
| 固定权重 CPU/NPU 高斯生成 | mvsplat/gaussian_generation/ |
| 常驻 CPU＋FPGA 渲染与网页 | mvsplat/rendering/ |

initialize/ 只协调预热和资源生命周期。结构、接口和数值政策见
[代码结构](../../docs/CODE_STRUCTURE.md)、[接口](modules/INTERFACES.md) 和 [主线指标](../../MAINLINE.md)。

## 保留的对照与工具

reconstruct 保留完整 SfM/冷启动 CPU 对照；其内部仍调用 video、pose 辅助阶段。
render 用于已有模型的冻结后端对照，validate 只检查已有文件合同，不执行训练。
modules/ 保留这些实际依赖与公共合同，不是第二套当前两视图算法。

gaussian/export 旧 OpenSplat 公共命令、run_cpu、stages、bounded、npu_frontend 和旧训练构建脚本已删除。
校验合同仍可识别历史 Gaussian/导出格式；这不表示训练命令重新启用。
旧实验、独立 NPU 诊断包和早期硬件脚本从活动树移除，可按
[清理归档](../../archive/history_20261102_cleanup/README.md) 恢复。
私人视频、原始输出与未提交工作目录不在本次删除范围。
