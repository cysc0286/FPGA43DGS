# 当前 MVSplat 前馈主线

四个业务模块保持 video_input/、pose_estimation/、gaussian_generation/、rendering/。
视频前预热在 initialize/；视频后协调在 scene_preparation.py、warm_pipeline.py。
先读 [架构](ARCHITECTURE.md)、[接口](../modules/INTERFACES.md) 和 [仓库当前主线](../../../MAINLINE.md)。

推荐固定权重 MVSplat 的 **NPU backbone_cnn＋depth_head、CPU 其余网络**，随后由
CPU＋冻结四路 FPGA 渲染。两张上下文及相机参数生成高斯，留出目标用 PnP 定位；
无逐场景训练，也不是完整在线 SLAM。

## 运行与回退

从重建目录运行 pipeline.py initialize，显式传 --live-renderer、--render-profile mainline，
以及 --backend npu --partitions backbone_cnn depth_head 和匹配的图/桥接库/验收记录。
[完整命令](../../../README.md)；[板端资格验收与网页操作](FINAL_INTEGRATION_20261102.md)。

NPU 启动前必须检查输入切换、共存地址边界和资产身份；backbone 默认执行两次 SDK
forward 规避输入残留，全部成本计入。改变库/运行策略须重新生成 receipt。
TF32 严格逐元素门槛仍失败，当前使用三留出视角的原生 FPGA 图像验收策略。
错误不会静默回退。CPU 对照用 --backend cpu，去掉 NPU 专用参数；
通用 CLI 默认 CPU 保留兼容。

左侧网页拖动仅选择观察点，确认后才请求 FPGA，右侧显示真实返回帧。
场景加载后换视角不再推理。pipeline.py reconstruct 保留冷启动/完整 SfM 回退，
不能与预热主线混作同一计时。

固定 MVSplat 提交 01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01，
re10k.ckpt SHA256 83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a。
权重、vendor、SDK、ARM 环境、私人视频和生成资产不纳入源码提交。

## 结果边界

2026-11-02、128×128、32768 点：同调度 CPU 首帧 24.512 s，
修正双子图 24.353 s，额外完整前向预热候选 23.933 s，各一次，20 s 目标未达到。
候选首帧对 CPU 为 55.332 dB / 0.999232，对真实留出图为 19.450 dB / 0.804061。
详细正负结果见 [NPU 深度覆盖实板记录](gaussian_generation/npu_branch/NPU_DEPTH_VALIDATION_20261102.md)。

冻结后端 [render_main](rendering/render_main/README.md) 保留原样；
render_branch 默认不调用。旧报告按日期阅读，不覆盖新的输入切换发现。
本次 Git 合并只新增软件核验；修正场景网页恢复、最后原始归档和板端一行验证元数据同步待接板补验。
