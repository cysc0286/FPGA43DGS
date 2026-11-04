# 当前主线与验证边界

更新：2026-11-04。结构整理没有新增板端测试；以下速度属于 11 月 2—3 日保存的历史实测。

发布确认：用户于 2026-11-04 确认 11 月 3 日夜间最后一次测试已核验通过，并授权提交当前代码。
该状态来源为用户确认；本轮不新增或推算该次测试的耗时、画质、资源指标。发布内容见 [发布说明](docs/RELEASE_20261104.md)。

## 当前部署选择

- CPU：视频解码、位姿、注意力／代价体等尚未迁移的网络运算、Gaussian 适配、投影和排序。
- NPU：backbone_cnn、depth_head、to_gaussians、to_disparity 和 transformer_ffn_0…5，共十个显式子图；并非全网 NPU。
- FPGA：沿用四路 200 MHz 冻结主线、Dense mode 2、精确指数 ROM 和有序合成。代码与硬件都在 rendering/render_main。
- 网页：独立 web 模块提供 USB 采集、文件上传和常驻场景视角控制；同一 NPU／渲染设备所有者，错误显式上报。

严格 TF32 逐元素门槛和 SDK 根因仍未关闭。已有资产绑定图像资格及受限规避，不代表所有场景数值通过。
连续 corr/refine U-Net 候选存在输入切换陈旧／不响应，不接默认链。CPU 对照和当前数值／内存／设备隔离门槛保留。
本轮移动代码路径后，执行源码身份会变化；重新部署必须重新资格验收，不能复用旧 receipt。

## 最近已保存结果

11 月 3 日同一旧公开视频，640×480、2.9 秒、30 帧，裁至 128×128、32768 高斯。三轮 HTTP 上传末字节到首帧为 52.049747、49.032153、46.839201 秒。
最后一轮完整前向 13.295171 秒，进程监测峰值 PSS 546.704 MiB、最低 MemAvailable 279.488 MiB；该 PSS 不是整个服务的合计。
三轮高斯和相应完整图像精确一致。每配置仅一次，不能归因或宣称稳定提升，20 秒目标未达到。
[双入口实测记录](examples/3dgs_reconstruction/mvsplat/video_input/DUAL_INPUT_VALIDATION_20261103.md)。

同日原生换视角 53.84／54.87 ms；独立网页确认 5°视角为板端 53.44 ms、请求到画面就绪 72.4 ms，非物理 paint。
这些是场景已准备后的结果，不是视频重建时间；本轮未复测。

11 月 2 日固定 prepared 输入十子图完整前向 10.369／10.547 秒，CPU 18.551 秒；这不是视频到首帧。
十子图对旧固定资产的三视角 PSNR 为 42.736–44.042 dB、SSIM ≥0.995357；新场景绝对门槛仍有失败，不能声称任意场景画质合格。
[六 FFN 与限制](examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch/NPU_EXPANSION_20261102.md)。

## 冻结硬件

BOOT SHA256：b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6。
历史全板资源：LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63.00%。
原 AI 脉宽及 GT／CDC 限制仍在，不能称全板无条件时序签核；本轮未重新综合、布线或更新固件。
[冻结包](examples/3dgs_reconstruction/mvsplat/rendering/render_main/README.md)。

真实 USB 新录环拍、完整 360°覆盖、长期连续 NPU 更新仍需独立实板验收。
NPU 利用率、真实 DDR 流量、功耗未测；网页方向控件可转满一圈不等于场景已重建全方位。
