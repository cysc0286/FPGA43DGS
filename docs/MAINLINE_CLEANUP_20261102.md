# 主线清理核验（2026-11-02）

用户明确要求删除与 NPU→FPGA→网页主线无关、容易误用的旧实验和重复入口，
并保留核心四模块、CPU 回退和冻结渲染包。

## 删除与保留范围

从活动树删除 250 个跟踪文件：

- bounded 受限 OpenSplat 训练、npu_frontend 旧特征匹配 NPU，以及 modules/gaussian_generation 旧训练/导出。
- run_cpu、stages、旧训练构建/出图/证据收集、旧路径与资源估算等入口。
- 独立 NPU deploy_20261021 诊断工程；当前 NPU 代码仍只在 gaussian_generation/npu_branch。
- 早期 batch/scene/CAT/compositor/flicker 研发脚本和测试，当前渲染改用已有冻结包复建。
- pipeline.py 不再接受 gaussian/export 的 OpenSplat 训练/导出命令，误用会直接报参数错误。

保留：

- video_input、pose_estimation、gaussian_generation、rendering 四模块以及 initialize 协调。
- NPU 正式入口、当前诊断与验收工具，显式 --backend cpu 回退。
- reconstruct 完整 SfM/冷启动对照，以及实际被它调用的 video/pose。
- 合同校验与已有模型 render 工具；modules/common.py、contracts.py 及冷对照依赖不误删。
- render_main、render_branch、releases 的全部字节；GraphDECO 许可与实际使用的 SSH helper。
- 另一个原 main 工作目录中的所有未提交硬件工作和私人数据。

## 历史保全与恢复

删除前归档 257 项：250 个删除文件及 7 个入口/说明的修改前快照。
原始数据共 1195945 字节，4 个 ZIP 共 590537 字节；每成员的原路径、长度、
SHA256 和包哈希保存在 archive/history_20261102_cleanup/index.json。
旧成功/失败记录均保留，不改写为新结果。

使用既有 tools/archive_history.py verify / restore 核验并恢复到新目录。
主线不加载 ZIP；恢复旧项目需配套原提交 f9af617 或独立历史 checkout。
Git 历史保留，所以这次不声称缩减远端仓库全部历史体积。

## 软件核验

- tools/check_core.py --mvsplat：128 个 Python 源码语法通过；
  接口 16、公共入口 3、MVSplat 146，共 165 项，161 通过、4 跳过。
- 跳过范围与上一轮相同：缺少 ffprobe 的 3 项视频测试、Windows 符号链接权限的 1 项。
- 删除旧训练调用测试，替换为 gaussian/export 均拒绝的回归；
  仍核对完整 SfM 对照路由、CPU/NPU配置转发和当前渲染器接口。
- 冻结 render_main 151 个、render_branch 197 个文件哈希通过；
  92 个调用源的主线隔离通过。测试使用真实冻结 render.py，
  不再依赖已删除的旧研发目录。
- 4 个历史 ZIP 的成员集合、CRC、原始长度、SHA256、内外清单核对通过。
- 源码包在本机 outputs 生成并逐成员核验，必须包含网页资源且不含已删实验。
- git diff --check 通过。本轮没有 SSH、NPU 执行、综合、布线或板端渲染。

原始软件日志和干净解压/历史恢复副本仅保留本机 outputs，不上传。

## 指标与评价

该清理不改变权重、布局计算、排序、NPU SDK 策略或 FPGA 电路，不能宣称加速。
沿用此前实板固定公开视频、128×128、32768 点：
CPU 首帧 24.512 s，修正双子图 NPU 24.353 s，额外预热候选 23.933 s，各一次；
对 CPU 首帧 55.332 dB / SSIM 0.999232，对真实图像 19.450 dB / 0.804061。
严格 TF32 tensor 门槛仍失败，采用已有三视角原生 FPGA 图像验收；SDK 根因仍未关闭。

全板历史资源 LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63%；
200 MHz，既有 AI 脉宽/GT/CDC 限制不变。本次资源、时序、DDR 流量和功耗均未重新测量。
优点是主线入口和代码归属更清楚，旧训练不再容易误启动；
代价是旧实验须先从归档恢复，不能沿旧目录直接运行。
修正场景网页恢复、最后原始归档和板端验证元数据同步仍待设备恢复后补验。
