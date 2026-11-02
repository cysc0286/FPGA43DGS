# 2026-11-01 摄像头免标定全链路试跑

用户明确同意先跳过实测标定，并要求开始测试；用户配合缓慢平移摄像头。
本轮完成实际 USB 短片 → 自动选帧/位姿 → 真权重 CPU MVSplat → 高斯导出 → 冻结 FPGA 首帧
→ 常驻交互。NPU 未启用，固件未改。完整初始化链已正常结束，`warm_result.complete=true`、
watchdog `status=passed`。原会话累计 377 次交互请求、错误 0，原生请求累计 15788.5018 ms；
该会话包含人工操作与空闲，不用会话时长计算持续 FPS。
随后为交付用户操作时间，释放模型进程并通过既有 LiveRenderer/serve_scene API 重新打开保存的同一场景，
未重新推理；回到原视角像素哈希与旧会话完全相同。当前保存场景会话继续运行。

## 版本、环境与复现

- 本机仓库 main 基线 `a6d621767b5b91c9142954837ee5a04197e8995f`，包含本次未提交摄像头增量。
- 板端候选 `/root/fpga43dgs_reconstruction/camera_chain_20261101`，最初部署 70 个源码文件并核对哈希。
  后补监控回归测试，最终清单 71 个文件；没有覆盖原重建发布目录，依赖和权重用软链接复用。
- 相机 `/dev/video0`，1280×720 MJPEG 60 fps，实际 VID:PID `05a3:9230`。
- ARM 环境、libgomp 预加载与以前板端约束相同；Torch 先于 OpenCV 初始化，推理 4 线程、准备 1 线程、串行准备。
- 真权重 `re10k.ckpt` SHA256 `83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a`。
- 原生程序 `render_exact_20261001_direct/source_verified_source/rendering/live_exact`（相对板端 reconstruction 根），
  SHA256 `d1220d46d0cb6830077b973a2cfb2587664e36d28357bc79c18fb2eb6dfaa672`，与主线清单匹配。
- SD 启动分区只读挂载、哈希核对后卸载，BOOT 文件 SHA256
  `b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6`。
  此项核对 SD 文件，不能单独证明当前 PL 装载身份；真实主线协议和首帧运行另行通过。
- 板卡系统时钟停在 2026 年 3 月，报告日期用电脑会话日期，阶段耗时用 monotonic，采集用封装 PTS。

相机内参明确为 `approximate`，没有生成实测标定文件：假设 95° 对角视场适用于 1280×720、
方形像素、居中主点、零畸变，焦距约 672.864 原图像素，`--focal-ratio 0.5256751337`。
该假设错误会影响位姿和新视角质量，当前试跑不构成正式镜头/米制精度验收。

在候选目录、既有 LD_LIBRARY_PATH/LD_PRELOAD/PYTHONPATH 环境下，本次核心入口为：

```sh
# 先停止占用相机的预览进程；须核对 PID 命令行，不能照抄旧 PID 杀进程。
cd /root/fpga43dgs_reconstruction/camera_chain_20261101/mvsplat
/usr/bin/python3 -m video_input.capture --device /dev/video0 \
  --width 1280 --height 720 --fps 60 --seconds 10 --settle 0 \
  --out ../capture_001

cd ..
# out 必须是新目录；实际本次已存在，不要覆盖/盲目重跑。
/root/fpga43dgs_reconstruction/arm_env/bin/python3 pipeline.py initialize \
  --video capture_001/capture.mkv --out camera_scene_001 \
  --auto-select --focal-ratio 0.5256751337 \
  --weights weights/re10k.ckpt --vendor vendor/MVSplat_reference \
  --renderer /root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928 \
  --live-renderer /root/fpga43dgs_reconstruction/render_exact_20261001_direct/source_verified_source/rendering/live_exact \
  --render-profile mainline --backend cpu --threads 4 --prepare-threads 1 --serial-prepare \
  --interactive --viewer-host 127.0.0.1 --viewer-port 8080
```

实际启动外层使用 `mvsplat/measure.py`，650 MiB RSS / 128 MiB 系统预留 / 1800 秒总时限。
该总时限包括初始化、重建和交互；原完整链路已正常结束。
当前保存场景查看器单独使用修正后的监控，并从本次重新打开起有新的 1800 秒时限。
旧内存统计的局限与修复见下节。当前服务仅监听板端 127.0.0.1:8080，
经核对主机公钥的 SSH 转发到电脑 127.0.0.1:18432，token 在本机临时会话文件中。
浏览器拖动/滑块旋转、滚轮缩放、Shift 拖动平移，点击“结束会话”释放资源并保存最终记录。
关闭浏览器标签本身不结束板端会话。原摄像头预览已停止，当前显示的是重建场景。

## 测量结果与范围

先对已有 NYU 视频跑一轮 CPU＋FPGA 基线，再录制新场景。两者输入和几何工作量不同，
下表用于分清阶段，不用于宣称加速或回退。

| 项目 | 已有文件基线（单次） | 本轮真实摄像头（单次） |
|---|---|---|
| 输入 | 4,900,963 字节历史 MP4 | 70,669,433 字节 MKV，593 个实际 JPEG 包 |
| 原始采集 | 不适用 | 封装流约 10.005 秒，PTS 计 59.2711 fps，最大帧间隔 68 ms |
| 采集检查全过程 | 未测 | 16.0428 秒，打开设备到封装/probe/哈希完成，非纯曝光时间 |
| 模型/设备预热 | 37.1041 秒，其中模型加载 23.1308 秒 | 39.5078 秒，其中模型加载 24.4593 秒 |
| VIDEO_COMPLETE→SCENE_READY | 26.0326 秒 | 48.1205 秒 |
| SCENE_READY→FRAME_COMPLETE | 0.2651 秒 | 0.2561 秒，含加载/调用/归档检查，不等于纯内核时间 |
| VIDEO_COMPLETE→FRAME_COMPLETE | 26.2976 秒 | **48.3766 秒**，不包含前置模型加载和录制 |
| 输出 | 32768 高斯、128×128 FPGA 首帧 | 32768 高斯、128×128 FPGA 首帧 |
| 完整录制开始→首帧 | 未测 | 未连续打点；分阶段启动含人为/调试等待，不拼加成端到端时间 |

摄像头输入通过几何初始门槛：上下文 `[422,592]`，留出目标 `507`；候选为
`[0,84,169,253,338,422,507,592]`。上下文特征 406/376，互相匹配 214，
有效 Essential/正深度点 205，正深度比例 1.0，三角化中位夹角 4.0271°（门槛 ≥1°）。
目标 152 个地标匹配、151 个 PnP 内点，中位重投影误差 **0.3602 px**（门槛 ≤2 px），
误差单位是 480×270 几何解码图像素，不是 720p 像素或米。失败候选保存在 input.json，未放宽门槛。

| 正确性/画质 | 结果 | 限制 |
|---|---|---|
| 文件交付 | 录像、context、gaussians、PLY、最终 framebuffer 的哈希链全部核对通过 | 证据已取回电脑 |
| 四阶段事件 | READY→VIDEO_COMPLETE→SCENE_READY→FRAME_COMPLETE | 文件入口，采集独立测量 |
| 实拍留出目标画质 | 单张 PSNR **21.6187 dB**、SSIM **0.8849** | 128×128 RGB，SSIM 为 11×11 高斯 σ=1.5、有效区域、RGB 平均 |
| 像素误差 | 平均绝对差 8.6517/255，最大绝对差 240/255 | 相对真实留出帧；局部误差仍明显，不能只报平均值 |
| LPIPS、标定后对照、其他场景 | 未测 | 当前不支持数据集级或准确几何结论 |
| 常驻视角请求 | 24/24 成功、请求 ID 和 128×128 输出尺寸匹配 | yaw 在 −10°/−5°/0°/5°/10° 范围内采样 |
| 返回原视角 | 8 次结果像素哈希完全相同 | 相同估算场景与同一原生程序，未与 CPU Dense 做新差分 |
| 实际浏览器 | 图像、旋转、滚轮、复位、刷新、320 px 布局共 6 类通过，脚本错误 0 | 测试未点击退出，保留给用户 |

随后原会话由受控 `/quit` 正常结束，保存完整验收记录，再重新打开保存场景；新查看器原视角输出与旧会话精确一致。

加载同一场景后连续 24 次请求，包含首次网页请求开销；无删除慢样本：

| 耗时（ms） | 均值 | 中位数 | 最小 | 最大 | P95 |
|---|---|---|---|---|---|
| 原生相机请求→完整 RGB | 36.2104 | 36.05 | 34.57 | 40.99 | 38.2935 |
| 本机 HTTP→PNG 收全 | 79.7677 | 67.3607 | 54.5017 | 158.1088 | 118.8809 |

第一行排除 PNG、SSH、浏览器显示；第二行包含 PNG/HTTP/SSH往返，排除浏览器绘制。
没有独立测浏览器持续显示 FPS 或摄像头曝光到显示延迟。不同点集覆盖和视角下，不能照搬这些数值。

## 内存统计兼容修复与资源边界

实机 `/proc/PID/task/TID/children` 不存在。旧 `measure.tree_rss` 捕获缺文件异常后只累计主进程，
会漏算原生渲染子进程。旧基线的 `634.09375 MiB` 不能表述为完整进程树峰值；
基线最小系统可用内存 268.9805 MiB 的系统级采样仍有效。

改为扫描 status 中 PPid 关系累计子孙进程，并处理进程采样期间退出、无 RSS 的父进程与环路。
本机 3 项新回归通过，板端系统 Python 同 3 项通过（0.016 秒仅为测试执行耗时）。
新函数现场核对原完整链会话进程树 **529.1445 MiB**（包括 watchdog），系统可用约 **359.8438 MiB**，
是单次快照；不是本次推理峰值。完整进程树推理峰值在修正后的启动器下尚未重新测量。
原完整链 watchdog 内存中仍为旧版本，最终 598.1289 MiB 是主进程采样峰值、系统最小可用内存 273.9883 MiB；
不能冒充全树峰值。其完整执行 1465.3113 秒含长时间交互和空闲，不是重建耗时。
当前保存场景查看器已使用修正后的 watchdog，进程树快照 **166.6211 MiB**、系统可用 **664.9336 MiB**，
模型推理进程已退出。新查看器真实进程树限制 650 MiB、系统预留 128 MiB、1800 秒，退出后写最终监控结果。
新增 `/proc` 扫描有采样开销，不改变模型或渲染算术。`board_sync` 已加入 measure.py 防止后续部署漏更新。

本轮无新增模型/渲染算术改动，前一轮 65 项相关回归的通过记录保留日期与范围，本轮未重复执行。
新监控回归单独执行；真实模型、FPGA 首帧、24 次视角和 6 类浏览器操作是本轮新增证据。
最终根文件系统剩余约 **220.08 MiB**；没有清理旧实验或修改分区，录像与模型结果已归档电脑。

硬件沿用主线 4 路、配置 200 MHz，未改 BOOT/RTL，未重新测频率或进行 LUT/FF/BRAM/DSP、
WNS/TNS/hold/脉宽签核。首目标原生程序返回 hardware_cycles=3,800,286，
native wall=50.6992 ms（只此一帧）；内核计数不能替代主机可见整帧时间。
同帧原生记录 input_bytes=3,133,760、output_bytes=262,144，upload=2.5482 ms、wait=8.4199 ms、read=0.6904 ms；
这些为程序计数字段，未独立抓取 DDR 总线或启用细粒度 DMA profiler，不把 payload 字节当作总 DDR 流量。
板端功率、能量/帧、NPU 性能均**未测**。

## 原始证据与优劣

电脑证据根：`D:\SyncFiles\1_Workspace\FPGA3DGS\tmp\board_camera_20261101\`。
`fullchain_evidence/` 包含两轮日志/事件、输入图、原始短片、真实 gaussians.npz、model.ply、framebuffer 和导出验证；
`live_scene_validation.json` 保存全部 24 次数据、几何与图像质量；`live_viewer_browser.json` 和截图为真实浏览器证据；
`fullchain_preflight.json`、源码清单、最终审计及 board_measure_tests.txt 记录环境与修复。
`viewer_handoff.json` 记录释放模型后的保存场景查看器，新的会话 URL 由 `viewer_session.json` 保存。
首次完整运行与修复后源码清单分别保留，避免用新监控版本冒充历史测量。
短片 SHA256 `f1472b1382f293c4e35f23f9ff824935c758aef7ad0d9001e73ce5304b1b1db8`；
首帧 framebuffer SHA256 `64e988500a64ffcabdeff7d452e71e874419ae83a6b414bcc545d33843a21d65`。

优势：不用等待棋盘格即可验证真实输入到 FPGA 与浏览器的整条路径，复用现有权重和硬件，
保持估算内参标记、几何门槛和哈希链，定位了板端监控漏算问题。
代价：内参/畸变假设限制几何和画质，当前仅两张上下文、一个静态场景、128×128 输出；
自动选帧和 CPU 推理耗时仍明显，首帧 48.38 秒不等于实时重建。网页压缩/传输开销高于纯渲染，
存储余量小。当前证据支持试跑可用，不支持新的加速、节能或标定精度结论。
下一步优先做多个静态场景重复采集、真实镜头内参对照和分阶段热点分析，再考虑提分辨率或前端加速。
