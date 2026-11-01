# USB 摄像头短片输入与交互渲染

更新：2026-11-01。实现目标为“采集几秒静态场景 → 重建一次 → 交互换视角”。
同日新增“清理历史记录（保留最新）”按钮：仅删除本页面 sessions 目录内的旧会话及配套日志，
保留当前短片、最近已验证首帧的成功场景及其来源录像；忙时拒绝，未知目录/链接保留。
不自动归档，不自动在每次录制后删除。实际清理范围、释放空间和保留验证见
[历史清理实测](BOARD_HISTORY_CLEANUP_20261101.md)。
2026-11-01 已验证真实板卡 UVC 采集、预览和短片解码；没有执行本次真实权重推理或 FPGA 渲染。
本机镜头 720p MJPG 实际只支持 60 fps，需显式传 `--fps 60` / `--camera-fps 60`。
独立预览不要求标定：`python -m video_input.preview --device /dev/video0 --fps 60 --log preview.log`。
2026-11-01 新增按钮入口：`python -m initialize.camera_ui ...`，开始/结束录制、计时与预览同页，
检查完成后点击“重建场景”，再打开交互场景。当前电脑入口为 http://127.0.0.1:18433/ ，
运行参数与验收边界见 [按钮录制实测](BOARD_CAMERA_BUTTONS_20261101.md)。
随后修复真实长录像失败：整段候选无法通过时，最多依次尝试末尾/中部/开头约 4 秒的局部窗口，
每轮候选仍最多 12，原始帧索引和全部几何门槛保持。按钮新录像默认上限改为 10 秒，
增加主动空间监控、短 Matroska cluster 和 Linux RLIMIT_FSIZE 硬保护；原片超量问题与
新实测边界见 [长录像修复](BOARD_LONG_RECORD_FIX_20261101.md)。
预览默认只监听板端 127.0.0.1:8081，可经 SSH 转发访问。实测详情见
[板端摄像头调试](BOARD_CAMERA_DEBUG_20261101.md)。下文 2026-10-29 回归记录保留其原始范围。

## 实现范围

`capture.py` 用 Linux UVC/V4L2 接收 MJPEG，默认 1280×720、30 fps、稳定 1 秒后保留 5 秒短片。
录制、probe 和时长/实际模式/帧率检查通过后，才把临时文件改名为 `capture.mkv`，
沿用原 `receive_file` 交付。`capture.json` 保存请求参数、生效参数、帧 PTS、哈希和失败原因；
PTS 是封装流时间戳，不等同于传感器曝光时刻。录制错误不发布 VIDEO_COMPLETE。
采集不重新编码，不把整段原始帧存到 RAM；默认解码保留 8 张候选帧，上限为 12。
帧率、模式不匹配、短片不足时长或实际帧率低于请求的 90% 都会拒绝本次输入。
短片存储上限为 256 MiB，并预留 64 MiB 文件系统空间；容量截断导致时长不足也会拒绝交付。
采用 Matroska 避免板端旧 FFmpeg AVI 的双倍帧率和空帧槽问题；标定/自动选帧使用实际 JPEG 包数，
不直接相信容器声明的帧数。

`calibration.py` 从棋盘格图像拟合针孔内参/畸变，并在选中帧上校正到主点居中的新相机。
输入模式尺寸必须与标定完全一致；焦点或传感器裁切变化后要重新标定。
resize 半像素、中心裁切和网络归一化的内参变换保持一致，适配冻结渲染器的居中主点 ABI。
95° 对角视场不作为正式内参；摄像头入口强制提供标定 JSON。

### 2026-11-01：用户授权的免标定试跑

用户同意先跳过实测标定。保持 `--camera` 的正式标定检查，先用 `video_input.capture`
录制短片，再向原 `--video` 入口传入 `--auto-select --focal-ratio 0.5256751337`，不传 `--calibration`。
该焦距比由 `sqrt(1280²+720²)/(2*tan(95°/2))/1280` 得到；假设 95° 对角视场适用于当前
1280×720 画幅、方形像素、居中主点和零畸变。原图等效焦距约 672.864 px，
解码/裁切后的内参仍由既有代码变换，不能把原图像素焦距直接填到缩小图。
`input.json` 保留 `kind=approximate` 与 `distortion=assumed zero`，不生成伪装为实测的标定 JSON。
此模式用于确认实际链路是否可运行；几何门槛仍保持，失败不靠放宽门槛或虚构位姿通过。
其视觉结果不能替代真实镜头标定、尺度或正式重建质量验收。
本轮实板已通过该入口得到真实首帧及交互，见 [摄像头全链路实测](BOARD_CAMERA_CHAIN_20261101.md)。

`geometry.select_views` 缓存候选 SIFT，按三角化质量选择两张上下文与一张不同的中间留出目标。
初始门槛：至少 30 内点、正深度比例 0.75、三角化中位夹角 1°、目标 PnP 中位误差 2 px；
模糊/低纹理检查使用解码图灰度 Laplacian 方差至少 15。这些是起始门槛，尚需按实拍场景评估。
`--min-parallax` 和 `--max-reprojection` 可显式调整；误差单位是默认宽度 480 的几何图像像素。
失败时给出原因，不回退到任意上下文或虚构位姿。

`rendering/viewer.py` 为常驻 LiveRenderer 提供浏览器控制，支持旋转、缩放、Shift 拖动平移、复位和结束。
场景只装载一次；浏览器只保留最新待处理视角，一次只发送一个渲染请求。
控制范围先限制为水平 ±30°、俯仰 ±20°、距离倍率 0.6–1.5，不能保证未观测区域完整。
显示原生分辨率；放大显示不等于高分辨率重建。默认仍为 128×128、CPU 生成、冻结 FPGA 主线。
渲染协议超时/异常会结束交互会话，避免把迟到的上一帧当成下一请求结果。

## 连接与板端前提

- 摄像头插开发板 USB-A HOST；MINI USB 用于串口。四个 USB-A 共享 USB2.0 上游带宽。
- 开发板保持原 12 V 供电；电脑用网线访问。摄像头掉线时先检查 USB/供电/曝光与驱动日志。
- Linux 需要匹配内核的 UVC/V4L2 支持、`v4l2-ctl`、`lsusb`、支持 V4L2/MJPEG 的 `ffmpeg`/`ffprobe`。
- Python 复用现有板端模型环境，OpenCV 需有 FFmpeg 后端及 `CAP_PROP_N_THREADS`。
- 已有权重、vendor、原生渲染程序和 BOOT 必须与当前主线匹配。此功能不会安装驱动或烧录固件。

## 先枚举和标定

以下命令在板端 `examples/3dgs_reconstruction/mvsplat` 目录执行。
设备路径按真实枚举填写，存在时优先使用 `/dev/v4l/by-id/` 稳定标识。

```sh
python -m video_input.capture --device /dev/video0 --probe
```

确认实际档位后，固定手动焦点；本次摄像头 1280×720 MJPG 使用 60 fps。控制曝光/白平衡前先看 probe 枚举的控件，
不要照抄其他相机的控件编号。准备 9×6 **内角点**、方格边长已知的棋盘格。
可用系统相机保存 20–30 张同模式标定图，也可先录一段棋盘格视频后低频抽帧：

```sh
python -m video_input.capture --device /dev/video0 --out runs/calibration_clip \
  --width 1280 --height 720 --fps 60 --seconds 25 --settle 2
mkdir -p calibration/images
ffmpeg -n -i runs/calibration_clip/capture.mkv -vf fps=1 calibration/images/frame_%03d.png
python -m video_input.calibration --images calibration/images \
  --pattern 9 6 --square-mm 25 --out calibration/hf868_720p.json
```

棋盘格需覆盖中心和四周，包含不同距离与倾角；`25` 必须换成实际毫米尺寸。
工具至少要求 10 张检出棋盘格的图，建议 20–30 张；输出全局和逐图 RMS。
拟合通过不等于独立标定验收，应检查新照片中的边缘残差与直线校正效果，建议初始误差目标约 0.5–1 px。
校正相机模型不会自动让现有单目场景坐标变成米制。

## 运行摄像头整链

在 `examples/3dgs_reconstruction` 下，复用原板端启动环境和真实路径；下面路径变量均为示例占位。
不要使用旧 `initialize.board_accept` 录制摄像头：那个入口固定历史文件，且有 600 秒测试监控。

```sh
BOARD_PY=/root/fpga43dgs_reconstruction/arm_env/bin/python3
WEIGHTS=/actual/path/re10k.ckpt
VENDOR=/actual/path/MVSplat_reference
FROZEN=/actual/path/3dgs_renderer_v1_20260928
RENDER_LIVE=/actual/path/render_live
CALIBRATION=/actual/path/hf868_720p.json

# 按已有板端部署设置 LD_LIBRARY_PATH / LD_PRELOAD / PYTHONPATH。
# 原参考设置见 mvsplat/initialize/board_accept.py；先核实本板对应路径。
"$BOARD_PY" pipeline.py initialize \
  --camera /dev/video0 --calibration "$CALIBRATION" \
  --camera-width 1280 --camera-height 720 --camera-fps 60 --capture-seconds 5 \
  --weights "$WEIGHTS" --vendor "$VENDOR" --renderer "$FROZEN" \
  --live-renderer "$RENDER_LIVE" --render-profile mainline \
  --out runs/camera_scene_001 --interactive --viewer-host 0.0.0.0 --viewer-port 8080
```

READY 后按回车开始采集；需要脚本自动启动时显式加 `--capture-now`。
拍摄静止、有纹理的场景，缓慢平移，保持重叠，避免只原地旋转或快速扫动。
摄像头模式自动启用选帧。电脑浏览器打开终端打印的 VIEWER_READY URL，把 `<board-IP>` 换成板卡实际地址，
保留 URL 中的会话 token。默认仅监听 127.0.0.1；上述 0.0.0.0 适合通过板卡网口访问。
点击“结束会话”或终端 Ctrl+C 退出并释放渲染资源；关闭浏览器标签页本身不会结束板端会话。

原 `--video` 路径仍可使用。将同一 `capture.mkv` 作为文件输入，并添加相同 `--calibration`、
`--auto-select` 和选帧参数，可复查输入链；不加标定/自动选帧时保留原近似焦距、显式索引路径。
摄像头和 `--video` 互斥；`--interactive` 要求显式提供原生 `--live-renderer`。

## 计时和证据

原四主事件和 VIDEO_COMPLETE→FRAME_COMPLETE 定义保持不变。
摄像头增加 READY 后、VIDEO_COMPLETE 前的 CAPTURE_STARTED / CAPTURE_COMPLETE。
`capture_seconds_record_only` 包含设备打开、稳定等待、采集、封装及采集检查；
实际帧时间戳/请求采集时长分别保存在 capture.json，不能把它们混称传感器曝光总时间。
同时记录采集开始/完成到首帧的时间，避免只报后处理时间。
交互 `render_ms` 从 native render_camera 请求到完整 RGB，排除 PNG 编码、网络和浏览器显示；
浏览器显示的时间不能称摄像头到画面的端到端延迟。

主要产物：`capture/capture.mkv`、`capture/capture.json`、`capture/ffmpeg.log`、`events.json`、
`input/{input.json,context.npz,images/}`、`inference/`、`renderer_input/`、首帧板端归档及退出后的 `warm_result.json`。
input.json 记录标定哈希、校正后的内参、选帧索引、门槛和失败候选。
`initialize/board_sync.py` 已补齐采集、标定、事件、交互和预览依赖；11 月 1 日仅上传隔离目录中的采集/预览子集。

## 本地验证记录（2026-10-29）

本机独立环境：`C:\FPGAtools\python-envs\camera-dev`，不替换系统 Python 或板端环境。
Python 3.13，NumPy 2.5.4、OpenCV 4.14.0、SciPy 1.18.1；完整版本保存在工作区验证日志。
本轮相关回归 **60 项通过**（新增 21 项、既有 39 项），无跳过；一次合并运行 5.059 秒，
这只是本机测试执行时间，不是摄像头/重建/FPGA 性能。另完成一次无头 Edge 浏览器验证，
覆盖 7 类操作、返回 5 张合成 RGB，页面脚本错误和服务端错误均为 0。
部署清单 64 个源码路径均存在；标定预检不会提前导入 OpenCV，保留板端 Torch 先于 OpenCV 的加载顺序。
原始日志位于本工作区 `tmp/camera_input_20261029/`，包含 unittest、browser、依赖版本和源文件哈希。

在 `mvsplat` 目录复现软件回归：

```sh
python -m unittest video_input.test_camera video_input.test_camera_pipeline \
  pose_estimation.test_camera_selection rendering.test_viewer \
  test_fast test_initialize test_pipeline_renderer test_live_rendering -v
```

覆盖真实 MJPEG 编解码/复封装/probe、12 张投影棋盘格标定拟合、带不同深度纹理的合成短片经真实 SIFT/Essential/PnP
生成 context.npz、解析射线校验、已知 120 个三维点的尺度与 PnP、生命周期、异常、HTTP RGB/编号、退出和超时停止。
集成测试中的模型、摄像头设备输入和 FPGA 使用替身；不声称真实网络或板端完成。
修正了既有 test_pipeline_renderer 的三处 mock 路径，使其指向重构后的实际 ModelRuntime 定义。

| 指标 | 同范围基线/门槛 | 本轮范围与状态 |
|---|---|---|
| 原文件输入与主线渲染控制 | 既有合同 | 本地相关回归，不能替代旧实板验收 |
| 校正映射与独立解析投影 | 误差绝对值 ≤3×10⁻⁵ 解码像素 | 软件测试通过门槛 |
| 裁切后网络内参 | 绝对容差 10⁻⁷ | 软件测试通过门槛，主点归一化为 0.5 |
| 合成棋盘格标定 | RMS <0.5 px，焦距相对误差 <3% | 软件测试通过门槛；非真实镜头标定 |
| 已知点 PnP | 中位重投影 <0.001 px、目标平移误差绝对容差 2×10⁻⁴ | 软件测试通过门槛；尺度为现有归一化坐标 |
| 合成短片自动选帧 | 夹角 ≥1°、留出目标中位误差 ≤2 px | 真实 OpenCV 求解的软件测试通过 |
| 浏览器 | 首帧、旋转、滚轮缩放、复位、刷新、320 px 布局、退出 | 本地无头 Edge＋合成 RGB；无页面脚本错误 |
| PSNR/SSIM/LPIPS | 真实留出图/固定模型 | 未测，未执行真实权重和板端渲染 |
| 首帧延迟、交互平均/P95、峰值 RSS | 当前摄像头＋30TAI 同范围 | 未测，本地回归耗时不作为板端性能 |
| LUT/FF/BRAM/DSP、200 MHz 时序 | 冻结 render_main | 未改硬件，未做新综合/布线/时序验收 |
| DMA/DDR 字节、等待、功耗、能量/帧 | 实板真实采集和渲染 | 未测 |

优点：复用既有文件交付和常驻渲染 ABI，采集缓冲有上限，真实标定与几何失败可追溯，旧文件入口仍可用。
代价：增加 MJPEG 解码/校正和多候选几何工作，中心裁切缩小有效视场；两视图仅覆盖有限区域，
CPU 推理、选帧成本和板端内存仍需实测。当前结果支持软件接线正确性，不支持新的加速或节能结论。
