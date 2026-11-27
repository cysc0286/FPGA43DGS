# 相机云台四阶段验收

2026-11-18 更新：此前俯仰小步复验突然大幅后仰，用户关闭舵机外供，零位和阶段01机械复验未通过。新驱动要求 ABI3 完整脉冲停止，最后板端核对仍为 ABI2，不得继续阶段02/03。ABI3 本机位流、时序审查和候选 BOOT 已生成，尚未刷入或完成仪器验收；厂家时钟脉宽限制仍保留，不宣称完整时序签核。详见 [MOTOR_INCIDENT_20261116.md](../MOTOR_INCIDENT_20261116.md)。

按顺序独立执行，上一阶段完成后再进入下一阶段。所有入口默认预览，不发送运动指令；采集视频及渲染输出放在仓库之外的私有测试目录。

| 阶段 | 目标 | 负责范围 |
| --- | --- | --- |
| 01 电机 | 两轴小角度、PWM与方向 | 现有 ARM 驱动与 FPGA PWM |
| 02 轨迹 | 三层弓字扫描与限位 | 复用正式轨迹规划与执行核心 |
| 03 同步采集 | 录像就绪后启动扫描 | 相机录制、运动与时间证据 |
| 04 重建渲染 | 使用阶段03原视频运行正式链路 | 板端 CPU、NPU、FPGA及网页 |

当前默认轨迹为俯仰+30°、0°、-30°，水平−30°至+30°交替扫描，两轴20°/s，计划16.66秒，含前后各0.5秒画面后计划录像17.66秒，目标15–20秒。上一轮30°/s快版及原46.06秒慢版独立保留；所有时间先为规划值，实际运动与画质须现场验证。

最新统一参数、60fps假设、约90°×90°前方扇区的覆盖口径及每步检查项见[SCAN_PLAN.md](SCAN_PLAN.md)。该角度是沿用当前裁剪方式的几何预期，不是已验收的高斯覆盖范围；第二、三步使用同一份配置，第四步单独检查三层及交界效果。

## 开始之前

本目录是测试入口，实际驱动、录制、标定和重建仍调用原模块，不另外维护一套实现。所有命令均在开发板现有工程的 `mvsplat` 目录运行。也可直接运行本目录四个编号 `.sh` 文件；它们会定位工程目录，`PYTHON` 环境变量可指定现有 ARM Python。

```sh
cd /实际工程位置/examples/3dgs_reconstruction/mvsplat
export PYTHON=/现有ARM环境/bin/python3
RUN=/root/fpga43dgs_private/camera_acceptance_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
```

下文 `--out` 必须是尚不存在的新目录；保留失败证据，重跑时换新目录名。入口默认预览，只有加 `--execute` 才会在 ARM 板上执行。不要把测试视频、访问令牌、部署配置和输出提交到 Git。

### 固件和硬件条件

- 使用已经编译的 J17 23/21 两路 PWM **候选固件**。它尚未完成上板波形验收，四阶段脚本不会自动刷写 BOOT，也不会重新综合。
- 水平信号接 J17 **物理23脚**，俯仰信号接 **物理21脚**，物理2脚共地；舵机外供7.4 V，俯仰中位安装为水平。原3/5脚方案不再使用。
- 新固件应同时提供冻结渲染器 FLK1 ABI 与 SRV1 ABI3；旧固件会被驱动拒绝运动，不应绕过检查。固件位置、约束和回退说明见上级 `CAMERA_SERVO.md`。
- 使用原 ICraft SDK 编译 ARM 驱动：`sh video_input/build_camera_servo_native.sh`。调整轨迹只改 JSON，不重新综合；这条编译命令也只编译 ARM 程序。
- 开始电机测试前结束正在占用硬件的推理/渲染进程；开始录像前结束占用同一摄像头的旧预览。保留设备独占锁，不用 `killall` 或删除锁文件抢占设备。

## 01 电机测试

入口：`stage01_motor.py` / `01_motor.sh`。先只读检查，再分别测试两轴。

```sh
sh video_input/acceptance/01_motor.sh --out "$RUN/01_preview" --action both
sh video_input/acceptance/01_motor.sh --out "$RUN/01_status" --action status --execute
sh video_input/acceptance/01_motor.sh --out "$RUN/01_pan" --action pan --execute --wiring-j17-23-21
sh video_input/acceptance/01_motor.sh --out "$RUN/01_tilt" --action tilt --execute --wiring-j17-23-21
```

每个单轴执行1500→1450→1500→1550→1500 µs的小幅动作，结束关闭 PWM。先在舵机外供关闭时用仪器测约20 ms周期和目标脉宽，再恢复外供观察小角度动作。通过条件是周期/电平正确、轴向正确、能回中、无干涉；寄存器读回和脚本退出码不能代替实物观察。保留示波器截图、`report.json` 和各轴 `.log`。

若报 `SRV1 is absent`，优先确认候选固件是否已加载；若报 `Board is in use`，让原硬件任务正常退出。不要通过放宽身份或电压检查继续试转。

## 02 预定轨迹

入口：`stage02_trajectory.py` / `02_trajectory.sh`。不打开摄像头。

```sh
sh video_input/acceptance/02_trajectory.sh --out "$RUN/02_preview"
sh video_input/acceptance/02_trajectory.sh --out "$RUN/02_live" --execute --wiring-j17-23-21
```

默认轨迹从俯仰+30°开始，水平−30°→+30°；右端降到0°，水平+30°→−30°；左端降到−30°，再水平−30°→+30°，最后回中。层内不停，只在两端换层。两轴均为标称20°/s，833点，计划16.66秒；Linux调度、启动检查及寄存器读写会使实耗时增加。更新仍为20 ms一点、最大5 µs步长，沿用10 µs限制及原PWM固件，不需要重新综合。

通过条件是三条水平扫描方向交替、仅在两端换层、机械不碰撞并最终回中。俯仰硬保护仍为±45°对应的1000–2000 µs；本轨迹目标仅1167–1833 µs。保存 `trajectory.csv`、同名 `.json`、`motion.csv`、`native.log` 以及实测角度/运动录像。角度仍须实物标定。

`--profile /绝对路径/自定义.json`可单独测试其他轨迹。`camera_scan_bow_fast15.json`保留30°/s、11.66秒快版；`camera_scan_bow_conservative.json`保留原水平10°/s、俯仰5°/s、46.06秒慢版。原单层逐点和螺旋配置也继续保留。三种速度都需实测舵机跟随、端点振动和运动模糊。

## 03 运动和录像同步

入口：`stage03_sync_capture.py` / `03_sync_capture.sh`。不运行模型，也不开始重建。

```sh
"$PYTHON" -m video_input.capture --device /dev/video0 --probe
sh video_input/acceptance/03_sync_capture.sh --out "$RUN/03_preview"
sh video_input/acceptance/03_sync_capture.sh --out "$RUN/03_live" --device /dev/video0 --width 1280 --height 720 --fps 30 --execute --wiring-j17-23-21
```

若探测确认同分辨率支持60fps，可将上一条命令的`--fps 30`改为`--fps 60`；仅预览参数时不要加`--execute`。不要同时运行两个采集入口。第四步直接使用这次的`capture_handoff.json`，不用也不应另外修改视频帧率。

以探测结果选择实际支持的MJPEG模式；确认支持1280×720、60fps后可显式加`--fps 60`，分辨率变化则必须重新匹配标定。当前默认仍为1280×720、30fps。录像采用原`CameraRecorder`和原生MJPEG/Matroska，无额外视频重编码；60fps下还需验证实际时间戳、USB吞吐和存储预算。

执行顺序是：等待新鲜预览→启动录像→确认录像已出帧→保留0.5秒前置画面→启动扫描→扫描退出后保留0.5秒→停止录像→检查帧率、时间戳、时长和SHA256→生成`capture_handoff.json`。默认计划录像17.66秒，加2秒调用余量的预检预算为19.66秒；目标由`--target-seconds 20`记录，实际长度由录像验证取得，报告中的`target_met`才表示是否在20秒以内完成，另行核对是否符合15–20秒期望。

默认`--max-seconds 20`是保护上限，并非固定录制20秒：轨迹完成后立即按尾段设置停止。默认轨迹不接受15秒截断；要比较上一轮快版，显式加`--profile video_input/camera_scan_bow_fast15.json --target-seconds 15 --max-seconds 15`，但实耗时超预算仍会终止运动，不把截断录像当全程成功。独立旧网页自动录制入口没有自动驱动此扫描。空间不足、相机断流、运动失败或超时均中止，保留失败原文件。

若快版实拍跟随或画质不佳，可用慢版对照，同步放宽录制上限：

```sh
sh video_input/acceptance/03_sync_capture.sh --out "$RUN/03_slow" --profile video_input/camera_scan_bow_conservative.json --max-seconds 60 --execute --wiring-j17-23-21
```

这是**软件顺序同步**，保证先有录像再发运动命令；不是传感器曝光与 PWM 的硬件同步。驱动在真正运动前仍有身份/时钟检查，日志明确记录的是录像就绪与运动进程派发时刻；严格脉冲延迟需仪器测量。

通过条件是录像包含到起点、三层扫描和回中全过程，没有提前结束；`report.json` 完成、运动日志完整且 `capture_handoff.json` 存在。它绑定的 `captures/clip_*/capture.json` 和 `capture.mkv` 必须一起保留。录像默认仅256 MiB存储预算，触及预算会失败，不能把它当成功的全程录像。

## 04 相机输入适配与 NPU/FPGA 渲染

入口：`stage04_reconstruction.py` / `04_reconstruction.sh`。沿用正式 `pipeline.py initialize`、NPU部署门槛、冻结 FPGA 渲染器和网页查看器。

先将 `deployment.example.json` 复制到工程外的 `$RUN/deployment.local.json`。把占位路径替换为**当前已验收部署**的 Python、权重、MVSplat参考代码、NPU bundle、动态库、oracle、资格记录、渲染器和当前相机标定文件的绝对路径，子图选择也沿用该资格记录绑定的准确顺序。示例中的十子图只是填写格式，不是重新核验后的资产声明；不要伪造资格文件或为通过测试切成CPU。若当前走严格门槛，则删除 `npu_acceptance` 字段，仍须真实通过该门槛。

```sh
sh video_input/acceptance/04_reconstruction.sh --out "$RUN/04_preview" --capture "$RUN/03_live/capture_handoff.json" --deployment "$RUN/deployment.local.json"
sh video_input/acceptance/04_reconstruction.sh --out "$RUN/04_pair" --capture "$RUN/03_live/capture_handoff.json" --deployment "$RUN/deployment.local.json" --scope pair --execute
```

第一条只校验交接文件、哈希、配置与相机模式；不启动NPU或FPGA。第二条先用单组旋转帧对检查相机→高斯→FPGA→网页。首帧后到 `pipeline.log` 查查看器实际地址和令牌；默认监听板卡18083端口，与旧18080入口区分。先完成单组效果确认，按 Ctrl+C 结束它并释放硬件，再测试整段稀疏多组：

```sh
sh video_input/acceptance/04_reconstruction.sh --out "$RUN/04_scan" --capture "$RUN/03_live/capture_handoff.json" --deployment "$RUN/deployment.local.json" --scope scan --execute
```

`scan` 是默认模式，使用现有真实旋转估计、64个候选、最多8组、65536总高斯预算。只传入本分支现有主线已支持的参数，不依赖未合入的选组实验。`pair` 使用同一固定中心旋转路径但仅一组，**不会误入要求平移视差的普通双目路径**。相机原地旋转不提供可靠的三角测量深度，结果定位为固定中心环视；不注入虚构位姿或虚构基线。

相机适配包括复用阶段03原始素材、哈希确认、核对采集尺寸与标定尺寸一致、保留原帧编号、复用正式去畸变/等比缩放/裁剪及内参变换。手机或其他镜头不能套用这份相机标定。

验收时分别观察左、中、右以及上、中、下的画面，记录黑边、孔洞、接缝、重影和模糊，重点检查俯仰约+15°和−15°附近的两处交界。当前输入裁剪对应的中央截面覆盖预期约水平91.3°、垂直91.2°，不是整齐矩形或完整360°。已有自动选组重点仍是水平方向，不能仅凭首帧通过宣称三层覆盖均匀；跨层均匀性必须实景确认。层间有效视场重叠约1.24°，若跨层表现差，保留素材用于后续选帧与融合调试。

程序只在收到正确顺序的 `READY→VIDEO_COMPLETE→SCENE_READY→FRAME_COMPLETE`，且找到匹配哈希的128×128实际RGB帧后记录首帧成功。完整调用仍受PSS 700 MiB、系统可用96 MiB和设备侧至少64 MiB隔离保护；不移除既有NPU数值/图像资格核验，也不声称已将全部计算迁往NPU。

### 时间和效果怎样回报

- 阶段02：计划时间与实际 `motion.csv` 分开；它们均不是重建时间。
- 阶段03：采集时长、录像就绪到运动派发延迟、运动进程实耗时分别记录；核对`actual_video_seconds`和`target_met`，默认计划17.66秒不能填成实测成绩。
- 阶段04：报告 `VIDEO_COMPLETE→FRAME_COMPLETE` 的场景准备时间，并分列高斯场景准备与首次渲染阶段；这是已有视频的手动回放，排除录像、预热和阶段间等待，不能冒充“点击停止录像到出图”的自动闭环时间。
- 单组与多组分开报告；首帧耗时不代表交互换视角耗时，网页显示也不等于画质合格。以实际渲染图、各方向截图和 `CHECKLIST.md` 的人工检查为准。
- 保存 `report.json`、`pipeline.log`、`measurement.json`、`scene/events.json`、NPU资格结果及 `scene/board/*/frame.bin`。网页运行时可直接看已写出的报告；退出后补齐进程树内存测量。交互查看器被手动结束会在测量记录留下中断状态，应与正常结束/保护触发区分。

## 本机回归和当前边界

```sh
python -m unittest video_input.acceptance.test_acceptance video_input.test_camera_scan video_input.test_camera_servo video_input.test_recording -v
```

本机65项回归中64项通过、1项因Windows符号链接权限跳过；C++共享执行核心25项检查通过。覆盖默认不动设备、833点默认版、583点快版及慢版回退、第二三步配置与轨迹一致、60fps参数和第四步原采集模式交接、录像先于运动、断流/超时停止、各版时间预算、原素材交接和哈希、错误标定、CPU伪装NPU拒绝、首帧事件与帧哈希核验，以及第四步两种范围与本分支真实主线命令行的兼容性。命令见[VALIDATION.md](VALIDATION.md)。实际波形、机械角度、USB录制、板端NPU和FPGA以及画质仍需队友逐项上板；未做的项目在检查表标“待测”，不标“通过”。
