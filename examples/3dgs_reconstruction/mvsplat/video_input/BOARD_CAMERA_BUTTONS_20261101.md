# 2026-11-01 网页按钮录制与重建

用户要求自己在 UI 中按“开始录制”和“结束录制”，替代聊天倒计时配合。
本轮已实现并部署真实板端服务；入口 **http://127.0.0.1:18433/**，交付时为 idle，
无已选测试短片。电脑 SSH 转发进程和板端服务需保持运行，重启后须重开。

## 功能与边界

1. “开始录制”：关闭原空闲预览流，启动单个 V4L2 输入；同一 FFmpeg 把原始 JPEG 包
   写入临时 MKV，并向页面发送预览 JPEG。页面先显示“正在启动”，出帧后显示录制状态与时长。
2. “结束录制”：向 FFmpeg 发 `q`，写好封装尾部，检查 JPEG 包时间戳、分辨率、帧率和至少 1 秒有效素材，
   通过后发布 capture.mkv/capture.json。检查期间显示保存状态，失败保留 partial 和日志。
3. 每段最长 20 秒，文件上限 min(256 MiB, 空闲空间−64 MiB)，触及边界会自动停止并说明原因。
   太短、低帧率、相机断流、空间不足等不交付重建；未到边界的意外提前断流不能伪装为空间自动停止。
4. 页面刷新从后端恢复真实状态，重复开始被拒绝；暂停显示/关闭页面不停止录制，仍受后端边界限制。
   按钮位于预览图上方的吸顶工具条，320 px 窄屏可换行；有计时、可用空间、短片摘要和错误提示。
5. 保存完成后由用户点击“重建场景”；协调入口启动现有 pipeline initialize，预览暂时释放以让出内存。
   真实模型和 FPGA 首帧完成后恢复预览，并提供“打开交互场景”链接。
   正在重建时禁止开始新录像，旧场景的链接与当前尚未重建短片有明确区分。
6. 再次重建会结束上一查看器、等待旧任务退出后启动新任务，避免两个 FPGA 会话并存。
   监听只在板端 loopback；写操作带随机会话 token，跨页面未授权请求被拒绝。

业务采集实现在 `video_input/recording.py`，复用 `preview.py` 的单源 JPEG 解析，
`initialize/camera_ui.py` 仅协调生命周期；四模块算术和冻结渲染包未改。
`measure.py` 新增 SIGTERM 清理处理，退出监控时先终止其隔离的子进程组，避免遗留模型/渲染进程。
`initialize/board_sync.py` 带上两个新入口。没有新增独立业务模块。

## 部署与复现

本机仓库 main 基线仍为 `a6d621767b5b91c9142954837ee5a04197e8995f`，本次增量未提交。
板端隔离候选 `/root/fpga43dgs_reconstruction/camera_buttons_20261101`，最终 74 个源码文件
与本机逐字节核对；模型、vendor、deps 软链接复用上次隔离部署。旧目录和原固件未覆盖。
相机 `/dev/video0`，1280×720 MJPEG、请求 60 fps。板端旧时钟仍在 3 月，录像文件名因此是 3 月；
报告日期为电脑会话日期，计时用 monotonic/封装 PTS，不修改板卡时钟。

板端保留原 libgomp 预加载、LD_LIBRARY_PATH 和 deps PYTHONPATH。在候选的 mvsplat 目录：

```sh
/root/fpga43dgs_reconstruction/arm_env/bin/python3 -m initialize.camera_ui \
  --device /dev/video0 --fps 60 --max-seconds 20 --host 127.0.0.1 --port 8081 \
  --out /root/fpga43dgs_reconstruction/camera_buttons_20261101/sessions \
  --weights ../weights/re10k.ckpt --vendor ../vendor/MVSplat_reference \
  --renderer /root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928 \
  --live-renderer /root/fpga43dgs_reconstruction/render_exact_20261001_direct/source_verified_source/rendering/live_exact \
  --focal-ratio 0.5256751337 --viewer-public-url http://127.0.0.1:18434/
```

电脑 loopback 18433 → 板端 8081，18434 → 8080；本机转发脚本
`D:\SyncFiles\1_Workspace\FPGA3DGS\tmp\board_camera_20261101\button_tunnel.py`。
当前配置仍使用用户授权的 95° FoV 估算、居中主点、零畸变。正式标定后可改传 `--calibration`；
两种内参入口必须二选一，不能隐式混淆。模型与渲染主线哈希沿用上一轮记录，未启用 NPU。
重建 watchdog：全树 RSS 650 MiB、系统保留 128 MiB、初始化/重建/交互合计 1800 秒；
录制服务本身持续运行，重建时限结束不自动重跑。输出仍为 128×128。

## 验证、指标与证据层次

| 项目 | 本轮结果 | 范围 |
|---|---|---|
| 本地相关回归 | **75 项通过**，无跳过，一次 8.103 秒 | 原 68 项加按钮录制 7 项；测试耗时不是板端性能 |
| 板端逻辑回归 | **10 项通过**，0.736 秒 | 7 项录制状态/HTTP/异常＋3 项进程树统计；设备用替身 |
| 真实浏览器录制 | **10 类检查通过，页面脚本错误 0** | 真实 UVC 出帧、开始、计时/录制中图像、重复开始禁用、刷新恢复、结束保存、重建可用、预览恢复、暂停继续、320px布局 |
| 真实手动录制第 1 次 | 202 帧、3.412 秒、58.9097 fps、14,010,809 字节 | 开始请求到封装/probe/哈希完成 5.4702 秒，含用户等待，非停止按钮延迟 |
| 真实手动录制第 2 次 | 188 帧、3.176 秒、58.8791 fps、12,756,729 字节 | 同范围 5.0101 秒；两次均是测试短片，画面偏黑，未作为重建样本 |
| 新协调入口真实重建 | 上次已成功的实拍片段回归通过 | 真权重 CPU＋FPGA，初始化→首帧→查看器→正常退出→预览恢复调用 |
| 新协调入口图像 | 原视角 RGB 与上一轮同场景像素哈希一致 | 相对固定旧输出最大/平均像素误差均 0；不是黑色按钮测试片段的画质结论 |
| SIGTERM 清理 | 1 项真实 Linux 子进程测试通过 | watchdog 写 interrupted，隔离测试子进程已消失，无遗留 |
| 最终交付 | idle，实时 JPEG 可取，无错误，74 个源码与部署一致 | 不预选测试短片，用户自行开始录制 |

本地复现命令在 mvsplat 下：

```sh
python -m unittest video_input.test_recording video_input.test_preview initialize.test_measure \
  video_input.test_camera video_input.test_camera_pipeline pose_estimation.test_camera_selection \
  rendering.test_viewer test_fast test_initialize test_pipeline_renderer test_live_rendering -v
```

自动化状态测试包括：重复点击、未录制不能停止/重建、短片不足 1 秒、低磁盘、相机断流、
封装检查失败保留证据、手动与自动时长上限、提前 EOF、HTTP token、重复录制与预览恢复。
自动时长上限/异常由逻辑替身验证；本轮未故意让真实板卡磁盘耗尽或拔掉相机。

新协调入口使用上轮真实短片做一次回归：预热 **46.7440 秒**，VIDEO_COMPLETE→FRAME_COMPLETE
**51.0532 秒**，查看器单次原生请求 **47.13 ms**。外层完整过程 106.7036 秒包含加载、归档、测试请求和退出；
不是纯重建延迟，未提供新多轮均值/P95，也不把它与前一轮 48.38 秒不同运行条件相减宣称回退。
新监控的全进程树峰值 **635.7344 MiB**（50 ms 采样、共享页相加），最低系统可用内存 **266.8477 MiB**，
650/128 MiB 门槛通过。该峰值不包含独立浏览器/转发进程，测试期间相机预览已释放。
最终 idle 摄像头服务树快照 **60.1445 MiB**、系统可用 **666.9453 MiB**，为单点而非峰值。
本次没有重新测浏览器持续显示 FPS、停止请求→保存完成的独立延迟、CPU 峰值或长期运行稳定性。

算法/画质未改：复用固定原实拍场景做像素一致性回归，新 UI 短片未评估 PSNR/SSIM/LPIPS，
未重新做实测标定。旧实拍 21.62 dB/0.885 的单图范围见前一轮报告，不能套到当前偏黑画面。
BOOT/RTL、FPGA 4 路主线配置未改；本轮没有新综合、LUT/FF/DSP/BRAM、WNS/TNS/hold/脉宽签核，
没有重测硬件时钟、DMA/DDR 总流量、板端功率或能量/帧，也没有 NPU 性能结论。

最终根文件系统约余 **156.02 MiB**；已把两段测试录像、日志和重建验证结果取回电脑，未删除板端旧实验。
空间限制可能使用户录像在 20 秒前自动结束，页面会显示原因；这不等于摄像头断流。

证据位于 `D:\SyncFiles\1_Workspace\FPGA3DGS\tmp\board_camera_20261101\`：
`button_ui_tests.txt`、`button_board_tests.txt`、`button_browser.json`、desktop/mobile 截图、
`button_capture_evidence/`、`button_job_verification.json`、`button_job_measurement.json`、`button_job_evidence/`、
`watchdog_cleanup_result.json`、`button_source_manifest_final.json`、`button_final_audit.json`。
部署初始和最终清单分别保存；不会把修复后的代码版本冒充先前运行版本。

## 优劣与下一步

改进：用户自主控制起止，录制时有同源预览，刷新可恢复状态；短片通过检查才进入原链，失败可重录；
重建后自动给交互链接，无需人工拼命令或聊天倒计时。
代价：预览轮询、双输出与状态服务有 CPU/传输成本，重建初始化仍需几十秒，板端剩余空间有限；
按钮改善操作便利性，不代表重建提速、提高画质或节能。
下一步优先在明亮、有纹理场景中由用户自行录 5～10 秒，核对实际操作体验；
需要连续大量使用时再安排持久存储和多场景/标定对照，不自动删除旧数据腾空间。
