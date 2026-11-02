# 视频—NPU—FPGA—网页联调与 main 整合

## 当前主线与证据边界

2026-11-02 已完成固定公开视频 → 真实 NPU backbone_cnn＋depth_head → CPU＋冻结四路 FPGA 首帧。
推荐入口默认包含 backbone 两次真实 SDK 执行的输入残留规避；仍需当前图像 receipt 与输入切换检查。
CPU 同调度 24.512 s，修正双子图 24.353 s，额外完整前向预热候选 23.933 s；
各一次，不宣称稳定加速，20 s 目标未达到。它不是全网 NPU，也不是完整在线 SLAM。
[最新实板正负结果与原始证据边界](gaussian_generation/npu_branch/NPU_DEPTH_VALIDATION_20261102.md)。

此前单 backbone 场景已做左右双栏网页实测；修正双子图三留出视角与完整视频首帧均已验证，
但该修正视频场景的网页恢复因断网尚未完成。最后原始证据下载与一行验证元数据板端同步同样待补。
本次用户要求合入 main；只执行软件核验与 Git 整合，不再连接/烧录板卡。
原 main 工作目录的未提交硬件实验保留。

## 调用链和代码归属

~~~text
本地视频 / 摄像头短片 / 连续帧窗口
  → video_input：CPU 解码、校正、上下文选择
  → pose_estimation：CPU 几何、目标位姿
  → gaussian_generation：常驻 MVSplat，NPU backbone＋depth_head，CPU 其余网络
  → Gaussian 参数校验与现有 rows/相机 ABI
  → rendering：常驻 CPU 投影/排序＋冻结 FPGA 求值/有序合成
  → 网页左侧选择观察点 → 确认 → 板端返回 → 右侧显示
~~~

四模块目录与交接保持不变，initialize 只预热和协调生命周期。
gaussian_generation/npu_branch/config.py 为视频、按钮、连续窗口共用后端配置，
qualify.py / acceptance.py 负责图像验收与资产绑定；
rendering/viewer.py、viewer.html、viewer.css、viewer.js 分开服务、页面、样式和交互。
软件入口贯通不代表相机 NPU 和长期连续 NPU 更新已完成实板验收。
NPU 错误明确报错，CPU 回退由调用者重新显式选择。

## NPU 就绪与图像门槛

默认严格逐元素门槛 rtol=2e-3、atol=2e-4 保留，TF32 仍未通过。
用户允许轻微图像变化时，可使用固定测试场景的原生 FPGA 图像验收：

1. 在实际保留 worker 中交错执行两轮 oracle，检查有限值、形状和重复哈希，保留严格误差。
2. 每个子图执行 B,B,A,A 输入切换，首次/再次结果必须一致且 A 与 oracle 对应；旧 receipt 不可绕过。
3. 同 prepared input，CPU 与 NPU 完整前向均用同一 FPGA 后端渲染至少三个留出视角；
   对 CPU 图像 PSNR ≥40 dB、SSIM ≥0.99，对真实图像 PSNR 降幅 ≤0.2 dB、SSIM 降幅 ≤0.005。
4. FPGA 渲染后重跑完整前向和 oracle，验证设备共存后高斯重复性。
5. receipt 绑定模型源码/权重、图、桥接库、SDK、渲染器、BOOT 文件和实际运行身份。

推荐 worker 默认 MGS_NPU_SETTLE_BACKBONE=1，backbone 每请求真实执行两次并计入全部成本。
显式 0 仅用于诊断；一轮 oracle 重复不能替代输入切换。SDK 根因和历史权重预检错误仍未关闭。
图像验收只证明固定输入的受控变化，不是任意新场景质量承诺。
BOOT 文件哈希检查也不是运行中 PL 身份的硬件证明。

## 板端复跑顺序

先用 initialize/board_sync.py 同步到新候选目录，复用既有 vendor、权重和 ARM 环境，不覆盖旧目录或 BOOT。
设备先开 FPGA 渲染器、再开 NPU；关闭时顺序相反。
worker 默认预留低 64 MiB PL DDR，桥接库必须支持并核对；旧不兼容库拒绝启动。

下面 B 为核实过的历史图/oracle 资产，C 为新候选目录，其他变量须替换为实际已配置路径。
Python 必须为现有 ARM MVSplat 环境，LD_LIBRARY_PATH 等沿用 initialize/board_accept.py。

~~~sh
cd "$C/mvsplat"
python -m gaussian_generation.npu_branch.build \
  --sdk /root/heterogs_npu/sdk_3.36.1 --out "$C/bridge_guard"
python -m gaussian_generation.npu_branch.qualify \
  --backend npu --partitions backbone_cnn depth_head \
  --partition-bundle "$B/candidate/compiled" --oracle-graphs "$B/candidate/graphs" \
  --npu-library "$C/bridge_guard/libmgs_npu.so" --worker-python "$ARM_PYTHON" \
  --weights "$C/weights/re10k.ckpt" --vendor "$C/vendor/MVSplat_reference" \
  --input "$B/candidate/input" --renderer "$FROZEN_RENDERER" \
  --live-renderer "$LIVE_RENDERER" --boot-image "$MOUNTED_BOOT" \
  --out "$C/qualification"
~~~

新构建桥接库哈希变化，须重新 qualify，不能复用旧二进制的 receipt。
三视角资格输入不能用仅一个目标的实时窗口代替。
通过后将上述 NPU 参数及 --npu-acceptance "$C/qualification/acceptance.json"
加到 pipeline.py initialize 或 initialize.camera_ui；完整命令见仓库 README。
默认不加 --warm-kernels，额外完整前向预热目前仅为候选。

## 网页和计时

- 左侧为最多 1500 高斯中心的预览，使用同一轨道相机计算；拖动/缩放/重置只修改待确认视角。
- 首次载入请求初始画面；之后点击确认才 POST /render，校验 request_id，忙时拒绝重复提交。
- 右侧显示真实返回帧、实际 RGB 尺寸、板端相机到 RGB 时间和浏览器请求到绘制时间。
- 左侧预览不承担完整 3DGS 渲染；手机窄屏上下排列，后续视角不重复 MVSplat 推理。

场景准备：VIDEO_COMPLETE → 首个留出视角 FRAME_COMPLETE。
预热、视频输入只记录；相机到 RGB 与浏览器 PNG/网络/绘制分开。
NPU 分项包括打包、IPC、SDK 提交/等待、输出转换；逻辑 I/O 字节不是实测 DDR 流量。
旧网页场景六次原生 40.63–45.95 ms，浏览器 57.5–67.2 ms；修正新场景待网页复测。
新视频候选对 CPU 首帧为 55.332 dB / 0.999232，对实拍为 19.450 dB / 0.804061。
资源沿用冻结硬件历史记录；本次未重新综合布线，功耗和真实 DDR 流量未测。

## 软件与归档

本次合并的软件回归、冻结包哈希、网页浏览器交互和源码包检查记录见
[合并核验](gaussian_generation/npu_branch/MAINLINE_MERGE_20261102.md)。
这些软件测试不替代板端 NPU 数值、真实视频和 FPGA 验收。
私人视频及其图像/高斯、权重、SDK、运行原始日志不进入提交。
