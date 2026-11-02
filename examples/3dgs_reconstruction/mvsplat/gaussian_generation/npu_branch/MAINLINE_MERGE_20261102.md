# NPU 主线合并核验（2026-11-02）

用户授权将现有 NPU 主线合入 main，保持原代码架构。
本次只整合代码、验收逻辑、网页和说明，不连接板卡或修改固件。

## 范围

- 四业务模块 video_input、pose_estimation、gaussian_generation、rendering 不变；
  initialize 继续负责预热、生命周期与部署协调。
- 推荐显式 backbone_cnn＋depth_head NPU，CPU 其余网络；CPU 回退保留。
- worker 环境默认启用已实测 backbone 两次 SDK 执行规避，仍累计两次耗时、
  检查 B,B,A,A 输入切换和 receipt 身份。显式诊断覆盖保留。
- 源码打包清单补入网页 HTML/CSS/JS，避免拆分页面后漏包。
- 同步 README、MAINLINE、结构/要求和最终联调文档，区分双子图新结果与旧网页场景。
- render_main、render_branch 和 BOOT 均保持原字节；原 main 目录未提交硬件实验不纳入。

## 软件核验

本次使用电脑检查环境，不是板端 NPU 或 FPGA 测量。

- tools/check_core.py --mvsplat：136 个 Python 源码语法通过；
  接口 16、兼容入口 3、MVSplat 146，共 165 项，161 通过、4 跳过。
  其中 3 项需要 FFmpeg/ffprobe，1 项需要本 Windows 会话不支持的符号链接创建权限。
- 新增默认 worker 策略及显式诊断覆盖测试；CPU 路径仍由既有集成测试检查。
- Playwright/Edge：9 项网页交互通过、页面异常 0；
  4 次合成 RGB 请求成功，HTTP 渲染错误 0。
  包括拖动/缩放不发请求、确认提交、预览/板端相机参数一致、双栏/窄屏布局、关闭。
- tools/check_render_packages.py：render_main 151、render_branch 197 个文件哈希通过；
  94 个调用源的主线隔离检查通过。
- git diff --check 通过。源码包从明确清单和 Git 跟踪文件生成，
  逐成员核对 SHA256 并检查网页资源包含、原始视频/生成数据排除。
  压缩包及原始软件日志仅保存在电脑 outputs，不上传。

## 沿用实板结果与评价

128×128、32768 高斯；同调度 CPU 首帧 24.512 s，修正双子图 24.353 s，
额外完整前向预热候选 23.933 s；各一次，不足以证明稳定加速，20 s 目标未达到。
修正首帧对 CPU 为 55.332 dB / 0.999232，对真实图像为 19.450 dB / 0.804061。
TF32 严格逐元素门槛仍失败，使用三留出视角原生 FPGA 图像政策；
输入残留采用两次执行规避，SDK 根因尚未关闭。

四路硬件历史资源：LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63%；
200 MHz，既有 AI 脉宽及 GT/CDC 限制未消除。本次未综合、布线或测功率/DDR 流量。
优点是 NPU 正式入口、失败行为和部署资产已统一，保持冻结渲染回退；
代价是预热/验收与两次 backbone 执行开销，首帧收益仍有限。

修正场景网页恢复、最后原始归档和板端一行验证元数据同步待设备恢复后完成。
此前网页换视角 40.63–45.95 ms（原生）/57.5–67.2 ms（浏览器）来自旧场景，
不能冒称修正后新场景已网页复测。完整正负结果见 NPU_DEPTH_VALIDATION_20261102.md。
