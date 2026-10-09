# 四路渲染主线冻结包

此包固定已经实板验证的 grouped-shared 四路、200 MHz、Dense mode 2 渲染器。
调用入口保持 `rendering.runtime.LiveRenderer.mainline(...)`，配置在上一级
`mainline.json`。实验版本单独保存在 `../render_branch/`，主线不调用它。

基准范围为场景已经加载后，相机请求到完整 RGB 返回。2026-10-08 同轮 60 帧
平均 43.575 ms；本次封装不重新宣称产生性能收益，也不自动更新开发板。

BOOT SHA-256：`b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6`。
## 内容与入口

- `hardware/`：已测 BOOT、PL 位流、原始集成清单和布线后报告。
- `rtl/`：从已测硬件冻结目录提取的精确 HLS RTL/ROM 文件。
- `fpga/`：旧版 HLS 源码及本地 include 依赖闭包，独立复建。
- `cpu/`：常驻 C++ 渲染源码及参考代码依赖快照。
- `profile.json`：四路主线软件参数快照，与 `../mainline.json` 一致。
- `validation/`：历史实板、质量、资源及本次封装检查。`manifest.json` 核验包内文件。

独立检查：仓库根目录运行 `python tools/check_render_packages.py`。
HLS 2018.3 复建：在本目录运行
`./build_hls.ps1 -Stage csim -Hls PATH_TO_VIVADO_HLS_BAT`；
`-Stage synth` 增加综合，`-Stage cosim` 增加 C/RTL 联合仿真。
这些命令不会重写 BOOT，也不会连接板卡。

ARM Linux 安装好配套 ICraft SDK 后运行 `sh build_native.sh`，产物为
`bin/live_exact`。可用 `ICRAFT_SDK_ROOT` 指定 SDK 根目录。
本次板卡 SSH 连接超时，未取回已测 ARM 可执行文件；仓库提供源码和编译入口，
不把重新编译出的程序自动当作哈希 `d1220d46...` 的已测程序。
旧性能对应原始已测二进制，重编译产物须再次做帧差分/板测。

硬件固定为悟净 30TAI Lite 的现有平台及 SD 启动配置；这里只封装产物，
不自动刷写。修改 FPGA 后的全板复建仍需要原厂平台、Vivado 2018.3 和配套工具，
不能把 HLS 通过当作新全板布线通过。保留已知 AI 脉宽违例，不称全板无条件签核。
