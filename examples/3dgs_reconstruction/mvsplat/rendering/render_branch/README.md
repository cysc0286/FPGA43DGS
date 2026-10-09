# PipeGS HGR 实验分支（不参与默认执行）

此目录位于 `rendering/` 内，与 `../render_main/` 并列。默认渲染仍使用旧版四路 grouped-shared；
本目录只保存新版两路 PipeGS HGR v4 的源码、固件及独立验证结果。
它不是新的默认渲染器，也不会在导入、初始化或正常渲染时自动烧录或调用。

2026-10-08 配对实板测试：32768 高斯、128×128、三个视角，每种固件 60 帧。
旧版平均 43.575 ms，新版平均 53.302 ms；新版慢 22.32%，因此保留为研究分支。
新版相对旧版的三视角平均 PSNR 变化 -0.00046 dB，平均 SSIM 变化 -0.000051。
这是已加载场景后相机请求到完整 RGB 的时间，不是视频重建时间。

## 独立结构

- `fpga/pipegs_hgr_v4_coord_2lane/`：新版两物理通道 HGR 核心、坐标转换和测试台。
- `fpga/hls/`、`fpga/ctu/`、`fpga/pipeline/`：完整 include 依赖，避免依赖本地实验目录。
- `rtl/`：与本次实板固件关联的冻结 HLS RTL 和 ROM。
- `hardware/`：候选 BOOT、PL 位流、布线后报告及原始集成清单。
- `validation/board_20261008/`：A-B-B-A 全部原始耗时、图像、安装/回退凭据。
- `validation/RESULTS.md`：速度、质量、资源和采用结论。
- `manifest.json`：固定候选身份、`default_enabled=false` 和逐文件 SHA-256。

使用上一级 `rendering/` 的共享 CPU 软件接口；本目录没有 Python 自动导入入口、
默认选择器或自动安装钩子。保留相同相机、Gaussian 和 DMA ABI。
正常运行 `pipeline.py`、预热和 `LiveRenderer.mainline(...)` 都不会调用此目录。
此处的 `render_branch` 是主仓库中的并列实验目录，不是默认 Git 分支切换。

## 单独复建

仓库根目录：`python tools/check_render_packages.py`。
本目录：`./build_hls.ps1 -Stage csim -Hls PATH_TO_VIVADO_HLS_BAT`。
可显式选择 `synth` 或 `cosim`；输出写入独立 `build/`，不会覆盖固件或旧版。
需要 Vivado HLS 2018.3；修改后的全板复建另需原厂平台。

此候选借鉴 PipeGS 的 HGR 思路，不是完整论文复现。四物理通道 HGR 候选此前
未完成布线，不包含可部署 BOOT；本目录封装的是已经完成实板对照的两路版本。
按当前结果，旧版仍是速度主线，新版供进一步算法和硬件研究。
