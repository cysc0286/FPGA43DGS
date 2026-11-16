# FPGA / NPU 板端交付说明

目标板卡为悟净 30TAI Lite。当前仓库保留视频到高斯生成、NPU 适配、CPU＋FPGA 渲染及网页入口；源码包额外携带已冻结并有哈希清单的 FPGA 板端产物。

## FPGA 渲染包

`examples/3dgs_reconstruction/mvsplat/rendering/render_main/` 保存当前渲染主线：

- `rtl/`：冻结实现的 Verilog/SystemVerilog RTL 与 ROM 初始化数据。
- `fpga/`：HLS/管线实现源码；`build_hls.ps1`、`build_hls.tcl` 提供 HLS 阶段复建入口。
- `hardware/BOOT.bin` 和 `hardware/ai7030_edif_top_disable_icap.bit`：冻结的板端启动镜像和 PL 位流，哈希受 `manifest.json` 保护。
- `hardware/reports/`：对应版本的 DRC、时序和资源报告。
- `cpu/`、`build_native.sh`：ARM 常驻渲染配套代码与 SDK 编译入口。

改动包通过 `tools/package_core.py` 读取 `render_main/manifest.json`，逐文件检查大小和 SHA-256 后写入 ZIP。此冻结包复用既有板测结果；它不是新的板端验收，也不会自动改写 SD 卡或烧录位流。

## 最新相机PWM整板候选

`examples/3dgs_reconstruction/mvsplat/video_input/firmware/j17_23_21_20261115/`提供已生成的BOOT、PL位流、完整性清单、相对父平台的集成补丁和已有实现报告，水平/俯仰信号为J17物理23/21脚。该镜像增加50Hz双路PWM，保留渲染逻辑；尚未上板验证，不能继承四路渲染旧镜像的整板验收结论。默认冻结渲染包不被此候选覆盖，脚本不自动烧录。

先运行`python tools/check_camera_firmware.py`核对文件和四个BOOT分区，再按`video_input/acceptance/README.md`完成电机、轨迹、同步录像和NPU/FPGA渲染四阶段验收。主机回归、RTL仿真、布局布线与实际机械运动分别记录。完整原厂平台和受许可工具仍须另行取得。

## NPU 生成端

`examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch/` 保留正式 NPU 子图导出、ICraft 编译入口、桥接代码、运行时与资格工具。匹配提交的 MVSplat 推理源码快照同时位于 `examples/3dgs_reconstruction/vendor/MVSplat_reference/`，并附文件级 SHA-256 清单。它是高斯生成模块的硬件后端；视角切换仍走 CPU＋FPGA 渲染。

部署时需在配套环境准备 ICraft/CustomOp SDK、板端 NPU runtime、与当前 checkpoint 对应的编译图、oracle 和资格记录。模型权重、SDK、生成的私有/设备相关图包及个人数据不作为仓库资产。当前 NPU 只覆盖已明确选择的子图，注意力、代价体等仍由 CPU 处理；已有记录中的严格逐元素精度门槛尚未关闭，不能将源码存在描述为全网 NPU 或无条件加速验收。

## 软件上板与重新构建边界

- `initialize/board_sync.py` 按六模块部署清单同步运行时代码；SSH 密码只从 `FPGA_BOARD_PASSWORD` 环境变量读取。
- `render_main/build_native.sh` 在 ARM Linux 上编译常驻渲染程序，需配置匹配的 ICraft SDK 根目录。
- 若要从源头重新生成全板位流，还需单独取得悟净 30TAI Lite 原厂 Vivado 工程/IP 与平台文件，并使用配套 Vivado 2018.3、Procise/ICraft 版本。当前仓库没有完整 XPR/Block Design/XSA/XDC 工程快照，因此不能只靠 ZIP 在一台干净电脑上复建整张板卡。

## 生成和检查交付 ZIP

```sh
python tools/package_core.py --output FPGA43DGS-board-deploy.zip
python tools/check_render_packages.py
```

打包器核验源码哈希与冻结 FPGA 清单，并将压缩 ZIP 限制在 200 MiB 内。ZIP 内的 `SOURCE_SHA256.json` 可用于逐文件检查。
