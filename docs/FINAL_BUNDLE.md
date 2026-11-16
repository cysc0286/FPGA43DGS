# 最终实现链路交付包

交付文件为 `FPGA43DGS-final-20261104.zip`。包内保留统一入口和六部分主线：视频采集／上传、位姿估计、MVSplat 高斯生成、CPU＋FPGA 渲染、网页交互，以及正式 NPU 后端代码。

## 可直接恢复的内容

- 固定到提交 `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01` 的 MVSplat 推理源码快照及 MIT 许可证；未附作者预训练权重。
- 当前六部分运行源码、网页静态资源、ARM 构建脚本、NPU 子图导出／编译／组装／桥接／运行／资格工具。
- 冻结的悟净 30TAI Lite 四路 FPGA 渲染包：RTL/SystemVerilog、ROM 初始化数据、HLS 源码、CPU 端源码、BOOT.bin、PL 位流及原有资源／DRC／时序／HLS 报告。所有冻结文件按 `render_main/manifest.json` 逐项验证。
- 历史整链结果与主线状态说明。`FILELIST.txt` 列出所有归档内容；`SOURCE_SHA256.json` 为各文件和清单提供 SHA-256。

解压后从 `examples/3dgs_reconstruction/` 查看统一入口及 CLI。顶层 `README.md`、`docs/CODE_STRUCTURE.md` 和本文件说明运行次序、接口与恢复边界。生成本机匹配的高斯权重、ICraft 编译图、NPU 动态库和资格记录时，应按 `docs/BOARD_DEPLOYMENT.md` 使用相同的 SDK／工具版本，并为新源码身份重新资格验收。

## 必须另行取得／生成的项目资产

为避免把不含必要运行资产的 ZIP 误称为“解压即运行”，以下内容未打包：

- MVSplat `re10k.ckpt` 预训练权重（从[作者提供的预训练模型目录](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU)获取；应核验 SHA-256：`83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a`）。
- 供应方 ICraft／CustomOp SDK、板端 NPU Runtime 与许可证；这些闭源工具需从供应方安装，不能从源码包恢复。
- 对应权重、ICraft 版本及板端 Runtime 生成的 NPU 编译图、`libmgs_npu.so` 与资格凭证。源码包内有导出、编译、打包和构建入口，可在匹配环境中重建；生成物不应与不匹配权重或 SDK 混用。
- 原厂悟净 30TAI Lite Vivado 平台工程和 IP，包括 XPR、Block Design、XSA、完整 XDC／板级文件及厂商 IP 仓库。仓库保存的是冻结部署二进制、提取的 RTL/HLS/CPU 源码和实现报告，并不包含原厂全板工程快照；因此可恢复并部署保存的 FPGA 主线，但不能仅凭此 ZIP 从空白环境重新生成全板位流。
- 与目标平台匹配的 Vivado 2018.3、Procise、Linux/ARM sysroot 与交叉编译 SDK，以及部署设备、网络和凭证。

USB 摄像头、视频、场景、导出结果、个人数据、Python 环境和本地密钥均未打包。最终 ZIP 以源码和冻结部署资产为主，压缩后限制为 200 MiB；不包含用户私有测试素材或大型模型权重。

## 文件清单与验证

解压前可查看 `FILELIST.txt`。`SOURCE_SHA256.json` 的路径相对于 ZIP 内 `FPGA43DGS/` 根目录；可用下列命令核对 ZIP 完整性和清单：

```sh
python tools/check_render_packages.py
```

归档器会拒绝缺失／改动的冻结 FPGA 文件、检查关键 NPU／FPGA 入口和报告、逐文件回读哈希、执行 ZIP CRC 检查，并确保压缩包小于 200 MiB。此次只做源码包检查；不会刷写板卡，也不会新增性能、画质、资源或时序结果。
