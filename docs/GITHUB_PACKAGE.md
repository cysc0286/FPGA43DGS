# 主线源码与板端部署包

当前业务结构见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。
默认打包收录主线源码，并按 `render_main/manifest.json` 逐项校验后加入冻结 FPGA 部署资产（RTL ROM、BOOT.bin、位流及对应报告）：

```sh
python tools/package_core.py --output /path/to/new_sources.zip
```

整理期间新路径尚未进入索引，可显式加 `--include-working-tree` 验证完整工作区；该选项遵守 Git 忽略规则和源码清单，并不暂存／提交文件。
打包后逐文件核对 SHA256，并拒绝超过 200 MiB 的压缩包。NPU 运行时代码、部署脚本与资格工具在包内；ICraft SDK、动态库、编译图、模型权重、oracle/资格凭证和 Python 环境需按板卡环境单独生成或安装，不从个人环境打包。
完整匹配硬件和回退资产见 `examples/3dgs_reconstruction/mvsplat/rendering/render_main/`；包内冻结资产与清单逐项一致。运行 `python tools/check_render_packages.py` 可独立核验冻结渲染包。
包内**没有原厂完整 Vivado 工程快照**（如 XPR、Block Design、XSA、XDC 与厂商平台工程）；仓库保存冻结 RTL/HLS 源、复建脚本与已验证 BOOT/位流。全板重新综合布线仍需要悟净 30TAI Lite 配套平台文件、Vivado 2018.3 和厂商工具。
用户素材、视频、场景、输出、权重、SDK、环境、密钥和私有部署凭证不进入 ZIP。仓库外的整理前备份仅用于本机恢复。
