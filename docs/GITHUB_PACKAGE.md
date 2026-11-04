# 源码交付

当前业务结构见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。
默认打包仅收录 Git 已跟踪且当前存在的合格源码：

```sh
python tools/package_core.py --output /path/to/new_sources.zip
```

整理期间新路径尚未进入索引，可显式加 `--include-working-tree` 验证完整工作区；该选项遵守 Git 忽略规则和源码清单，并不暂存／提交文件。
打包后逐文件核对 SHA256。用户素材、输出、权重、SDK、环境和 BOOT／位流不进入源码 ZIP。
完整匹配硬件和回退资产单独从 `rendering/render_main/` 获取，并运行 `tools/check_render_packages.py`。
仓库外的整理前备份只用于本机恢复，不提交 GitHub。历史旧包和候选已从当前树删除。
