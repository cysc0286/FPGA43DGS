# 当前代码与归档交付

更新：2026-10-09。主入口见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md) 和根 README。

## 三种内容分开

1. **主线源码**：四模块、预热/场景协调、接口校验、当前回归代码、当前后端源码。
   `tools/package_core.py` 依据 `core_sources.json` 与 Git 跟踪清单选择文件；不扫本地试验树。
2. **完整冻结渲染包**：`mvsplat/rendering/package/`，独立保留四路硬件、CPU 源码、构建配置、资源和实板证据。
   `mvsplat/render_branch/` 为独立 HGR v4 候选，不默认调用，也不进入主线协作源码 ZIP。
3. **历史**：`archive/history_20261009/` 的 ZIP 与索引纳入 Git；原始路径可以在新目录恢复。
   本机大量运行目录在 `archive/local_workspace_20261009/`，不上传。

## 创建协作源码包

从 Git checkout 根目录，在新增主线文件已加入 Git 索引后运行：

```text
python tools/package_core.py --output /path/to/new_core_sources.zip
```

输出必须是新路径。打包器逐文件保存/核对 SHA256；它不编译或运行算法。
主线源码 ZIP 不含预训练权重、SDK、vendor、Python 环境、位流或场景数据，
也不包含已隔离的 OpenSplat 训练试验和旧 NPU 特征实验。
老 CLI 为兼容保留，若主动回放旧训练，使用完整仓库及对应历史依赖，不用精简主线 ZIP。
完整冻结包请另外取 `rendering/package/`；不要对只含源码的子集声称完整固件清单通过。

## 本次状态

已拆分视频、位姿、高斯生成代码；历史 ZIP 与文件哈希已核对。
冻结主线配置、C++/RTL、硬件和画面证据保留，未提交硬件工作未混入整理提交。
没有新上板、编译、推理或单元测试结果；此前验证数量只属于对应历史版本。
旧发布说明的完整内容已保存到 `documentation_history.zip`。
