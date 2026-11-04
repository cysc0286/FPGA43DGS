# 视频到高斯到渲染

`pipeline.py` 是唯一整链 CLI：`initialize`、`web`、`view`。
六部分源码全部在 `mvsplat/`，具体归属与接口见 [代码结构](../../docs/CODE_STRUCTURE.md)。
当前默认 NPU 网页入口与显式 CPU 对照均保留；NPU 路径需匹配的图、SDK、权重及当前代码资格。
旧 OpenSplat 训练和整段 SfM 冷链不再作为入口。
预热、视频输入只记录，视频完成到首张目标帧计为场景准备；后续视角复用常驻渲染器。
