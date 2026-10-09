# 高斯到渲染

当前交互渲染主线在 [mvsplat/rendering](../../mvsplat/rendering/README.md)，
发布冻结包在 [rendering/render_main](../../mvsplat/rendering/render_main/README.md)。
并列的 `render_branch` 只作实验归档，不由此模块调用。
本模块原有显式冻结包适配器保留兼容。


输入 RenderInput；render() 调用外部冻结包 render.py。独立命令 pipeline.py render。后端研发源码位于 examples/3dgs_flicker_hw，本目录只接接口；ARM 二进制在电脑上只允许 --plan。

共用字段与运行命令见 [接口文档](../INTERFACES.md)。
