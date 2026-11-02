# 共用板卡传输依赖

本目录仅保留 board/remote.py，供当前主线部署与诊断调用。
旧合成器、仿真、烧录和测试入口已从活动树删除，原文件按路径/哈希保存在
[清理归档](../../archive/history_20261102_cleanup/README.md)。

它是共用 SSH 基础工具，不是当前渲染模块；真实后端位于
[render_main](../3dgs_reconstruction/mvsplat/rendering/render_main/README.md)。
