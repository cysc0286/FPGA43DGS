# 常驻 CPU＋FPGA 渲染

正式入口是 `LiveRenderer.mainline(...)`，由 `mainline.json` 选择四路冻结配置。
`live_renderer.cpp` 常驻场景并缓存视角无关数据；每次新相机只执行投影、分组排序、FPGA 求值合成和完整 RGB 回传。
`pipeline_adapter.py` 接收生成模块的内存高斯行；`camera.py` 管理相机数学，`contracts.py` 校验文件交换 ABI。
网页／HTTP 已迁至 web；渲染模块不读取 HTML，也不依赖网页包。

`render_main/` 保存冻结硬件、CPU／RTL、资源与实板证据，151 个清单文件保持原字节。
旧独立 render_branch 和 releases 已从当前代码树删除，整理前备份在仓库外。
`resident_runtime.py` 和 `*_resident.cpp` 是同一渲染接口的兼容回退，保留外部匹配运行包的要求。
构建／验证脚本也归本目录；从 mvsplat 用 `python -m rendering.check_resident --help` 查看工具。
`tests/display_reference/render.py` 只保留独立标量图像转换 oracle，没有另一份生产入口。

现有资源、时间和画质见根 MAINLINE.md。本轮仅目录整理和软件核验，不声称 FPGA 内核或时序改善。
