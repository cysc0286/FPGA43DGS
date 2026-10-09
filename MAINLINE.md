# 当前主线和验收边界

更新：2026-10-09。架构入口见 [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)，
四个数据交接见 [INTERFACES.md](examples/3dgs_reconstruction/modules/INTERFACES.md)。

## 只保留一个推荐渲染配置

- 软件配置：`examples/3dgs_reconstruction/mvsplat/rendering/mainline.json`。
- 发布包：同目录 `package/`，含接受过实板验证的四路 grouped-shared 硬件、源码和证据。
- 入口：独立交互用 `LiveRenderer.mainline(...)`；整链显式指定 `--live-renderer`，由 `PipelineRenderer(profile="mainline")` 调用同一配置。
- CPU：常驻 C++、四线程、缓存场景、稳定深度基数排序、direct_collect、NEON 封装。
- FPGA：四路、200 MHz、Dense mode 2、有序合成、精确指数 ROM；不额外抽点。
- `render_branch/` 只保存 HGR v4 候选，默认不调用；`npu/` 不进入当前渲染后端。

## 四模块与预热

`initialize` 仅负责视频前的权重/运行环境/渲染资源，不计算新视频的位姿或高斯。
随后依次由 `video_input`、`pose_estimation`、`gaussian_generation`、`rendering` 处理。
`scene_preparation.py` 和 `warm_pipeline.py` 组织现有同步点及数据交接，不重复实现算法。
老 `modules/` 保留 COLMAP 冷路径和既有文件合同兼容；新的数值实现不再往那里新增。

## 时间和数据

| 指标 | 起止点 | 已保存证据 |
|---|---|---|
| 预热、视频输入 | 各自开始到结束 | 只记录，不作为系统性能成绩 |
| 场景准备 | VIDEO_COMPLETE → 首帧 FRAME_COMPLETE | 历史旧渲染接线约 26–28 s；当前统一主线接线尚未整链复验 |
| 后续换视角 | 已加载场景，相机请求 → 完整 RGB 返回 | 2026-10-08 四路平均 43.575 ms、中位数 42.183 ms、P95/P99 52.893/62.080 ms |

换视角输入固定 32768 高斯、SH3、128×128、相机 7/15/22，A–B–B–A 各版本 60 帧。
同轮候选为 53.302 ms。四路画质 PSNR 19.4519 / 20.2223 / 21.3486 dB，
SSIM 0.804464 / 0.785135 / 0.784232；LPIPS、连续路径闪烁与功耗未测。
全板历史资源：LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63.00%。
继承既有 AI 脉宽及 GT/CDC 限制，不能称全板无条件时序签核。
这些均不是本轮新测；目录整理也不构成新的性能或画质收益。

[完整同轮结果](examples/3dgs_reconstruction/mvsplat/render_branch/validation/RESULTS.md)。
[历史试验索引](archive/history_20261009/index.json)。

## 当前整理的核验范围

完成历史 ZIP 与各成员哈希核对、移动前源文件哈希核对、算法定义静态搬迁核对。
保留冻结包、主线配置、C++/RTL、未提交硬件工作；不运行板卡、编译器或算法测试。
后续恢复测试时，先验证旧 import 兼容、四模块交接、预热首帧，再做同输入的换视角回归。
