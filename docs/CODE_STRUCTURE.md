# 当前代码结构

更新：2026-10-09。**四个业务模块都在 `examples/3dgs_reconstruction/mvsplat/` 下。**
`initialize` 是视频前预热；两个协调文件管理调用顺序。它们不是第五种重建算法。

| 模块 | 实际实现 | 负责什么 | 推荐执行单元 |
|---|---|---|---|
| 视频输入 | `video_input/receipt.py`、`video_input/decode.py` | 接收已完成视频、选帧、缩放，保留原帧索引 | CPU |
| 位姿估计 | `pose_estimation/geometry.py` | 两视图 SIFT/匹配/Essential/三角化，留出视角 PnP | CPU |
| 高斯生成 | `gaussian_generation/runtime.py`、`adapter.py` | 固定权重前向、参数校验、协方差/SH/相机 ABI 转换 | 当前 CPU；NPU 另行验证 |
| 渲染 | `rendering/runtime.py`、`live_renderer.cpp`、`package/` | 常驻场景、投影、排序、FPGA 求值合成、返回 RGB | CPU＋FPGA |

这些是实际算法源码位置，不是再加一层同名包装。解码和几何从原 `video_input/prepare.py`
拆出；ModelRuntime 从 `initialize` 移出；Gaussian 导出从根目录移入生成模块。
原函数体保持不变；`export.py`、`initialize/model_runtime.py`、`video_input/prepare.py`
仅保留旧命令/import 兼容，不再维护第二份实现。

## 如何读代码

1. `pipeline.py initialize`：统一 CLI。视频前资源由 `initialize/session.py` 管理。
2. `initialize/run.py`：接收视频与事件记录，驱动 `warm_pipeline.py`。
3. `scene_preparation.py`：调用解码模块和位姿模块，发布上下文与首目标交接。
4. `gaussian_generation/runtime.py`：复用已加载模型生成世界坐标 Gaussian。
5. `gaussian_generation/adapter.py`：Gaussian 转 `N×62` rows，相机转 FLCAM001。
6. `rendering/pipeline_adapter.py`：首帧合同连接到 `LiveRenderer.mainline(...)`。
7. 后续视角直接调用 `LiveRenderer.render_camera(...)`，不重复前四个模块。

完整数据字段、数值含义和所有权见 [四模块接口](../examples/3dgs_reconstruction/modules/INTERFACES.md)。

## 哪些目录不属于默认执行路径

| 位置 | 定位 |
|---|---|
| `archive/history_20261009/` | 823 个历史文件条目的 ZIP 与哈希清单，含失败/负收益结果 |
| `archive/local_workspace_20261009/` | 本机未跟踪原始实验、日志、生成物；忽略、不上传 |
| `mvsplat/render_branch/` | 保留的 PipeGS HGR v4 独立冻结候选；主线不调用 |
| `mvsplat/npu/` | 前馈网络 NPU 候选、编译与诊断；没有新的数值验收 |
| `releases/3dgs_renderer_v1_20260928/` | 较早冻结回退及现有 CPU 契约依赖，不能当作最新位流 |
| `reconstruction/modules/` | 原 COLMAP/OpenSplat 文件接口兼容；`contracts.py` 仍被导出/校验使用 |
| `reconstruction/bounded/`、`npu_frontend/` | 历史训练/特征实验，不进入主线协作源码包 |
| `examples/3dgs_flicker_hw/` | 旧硬件开发工作树与未提交工作；日常复建使用当前冻结包 |
| `examples/3dgs_compositor/board/remote.py` | 现有板卡传输工具依赖，不是渲染算法 |

未提交硬件改动原地保留。本轮不把它们混入整理提交，也不把它们称为已验收主线。

## 打包规则

`tools/core_sources.json` 明确主线包含项。`package_core.py` 只选 Git 跟踪的合格源码，
避免递归扫进本地实验。历史 ZIP、`render_branch`、旧训练实验、运行结果不进入主线源码包。
主线源码包不含权重、SDK、模型、BOOT/bit；完整硬件复建/回退资产请另取 `rendering/package/`。
历史恢复必须解压到新目录，按 [归档说明](../archive/history_20261009/README.md) 操作。

本轮只有静态整理与文件完整性核对，未运行功能测试、编译或上板测速。
