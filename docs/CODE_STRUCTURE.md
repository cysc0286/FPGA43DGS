# 当前代码结构

更新：2026-11-02。**四个业务模块都在 `examples/3dgs_reconstruction/mvsplat/` 下。**
`initialize` 是视频前预热；两个协调文件管理调用顺序。它们不是第五种重建算法。
目录约束见 [项目要求](PROJECT_REQUIREMENTS.md)，执行规则同时写入根目录 `AGENTS.md`。

| 模块 | 实际实现 | 负责什么 | 推荐执行单元 |
|---|---|---|---|
| 视频输入 | `video_input/receipt.py`、`video_input/decode.py` | 接收已完成视频、选帧、缩放，保留原帧索引 | CPU |
| 位姿估计 | `pose_estimation/geometry.py` | 两视图 SIFT/匹配/Essential/三角化，留出视角 PnP | CPU |
| 高斯生成 | `gaussian_generation/runtime.py`、`adapter.py` | 固定权重前向、参数校验、协方差/SH/相机 ABI 转换 | NPU backbone＋depth_head＋to_gaussians＋to_disparity；CPU 其余网络 |
| 渲染 | `rendering/runtime.py`、`live_renderer.cpp`、`render_main/` | 常驻场景、投影、排序、FPGA 求值合成、返回 RGB | CPU＋FPGA |

2026-10-29：视频输入内部新增 `capture.py`（Linux UVC 短片）与 `calibration.py`（针孔标定/校正）；
位姿模块 `geometry.py` 增加候选对选择；`rendering/viewer.py` 提供常驻场景浏览器控制。
原模块边界和冻结渲染主线不变。以上为 10 月 29 日范围；11 月 1 日已按用户授权使用估算内参实板试跑。
11 月 1 日新增 `video_input/recording.py` 按钮开始/结束、预览同源、封装校验；
`initialize/camera_ui.py` 只协调既有录制、重建子进程与查看器生命周期。
11 月 1 日后续获准实现连续框架：`video_input/streaming.py` 为有界近期 JPEG 窗口及上下文序列化；
`pose_estimation/streaming.py` 固定初始参考并复用地标定位；`rendering/streaming.py` 串行交换模型和相机请求；
`initialize/streaming.py` 协调一个常驻模型、采样器和 FPGA 所有者；后端共用 NPU 配置，但长期 NPU 连续更新尚未实板验收。
局部点云随有效窗口更新，高斯按批更新，均不构成全局 SLAM 或全局高斯融合。
操作和当前验证见 [连续框架](../examples/3dgs_reconstruction/mvsplat/video_input/STREAMING.md)。
本轮已做真实短片按钮和原实拍片段重建回归，详见
[按钮录制实测](../examples/3dgs_reconstruction/mvsplat/video_input/BOARD_CAMERA_BUTTONS_20261101.md)。

这些是实际算法源码位置，不是再加一层同名包装。解码和几何从原 `video_input/prepare.py`
拆出；ModelRuntime 从 `initialize` 移出；Gaussian 导出从根目录移入生成模块。
原函数体保持不变；`export.py`、`initialize/model_runtime.py`、`video_input/prepare.py`
仅保留旧命令/import 兼容，不再维护第二份实现。

## 如何读代码

2026-11-02 内存增量：`memory_runtime.py` 只负责现有代价体调度；
`gaussian_generation/runtime.py` 管理内存/磁盘待归档对象与执行源码身份；
`initialize/result_retention.py` 管理同协议合格结果保优回收，非新增业务模块。
实板对照和当前持久部署见 [内存与存储记录](../examples/3dgs_reconstruction/mvsplat/initialize/NPU_MEMORY_STORAGE_20261102.md)。

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
| `mvsplat/rendering/render_branch/` | 保留的 PipeGS HGR v4 独立冻结候选；主线不调用 |
| `releases/3dgs_renderer_v1_20260928/` | 较早冻结回退及现有 CPU 契约依赖，不能当作最新位流 |
| `reconstruction/modules/` | 完整 SfM 冷对照与共用文件校验；旧 OpenSplat 训练/导出实现已删除 |
| `archive/history_20261102_cleanup/` | 从活动树删除的训练、特征 NPU、独立诊断与早期渲染实验；按路径/哈希恢复 |
| `examples/3dgs_compositor/board/remote.py` | 现有板卡传输工具依赖，不是渲染算法 |

原 main 工作目录的未提交硬件改动原地保留；本次只从当前干净工作树删除已归档的跟踪旧实验。

## 候选分支放在哪里

```text
mvsplat/
├── initialize/                 仅预热、资源生命周期与整链部署协调
├── video_input/
├── pose_estimation/
├── gaussian_generation/
│   ├── runtime.py              常驻模型，CPU 或经资格验收的 NPU 子图
│   ├── adapter.py              Gaussian/相机到渲染 ABI
│   └── npu_branch/             推荐 NPU 后端及编译、诊断、资格验收
└── rendering/
    ├── render_main/            当前四路冻结主线
    └── render_branch/          独立 HGR 候选，默认不调用
```

原 `mvsplat/npu/` 已整体迁入 `gaussian_generation/npu_branch/`，不保留同级副本。
从 `mvsplat` 目录调用 NPU 工具时使用 `python -m gaussian_generation.npu_branch.<工具名>`。
推荐显式 --backend npu --partitions backbone_cnn depth_head；通用 CLI 默认 cpu 保留兼容。
严格逐元素门槛仍保留。TF32 未通过时需当前资产绑定的三留出视角图像验收，
并通过 B,B,A,A 输入切换；详见 [联调说明](../examples/3dgs_reconstruction/mvsplat/FINAL_INTEGRATION_20261102.md)。
默认 backbone 两次真实 SDK 执行规避输入残留，累计全部成本，不代表根因修复。
板端候选同步和独立诊断脚本随同更新包路径，不能只上传一个脚本后依赖旧 `npu/`。
新增代码、测试和候选实现都必须归入对应现有模块；新独立模块需先获用户明确批注。

## 打包规则

`tools/core_sources.json` 明确主线包含项。`package_core.py` 只选 Git 跟踪的合格源码，
避免递归扫进本地实验。历史 ZIP、`render_branch`、旧训练实验、运行结果不进入主线源码包。
主线源码包不含权重、SDK、模型、BOOT/bit；完整硬件复建/回退资产请另取 `rendering/render_main/`。
历史恢复必须解压到新目录，按 [归档说明](../archive/history_20261009/README.md) 操作。

2026-11-02 合并继续保留原四模块，网页拆成 rendering/viewer.py、viewer.html、viewer.css、viewer.js，
共用后端配置位于 gaussian_generation/npu_branch/config.py，initialize 仅协调生命周期。
源码包纳入网页静态资源；不包含私人输入、生成资产、权重或 SDK。
本次执行软件回归和冻结文件检查，未新增上板测速；实板证据及待补项见 MAINLINE.md。

## 2026-11-02 清理后的入口

公共 CLI 删除 gaussian/export 训练和旧导出命令；仍保留 initialize（推荐 NPU）、
reconstruct（完整 SfM CPU 对照），及实际被引用的 video/pose/render/validate 工具。
modules/gaussian_generation、bounded、npu_frontend、run_cpu、stages、旧构建/出图脚本、
独立 NPU deploy_20261021 和早期硬件实验已从活动树删除，详见
[清理核验](MAINLINE_CLEANUP_20261102.md)。冻结资产及当前 NPU 诊断工具继续保留。
