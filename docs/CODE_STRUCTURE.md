# 当前代码结构与入口

更新：2026-10-09。这里描述实际主线和源码位置；历史报告按各自日期和输入范围阅读。

## 主线流程

```text
initialize 视频前预热
  → READY
  → 接收已完成的视频 → VIDEO_COMPLETE
  → 抽帧和相机位姿
  → MVSplat 固定权重前向生成 Gaussian
  → 格式转换与场景装入 → SCENE_READY
  → CPU 投影/Tile 分组/排序 + FPGA 求值与有序合成
  → 首帧 FRAME_COMPLETE
  → 后续重复提交相机，返回新的完整 RGB 帧
```

预热及视频输入时间只记录；场景准备从 VIDEO_COMPLETE 到首帧 FRAME_COMPLETE。
已加载场景的后续换视角时间另计。此代码结构说明没有新增实时摄像头、逐场景训练
或 NPU 真机验收结果。

## 目录树

```text
FPGA43DGS/
├── README.md / MAINLINE.md            主线和历史结果入口
├── docs/                             结构、交付说明、历史发布记录
├── examples/
│   ├── 3dgs_reconstruction/
│   │   ├── pipeline.py               统一 CLI：initialize / reconstruct / 分阶段入口
│   │   ├── modules/
│   │   │   ├── contracts.py          FrameSet/PoseSet/GaussianScene/RenderInput/RenderResult
│   │   │   ├── INTERFACES.md          文件和数值交接合同
│   │   │   ├── video_input/          视频模块适配层
│   │   │   ├── pose_estimation/      位姿模块适配层
│   │   │   ├── gaussian_generation/  旧训练链适配与对照，不是当前主线生成器
│   │   │   └── rendering/           冻结渲染包的分阶段调用适配
│   │   └── mvsplat/
│   │       ├── initialize/          视频前加载固定权重、建立运行时
│   │       │   ├── run.py           预热 CLI 与事件驱动入口
│   │       │   ├── session.py       WarmSession，统一选主线渲染配置
│   │       │   └── model_runtime.py 常驻 MVSplat 模型
│   │       ├── video_input/         文件接收、抽帧、快速位姿准备
│   │       ├── warm_pipeline.py     视频后准备、推理、导出和首帧调度
│   │       ├── board_pipeline.py    reconstruct 冷启动路径
│   │       ├── infer.py             模型前向参考入口
│   │       ├── export.py            Gaussian 到渲染格式转换
│   │       ├── rendering/           当前 CPU＋FPGA 渲染主线
│   │       │   ├── mainline.json    唯一主线配置
│   │       │   ├── runtime.py       LiveRenderer：加载场景、相机到 RGB
│   │       │   ├── pipeline_adapter.py 视频首帧适配，默认 mainline
│   │       │   ├── live_renderer.cpp  常驻原生计算与 FPGA 调用
│   │       │   └── package/         四路主线冻结源码、固件和证据
│   │       ├── render_branch/       两路 PipeGS HGR v4，独立保存，默认不调用
│   │       ├── npu/                 MVSplat NPU 分区部署候选
│   │       ├── test_*.py            3DGS 接口、生命周期及协议回归
│   │       └── results/             按日期保存的结果与核验记录
│   ├── 3dgs_flicker_hw/             FPGA 后端研发：HLS、RTL、流水、板端工具
│   ├── 3dgs_flicker_cat/            3DGS 算法/数值参考与依赖
│   ├── 3dgs_compositor/board/       主线仍使用的共享板卡工具
│   └── 3dgs_scene/                  场景及渲染前处理的共享代码/历史对照
├── releases/                        旧版回退包与发布入口
├── tools/                           源码打包、合同检查、冻结包检查
└── third_party/                     第三方版本、来源与许可记录
```

`platform/`、`build/`、`papers/`、原始数据及本地 `npu_3dgs/` 工作区不等于 GitHub
发布内容；部分内容被 Git 忽略或未跟踪。不要按本机资源管理器的目录数量判断主线模块数。

## 给前端与后端开发者的边界

| 部分 | 主要修改位置 | 当前执行位置/状态 |
|---|---|---|
| 视频和位姿 | `mvsplat/video_input`，传统路径对应 `modules/video_input`、`modules/pose_estimation` | ARM CPU；快速路径抽帧和位姿仍集中在 `prepare.py`，不是两个独立常驻服务 |
| 高斯生成 | `mvsplat/initialize/model_runtime.py`、模型运行时及 `infer.py/export.py` | 当前主线为 ARM CPU 的预训练 MVSplat 前向；NPU 候选另行验证 |
| 渲染 | `mvsplat/rendering` | CPU 投影/分组排序，FPGA 求值和有序合成；主线四路 grouped-shared |
| FPGA 核研发 | `examples/3dgs_flicker_hw` | 构建与实验工作区；已接受产物在 rendering/package，实验修改不自动进入发布包 |
| NPU | `mvsplat/npu` | 部署与数值问题待解决；不属于当前渲染必选项 |

前端向主线渲染交付 PLY 或 `float32[N,62]` 高斯行和相机参数；原生相机协议为
136 字节 FLCAM001。换场景装载一次，后续只更新相机并返回完整 RGB。首帧适配器
继续保留归档核验，交互帧可以在完成返回之后显式归档。

`modules/` 保留四个逻辑模块的文件接口，不代表暖启动链会逐个调用所有旧模块。
特别是 `modules/gaussian_generation/training.py` 属于旧训练对照，当前高斯生成以
`mvsplat` 为准。`3dgs_flicker_hw/frontend` 指渲染前处理，不是视频重建前端。

## 本次删除与保留

- 删除 Git 中 26 个旧基础 ALU 源码、测试平台、板端脚本和归档元数据文件。
- 删除本地已退出 Git 的 5 个独立加法器 smoke test 源码/入口文件。
- 删除 `tools/core_sources.json` 的旧归档文件和 ALU 目录条目，并修正当前导航。
- 本地忽略的旧生成物和实验证据保留，不再属于源码包。
- 保留 3DGS 自身的正确性与协议回归，以及主线共享的板卡工具。
- 保留冻结包和平台的 `adder_top/legacy_adder_top`：它们仍承载原厂寄存器/DMA
  兼容接口，名字包含 adder 不等于可以当作独立测试删除；删掉会破坏现有构建或冻结包。
- 历史发布回执及报告中的旧路径作为历史记录保留，不用于当前打包。

递归清理整个本地旧目录被自动审批拦截，因此采用精确文件级源码删除，未强行清空
生成物。用户要求板卡离线期间停止测试，本次仅清理源码、打包配置及文档；不运行
软件测试、RTL 仿真、综合、布线或板端程序。算法与位流未修改，没有新的耗时、画质
或资源指标；上一轮 86 项本地测试是前一次整合修复的结果，不作为本次清理后的新验收。
