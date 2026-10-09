# MVSplat 主线架构

```text
initialize/session.py（视频前：权重、运行环境、渲染器）
  READY
video_input/receipt.py（视频完成）
  VIDEO_COMPLETE
scene_preparation.py
  video_input/decode.py → pose_estimation/geometry.py
  context.npz + input.initial.json / input.json
warm_pipeline.py
  gaussian_generation/runtime.py → gaussian_generation/adapter.py
  → rendering/pipeline_adapter.py → rendering/runtime.py
  → 常驻 C++：投影、排序、提交 → 四路 FPGA：求值、合成
  SCENE_READY → FRAME_COMPLETE
```

四个业务模块有独立代码位置和交接格式，详见 [当前接口](../modules/INTERFACES.md)。
`initialize` 只处理视频前预热；场景准备、目标 PnP 与 Gaussian 生成发生在视频后。
`warm_pipeline.py` 负责同步、计时和证据归档，数值算法放在对应业务模块。

## 推荐后端

显式指定 `--live-renderer` 和 `--render-profile mainline`。
`PipelineRenderer` 调用 `LiveRenderer.mainline(...)`，统一读取 `rendering/mainline.json`。
场景只装入一次，此后相机直接交给常驻 C++，返回完整内存 RGB。
旧 `initialize/renderer_runtime.py` 仍作为兼容实现和 C++ 依赖保留；没有提供 live binary
时旧 CLI 会使用它，因此不能把省略 `--live-renderer` 的结果标成当前主线成绩。

## 验收边界

- 当前 CPU 运行 MVSplat；NPU 候选保持数值门禁，不默认参与，也没有本轮新增验收。
- CPU 负责渲染投影与分组排序，四路 FPGA 负责高斯求值及有序合成。
- 四路渲染冻结包和 HGR v4 候选包保持独立。历史时间与画质统一见仓库根 `MAINLINE.md`。
- 本轮拆分移动算法定义，不改权重、算术、排序、C++/RTL、主线配置或位流。
- 本轮无功能测试或实板运行；源码搬迁静态核对不能替代整链运行验收。

`test_*.py` 是当前维护所需的回归代码，继续保留；旧实验运行目录与结果移入仓库根 `archive/`。
