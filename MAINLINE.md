# 3DGS 主线与目录入口

更新：2026-09-28。最终目标是全链在悟净 30TAI Lite 上执行；电脑用于开发、构建和参考验证，不能将电脑训练记录称为板端训练成功。

```text
视频
  ↓ video_input
FrameSet：帧图像 + 帧号/时间戳清单
  ↓ pose_estimation
PoseSet：去畸变图像 + 内参/位姿 + 稀疏几何
  ↓ gaussian_generation
GaussianScene：优化后的高斯参数
  ↓ export（高斯模块内部格式适配）
RenderInput：model.ply + FLCAM001 相机
  ↓ rendering（调用已有冻结后端）
RenderResult：frame.bin + frame.ppm + result.json
```

## 四个模块

| 模块 | 当前实现路径 | 职责与状态 |
|---|---|---|
| 视频输入 | [video_input](examples/3dgs_reconstruction/modules/video_input/) | 从已有视频均匀抽帧；实时摄像头不是当前接口 |
| 位姿估计 | [pose_estimation](examples/3dgs_reconstruction/modules/pose_estimation/) | COLMAP CPU 特征/匹配/位姿/三角化、去畸变、内参适配；输出包含稀疏点，不仅是位姿 |
| 高斯生成 | [gaussian_generation](examples/3dgs_reconstruction/modules/gaussian_generation/) | OpenSplat CPU 优化、资源控制、PLY/相机导出；板端训练成功尚未核验，电脑参考与板端结果分开记录 |
| 高斯渲染 | [rendering](examples/3dgs_reconstruction/modules/rendering/) | 校验模型/相机接口并调用已完成的 CPU＋FPGA 渲染包；不重新实现光栅化 |

统一分阶段入口：[pipeline.py](examples/3dgs_reconstruction/pipeline.py)。详细字段与命令：[接口文档](examples/3dgs_reconstruction/modules/INTERFACES.md)。旧 `stages.py` / `run_cpu.py` / `bounded/run.py` 继续使用同一套模块实现，不保留第二份计算代码。

模块独立交付使用 `pipeline.py validate --only`，整段检查保留 `--through`。

2026-09-28 新分析目标：以质量合格后的总耗时最短为优先，允许充分使用板上资源；NPU 的前馈高斯、特征和批量属性候选见 [加速分析](examples/3dgs_reconstruction/NPU_ACCELERATION_ANALYSIS_20260928.md)。这些是候选研究，尚未切换默认重建方法或重新上板验证。

四模块采用运行目录中的文件交接，可单独运行和替换实现。它们不是四个必须同时常驻的服务；板端优先顺序执行，避免叠加训练、SfM 和渲染内存。这个整理没有新增实时建图、NPU 加速或两单元硬件。

## 已完成渲染后端

- 冻结运行包与回退基线：`releases/3dgs_renderer_v1_20260928/`，保留程序、源码、模型、相机、位流与清单。
- 后端研发源码：`examples/3dgs_flicker_hw/`；其中 `frontend/` 是**渲染前处理**，不是视频重建前端。
- CAT/PR 软件参考：`examples/3dgs_flicker_cat/`。
- `platform/` 和 `build/` 保持原路径，避免破坏已有厂商工程和绝对路径。
- 后续两单元轻量版应独立构建；本轮只整理模块，没有改变四单元硬件或数值规则。

## 实验与共享依赖

基础 ALU 源码与历史清单在 [archive/basic_demos_20260928](archive/basic_demos_20260928/README.md)；原始压缩包与实验输出仅保留在本地。

以下目录不作为主线入口，但暂不整目录移动：

| 路径 | 保留原因 |
|---|---|
| `examples/3dgs_compositor/board` | 当前 FLICKER 构建、上传、测量脚本仍导入其板卡工具；整目录搬走会破坏主线 |
| `examples/3dgs_reference`、`3dgs_tile_cpu`、`3dgs_scene`、`3dgs_batch`、`3dgs_fpga_eval` | 历史对照、模型/验证数据和可能的构建引用；下一次迁移需按依赖拆出共享资产，不能按“旧目录”删除 |
| `npu_3dgs` | 历史 NPU/异构实验和对照记录；不加入默认执行路径 |
| `examples/3dgs_reconstruction/npu_frontend` | 位姿估计模块的候选匹配实现，单独开发/验证；默认 CPU 链未切换 |

历史 evidence、runs 和 release 均保持可追溯；新目录整理的验证在 [本轮记录](examples/3dgs_reconstruction/evidence/module_refactor_20260928/VALIDATION.md)。

## 后续工作顺序

1. 使用四模块接口继续完成 ARM 训练的依赖、内存和输出核验，最终生产链不依赖电脑训练。
2. 模型与相机通过现有 ABI 交给冻结渲染后端，同一模型比较 CPU/FPGA。
3. 后端轻量化另建候选，保留四单元冻结版；减少逻辑资源不能等同于增加 Linux 内存。
4. 根据板端实测再选择前端 FPGA/NPU 候选。每轮保留正确性、画质、时延、内存和资源证据。
