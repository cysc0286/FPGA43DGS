# 四模块文件接口 v1

当前整链主入口是 `python examples/3dgs_reconstruction/pipeline.py reconstruct --video INPUT.mp4 --out NEW_RUN`。它在 ARM 上调用 MVSplat，默认完整 SfM 位姿，输出 `renderer_input/model.ply`、相机、`manifest.json`、各视角图像和 `pipeline_result.json`。`reconstruct` 使用 `--video/--out`，以下 `--run` 分阶段示例属于旧 OpenSplat 对照。MVSplat 的 `context.npz`、`input.json` 和网络中间产物见 [MVSplat 接口](../mvsplat/README.md)；它不生成本页旧训练合同中的 `splat.ply`/`training_request.json`，因此不能对其直接运行旧 `validate --through gaussian`。

本次沿用既有产物格式。`contracts.py` 提供 `FrameSet`、`PoseSet`、`GaussianScene`、`RenderInput`、`RenderResult` 校验器；它们是本地文件接口，不是网络服务或新的硬件 ABI。

## 模块交接

| 生产模块 → 消费模块 | 必需产物 | 必须保持的语义 |
|---|---|---|
| 视频 → 位姿 | `frames/*.png`、`frames.json` | 帧名唯一，帧号和秒时间戳严格递增，尺寸有效；当前 SfM 至少 5 帧 |
| 位姿 → 高斯 | `project/images/`、`project/sparse/0/{cameras,images,points3D}.bin`、`project.json`、`sfm.json` | COLMAP 图像、内参、位姿和稀疏点属于同一重建；单目尺度未标定；保持留出图像标识 |
| 高斯优化 → 导出 | `splat.ply`、`cameras.json`，以及上一步 PoseSet | 高斯包含位置、SH、opacity、scale、rotation；普通 XYZ/RGB 稀疏点不是训练完成的高斯模型 |
| 导出 → 渲染 | `renderer_input/model.ply`、指定 `.bin` 相机、`manifest.json` | 62 个 float32 的 binary little-endian PLY；相机 `FLCAM001`，136 字节 |
| 渲染 → 展示/评价 | `frame.bin`、`frame.ppm`、`result.json` | `GSSOUT01`，每像素 RGB/T/last 共 20 字节；完成状态、尺寸和 frame 哈希须正确 |

各模块通过命令显式接收 `--run`，运行目录可整体复制。PoseSet 可仅复制表中必需产物，不需要原视频、数据库、特征缓存；训练代码不会重跑 SfM。渲染模块只需要导出目录及已有冻结包，不依赖 pycolmap、OpenCV、Torch。

`frames.json` 保留输入视频 SHA256 和原帧索引；`project.json` 保留校准与留出范围；`training_request.json` 保存独立训练命令、配置和输入/程序哈希；已有一键执行器使用原 `config.json` 和阶段 receipt。文件校验器是结构与一致性检查，不等价于重新估计几何或实板性能验收。

## 坐标与数值

- SfM 项目保留 COLMAP 相机约定。训练前去畸变并变换图像和内参，使主点为 `((W-1)/2,(H-1)/2)`；不能仅改参数而不变换图像。
- FLCAM001 存 camera-to-world 旋转矩阵（文件行优先）、相机位置、焦距和输入/输出分辨率，局部轴为 x 右、y 下、z 前。目标视角需位置、朝向和内参，只有位置不够。
- 高斯 opacity 为 sigmoid 前存储值，scale 为 exp 前存储值；不能重复激活。SH0 导出用零填充高阶系数，保持冻结包格式，不能宣称因此减少后端存储或运算。
- 保持原有深度排序、alpha 合成、精度和提前结束规则。完整后端定义见 [冻结接口](../../../releases/3dgs_renderer_v1_20260928/INTERFACE.md)。
- 留出图像不参与高斯优化，但参与 SfM。质量指标必须披露这一范围；插值新视角没有对应真实照片。

## 旧 OpenSplat 分阶段运行

从项目根目录执行，使用已经配置依赖的 Python；下面 `python` 在板端应指向板端环境。电脑运行仅用于参考验证。

```text
python examples/3dgs_reconstruction/pipeline.py video --video INPUT.mp4 --run RUN_DIR
python examples/3dgs_reconstruction/pipeline.py pose --run RUN_DIR
python examples/3dgs_reconstruction/pipeline.py gaussian --run RUN_DIR --opensplat PATH_TO_BOUNDED_OPENSPLAT
python examples/3dgs_reconstruction/pipeline.py export --run RUN_DIR
python examples/3dgs_reconstruction/pipeline.py render --run RUN_DIR --renderer PATH_TO_FROZEN_PACKAGE --out NEW_RENDER_DIR --backend fpga
```

默认输入 30 帧，SfM 图像宽 480，训练宽 160；独立训练采用 `bounded/profiles.json` 的 `arm_candidate`（1000 步、6000 点上限、2 线程）。这些是既有候选设置，不是本轮测得的板卡最优参数。

`gaussian --plan` 只验证 PoseSet 并输出命令。`render --plan` 校验模型/相机并输出命令，不执行 ARM 二进制。已有产物禁止覆盖；失败记录保留，修复后使用新的运行目录或已有受控续跑方式。

```text
python examples/3dgs_reconstruction/pipeline.py validate --run RUN_DIR --through export
python examples/3dgs_reconstruction/pipeline.py validate --run RUN_DIR --through render --out RENDER_DIR
```

`validate` 验证产物接口，不能把历史产物检查写成“本轮重新上板通过”。

模块独立交接时用 `--only`，不要求接收人带上不需要的上游产物：

```text
python examples/3dgs_reconstruction/pipeline.py validate --run POSE_RUN --only pose
python examples/3dgs_reconstruction/pipeline.py validate --run GAUSSIAN_RUN --only gaussian
python examples/3dgs_reconstruction/pipeline.py validate --run RENDER_RUN --only export
python examples/3dgs_reconstruction/pipeline.py validate --run RENDER_RUN --only render --out RENDER_OUTPUT
```

`--only` 与 `--through` 互斥；不指定时仍检查从 video 到 export 的整段产物。`--only export` 检查默认 `novel_midpoint.bin`；其他目标相机使用 `render --camera NAME.bin --plan` 检查。独立检查只是当前边界的验收，不证明其上游重建准确、所有文件具备跨版本溯源或整链已经跑通。

## 旧入口、打包和续跑

- `stages.py frames/sfm/export/validate_adapter` 保持原参数与默认值，通过延迟导入调用四模块。
- `run_cpu.py` 和 `bounded/run.py` 继续可用；CPU 训练命令改为共享 `gaussian_generation/training.py`。
- `bounded/package.py` 已加入 `modules/` 和 `pipeline.py`；避免源码包缺少拆分依赖。
- bounded 的源码指纹包含新模块。整理前的历史 run 不允许用修改后的源码静默续跑；使用冻结源码或新 run。
- 独立模块入口使用标准产物接续，一键受控执行器另有完整的阶段 receipt，二者不能互冒充断点恢复记录。
