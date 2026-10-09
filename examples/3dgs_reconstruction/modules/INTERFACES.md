# 当前四模块接口：视频 → 位姿 → Gaussian → 渲染

此页是 **MVSplat 前馈主线**的交接说明。实际四模块位于相邻 `mvsplat/` 目录；
本 `modules/` 目录的旧类/命令只承担兼容。旧 OpenSplat 文件合同保存在
[LEGACY_INTERFACES.md](LEGACY_INTERFACES.md)，不得把旧训练的 `splat.ply` 当作新链必需产物。

## 四个接口

| 交接 | 生产者与消费者 | 数据与不变量 |
|---|---|---|
| 视频 → 位姿 | `video_input.decode.decode_selected` → `pose_estimation.geometry` | `{原帧索引: BGR uint8 H×W×3}`、总帧数、FPS、2 个上下文索引、至少 1 个留出目标索引；索引不重复，尺寸一致 |
| 位姿 → 高斯 | `scene_preparation.main` → `gaussian_generation.runtime.ModelRuntime.infer` | `context.npz`：image float32 `[2,3,S,S]` RGB `[0,1]`；extrinsics `[2,4,4]` c2w；intrinsics `[2,3,3]` 归一化内参；near/far `[2]`；输入 SHA256 绑定 |
| 高斯 → 渲染 | `ModelRuntime.take_gaussians` → `gaussian_generation.adapter.gaussian_rows` → `LiveRenderer.load_rows` | 内存 Gaussian：means `[N,3]`、covariances `[N,3,3]`、harmonics `[N,3,25]`、opacities `[N]`；转换为 little-endian float32 `[N,62]` 连续 rows，一次加载 |
| 视角 → 图像 | `adapter.camera_bytes` → `LiveRenderer.render_camera` | FLCAM001 136 字节相机 → `LiveFrame.rgb` uint8 `[H,W,3]`＋元数据；需要原始缓冲时显式 include_raw |

## 必须保持的语义

- 单目几何尺度非米制；第一上下文相机为原点，两视图正深度中位数归一到 10。
  后续目标 PnP 与 Gaussian 必须使用同一坐标系，不能分别重新归一化。
- `image` 是 RGB 通道优先；视频解码的 BGR 只在视频/位姿内部使用。
  c2w 方向和归一化内参不得与 w2c、像素内参混用。
- 网络输出 opacity 已激活；协方差是世界坐标协方差。
  adapter 把 opacity 转 logit、尺度转 log、旋转转 wxyz，渲染器按原 ABI 读取。
- 网络 SH4 有 25 个系数；既有渲染 ABI 保留前 16 个系数（SH3）。本轮没有改变这项裁剪或渲染精度。
- rows 字段顺序：XYZ 3、法线占位 3、DC 3、高阶 SH 45、opacity 1、log-scales 3、wxyz 4，共 62。
- 目标视角不参与 MVSplat 两张上下文输入，但会通过图像匹配和 PnP 定位；不能省略 PnP 后声称同样的新视角验收。
- `take_gaussians(expected_sha256)` 消费当前推理的待交付结果；不能重复拿旧场景输出。
  首帧后仍需保存并校验延后归档，不能靠取消正确性核对提速。

## 调度和所有权

`initialize/session.py` 创建并持有模型和渲染器；四模块不互相重新创建运行环境。
`scene_preparation.py` 发布 context ready / first target ready 回调，`warm_pipeline.py`
控制是否重叠、何时装入场景与归档。默认仍保留既有串行准备配置，重叠为显式选择。
改变交接字段时必须同时更新生产者、消费者和既有接口回归，不能只改一端。

视频前 `READY`；视频完成 `VIDEO_COMPLETE`；场景装入 `SCENE_READY`；首帧完成 `FRAME_COMPLETE`。
场景准备时间为 VIDEO_COMPLETE → FRAME_COMPLETE；预热和视频输入只记录。
后续视角只重复最后一个接口，单独统计相机请求到完整 RGB 的延迟。

## 主线与独立开发

- 视频/位姿优化：修改 `video_input/decode.py`、`pose_estimation/geometry.py`，保持上述上下文与目标相机合同。
- 高斯生成优化：修改 `gaussian_generation/`；NPU 代码统一放在其 `npu_branch/`，通过已有 PartitionRuntime 挂接，不改变 Gaussian 交接字段。
- 后端优化：修改 `rendering/` 的候选实现；当前默认固定 `mainline.json`，冻结包内容不可原地改写。

旧 `video_input/prepare.py`、`initialize/model_runtime.py`、根 `export.py` 只是兼容入口。
新增功能应修改真实实现，不复制第二套算法。主线没有默认调用 `render_branch`。
目录和模块新增约束见 [项目要求](../../../docs/PROJECT_REQUIREMENTS.md)；
新增独立模块必须先取得用户明确批注/同意。
本轮为源码位置与接口说明整理，功能/板端回归待恢复测试后执行。
