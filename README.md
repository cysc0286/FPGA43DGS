# HeteroGS：板端视频重建与 NPU＋FPGA 渲染

## 当前版本：手动采集与 CPU＋FPGA 演示

[2026-11-27 更新说明](docs/RELEASE_20261127.md)汇总本次相机、覆盖选帧和云台源码增量，沿用下方六部分目录。当前已验证部署由 **ARM CPU 生成高斯、FPGA 渲染**；NPU 十子图实现保留，当前服务未启用，也未在本轮重新资格验收。

- USB 录制上限 **15 秒**，支持手动曝光、增益和亮度；默认按真实时间戳与实测帧率接收，不强制 60 fps，不补帧。
- 电脑视频输入沿用 `/upload`；与 `/capture` 共用同一处理链，本地视频不套用 USB 标定。
- 网页默认自动选择 **最多 2 个帧对、64 个候选帧**。`coverage` 策略支持跨帧选择有重叠的真实视角；主分支原有 `balanced` 与 `legacy` 策略继续保留。
- 当前拍摄由用户手动转动云台；电机测试和 ABI3 固件保留源码及本机候选记录，不自动启动运动或刷写。

同一历史录像的裁剪后水平视锥覆盖估计曾从 42° 扩大到 58°，该例实际只需一对。覆盖估计不等于完整三维、360°或画质验收，多对仍可能触及内存保护。详细时间、负结果与边界见 [MAINLINE.md](MAINLINE.md)。本次汇总没有新增板测成绩。

## 已有固件与画质流程

[2026-11-16更新说明](docs/RELEASE_20261116.md)汇总本次固件、渲染与画质代码。已验收的200MHz四路渲染包位于`rendering/render_main/`；最新J17 23/21相机PWM整板候选的[BOOT、位流和报告](examples/3dgs_reconstruction/mvsplat/video_input/firmware/j17_23_21_20261115/)已随源码提交，候选尚未完成实板运动验收。

环视流程加入真实视野驱动选组、邻近高斯归并、重叠区平滑过渡和上下文低频颜色修正，仍由一个合并场景经CPU投影排序与FPGA渲染。画质修正在场景准备阶段修改高斯参数，不替换FPGA返回画面；NPU覆盖范围和既有资格门槛保持不变。

仓库根目录可执行`python tools/check_render_packages.py`与`python tools/check_camera_firmware.py`核对固件及配套源码。六部分结构和四步相机测试继续保留。

## C++ 与 FPGA 源码入口

主业务代码沿用 `examples/3dgs_reconstruction/mvsplat/` 路径。该目录是正式应用代码，已通过 `.gitattributes` 纠正 GitHub 对 `examples/` 的文档分类；第三方和自动生成代码仍按原规则识别。右侧语言比例不是源码文件清单。

| 内容 | 源码入口 |
|---|---|
| ARM 常驻渲染、投影与排序（C++） | [live_renderer.cpp](examples/3dgs_reconstruction/mvsplat/rendering/live_renderer.cpp)、[depth_sort.hpp](examples/3dgs_reconstruction/mvsplat/rendering/depth_sort.hpp) |
| FPGA 渲染流水线（HLS C++） | [pipeline.cpp](examples/3dgs_reconstruction/mvsplat/rendering/render_main/fpga/pipeline/pipeline.cpp)、[HLS/管线目录](examples/3dgs_reconstruction/mvsplat/rendering/render_main/fpga/) |
| FPGA 顶层、寄存器、DMA 与生成的运算 RTL | [flicker_top.sv](examples/3dgs_reconstruction/mvsplat/rendering/render_main/rtl/flicker_top.sv)、[RTL 目录](examples/3dgs_reconstruction/mvsplat/rendering/render_main/rtl/) |
| 相机云台 PWM、总线接口与约束 | [camera_servo_pwm.v](examples/3dgs_reconstruction/mvsplat/video_input/rtl/camera_servo_pwm.v)、[RTL/XDC 目录](examples/3dgs_reconstruction/mvsplat/video_input/rtl/) |

NPU 的 SDK 调用桥接也使用 C++，见 [bridge.cpp](examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch/bridge.cpp)。上述源码与固件随同一仓库版本管理。

## 四步相机测试

按顺序运行四个独立入口，详见[测试说明](examples/3dgs_reconstruction/mvsplat/video_input/acceptance/README.md)和[检查清单](examples/3dgs_reconstruction/mvsplat/video_input/acceptance/CHECKLIST.md)。全部命令默认预览；确认现场条件后才加 `--execute`。

1. **电机测试**：`video_input/acceptance/01_motor.sh`，核验两轴小角度运动、方向和限位。
2. **轨迹测试**：`video_input/acceptance/02_trajectory.sh`，执行三层弓字扫描，不打开摄像头。
3. **同步录制**：`video_input/acceptance/03_sync_capture.sh`，录像出帧后启动运动，输出原视频交接文件。
4. **重建渲染**：`video_input/acceptance/04_reconstruction.sh`，调用现有 NPU＋FPGA 主线并提供网页。

命令在 `examples/3dgs_reconstruction/mvsplat/` 运行。默认两轴20°/s、俯仰 +30°/0°/−30°、水平 ±30°；计划运动16.66秒、录像17.66秒，20秒保护上限。参数与覆盖估算见[扫描方案](examples/3dgs_reconstruction/mvsplat/video_input/acceptance/SCAN_PLAN.md)。这些是规划值，真实运动、录像和画质须上板验收。

本分支提交源码、配置及说明；正式 NPU 资产、板端环境和已匹配 PWM 固件须沿用部署，视频与运行输出保存在仓库之外。

平台为悟净 30TAI Lite。主线：视频输入 → CPU 位姿估计 → MVSplat 生成 Gaussian → CPU＋FPGA 渲染 → 网页显示。
固定预训练权重在视频前加载；新场景执行前向推理，换视角复用已生成的场景。

## 六部分代码

所有业务代码位于 `examples/3dgs_reconstruction/mvsplat/`：

```text
initialize/           预热、生命周期、场景准备协调、计时和部署
video_input/          USB 录像／文件接收、解码、选帧和标定
pose_estimation/      特征、匹配、两视图几何和目标视角 PnP
gaussian_generation/  常驻模型、NPU 子图、CPU 剩余计算与 Gaussian 适配
rendering/            常驻 CPU＋FPGA 渲染、相机 ABI、冻结 render_main
web/                  采集／上传页面、视角控制、HTTP 和静态资源
```

[代码与接口](docs/CODE_STRUCTURE.md) · [项目约束](docs/PROJECT_REQUIREMENTS.md) · [当前主线与实测边界](MAINLINE.md) · [整理核验](docs/CORE_VALIDATION.md)

## 唯一整链入口

从 `examples/3dgs_reconstruction/` 运行：

```sh
python pipeline.py initialize --help
python pipeline.py web --help
python pipeline.py view --help
```

`initialize` 管理视频前预热及视频后首帧；`web` 提供 `/capture`、`/upload` 和交互查看器；`view` 只恢复已有场景。
原 OpenSplat／整段 SfM 冷链、旧发布包、独立渲染候选和重复兼容入口已移出当前代码树。

当前网页演示配置如下。依赖路径均须替换为板端实际资产，命令不会烧录固件：

```sh
python pipeline.py web --out /path/to/new/sessions \
  --weights /path/to/re10k.ckpt --vendor /path/to/MVSplat_reference \
  --renderer /path/to/matching/renderer --live-renderer /path/to/live_exact \
  --calibration /path/to/camera_calibration.json --backend cpu \
  --max-seconds 15 --ring-pairs 2 --ring-auto-coverage --ring-min-pairs 1 \
  --ring-candidates 64 --ring-selection-policy coverage --ring-coverage-target 1.0 \
  --ring-max-gaussians 65536 --ring-appearance-strength 0 \
  --host 127.0.0.1 --port 8081 --viewer-port 8080 \
  --viewer-public-url http://127.0.0.1:18080/
```

回环地址需要已有 SSH 端口转发；查看器公开 URL 应指向浏览器可达的对应端口。后台进程仍有 30 分钟总时限，从预热启动即计时；超时保留场景，刷新页面不会延长进程寿命。

NPU 历史配置显式选择十个子图，重新启用前必须验证当前固件、SDK、图包与验收记录身份：

```sh
python pipeline.py initialize --video INPUT.mp4 --out NEW_RUN \
  --weights /path/to/re10k.ckpt --vendor /path/to/MVSplat_reference \
  --renderer /path/to/matching/renderer --live-renderer /path/to/live_exact \
  --render-profile mainline --backend npu \
  --partitions backbone_cnn depth_head to_gaussians to_disparity \
    transformer_ffn_0 transformer_ffn_1 transformer_ffn_2 \
    transformer_ffn_3 transformer_ffn_4 transformer_ffn_5 \
  --partition-bundle /path/to/compiled --oracle-graphs /path/to/graphs \
  --npu-library /path/to/libmgs_npu.so --worker-python /path/to/arm/python \
  --npu-acceptance /path/to/current/acceptance.json --interactive
```

这些路径为占位，实际权重、SDK、图和硬件需另行配置。NPU 失败不静默回退。
当前 CPU 部署显式选择 `--backend cpu` 并去掉 NPU 专用参数；网页默认值也与该部署一致。
注意力、代价体等部分仍在 CPU；不能将十子图方案称为全网 NPU。
严格 TF32 张量门槛仍未关闭；图像资格、输入切换、资产身份、设备隔离和内存检查继续保留。
目录变化会使旧源码身份失效，重新部署需重新生成匹配资格，不能复用旧 receipt 充当新验收。

源码不包含私人录像、场景、权重、SDK 或 Python 环境。当前整理仅做本机软件核验，未新增板端性能数据。
