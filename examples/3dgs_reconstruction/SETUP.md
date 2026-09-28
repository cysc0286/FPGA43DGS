# CPU 参考链环境与复跑

当前 Windows 环境已实际配置在本目录 `.venv` 和 `vendor`，不需要 GPU、CUDA、NPU 或全局环境变量。全部重建参数与单次执行证据进入新的 `runs/<名称>`，不覆盖旧运行。

## 已有环境：直接使用

在仓库根目录 PowerShell 执行：

```powershell
# 下载并校验同一测试视频；已存在时只校验。
& examples/3dgs_reconstruction/.venv/Scripts/python.exe examples/3dgs_reconstruction/fetch_video.py

# 完整电脑 CPU 链：视频 -> 帧 -> 位姿/点 -> 高斯 -> PLY/相机/验证图像。
& examples/3dgs_reconstruction/run_cpu.ps1 -RunName my_nyu_test

# 后续只替换视频，沿用相同链路。要求静态主体、有平移视差和连续视角覆盖。
& examples/3dgs_reconstruction/run_cpu.ps1 -Video 'D:\my_video.mp4' -RunName room_a_v1 -Frames 30 -Iterations 1000 -MaxGaussians 20000
```

`Iterations` 为不小于 200 的 10 的倍数；原版 OpenSplat 每 10 步保存验证图。默认均匀抽帧，尚无自动去模糊/视差选择。对长视频可先裁剪一个连续、具有重叠的片段；不能假设任意视频抽取 30 帧都能重建。

入口中 COLMAP 特征/匹配显式 `Device.cpu`，BA 关闭 GPU；OpenSplat 编译为 `GPU_RUNTIME=CPU`，运行还传入 `--cpu`。SH0，黑背景，最多 20,000 个高斯，固定随机种子来自锁定源码。预算上限不代表每次恰好生成 20,000 个点。

## 对接已部署的板端 CPU 渲染器

`board_render.py` 复用原有 SSH 主机公钥校验和环境密码机制。认证使用会话环境中的 `FPGA_BOARD_PASSWORD`，不写入代码/报告。只在新的 `/root/fpga43dgs_reconstruction/reconstruction_<时间>` 上传当前模型与两台目标相机。

```powershell
& D:/Tools/Python31210/python.exe examples/3dgs_reconstruction/board_render.py --run examples/3dgs_reconstruction/runs/my_nyu_test
& examples/3dgs_reconstruction/.venv/Scripts/python.exe examples/3dgs_reconstruction/evaluate.py --run examples/3dgs_reconstruction/runs/my_nyu_test
```

默认调用已冻结 `3dgs_renderer_v1_20260928` 的 `cpu_dense` 后端。这里没有 NPU 计算、FPGA 加速测量、BOOT 写入。板端输出包括留出帧相机和两相邻相机之间的中点视角；中点视角没有对应真值，不能给它伪造 PSNR。

## 干净电脑上的环境重建

本轮实测组合：Python 3.12.10、pycolmap 4.2.0（无 CUDA）、CPU PyTorch 2.7.1、MSVC 19.44.35229、Windows SDK 10.0.26100.0、OpenCV C++ 4.10.0。Python 依赖精确版本见 `requirements-windows-lock.txt`；注意它不是 ARM 板端安装清单。

1. 创建独立 `.venv`；PyTorch 从 `https://download.pytorch.org/whl/cpu` 安装 `torch==2.7.1`，其余包按锁定文件安装。OpenCV Python 包只用于抽帧/评估，不能替代下一步的 C++ 开发库。
2. 克隆 [OpenSplat](https://github.com/WebODM/OpenSplat) 到 `vendor/OpenSplat`，检出提交 `62fd86fe3b62644b65890557ad9bf397abdc7cfb`。这是原作者未修改的 1.2.2 源码，不是我们自行重写的训练器。
3. 将 [OpenCV 4.10.0 Windows 官方包](https://github.com/opencv/opencv/releases/tag/4.10.0) 解压为 `vendor/opencv/build`。
4. 准备 MSVC x64 与 Windows SDK。此次使用 [portable-msvc 作者脚本](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977) 从微软下载经哈希校验的工具包，放入 `vendor/msvc`；本机网络另用 curl 替换下载函数，编译器与 SDK 内容未修改。脚本来源和下载文件哈希见 `evidence/sources.json`。也可适配已有 VS Build Tools 环境。
5. 运行 `.venv/Scripts/python.exe build_opensplat.py`。它生成 `runtime_paths.json`，只对当前子进程设置动态库路径。

构建日志保留于 `evidence/build_0.log`、`build_1.log`。OpenSplat 使用 AGPLv3，代码/依赖/测试视频保留各自原始授权与来源；本仓库不打包上传这些大文件。

## 数据接口与正确性边界

- 只有 RGB 视频参与计算，没有使用 NYU 深度、已有相机位姿或预训练高斯。
- COLMAP 先估计 SIMPLE_RADIAL 相机，然后去畸变。训练图像再重采样为 320 像素宽，同时更新 PINHOLE 内参和观测点，使主点严格匹配冻结 `FLCAM001` 的 `(W-1)/2,(H-1)/2`。
- OpenSplat 保存回 COLMAP 坐标系；适配器使用同一坐标系的相机到世界旋转与位置。SH0 高阶系数补零，转换为原接口的 62 个 float32 字段；这只兼容旧接口，不让旧解析器自动变成紧凑 SH0 存储。
- 留出 1 帧不参与高斯优化，但仍参与 SfM 和初始点颜色估计。这是有限范围的新视角检查，不是严格完全隔离的泛化测试。
- OpenSplat 的最终 RGB 会裁剪到 1；板端原始 framebuffer 保留超出 1 的数值，PPM 显示时才裁剪。图像质量统一在 `[0,1]` 比较，原始未裁剪结果仍独立保留，避免比较口径不一致。
- 本轮是电脑 CPU 建图＋板端 CPU 渲染。不能将电脑时间当作 ARM 板端时间，也不能称完整训练已部署在板上。
