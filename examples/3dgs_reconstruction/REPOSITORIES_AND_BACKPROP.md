# 可复用仓库与反向传播归属

更新：2026-09-28。当前目标：质量合格后总耗时最短，允许充分使用板上资源。本文明确代码复用和模块归属，不切换默认重建链或宣称 NPU 已部署。

## 仓库选择

| 仓库 | 用途 | 与当前工程的关系 |
|---|---|---|
| [MVSplat 作者仓库](https://github.com/donydchen/mvsplat) | 已知内外参的多视图图像，前馈预测高斯；提供预训练权重和测试入口 | 速度优先的首选候选。保留 A 的 PoseSet，复用作者 B 的网络，适配高斯输出后接已有 C。源码已独立取得，未执行 |
| [OpenSplat](https://github.com/pierotofy/OpenSplat) | 逐场景高斯优化，支持 CPU 路线 | 当前 B 已经使用其锁定源码和资源补丁；电脑训练已跑通，ARM 完整训练尚未核验，是现有参考链 |
| [NoPoSplat 作者仓库](https://github.com/cvg/NoPoSplat) | 未给位姿的少量图像预测规范坐标中的高斯；提供2/3视图权重 | 更激进的后续路线。会改变 A/B 边界，长视频的窗口对齐和全局一致性仍需实现；本次未下载完整仓库或运行 |

优先复用 MVSplat 的 encoder、权重、配置、输入归一化与评价代码，不自行重写网络。作者 decoder 使用 CUDA，板端将由已冻结的 CPU＋FPGA 渲染器替换；NPU 算子适配与模型输出格式转换仍需开发。默认 SH4 与现有 SH3 接口、坐标系、scale/opacity 激活、深度范围必须显式适配，不能只把张量改名为 PLY。

本地 MVSplat 位于 `vendor/MVSplat_reference/`，包含138个源码/配置等文件；来源为作者 GitHub `main` ZIP，压缩包 SHA256 为 `55b9dd485902313366c304d435cc32b5ad3000efaf1274aa26d53cee5a6da9fe`。Git 克隆因本机失效代理及直连重置失败，改用作者 codeload 源码归档成功；没有修改全局代理。归档不含 Git 历史，本轮未取得 commit SHA，因此以归档和逐文件哈希固定来源。

作者原始 LICENSE 保留。未安装其依赖、未下载权重和数据集、未运行推理/训练、未编译 ICraft 或更新位流；不能称“下载后已直接在板上跑通”。`vendor/` 被 Git 忽略，团队交接需要同时提供此源码包和清单，不能仅发送主仓库链接。

建议实施顺序：先用作者测试输入/预训练权重核验原仓库结果；再导出同一高斯接冻结后端，比较图像；最后将 encoder 子图迁移 NPU，计入回退、变换、搬运、CPU 几何和 FPGA 渲染，比较第一张合格图像总时延。现有 CPU/OpenSplat 参考保留。任何用于原作者基准的 GPU 运行不代表板端完成，最终目标仍是板上执行。

## 反向传播属于高斯生成/训练模块

高斯生成模块内部包含一个可微渲染的训练循环：

```text
输入：视频帧 → 位姿/内参/稀疏点
                     ↓
训练：初始化高斯 → 可微正向渲染 → 与实拍图计算损失
       ↑                              ↓
       └── 更新高斯参数 ← 反向传播计算梯度
                     ↓ 训练结束
              导出高斯模型＋相机
                     ↓
输出：CPU 前处理/排序 → FPGA 正向渲染 → 目标视角图像
```

反向传播计算损失对高斯位置、尺度、旋转、opacity 和颜色/SH 的梯度；参数更新由优化器完成，增密/裁剪是额外的点集合管理。SH0 当前只学习常量颜色，不代表位置/尺度等不训练。三维点数量与结构可能随训练变化。

本地实际代码（相对 `vendor/OpenSplat_bounded_v1/`）：

| 工作 | 位置 |
|---|---|
| 正向渲染 | `opensplat.cpp:244` 调用 `model.forward` |
| 损失 | `opensplat.cpp:249` 调用 `model.mainLoss`；实现 `model.cpp:1042` |
| 反向传播入口 | `opensplat.cpp:250`：`mainLoss.backward()` |
| 参数更新 | `opensplat.cpp:259` 调用 `optimizerStepCadence`；`model.cpp:258` 使用 Adam 并清梯度 |
| CPU 光栅化反向 | `rasterize_gaussians.cpp:220` 的 `RasterizeGaussiansCPU::backward`，调用 `rasterizer/gsplat-cpu/gsplat_cpu.cpp:328` |

`modules/gaussian_generation/training.py` 是调度层，通过 `--cpu` 启动上述程序；它本身不包含 Python `loss.backward()`。当前训练使用 OpenSplat CPU 的可微渲染路径，不是调用已冻结 FPGA 渲染包来求梯度。电脑执行证据不能替代 ARM 完整训练的验收。

当前冻结渲染后端只有正向计算，没有训练反向协议、保存激活/可见贡献供求导的接口、梯度归约和优化器。后续若用 FPGA 加速高斯训练，需要单独实现兼容的前向和反向合同；不能将“先有一个 FPGA 正向渲染器”写成“训练反向也已具备”。位姿估计模块的束调整也会做数值求导/优化，但属于几何求解，和高斯图像损失反向要分开描述。

## 使用预训练 MVSplat 后是否还反向传播

有两种运行模式：

- **作者训练或我们重新微调网络：**图像/位姿 → encoder → 高斯 → 可微 decoder → 损失 → 反向经过渲染器和 encoder → 更新网络权重。反向仍属于高斯生成模块的训练过程，需可微 renderer。作者 `src/model/model_wrapper.py:123` 的 `training_step` 返回 `total_loss`，由 Lightning 的训练执行器处理反向与优化；不能因文件里没有显式 `.backward()` 就认为没有反向。
- **加载预训练权重做场景推理：**图像/位姿 → 固定 encoder → 场景高斯 → 现有后端渲染；常规推理不计算训练梯度、不更新网络权重。作者 `model_wrapper.py:179` 的 `test_step` 为测试路径。若后续额外加入逐场景细化，细化部分才重新需要反向。

两种训练对象不同：当前 OpenSplat 优化的是每个场景的高斯参数；MVSplat 离线预训练优化的是跨场景预测网络的权重。前馈部署不是无中生有地省略必需计算，而是复用已学到的场景先验；新视频上的质量必须验证。两者的最终输出都可以通过适配交给渲染后端。

## 本次证据与限制

[源码取得记录](evidence/repo_handoff_20260928/source_acquisition.json)、[源码完整性与代码定位](evidence/repo_handoff_20260928/verification.json)。本轮是仓库与调用路径核对，没有新增速度、PSNR/SSIM、功耗或资源数据；预计收益仍待同输入、质量合格条件下的完整测量。
