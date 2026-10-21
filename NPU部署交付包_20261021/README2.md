# 悟净 Lite 板端 NPU 部署 · 交付包

生成日期：2026-10-21
平台：复旦微 JFMQL30TAI676H（悟净 30TAI Lite）／NPU：ZG330（AI_Mate）／模型：MVSplat 高斯生成

---

## 这个包是什么

把 MVSplat 的**部分子图**部署到了板载 NPU，其余子图保留在 CPU（混合部署）。
本包包含：**实测报告、可复现代码、过程文档、原始证据日志**。

## 先看哪一个

| 想知道什么 | 打开 |
|---|---|
| **结论与证据**（给负责人看） | `01_报告/NPU部署实测报告.html` |
| 怎么复现 | `02_代码/` + 本文末尾的「复现命令」 |
| 过程中试过什么、否掉了什么 | `03_文档/` |
| 数字从哪来（原始日志） | `04_原始证据/` |

---

## 一句话结论

> NPU 确实在板上执行了（设备识别成功、`npu_executed: true`、单子图快 57 倍）。
> 可复现的端到端收益是 **−1.33 s / 首帧 28.07 → 26.74 s（−4.7%）**，画质差异 63.6 dB（肉眼不可分辨）。
> 仓库现行的逐元素数值门禁（`rtol=2e-3, atol=2e-4`）**判 7 个子图全败，但画质全部合格**——
> 该门禁测的对象与项目目标不一致，建议改为图像级验收。

---

## 目录说明

### 01_报告
| 文件 | 内容 |
|---|---|
| `NPU部署实测报告.html` | **主报告**，单文件、图片已内嵌，浏览器直接打开即可。含 A 部分（NPU 真实执行的证据）与 B 部分（门禁应改的论证） |
| `npu_vs_cpu_frame7.jpg` | 报告里的对比图原图：留出实拍 / CPU 高斯 / NPU 高斯 / 差异×10 |

### 02_代码
| 文件 | 作用 | 备注 |
|---|---|---|
| `verify_diagnostic.py` | `verify.py` 的诊断副本：仅去掉两处 `raise`，其余一致；产出 `cpu/` 与 `hybrid/` 两套高斯 + `verification_diagnostic.json`（`diagnostic_only: true`） | **不改仓库原文件** |
| `image_probe.py` | 用同一个 NumPy SH3 渲染器渲染两套高斯，与留出帧比较，输出 PSNR/SSIM | 自研，用于相对比较 |
| `pack_figure.py` | 生成三列对比 + ×10 差异热图（JPEG，便于传输） | |
| `channel_group_attribution.py` | 按输出通道组归因误差，判断"局部权重坏"还是"整体精度不足" | 只依赖 numpy |
| `magstats.py` | 量值诊断：相对误差分位数、TF32/FP16 格点检验 | |
| `serial_console.py` | 串口控制台：`--list / --log / --send / --script / --wait-port` | PC 侧工具 |
| `sweep_partitions.sh` | 批量跑多个子图（每个单独一轮 + 出图） | 板端脚本 |

### 03_文档（按阅读顺序）
| 文件 | 内容 |
|---|---|
| `NPU部署方案.md` | 最初的方案与目标阶梯（G0–G5）、根因排序 |
| `上板检查清单.md` | 从开机到跑通的九步，含串口/驱动踩坑记录 |
| `代码审计结论.md` | 逐文件审计 + 多会话污染假设（H1）的推导 |
| `仓库已有能力清单.md` | 仓库已经做过什么，避免重复造轮子 |
| `首次上板实测.md` | 第一次上板的完整实测，含通道组归因 |
| `图像级验证结果.md` | **7 个子图的画质扫描**，以及"数值排序 ≠ 画质排序"的发现 |
| `分区分速与策略.md` | 各子图的加速收益表与部署优先级 |
| `口径对齐.md` | 我们的数字如何换算到仓库的"首帧 28.066 s"口径 |

### 04_原始证据
| 文件 | 对应什么 |
|---|---|
| `npu_probe.log` | `icraft_probe` 输出：设备识别成功 |
| `attribution_backbone_nchw.log` | backbone 按 8 个通道组归因（全组均匀，排除权重布局 bug） |
| `magstats_backbone.log` | 相对误差分位数、TF32/FP16 格点检验 |
| `image_probe_result.log` | 3 个留出视角的 PSNR/SSIM |
| `sweep_summary2.log` | 7 个子图的数值误差 + 画质汇总 |
| `speed_extract.log` | 各子图 CPU 与混合推理耗时 |
| `combo_evidence.log` | 两子图组合失败的直接证据（`Nonfinite output`） |

---

## 复现命令

板端（`B=/root/fpga43dgs_reconstruction/npu_board_v2_20260929`）：

```bash
cd $B/mvsplat
PYTHONPATH=$B:$B/deps \
LD_LIBRARY_PATH=/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu:/root/fpga43dgs_reconstruction/arm_env/lib \
OMP_NUM_THREADS=4 /root/fpga43dgs_reconstruction/arm_env/bin/python3 \
  -m npu.verify_diagnostic --backend npu \
  --graphs $B/candidate/graphs --bundle $B/candidate/compiled \
  --input $B/candidate/input --weights $B/weights/re10k.ckpt \
  --vendor $B/vendor/MVSplat_reference --out $B/<新目录> \
  --library $B/arm_bridge/libmgs_npu.so \
  --worker-python /root/fpga43dgs_reconstruction/arm_env/bin/python3 \
  --threads 4 --partitions backbone_cnn
```

出图（需 `LD_PRELOAD` 解决 cv2 的 libgomp TLS 问题）：

```bash
LD_PRELOAD=/root/fpga43dgs_reconstruction/arm_env/lib/libgomp.so.1:/root/fpga43dgs_reconstruction/arm_env/lib/python3.11/site-packages/torch.libs/libgomp-58a43326.so.1.0.0 \
LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib \
PYTHONPATH=$B:$B/deps <python> -m npu.image_probe \
  --input $B/candidate/input --cpu <out>/cpu/gaussians.npz \
  --npu <out>/hybrid/gaussians.npz --out <out>/image
```

---

## 必须如实告知的四点局限

1. **混合部署，非全部子图上 NPU**：两个及以上子图同时上 NPU 会产生非有限值（权重在 PL DDR 互相覆盖，属 SDK 层）。当前只上 1 个。
2. **`to_gaussians` 曾成功一次、目前不可复现**：2026-10-20 成功跑通（−5.38 s，画质 63.7 dB，产物齐全），
   10-21 连续 7 次失败（含断电重启后首次）。原因未定位。因此本包给出的**可复现数字只有 `backbone_cnn`**。
3. **画质对比用自研渲染器**：NumPy SH3 泼溅，非 FPGA 渲染器、非官方 CUDA，仅用于相对比较。
4. **线程口径不同**：仓库 `verify.py` 写死 2 线程，默认入口是 4 线程，基线相差 1.54 倍。本包数字均标注了线程数。
