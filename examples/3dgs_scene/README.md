# 完整图像大场景光栅化

结果与边界先看 [板端验证报告](VALIDATION.md)；[输入与正确性契约](CONTRACT.md)冻结实际执行阶段、官方门槛和不同高斯计数。四个规模不是复制小 Tile 点的压力测试，均从官方 train 场景选择不同 Gaussian，完整模型另外测试第二视角和 640×356 图像。

在仓库根目录、云 GPU 与板卡已连接且 GSB1 BOOT 仍在运行的前提下，用项目环境调用。SSH 凭据只通过临时环境变量给脚本，勿写入工程文件：

```powershell
& npu_3dgs\.venv\Scripts\python.exe examples\3dgs_scene\prepare.py
& npu_3dgs\.venv\Scripts\python.exe examples\3dgs_scene\run_board.py --phase calibrate
& npu_3dgs\.venv\Scripts\python.exe examples\3dgs_scene\run_board.py --phase measure --reuse-build examples\3dgs_scene\evidence\calibrate_YYYYMMDDTHHMMSS --cpu-repeats 3 --fpga-repeats 3
& npu_3dgs\.venv\Scripts\python.exe examples\3dgs_scene\register_results.py
```

`prepare.py` 需 `GS_CLOUD_PASSWORD`，`run_board.py` 需 `FPGA_BOARD_PASSWORD`；相应 SSH 主机密钥须已预置。GPU 来源和导出文件 hash 逐项核查。`run_board.py` 只传场景输入到板卡，官方答案不上传；输出回到 PC 再比。`make_previews.py` 用 Pillow 从实际板端读回生成 PNG，`evaluate_readbacks.py` 在云 GPU 上调用锁定的 GraphDECO SSIM/LPIPS 实现做**同场景官方输出误差**评价，不是与实拍照片评价训练质量。

本次冻结场景在 `data/20260926T135717`，两次校准与两次重复计时在 `evidence/` 的对应时间戳目录。`data/`、`evidence/` 内大体量二进制和预览图为本机实测证据，应保留但默认不把几十到上百 MB 的场景文件放进 Git。`npu_3dgs/benchmarks/scene_versions.csv` 与 `scene_registry.json` 保存可读版本索引；使用这些索引时应同步保留哈希指向的原始目录。
