# 稀疏旋转候选的内存交接

此增量移除每组 Gaussian 的 NPZ 落盘、重读校验和文件哈希扫描；采用同一个已加载推理会话，每组直接交出经过校验的数组。原角度归属、点预算、顺序、坐标、协方差、SH 和透明度计算均不变。未新增实板性能或画质测量。

## 接口

`generate_ring(runtime, prepared, out, defer_archive=True)` 调用 `runtime.infer(..., memory_only=True)`，随后立即 `take_gaussians`、复制保留点、释放该组完整数组。此路径不序列化任何 Gaussian NPZ，也不创建 `_pending_archive`。只保留各组的小型来源/运行记录与最终合并数组。

首帧完成后，由协调器调用 `persist_ring_archive(record, gaussians, out)`：保存最终 NPZ，重读并逐字段精确比对，然后发布文件哈希及归档完成状态。原有 `defer_archive=False` 仍会在函数返回前完成最终归档。

延迟归档时，`gaussian_sha256` 与 `gaussian_content_sha256` 都代表规范数组内容，`gaussian_identity_kind=array_content_v1`；归档后这个身份不变，NPZ 文件哈希单独写入 `gaussian_archive_sha256`。立即归档保留 `gaussian_identity_kind=npz_file_v1`。读取导出或恢复结果的代码必须区分内容身份与文件身份。

## 估算的 I/O 减少

每个 32,768 点输出包含 88 个 FP32 值/点，数组负载约 11 MiB。旧的每组保存流程写一次、重读校验一次、扫描文件哈希一次，共约 33 MiB 逻辑存储负载；两组可省约 66 MiB，四组约 132 MiB。此估算不含 ZIP 头、文件系统/缓存效果、内容哈希的内存读取以及最终归档，**不是实测 DDR 流量或耗时收益**。

最终合并 NPZ 仍然保存和校验，显式延迟模式将这些工作移到 `FRAME_COMPLETE` 之后。内容哈希仍需要读取数组，因此不能说完全没有数据扫描。

## 验证与边界

8 项 ring 测试通过：新路径首帧前禁止调用任何 `np.savez`；两组共用一个模型；弱引用检查确认完整前组数组在下一组开始前释放；各字段逐位一致；最终 NPZ 与原合并逻辑产生的文件 SHA 相同；内容变异、未知已有文件和校验失败均拒绝；归档重试不重写 Gaussian 数据。

6 项内容交接测试通过：字段顺序、大小端、strided/Fortran 存储等价时身份稳定；值、点数和浮点精度变化改变身份；非法形状、NaN/Inf、透明度和整数数组被拒绝。fake tensor/model 验证连续两次 memory-only 推理均不序列化 NPZ、不产生 pending archive，交接只能消费一次，互斥参数和已有归档状态仍受保护。

关联相机、ring 与内容交接测试共 43 项通过（1 项跳过）。测试未加载完整预训练模型、未使用真实 NPU/FPGA，未测首帧提速、PSNR/SSIM、功耗或真实内存峰值。新增运行代码会改变资格身份，需重新实板资格后再部署，不能沿用旧 receipt 声称通过。
