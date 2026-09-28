# DDR 输出握手修复（2026-09-27）

首轮板级实现没有生成可用位流，开发板仍为 GSB1。失败证据保留在 `build/board_20260926_233846`：完整路由 DCP、DRC、时序及日志。禁止对 `LUTLP-1` 添加豁免。

根因是 HLS FIFO 接口与厂商仲裁器的组合依赖：原 `kernel_full_n` 直接依赖 DDR `awready`，HLS `kernel_write` 可依赖 `kernel_full_n`，DMA 又用 `kernel_write` 生成 `awvalid`。厂商 `UserCustomOp_0_awwready` 的组合路径包含 `awvalid`，最终形成六个 LUT 的环路。

此前独立仿真让 ready 独立于 valid，无法发现这个问题。将接收端改为“观察到 valid 后才提供 ready”，旧 DMA 在 `evidence/integration_20260927T003514` 可重复无法完成，状态为 `0x3`。该行为在一般 ready/valid 协议中合法，发送端必须独立提供有效数据。

修复在 `rtl/flicker_dma.sv`：增加四条 512 位输出缓冲。HLS 的可写条件只依赖缓冲占用和已接收数量；DDR valid 只依赖缓冲内已有数据，和 ready 之间没有组合反馈。只有 DDR 接受数据后才递增 `write_count`；内核产生数量 `produced` 单列。仍在全部数据写入后执行末地址读屏障，才允许完成通知。

修复后同一握手回归 `integration_20260927T003628` 通过：两排队任务、512 个输出像素、计数、完成 ACK、保护区。随后增加缓冲满状态与暂停期间地址/数据保持检查；真实场景参考仍要求 768 像素全部原始位一致。对应测试源码已开始按次保存哈希和副本，不能用后续源码解释旧报告。

真实场景 `integration_20260927T003713` 与强化缓冲满/输出保持 `integration_20260927T004154` 随后均通过。第二轮物理实现 `build/board_20260927_004328` 正在进行；不能只凭 RTL 回归通过宣布厂商平台 DRC 已通过。

首轮路由报告 WNS=-0.451 ns、WHS=+0.029 ns、WPWS=-0.409 ns。最差整板 setup 在既有视频 resize；经过新增模块的路径还包含上述握手环，不能据此签署新增接口时序。修复后必须重新实现并重新审查，不沿用这些数值作通过证据。
