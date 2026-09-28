# 本轮验证记录

工程：`platform/basic_alu_fpga`（独立副本）；硬件：悟净 30TAI Lite。

| 层级 | 当前结果 | 证据 |
|---|---|---|
| 核心 RTL 仿真 | 2,800/2,800，协议及复位检查通过 | `build/sim_20260924_235524/simulation.log` |
| 实际顶层寄存器仿真 | 2,800/2,800，背压与 DMA 隔离检查通过 | `build/register_sim_20260924_235447/simulation.log` |
| ARM64 板端程序编译 | 通过，动态库均找到 | `build/board_native_build.log` |
| 旧位流识别保护 | 正确拒绝未加载 ALU1 的位流，未提交运算命令 | `build/baseline_capability_guard.log` |
| 综合、布局布线、位流 | 完成，Lite R6/R5 匹配，ALU 保留在实现中 | `build/board_20260924_235147/status.txt` |
| 新位流时序验收 | **整板未通过**；详见下文 | `build/board_20260924_235147/timing_summary.rpt` |
| JTAG 临时加载 | Procise 下载完成，板端识别 ALU1 | `build/program_20260925_002038.log` |
| 真实 FPGA 数值比对 | **2,800/2,800 全部逐位一致** | [实测原始输出](evidence/hardware_test.txt) |
| 原 DMA 加法器回归 | **16/16 正确** | [原加法器输出](evidence/original_adder_regression.txt) |
| 运行原加法器后再次测 ALU | **2,800/2,800 正确**；同一数据集的重复运行 | [再次实测输出](evidence/alu_after_adder.txt) |

仿真日期为 2026-09-24；板端实测完成于 **2026-09-25 00:23（北京时间）**，板端日志使用 UTC，显示为前一天 16:23。`build/` 是本机生成的被忽略目录；仓库保留代码、向量和上述原始输出，不包含供应商工程文件。[机器可读记录](evidence/result.json)包含源码、向量和位流 SHA256。

## 实测结果与时序边界

有效操作各 233 组，ADD32 额外 1 组用于非法操作后的恢复，因此 ADD32 为 234 组；非法操作 3 组。所有 2,800 组均在真实 FPGA 上执行，CPU 通过 ICraft XRT 寄存器访问提交、读取并核对预先生成的 Golden。

新增 ALU 在当前约束下工作于 **100 MHz**，实现资源为 **724 LUT、396 FF、0 DSP、0 BRAM**。经过 ALU 的最差建立时间裕量为 **+4.560 ns**；对 ALU 寄存器输入、输出的保持时间检查最差为 **+0.085 ns**。

整板时序仍不合格：WNS **−0.090 ns**，TNS **−0.158 ns**，2 个建立时间端点失败。补充报告确认这两条路径终点是原视频读出模块的 `vout_data_r_reg[10]` 和 `[9]`，分别为 −0.090 ns 和 −0.068 ns；本次加入 ALU 后的整板布局变化使它们出现违例，不能说整板时序没有退化。保持时间 WHS **+0.029 ns**，无失败端点；原有周期/脉宽问题仍为 WPWS **−0.409 ns**、2 个失败端点。总线偏斜报告未发现违例。

这些结果支持本轮 ALU 的开发性功能验证，**不代表整板时序验收、视频功能或长期稳定性已经通过**。没有放宽时钟、增加 ALU 假路径或更改原约束来消除违例。后续发布常驻版本前需修复视频路径时序，并核实厂商 AI 时钟与器件库约束。

## 当前板卡状态与重跑

本次仅通过 JTAG 临时加载新 PL，**没有修改 SD 的 BOOT.bin**。加载前后 Linux boot ID 相同，SSH 未断开。板卡当前保留 ALU 测试位流；重启后会恢复 SD 中已验证的原厂 Lite 加法器位流，此时 ALU 测试会因为模块标识不匹配而停止。

当前板卡可直接重跑：

```sh
cd /root/fpga43dgs_basic_alu
timeout 120 ./test_basic_alu vectors.txt
```

此次位流 SHA256：`dcf13e3c744e45e0b5aa030f4cb77b9ecea9349cc397e69699345679a0639875`。本轮没有验证视频输出、NPU 推理、3DGS 渲染、长期压力运行或温压角条件，也没有把约 0.012 秒的整组寄存器测试耗时作为 FPGA 核心延迟。
