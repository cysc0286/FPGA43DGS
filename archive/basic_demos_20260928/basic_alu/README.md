# 基础整数运算器：仿真与 30TAI Lite 上板测试

本例在一个 `basic_alu` RTL 模块中实现 12 种运算，采用“CPU 写操作数 → FPGA 运算 → CPU 读结果 → Python Golden 精确比对”的验证流程。它是基础算术与寄存器链路测试，尚不是 3DGS 渲染器或吞吐优化实现。

想对照源码、状态机波形和布线后时序学习，可阅读 [运算器源码与时序导读](WALKTHROUGH.md)，并运行 `open_waveform.ps1` 打开已校验周期数的短波形。

## 运算与数值约定

两个输入均为 32 位原始比特，结果寄存器为 64 位。有符号解释采用二进制补码。除乘法外，结果只使用低 32 位，高 32 位为零。

| 操作码 | 运算 | 约定 |
|---|---|---|
| 0 | ADD32 | `a+b`，溢出按 32 位回绕 |
| 1 | SUB32 | `a-b`，下溢按 32 位回绕 |
| 2 | MUL_U32 | 无符号 32×32，保留完整 64 位结果 |
| 3 | MUL_S32 | 有符号 32×32，保留完整 64 位补码结果 |
| 4 / 5 / 6 | AND / OR / XOR | 逐位与、或、异或 |
| 7 | SHL32 | 逻辑左移，移位量为 `b & 31` |
| 8 | SHR32 | 逻辑右移，移位量为 `b & 31` |
| 9 | SAR32 | 算术右移，移位量为 `b & 31` |
| 10 / 11 | MIN_S32 / MAX_S32 | 两个有符号 32 位数的最小值、最大值 |
| 其他 | 非法操作 | 返回 0，并置错误标志；下一条有效命令清除错误 |

乘法使用 32 次移位累加，负结果额外进行一次取补码。该实现便于核验全宽结果；后续若需要吞吐率，可以单独比较 DSP 乘法及流水实现。本例没有实现浮点、除法、指数函数或 NPU 算子。

## 测试数据和判定

`generate_vectors.py` 用独立 Python 整数运算生成 `data/vectors.txt`。每种操作测试 13 个边界值的全部两两组合与 64 个固定种子随机组合，共 233 组；另有 3 组非法操作和 1 组错误后的恢复操作，总计 **2,800 组**。边界包含零、全一、正负极值、交替位型和跨越 31/32 的移位量。

每个结果要求逐比特完全相等，容差为零；完成序号、忙标志、错误标志也必须正确。向量文件及 SHA256 见 `data/manifest.json`。测试程序拒绝空文件、截短文件和格式错误。

- `sim/tb_basic_alu.sv`：核心运算、提交前不执行、接受后锁存输入、重复序号不重跑、运算中复位及复位后恢复。
- `sim/tb_register_alu.sv`：经过本地供应商 `adder_top/reg_ctrl` 的实际地址译码测试，包含读返回背压、版本寄存器检查及 ALU 命令不会误触发 DMA。
- `board/test_basic_alu.cpp`：通过板端 ICraft XRT 访问真实 FPGA 寄存器，逐条核对同一份 Golden；不在 CPU 上代算实际结果。

## 1. 独立 RTL 仿真

在仓库根目录执行：

```powershell
python .\archive\basic_demos_20260928\basic_alu\generate_vectors.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\archive\basic_demos_20260928\basic_alu\run_sim.ps1
```

Vivado 默认位置为 `D:\Xilinx\Vivado\2018.3\bin`，可用 `-VivadoBin` 覆盖。成功标志为 `PASS: basic_alu (2800 golden vectors, protocol and reset checks)`。

## 2. 接入本地板卡工程

本步骤需要已经配置并验证的本地供应商工程 `platform/fpai_demo_fpga`。供应商包不随本例发布。准备脚本复制到 `platform/basic_alu_fpga`，只在副本中连接 ALU；保留原 DMA 加一器和其版本寄存器，记录原文件及新 RTL 的 SHA256。已有副本时脚本会拒绝覆盖。

```powershell
python .\archive\basic_demos_20260928\basic_alu\board\prepare_project.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\archive\basic_demos_20260928\basic_alu\board\run_register_sim.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\archive\basic_demos_20260928\basic_alu\board\build.ps1
```

编译使用 Vivado 2018.3 与现有复旦微补丁/Procise。过程日志、引脚、时序和资源报告写入 `build/`；**生成位流与时序验收是两个独立结论**，须检查 `status.txt` 和 `timing_summary.rpt`。本例没有为新运算器增加时序例外。

## 3. 寄存器 ABI

基地址 `0x400C0000`，32 位寄存器。以下均为字节偏移。

| 偏移 | 方向 | 内容 |
|---|---|---|
| `0x0C` | 写 | A |
| `0x10` | 写 | B |
| `0x14` | 写 | 操作码 |
| `0x20` | 写 | 请求序号；必须最后写入 |
| `0x84` | 读 | 结果低 32 位 |
| `0x88` | 读 | 结果高 32 位 |
| `0x8C` | 读 | 已完成序号 |
| `0x90` | 读 | bit 0：忙；bit 1：非法操作；其他为零 |
| `0x94` | 读 | 模块标识 `0x414C5531`，即 `ALU1` |
| `0xC0` | 读 | 保留原加法器版本 `0x20230628` |

每次仅允许一个未完成请求。先写 A/B/操作码，再提交与已完成序号不同的非零序号，轮询完成序号后读取结果。序号相同不会重新执行；本测试程序读取已有序号，因此可重复运行。结果在下一条命令完成前保持不变。复位时请求与完成序号均为零。

不要同时运行多个访问该模块的进程；模块无软件锁或命令队列。不要修改保留给原 DMA 示例的 `0x00/0x04/0x08` 寄存器。

## 4. JTAG 加载和板端验证

先确认使用 **30TAI Lite**、R6/R5 引脚报告匹配，并检查时序报告。`program.ps1` 的 `-ReportDir` 指向本次成功编译产生的 `build/board_<时间>`。脚本核对源码和报告后，使用 Procise 经 JTAG 临时加载 `ai7030_edif_top_disable_icap.bit`。默认下载器序列号为本项目已验证的 `210251784595`，可以通过参数更换。

若报告存在已经审查的时序违例，开发性功能试验需要显式传入 `-AllowTimingFailure`；该参数不会将时序验收变成通过。

将 `board/test_basic_alu.cpp`、`board/build_native.sh` 和 `data/vectors.txt` 上传到板卡同一目录，在该目录执行：

```sh
sh build_native.sh
timeout 120 ./test_basic_alu vectors.txt
```

依赖已安装的 ARM64 ICraft XRT 3.33.1、开发头文件、g++、libdw/libelf 等。编译采用 SDK 所需的 `gnu++17`。程序在执行任何运算前核对 ALU 标识及原加法器版本，版本不匹配立即失败。成功标志必须是 `PASS: FPGA basic_alu 2800/2800 exact matches`。

JTAG 加载不写 SD 卡；重新启动会恢复 SD 卡中原有的启动位流。可以再运行原 `adder_demo` 核对两者共存。程序输出的总耗时包含 SDK 寄存器访问和轮询，不表示 FPGA 核心延迟或渲染性能。

也可以在已安装 Python `paramiko` 的电脑端运行 `board/run_remote.py --host <板卡IP> --known-hosts <已核验的known_hosts文件>`，自动上传、检查 SHA256、编译及运行测试，日志放入 `build/hardware_<时间>/`。它使用已有 SSH 密钥或进程环境变量 `FPGA_BOARD_PASSWORD`，拒绝未知主机密钥，不会自动烧录 FPGA。密码只放在本次进程环境中，不写入源码或日志。

## 当前验证记录

实时结果及本机日志见 [VALIDATION.md](VALIDATION.md)。仿真、位流生成、时序验收、真实上板分别记录。
