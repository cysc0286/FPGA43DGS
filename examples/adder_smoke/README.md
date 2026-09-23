# 32 位加一器：最小 RTL 仿真示例

`add_one` 在时钟上升沿采样输入。当 `in_valid=1` 时，输出寄存器更新为输入加一，并将 `out_valid` 置为 1；当 `in_valid=0` 时，`out_valid` 清零，数据寄存器保持原值。`rst_n` 为低有效异步复位。所有加法按无符号 32 位回绕，`0xffffffff + 1 = 0`。

## 运行

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\examples\adder_smoke\run_sim.ps1
```

指定安装位置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\examples\adder_smoke\run_sim.ps1 -VivadoBin 'D:\Xilinx\Vivado\2018.3\bin'
```

脚本依次调用 Vivado 的 `xvlog`、`xelab`、`xsim`。日志位于本目录的 `build/<时间戳>/`，不纳入 Git。仿真错误、超时或缺少 PASS 标记时，脚本以错误退出。

## 验证范围

测试覆盖启动复位、连续有效输入、无效周期数据保持、整数边界、溢出回绕、256 组确定性伪随机输入，以及运行中异步复位后的恢复。成功输出为 `PASS: add_one (303 checks)`。

这是新增的独立教学示例，不是厂商板端 `adder_demo`。本例没有管脚约束、时钟约束、CPU 寄存器接口或 DMA，也不生成 BOOT；RTL 仿真通过不代表 30TAI 上板、时序收敛或 3DGS 部署通过。
