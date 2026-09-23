# FPGA43DGS

面向悟净 30TAI Lite 的基础 FPGA / 3DGS 开发项目。

首个提交提供独立的 [32 位加一器示例](examples/adder_smoke/README.md)，用于验证代码托管和 RTL 仿真流程。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\examples\adder_smoke\run_sim.ps1
```

需要已安装 Vivado；默认路径为 `D:\Xilinx\Vivado\2018.3\bin`，可通过 `-VivadoBin` 指定其他路径。运行成功时打印 `PASS: add_one`。

当前示例验证复位、有效信号、空闲周期和 32 位溢出回绕。它是纯 RTL 仿真示例，没有板卡管脚、DDR/DMA 或 ICraft 依赖；通过仿真不表示已完成上板或 3DGS 部署。
