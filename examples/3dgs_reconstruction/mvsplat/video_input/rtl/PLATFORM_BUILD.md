# 30TAI 舵机候选平台构建

原平台现存于旧仓库的 Git 忽略目录：
`D:/SyncFiles/1_Workspace/FPGA3DGS/FPGA43DGS/platform/flicker_groupshared_fpga`。
普通 `rg --files` 会漏掉该目录，查找工程时使用 `rg --files --hidden --no-ignore -g '*.xpr'`。
历史 `D:/ADProjects/...` 路径不能直接当作当前工程位置。

本轮板端测试复用既有 ABI2 BOOT，重新编译 ARM 程序。ABI3 正常停止修改需要重新综合、实现和打包；只更新 ARM 程序无法修正旧 FPGA 的末脉冲截短。

## 复制并接入

以下命令从本分支 `examples/3dgs_reconstruction/mvsplat/video_input/rtl` 目录执行，Python 使用本机已配置环境：

```powershell
& 'C:/FPGAtools/python-envs/camera-dev/Scripts/python.exe' ./prepare_camera_servo.py `
  'D:/SyncFiles/1_Workspace/FPGA3DGS/FPGA43DGS/platform/flicker_groupshared_fpga' `
  'D:/cs3/camera_servo_platform_j17_23_21'
```

目标必须为新目录，脚本拒绝覆盖。复制前检查设计依赖均在原平台内，复制后逐文件核对 SHA256，写入 `servo_preparation.json`，并再次检查原平台未变。缺失的展开 XCI 只在同名 XCIX 容器内确实存在时允许。不存在硬链接或外部源文件共享。

修改包括顶层新增 PWM 端口及寄存器路由、新增两个 RTL 与 XDC，以及六个 HLS ROM 包装文件中的数据路径重定位。ROM 数据、计算逻辑、渲染器数据总线和原始 IP 来源不变；厂家钩子会在候选内执行必需的平台补丁。原始来源共 891 文件，约 514 MiB；这只是准备结果，不能代替完整构建。

旧 HLS 生成文件的六处 `$readmemh` 含有 `D:/ADProjects/...` 绝对路径，不能只修改 XPR 文件引用。2026-11-18 首次整板综合发现找不到 ROM 后立即停止，未采用该结果。准备脚本现已将六处数据路径指向独立候选内的原始 `.dat`，逐文件 SHA256 确认数据未变；构建脚本在综合前检查恰有六处、文件可读且位于候选内，再重新综合。

## 构建与审查

```powershell
$env:JFM_PATH='D:/cs3/camera_servo_platform_j17_23_21'
$env:FMSH_PROCISE_PATH='C:/FudanMicro/Procise'
$env:PROCESSOR_ARCHITECTURE='AMD64'
$env:RDI_PLATFORM='win64'
& 'D:/Xilinx/Vivado/2018.3/bin/vivado.bat' -mode batch `
  -source ./build_camera_servo.tcl `
  -tclargs 'D:/cs3/camera_servo_platform_j17_23_21'
```

中断后先检查 `Get-CimInstance Win32_Process -Filter "Name='vivado.exe'"` 和任务日志，确认没有仍在运行的构建，才可追加 `--resume`。此选项保留已完成 IP；重置未完成综合任务，或已完成综合后的未完成实现。它不是从放置/布线任意阶段恢复，也不应与原构建并行。

完成后另用 `review_camera_servo.tcl` 审查经过新增路由的 setup/hold、100 MHz 时钟和 AE13/W13 LVCMOS33。保留整板 timing、DRC 和资源报告；不得把“生成 bit”自动认定为时序通过。

## 保留原启动内容的打包

先按平台内 `procise_run_disable_icap.tcl`、`call_procise_disable_icap.bat` 处理新的位流，核对执行成功及输出来源。不要再次把已处理位流作为输入反复打补丁。

当前 Procise 2025.1.1 对原始 PS 二进制重新封装会增加填充；ELF 包装试验虽然保留载荷，也改变 PS 分区属性。因此本轮没有采用重建 PS 的整包结果。`package_camera_servo.py` 仅接受哈希固定的已验收 BOOT、厂家生成的单 PL 图像和对应 bit；严格核对无加密/认证/数据校验表、分区头校验、链表边界、相同 PL 长度、bit 字节序及 NOP 填充，再只替换原 BOOT 的 PL 载荷。原头部、PS 分区数据和属性、全部 PL 外字节保持一致。任一条件不符即拒绝，不能用于其他启动布局。

厂家打包 Tcl 的内容格式如下（路径替换为本次候选；不是板端执行命令）：

```tcl
# 单 PL 的 candidate_pl.bif 内容：
# the_ROM_image:
# {
#   [destination_device=pl] D:/candidate/ai7030_edif_top_disable_icap.bit
#   [fsbl_config]apu_x32
# }
create_boot_image D:/candidate/candidate_pl.bif D:/candidate/PL.image.bin -fmql
exit
```

通过平台内批处理相同的 Procise 环境运行此 Tcl，须等待进程结束、检查退出码及日志，再运行：

```powershell
& 'C:/FPGAtools/python-envs/camera-dev/Scripts/python.exe' ./package_camera_servo.py `
  'D:/SyncFiles/1_Workspace/FPGA3DGS/FPGA43DGS_test-branch/examples/3dgs_reconstruction/mvsplat/rendering/render_main/hardware/BOOT.bin' `
  'D:/candidate/PL.image.bin' `
  'D:/candidate/ai7030_edif_top_disable_icap.bit' `
  'D:/candidate/BOOT.abi3.bin'
```

输出与同名 `.json` 证据必须不存在。打包不修改板卡或原 BOOT。本机以冻结原 bit 经厂家单 PL 打包后重组，得到与原 BOOT **全部字节和 SHA256 均相同**的结果。另做 12 项文件级核对，包括损坏载荷、加密/认证、错误基线与截断拒绝，以及仅在内存中的一致单字节改动确实只替换 PL 的一字节。它们只能证明此固定布局下的打包与保留逻辑，不能替代新 bit 的实施、启动或硬件功能验收。

实板渲染回归和示波器首尾脉冲检查是后续独立步骤。所有本机脚本均不烧写开发板。冻结 `render_main/hardware` 不用作新 BOOT 输出目录。

## 2026-11-18 本机实现记录

首次重新布线前会话中断，完成的 `ai7030_edif_top_physopt.dcp` 已保留。恢复脚本直接打开该物理优化检查点，核对 AE13/W13 和 100 MHz，再执行原生成脚本的 `NoTimingRelaxation` 布线及厂家位流步骤；未重做已完成的综合或放置。恢复脚本、日志和检查点哈希保存在工作区 `tmp/motor_integration_20261116`，不是通过修改综合输入假装恢复。

首版最终布线仍有 3 个 setup 失败端点，WNS/TNS 为 -0.059/-0.178 ns，位于原 `frame_to_ps_inst0/cmos_resize_24bit_u0` 通路。新增逻辑六组 setup/hold 检查全部通过，但首版整板结果不作为最终产物。随后只执行布线后 `phys_opt_design -directive AggressiveExplore`，未改变 RTL、时钟或增加时序例外；保存独立 `servo_timing_repair/postroute_physopt.dcp`。优化后整板 WNS/WHS 均为 +0.029 ns，TNS/THS 均为 0，setup/hold 失败端点均为 0；150838 条可布线网络全部完成，路由错误为 0。

整板脉宽检查仍有 2 个失败端点，WPWS/TPWS 为 -0.409/-0.479 ns，属于厂家 `clk_out1_AI_CORE_CLK_ST` 时钟；历史 ABI2 记录也有相同最差限制，不能据此宣称完整时序签核。厂家位流前钩子针对 xc7z030 禁用的 GTX 检查也保留原实现，不将该厂家处理写成独立审查无 DRC 问题。

修复后的独立审查命令为：

```powershell
& 'D:/Xilinx/Vivado/2018.3/bin/vivado.bat' -mode batch `
  -source ./review_camera_servo.tcl `
  -tclargs 'D:/cs3/camera_servo_platform_j17_23_21' `
    'servo_timing_repair/postroute_physopt.dcp' 'servo_reports_repaired'
```

本次最终位流输出目录为 `fpai_demo_vivado.runs/abi3_repaired`，与首版 `impl_1` 分开。厂家位流后钩子会自动生成 `_disable_icap.bit`，应检查元数据 `Patched=DISABLE_ICAP,VCCM` 及执行结束，不要再重复执行补丁。单 PL 的 BIF 必须引用修复后目录，不能误取 `impl_1` 首版或原 bit 往返试验的文件。

已生成的候选包在工作区 `tmp/motor_integration_20261116/abi3_firmware_20261118`。`BOOT.abi3.bin` SHA256 为 `4a90e337ae0d321a3f883f09452518727ef9cbf67bbef110f7c2eb9dae278e19`，补丁 bit 为 `d20d5e7f07da068ee97b5c966aca85c95b11d899205e387ca98be41097897cc3`。`BUILD_MANIFEST.json`、原始报告及脚本一同保留；厂家 `dump_boot_image` 再次核对新旧启动头和 PS 内容相同、仅 PL 不同。此恢复只交付 bit/BOOT，不将缺少本地 hwdef 的可选 sysdef 导出写成 SDK 平台导出成功。

用户明确要求先完成本机构建，本轮没有连接、刷写开发板或发送 PWM。新固件启动、渲染回归及实测首尾脉冲仍待上板；俯仰异常记录见 [MOTOR_INCIDENT_20261116.md](../MOTOR_INCIDENT_20261116.md)，通电动作仍暂停。
