# test-branch 本机验证记录

日期：2026-11-16。基于远端 main 的459dc8b整理四步入口及必要的驱动、轨迹、配置和RTL源码；现有NPU推理与冻结渲染实现保持不变。

在 `examples/3dgs_reconstruction/mvsplat/` 运行：

```sh
python -m unittest video_input.acceptance.test_acceptance video_input.test_camera_scan video_input.test_camera_servo video_input.test_recording -v
```

结果为65项测试，64项通过、1项因本机Windows无法建立符号链接而跳过。两种阶段04模式都交给本分支真实initialize参数解析器校验，在打开硬件之前截停；已移除仅存在于本地未发布实验中的选组参数，避免克隆分支后出现未知参数错误。

共享C++执行核心可独立验证，不加载SDK或打开设备：

```sh
python -m video_input.acceptance.stage02_trajectory --out /工程外新目录/trajectory
g++ -std=c++11 -O2 -Wall -Wextra -Werror video_input/test_camera_servo_scan.cpp -o /工程外目录/test_camera_servo_scan
/工程外目录/test_camera_servo_scan /工程外新目录/trajectory/trajectory.csv
```

本机MinGW编译成功，25项检查通过，覆盖默认833点轨迹及共享执行核心。计划运动16.66秒、录像17.66秒均为规划值，不是实测性能。

本轮未连接板卡、刷写固件或驱动舵机；ARM SDK完整版编译、真实PWM波形、USB录像、NPU/FPGA整链及各方向画质仍待四步现场验收。硬件资源、板端时延、带宽与功耗本轮未测，不产生加速或画质提升结论。源码、参数和说明直接在仓库提供，不提交压缩包、视频、部署私密配置或运行输出。
