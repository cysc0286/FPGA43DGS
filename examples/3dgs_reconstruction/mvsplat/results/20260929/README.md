# 2026-09-29 板端结果副本

`full_sfm`、`fast_separate`、`fast_fused`、`focal_control` 为上一轮四组；`speed_threads` 为本轮两线程与四线程的同口径复测。`pipeline_result.json` 记录 ARM aarch64 运行、视频 SHA256、阶段与渲染计时；`input.json` 记录选帧和相机约定；`inference.json`、`adapter_validation.json`、`quality_and_timing.json` 记录模型、转换和画质。`comparison.png` 是视频目标、完整 SfM FPGA、快速 CPU Dense、快速 FPGA 的并排效果图。没有 `comparison.png` 的快速分阶段目录仍保留数值报告。

这些是从被忽略的本地原始 `runs/` 复制的精简副本。原始权重、视频、PLY、相机、全部帧缓冲、监控日志和板端运行环境未提交；图像只是检查效果的预览，数值以 JSON 和 [验证报告](../../FAST_VALIDATION.md) 为准。`final_board_selection.json` 是更改自动选帧后的一次 ARM 准备阶段回归，不是完整链路再次测量。

原视频为 3 秒、30 帧，SHA256 `0a7e3c684e917dd937ae1ec6c64f770c111b939f08f020b66965c1fccf9ae86a`。快速路径用第 0、29 帧预测 32,768 个高斯，第 7、15、22 帧用于相机求解和图像评价。最快单次首图 66.68 秒，但第 7 帧 PSNR 19.45 dB，未过 20 dB 门槛；完整 SfM 质量合格但首图 249.43 秒。当前结果不证明稳定一分钟达标或全视频场景覆盖。

`speed_threads/threads2` 与 `speed_threads/threads4_repeat` 是新增的板端视频接收完成事件到首张 FPGA 帧 SHA256 核验同环境复测，分别为 74.04 与 71.74 秒；使用同一预置视频、相同权重和冻结后端，每组只做一次完整整链运行。四线程比两线程短 2.31 秒（3.1%），但第 7 帧质量仍未达标。`threads4` 保留前期不同预加载配置的 69.88 秒探索记录，不从中挑最快单次冒充稳态结果。旧目录没有视频完成事件，66.68/249.43 秒不属于这一口径。`prepare_t*_measurement.json` 保存 2/4 线程各两次独立准备进程耗时，`inference_diagnostics` 保存独立推理与未采用的 eager 候选负结果；详见 [快速链路分析](../../FAST_VALIDATION.md)。
