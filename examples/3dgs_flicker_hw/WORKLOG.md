# 执行记录（保持历史版本）

## 2026-09-28 渲染封装及视频重建路线评估

用户将当前成果独立命名为3DGS渲染，后续研究已有视频抽帧、生成高斯、目标视角渲染，允许有界画质折中。已生成 `releases/3dgs_renderer_v1_20260928`、ZIP约121.934MB和tar.gz约121.869MB；固定三个ARM二进制/完整示例模型/两相机/参考图/CPU与硬件源/BOOT/清单/输入输出合同。独立入口为包内 `render.py`，不写BOOT，不修改原实验树或数值。

从实际tar.gz上传到板端新目录 `/root/fpga43dgs_releases/20260928T004334`，解压后从 `/tmp` 调用，两视角FPGA和v0 CPU Dense均逐字节匹配冻结结果；验证证据 `evidence/package_smoke_20260928T004334`。单次调用分别3728.716/2963.170ms（FPGA）和4521.089ms（CPU Dense），仅功能复跑，不取代原预热1/测3的3163.533/2962.008ms。归档SHA见 `releases/3dgs_renderer_v1_20260928_archives.json`。最后引擎idle、DMA无错误、tmpfs为空、内存可用747MiB。

只读资源检查与文献核对完成；`examples/3dgs_reconstruction/README.md`记录新链、论文定位、板端限制、资源/质量控制和分阶段验收。当前ffmpeg可用，colmap未在PATH；没有运行视频重建或训练。内存预算表是参数+梯度+Adam状态解析估算，非实测峰值。建议离线COLMAP+原始3DGS流程、Taming预算控制候选，FLICKER包仍为渲染后端；并非复现全部候选论文。源代码包、历史指标与尚未验证的后续方案不得混称已部署。

## 2026-09-28 当前状态：完整基础板端链已验收

split II16已安装并获得同输入同输出2.167x/2.091x加速，随后按模块完成CPU属性、分组排序和整链对照。当前基础路径是PLY＋相机→ARM投影/SH/分组/排序→SDK双缓冲→FPGA AABB/CAT/求值/有序合成，运行时不再需要GPU。BOOT与原渲染程序未再次改变。命令见 `frontend/README.md`，完整指标/论文对应见 `frontend/VALIDATION.md`。

选择整数深度-ID排序 `frontend_group_20260927T233920`，对原逐Tile稳定排序输出逐字节一致，单次排序核1143.844/996.895→86.946/73.265ms。四版源码/二进制/列表及负结果见 `frontend/versions.csv`。板端有效Gaussian ID与官方一致；v0列表成员相同，v10少4个Tile实例并有次序差异，已披露。

正式 `full_benchmark_20260927T234919`：CPU base4/CPU Dense4/FPGA Dense各预热1测3，单调时钟、tmpfs中间文件，包含PLY读入、CPU前处理排序、进程和SDK初始化、输出；不含SSH与显示。均值ms：FPGA3163.533/2962.008，CPU Dense4569.384/3916.531，CPU base8250.038/6928.529；对优化CPU加速1.444x/1.322x。各后端重复输出无漂移。原始CSV、分析、读回图和实际BOOT只读来源绑定均归档；样本不足以声称尾延迟或稳态视频吞吐。

新列表HLS参考在 `frontend_hls_v0_20260927`、`frontend_hls_v10_20260927`；raw实板检查 `cat_raw_20260928T000229/000236` 各983040B零差异，计时最终图也逐位匹配同契约Golden。PSNR46.2075/47.9167、SSIM0.9982361/0.9989669通过当前近似质量门槛，官方严格门槛仍失败；CPU base新前端也有微小差异。LPIPS/功耗未测。

基础功能链完成不等于整篇FLICKER完成：还缺混合FP8/FP16 CTU、聚类/选择读取、模型剪枝/微调和跨帧常驻流水。CPU属性和分组约占67%，下一增量优先复用模型静态量、常驻内存和CPU前处理并行。Slice99.99%限制进一步下沉。NPU与下游任务仍延后，无待确认事项。构建和测量会话均已结束；下面是历史记录，不要依旧“在途”状态重启工作。

## 2026-09-27 22:52 当前进展：II16 split 已实板闭环

FLK1 主候选 `platform/flicker_splitopt_fpga` 已生成位流并通过路由审查，报告 `build/board_cat_20260927_210933`。BOOT `9918f9b69525bad88d8a9f06c5ab0940582df341ca32edc786d4377d3ba137cc` 经哈希保护更新，旧 BOOT 与回滚命令保存于 `evidence/boot_20260927T223617`；当前 boot_id `8be7c81f-773a-4ff4-91e8-10a6128c9acf`。SDK 加法与15个块传输检查、协议小输入及两完整视角 Dense raw 逐字节检查均通过。正式五次组 `cat_measure_20260927T224323`、串行组 `224856`、连续10帧组 `225121` 已登记且绑定实际 BOOT/程序来源；前后比较见 `evidence/split_board_comparison_20260927/REPORT.md` 和 `VALIDATION.md`。

两视角旧→新 FPGA Dense 均值 1418.217→654.365 ms、1328.043→635.104 ms，输出逐位一致，分别 2.167x/2.091x；速度仍仅 1.528/1.575 FPS。新版同轮 CPU4 5868.965/4671.431 ms、CPU Dense4 2093.690/1636.969 ms，精度与部分筛选算法不同，不能称同精度硬件加速。新位流串行 876.480/814.129 ms，连续10帧 651.945/634.312 ms 且输出无漂移。FPGA Dense 对官方图 PSNR 45.451/47.924 dB、SSIM 0.9978197/0.9989729，原官方 FP32 严格输出门槛仍不通过。整板 Slice 19,649/19,650（99.99%）是后续扩展瓶颈；新200MHz路径 setup/hold +0.310/+0.054ns，原厂 AI 脉宽两违例仍在。前处理/排序仍是离线 GPU；NPU 和下游任务未纳入本轮。

低寄存器占用的独立 II32 备选 `platform/flicker_split32_fpga` 已完成物理构建，会话11056结束，报告及冻结文件在 `build/board_cat_20260927_215430`。HLS/RTL 集成通过；布局 19,646/19,650 Slice（99.98%），只比实测 II16 少3个 Slice，整板 setup/hold +0.030/+0.019 ns，原厂 AI 脉宽两违例仍在；小输入 Dense RTL 5071 vs II16 5042 周期。未安装或测整帧，不能给 II32 写板端 FPS。因不能实质释放扩展空间且没有性能优势证据，保留当前 II16 位流，下一模块按顺序补板端 CPU 前处理和排序。两个候选构建会话及本轮测量会话均结束；不要重启或覆盖。

## 2026-09-27 23:15：板端属性前处理已验证，排序仍未迁移

新增 `frontend/attributes.cpp` 和 `frontend/stage.py`，在当前 II16 位流对应的板端 Linux ARM CPU 上读取锁定的官方 559,263 点 PLY 与 FLCAM001 相机输入，计算 view-space depth、屏幕均值、二维协方差/conic、opacity 和 degree-3 SH RGB。视角0/10总耗时 1490.120/1436.692 ms，其中投影 1111.873/1076.050 ms；有效高斯 457,115/384,680 与官方导出一致，遗漏0。字段均值误差保持在 1e-6 到 1e-8 量级，完整字段误差见 `evidence/frontend_attr_20260927T231255/validation.json`。

属性替换混合渲染 `frontend/hybrid_render.py` 通过质量门槛：沿用官方 GPU active ID、Tile ranges、深度排序键，只替换板端属性；PSNR/SSIM 为 46.208/0.998236 和 47.917/0.998967，对当前 Dense 基线的回归在冻结的约1 dB/0.001内。混合总时延 669.569/649.679 ms；这是单次隔离测试，不能把其与旧输入五次均值654.365/635.104ms的差直接归因为属性修改；该结果包含30次 FPGA任务，仍不含板端分组/排序，不能称 Camera Pose→Framebuffer。

一次将板端计算 depth 直接替换排序键的尝试被输入校验拦截（相邻高斯因 ARM/GPU 浮点差异不再满足有序合同），失败日志保留；当前隔离测试继续使用官方排序 key，并单独记录 depth 误差。下一模块实现板端 Tile 分组与稳定排序，先比较候选列表与官方列表的排序/覆盖差异，再接入完整渲染；不跳过该验证。NPU 仍延后。

## 2026-09-27 split 优化启动：FPGA 流程与速度优先

21:55 在途状态更新（优先读本段）：用户两项澄清均已回答，无待确认。主候选 II16 在 `flicker_splitopt_fpga` 布局 Slice=19,649/19,650（99.99%），BRAM=200.5/265，随后布线拥塞，仍在 route global iteration 1，**未完成、未上板**。会话 69776，日志/报告见下文。因实际拥塞，另建仅把 prepare_geometry II16 改成 II32 的独立源树 `evidence/split32_candidate_20260927/sources`；主工作源码仍为 II16。II32 HLS C/RTL PASS，几何 FF 估计 11,627→8,597，冻结 `hls_pipeline_split32_20260927T215423`。备用平台 `platform/flicker_split32_fpga` 已启动实现（会话11056，日志 `build/split32_board_console_20260927.txt`）；真实输入集成仿真会话5957，日志 `build/split32_integration_console.txt`。不要重新启动/覆盖两个平台。新增 freeze_hls.py --source-root 记录各候选实际源树，避免混入主目录不同 II 的源码。

旧版带计数 RTL 重跑 `pipeline_integration_20260927T211443` 已 PASS。同51记录/3 Tile，mode0/1/2 旧周期 33429/31368/13191，II16 新 30888/28174/5042，Dense 2.616x；属于小规模 RTL 协议场景，非板端指标。差分与两个仿真会话均已结束。`split_profile_comparison_20260927.json` 已保存。正式旧板五次组已分析画质（analysis.json），待新板测量后用 `pipeline/compare_split.py` 汇总；`record_provenance.py` 只读核对实际 BOOT/boot_id。新板要求六模式两完整视角、同程序、预热1测5，再串行消融和Dense连续10次。新增 `SPLIT_OPTIMIZATION.md`、`split_acceptance_20260927.json` 记录边界，不称完整 Camera Pose→Framebuffer。

后续前处理的只读准备：官方模型已在 `npu_3dgs/data/official/train/point_cloud/iteration_7000/point_cloud.ply`（138,698,755 B），相机同目录。已按锁定提交获取 GraphDECO forward.cu/auxiliary.h/config.h/LICENSE.md 于 `evidence/frontend_source_20260927`；旧源哈希使用 CRLF，下载 LF→CRLF 后精确匹配，变换写入 provenance.json。**尚未实现/计时新板端前处理**，不要写成已补齐流程。先完成当前 split 单模块实板验收再推进。

用户确认无需复制 ASIC 配置，以 FLICKER 的 FPGA 处理流程完整性和速度为目标，允许可控质量折中。当前单模块先优化 split，不改输入 ABI、六模式运算顺序或 FP16 Golden。板卡 boot_id 与已安装 FLK1 相同，空闲且可用内存 749 MiB。复测 `cat_measure_20260927T205621` 主视角 mode0 3.937786 s、Dense 1.430015 s，各自逐位匹配 Golden；这是诊断单次样本，不替代正式重复测量。源快照 `split_baseline_20260927T*` 保留。下一步 HLS、背压回归、实现及新旧板端对照；未完成项不能登记为已部署。

用户进一步明确：CPU/FPGA 分工可以都尝试，以效果优先；只研究硬件部署层，不开展定位导航或检测任务验证。

21:19 接续记录：`pipeline.cpp::split` 已拆为 frame_gaussians→prepare_geometry（II=16）→expand_subtiles（内层 II=4），前后用有序 BRAM FIFO 解耦；其余 dispatch/CTU/render/gather 源码与快照相同。不是跨 Tile Gaussian 缓存，也不是把几何移到 CPU。HLS `hls_pipeline_splitopt` 完整 C/RTL PASS，冻结于 `hls_pipeline_splitopt_20260927T210845`；独立前端对旧源码差分 `split_differential_20260927T211400` PASS（68 Gaussian、5 Tile、六模式、465 输出记录，含空/边缘/非正定 conic）。真实 RTL 集成 `pipeline_integration_20260927T210941` 六模式 PASS，增加仿真专用阻塞/握手计数，未改硬件 ABI。旧版同输入带计数重跑 `pipeline_integration_20260927T211443` 当时仍在执行。

独立平台 `platform/flicker_splitopt_fpga` 正在物理构建，报告 `build/board_cat_20260927_210933`，主日志 `build/splitopt_board_console_20260927.txt`。启动会话 69776，旧版集成仿真会话 50363；不要重复启动或覆盖项目。仍未更新板上 BOOT。候选安装用新 `pipeline/install_candidate.py --expected-boot-sha256 <已核对旧值>`，旧 `install_boot.py` 仍只适用 FLK0→FLK1。更新须先完成物理审查、package、freeze_board、核对 integration 哈希；当前已知旧 BOOT 为 `853d016f99b9f92f076c9cec13f519cf74616244b9f34e7c444c1ab543938e0f`。SDK 路径和板端可执行程序不变。

正式更新前对照已完成：`cat_measure_20260927T211227`（两视角，每组预热1/测5，cpu4/cpu_dense4/mode0/1/2），登记 `FLK1_BEFORE_SPLITOPT_20260927`。v0/v10 平均 ms：CPU4 5757.647/4659.023；CPU Dense4 2085.543/1629.742；mode0 3937.258/3555.099；mode1 1847.805/1820.241；Dense 1418.217/1328.043。`record_provenance.py` 已以只读挂载核对旧 BOOT，并绑定 boot_id、报告、程序/平台哈希。诊断单次样本也已单独登记，不混入五次组。下一步新板同条件重复、六模式整帧回读、串行消融/连续10帧、画质/资源/时序/优劣汇总。功耗和 LPIPS 未测。

## 2026-09-27 后续范围确认：CPU+FPGA 优先

用户明确先完善 CPU 和 FPGA 协同，再考虑 NPU。现状和精度重新核对了源码与 `cat_measure_20260927T194410/analysis.json`，未新增板端计时。CPU 当前不实时完成投影/SH/Tile 分组/排序；这些仍为离线 GPU 准备。FPGA 保留深度有序合成，FLICKER 筛选不使算法变成无序。优先 `split` 单模块调度/复用及停顿统计，随后评估 VRU 持续流水、CPU 参数准备和整条板端前处理。NPU 仅保留后备分析，详见 `pipeline/IMPROVEMENT_PATH.md`。

## 2026-09-27 本轮收尾：FLK1 完整实板对照已完成

当前没有运行中的构建或测试会话。FLK1 lean 已安装；BOOT SHA256 `853d016f99b9f92f076c9cec13f519cf74616244b9f34e7c444c1ab543938e0f`，boot_id `39319f54-9b83-40c9-beb2-8f82407c2ec9`。最后只读健康探针 `post_flk1_20260927T195500`：cap 464c4b31、ABI 20000、idle=0、DMA error=0、749 MiB available、16 GiB 磁盘可用；没有异常重启。

- `cat_measure_20260927T194410`：两完整视角 × CPU1/CPU4/CPU Dense4/六 FPGA 模式，预热1/实测3，全组已完成。
- `cat_measure_20260927T195129`：同两视角 × mode0/1/2，串行预热1/实测3，输出与流水及 Golden 一致。
- `cat_measure_20260927T195312`：Dense 流水每视角预热1/实测10，无漂移，逐位匹配 Golden。
- 三组均完成版本登记与 BOOT/程序来源侧录；`board_cat_summary_20260927/REPORT.md`、`comparison.json` 从冻结原始数据生成。主矩阵另有 `analysis.json`、`board_comparison.png`，已检查实际回读图。

Dense 平均 1.419/1.328 s，对四核 CPU 为 4.077/3.530 倍，对 CPU Dense 为 1.478/1.235 倍；FP32/FP16 和 AABB 算法差异已注明。同位流 mode1→2 为 1.302/1.370 倍，串行→流水降延迟 17.85%/15.74%。模式0反而比旧 FLK0 慢约1.88/1.84倍，Sparse/adaptive 几乎无额外提速且误差更大；保留负结果，首选 Dense。

所有硬件输出符合各自 FP16 契约，但官方严格 FP32 门槛失败；Dense PSNR 45.451/47.924 dB、SSIM 0.9978197/0.9989729，相对于官方渲染而非照片。新路径200MHz通过，整板原厂AI脉宽/I/O/CDC限制与Slice99.80%风险保留，不能写为整板完整签核。当前仍不是 Pose→Frame 全链，论文前处理/排序/聚类/混合精度/模型剪枝微调未完成。

下一模块先在同一 FLICKER 主线优化 split/VRU 持续调度并补停顿计数，保持公式、顺序及当前 FP16 Golden；不得直接跳到新的论文或 NPU。每模块闭环后再推进，历史平台和原始测量不得覆盖。实板 status 使用 `stagecat_20260927T122707/render status`，当前不要重跑只接受旧 FLK0 BOOT 的安装器。详见 `pipeline/VALIDATION.md` 和 `pipeline/README.md`。

## 2026-09-27 19:45：FLK1 已安装，完整视角原始输出通过

`board_cat_20260927_182153` / lean 平台已完成布线和位流。整板 setup/hold +0.030/+0.029 ns，新渲染/DMA 200 MHz +0.373/+0.055 ns；10 个已有总线偏斜约束最差 +2.523 ns，146,159 个可布线网络全部完成，0 连线错误、0 组合环路、0 无约束内部端点。仍有原厂 AI 时钟两个脉宽违例（−0.409/−0.479 ns）、厂商 SDI GTX 豁免和原有 I/O/CDC 覆盖限制，详见 `timing_review.json`，不是整板无条件签核。最终整板 61,839 LUT / 94,751 FF / 269 DSP / 151 RAMB36+73 RAMB18；Slice 19,611/19,650（99.80%），余量很小。FLK1 为 25,844 LUT / 44,151 FF / 261 DSP / 62 RAMB18。

BOOT `853d016f99b9f92f076c9cec13f519cf74616244b9f34e7c444c1ab543938e0f` 已通过分区验证、冻结和网络安装，证据 `boot_20260927T193531` 保留旧 BOOT 与回滚命令。重启后 boot_id `39319f54-9b83-40c9-beb2-8f82407c2ec9`，cap `464c4b31` / ABI `20000` / idle 0。`lean_block_20260927T194112` 三组原厂加法与十五组块传输通过。

协议小输入 mode0/2 在 `cat_raw_20260927T194221`、`194222` 通过。完整 240 Tile 两视角原始输出均逐字节匹配冻结 HLS 参考且保护区通过：v0 mode0 `194245`、mode2 `194256`；v10 mode0 `194303`、mode2 `194313`。每份输出 983,040 B，零差异；Gaussian 输入分别 1,072,038 / 980,729 条含头。raw 硬件周期仅用于正确性阶段，不代替含搬运的总耗时。

当前正式测量 session **4671**，证据 `cat_measure_20260927T194410`：两个完整视角，cpu1/cpu4/cpu_dense4/mode0–5，warmup1/repeats3，流水调度。尚需等其完成 → 登记/质量分析 → 同硬件 serial 对照及连续帧稳定性 → 汇总优缺点。此前所有构建 session 已完成或取消，不要重复构建或再次运行 FLK0→FLK1 安装。后续 status 使用 `stagecat_20260927T122707/render status`，不能再用旧 FLK0 程序。

## 2026-09-27 18:54 当前有效构建

只继续 `platform/flicker_cat_lean_fpga` / `build/board_cat_20260927_182153`，session **37315**，日志 `build/platform_cat_lean_console.txt`。综合通过，进入整板网表优化；综合报告 52,966 LUT / 83,788 registers / 269 DSP，较相同综合阶段 compact 减少 2,314 LUT / 2,684 registers / 8 DSP。最终面积与时序待布局布线，不能套用前一个候选的 placed_audit。

compact 的 session 73018 **已经结束**。其路由耗时 65 分钟仍有 1,816 overlap，时序中间结果 WNS −0.694 / TNS −21.106 ns；已核对进程归属后主动停止，源文件、opt/placed/physopt DCP、日志及报告冻结于 `evidence/routing_congestion_compact_20260927T185259`，`cancellation.json` 明确区分人为停止与工具自然失败。没有该候选的 routed DCP / 可安装 BOOT。不要再轮询 73018，也不要宣称此候选已通过路由。

18:36 板卡只读检查通过，证据 `pre_flk1_20260927T183611`：boot_id `c75f1046-cf09-453e-a45f-593e0140dd69`、FLK0 idle、729 MiB available，16 GiB 磁盘可用。仍未安装 FLK1。lean 使用 `pipeline/probe_block.py` 做安装后 SDK 回归，协议证据仍为 `pipeline_integration_20260927T164003`、板程序仍为 `stagecat_20260927T122707`。

## 2026-09-27 18:24：布线拥塞与独立精简候选

`board_cat_20260927_165427` / `platform/flicker_cat_compact_fpga` 的布局通过，但最终放置报告为 19,562 / 19,650 Slice（99.55%），63,985 LUT、97,436 registers。路由发生 Route 35-447 拥塞警告，已进入 Global Iteration 1，仍在执行（session 73018）。尚无最终时序、位流或新上板结果。`placed_audit` 只代表放置 DCP，不能用于安装验收。不能把新出现的旧 GSB1 setup 违例与原厂 AI 脉宽违例混为一谈。

为减少拥塞，新增独立 `platform/flicker_cat_lean_fpga`，构建报告 `build/board_cat_20260927_182153`、日志 `build/platform_cat_lean_console.txt`、session 37315。渲染器、四 VRU、CTU II8、DMA 均不变，仅删除此候选中的早期 GSC1/GSB1 计算核，保留原厂 adder/DMA/寄存器/复位；历史平台不改。`legacy_trim_sim_20260927T182010` 的差分仿真通过：16/64/256 配置产生既有 count+1 的 17/65/257 个正确输出，周期性停顿下原厂 DMA 周期相同，寄存器读背压保持；移除的 capability 返回 0/未映射，未伪造可用标志。此前两次仿真启动失败分别是漏列 AVR 依赖与缺省 timescale，日志保留；没有核逻辑修改来绕过功能失败。

选用 lean 时须运行 `pipeline/probe_block.py`，该脚本只修改旧块传输探针的平台识别条件，保留原有加法/保护区/15 组块回读检查。安装程序还会验证 lean 清单中的差分仿真记录与 RTL 哈希。标准 compact 仍使用 `board/probe_block.py`。两候选共用协议证据 `pipeline_integration_20260927T164003` 和程序 `stagecat_20260927T122707`；安装前必须审核最终路由 DCP 并核对实际选中平台。

板卡仍为 FLK0；未安装任何 FLK1 BOOT。后续仍是最终时序审查 → 打包/冻结 → 哈希保护安装/网络重启 → capability/SDK/raw 两完整视角 → CPU 与六模式/串行流水对照。不能因出现 lean 候选就覆盖或遗忘当前拥塞版本。

## 2026-09-27 17:32 当前执行位置

当前实际构建 `build/board_cat_20260927_165427`（平台 `platform/flicker_cat_compact_fpga`）已经越过此前失败的详细布局，处于 Phase 4.1.1 放置后时序优化。不能把此进展写成最终布线/上板通过。工具进程持续计算，当前会话构建 session 为 73018；日志 `build/platform_cat_storage_verified_console.txt` 及平台 `fpai_demo_vivado.runs/impl_1/runme.log`。

本次应使用协议证据 `evidence/pipeline_integration_20260927T164003`：包含 LUTRAM DMA、CTU II8、四 VRU，六模式 4,608 像素与原参考及周期均一致。板端可执行文件仍用已核对源码/二进制哈希的 `evidence/stagecat_20260927T122707`。接续顺序为：等物理实现完成 → `pipeline/audit.tcl`（FLK_REPORT_DIR 指向本次报告）审核新路径 setup/hold 与原厂约束 → `package.ps1` → `freeze_board.py` → `install_boot.py --integration <164003目录>` → 网络重启/新 boot_id/cap/SDK 回归 → 两完整视角 raw mode0/2 → CPU 与六模式、串行/流水和连续性对照 → 登记及质量分析。当前没有新 BOOT 安装，板端仍为 FLK0。

实际优化网表为 66,272 LUT / 97,294 FF / 277 DSP；DMA 为 1,062 LUT（536 LUTRAM）/ 414 FF。最终布线数待报告，不以此推断最终时序或性能。失败基线和 CTU-only 失败均已独立冻结，禁止丢弃或覆盖。

## 2026-09-27 16:55 接续状态

- CTU II=8 实际优化网表减少 5,911 LUT、10,496 FF、112 DSP，但详细布局仍差 947 Slice（原先差 3,132）。失败全量冻结 `evidence/placement_failure_ctu8_20260927T165028`，不存在可安装位流。
- 已接入仅属于 FLK1 的 `pipeline/rtl/flicker_dma.sv`，修正无复位存储写进程以推断 LUTRAM；FLK0 原文件保持不变。对应 `pipeline_integration_20260927T164003` 六模式 4,608 像素精确通过，与原实现周期完全一致。
- 当前实际构建 `build/board_cat_20260927_165427`，平台 `platform/flicker_cat_compact_fpga`。清单含上一失败快照与所用协议验证，全部源哈希已经核对。先前 `165302` 同步失败引发的错误启动已停止并记入 status，不视为新候选。
- 板卡仍为 FLK0，SSH/SDK 正常，当前未更新 BOOT。本轮新性能、画质、最终资源和时序尚待物理实现与实板验收。

## 2026-09-27 15:48 接续：修复 FLK1 容量失败

- 实际工具与本地工程均可访问，不再要求用户重复提供。板端只读探针在 15:06 正常，仍是 FLK0，没有安装未通过的新设计。
- 13:16 的 FLK1 布局失败，证据 `placement_failure_20260927T131658` 完整保留；原工程 `platform/flicker_cat_fpga` 不覆盖。
- 固定连线输出打包候选 C/RTL 已通过，但 HLS 面积下降不能代替物理结果。新的 `hls_pipeline_ctu8` 仅进一步放宽 CTU 发射间隔，复用算术，保持四 VRU 和数字契约；后续使用独立 `platform/flicker_cat_compact_fpga`。
- 协议集成、平台清单增加 HLS RTL 哈希链；安装要求仿真与实际构建使用相同 RTL，避免验证旧版本却安装新版本。
- 详细资源、负结果和待验收项目见 `pipeline/VALIDATION.md`。本次尚无新板端画质、延迟或功耗测量。
- 15:54 七次 C/RTL 事务通过，冻结 `hls_pipeline_ctu8_20260927T155459`；16:03 六模式真实输入协议回归全部通过，4,608 像素精确一致，证据 `pipeline_integration_20260927T155536`。仿真与待构建平台的 90 个 HLS 文件哈希相同。
- 构建 `board_cat_20260927_155527` 继续中。隔离 DMA 存储推断试验 `dma_storage_probe_20260927T160624` 发现将无复位的数据存储写操作与异步复位控制分开，可让综合器推断 LUTRAM；独立综合为 5,764→1,200 LUT、18,940→414 FF。该试验尚未集成、未功能验收，不能把独立模块数字当作整板节省或性能收益。

## 2026-09-27 12:45 接续：FLK1 集成验收中

以下是最新状态；下文早期“尚未上板”等记录仅描述对应历史时间。

- FLK0 三项架构修正已经实板验收，两个完整视角额外连续各 10 帧输出稳定；记录 `measure_20260927T012130`，不将这些重复帧当作新增独立场景。
- 独立 `pipeline/` 已接 AABB、CTU、四个 mini-tile FIFO/VRU，定义 FLK1 / ABI 2；FP32 参数在 FPGA 内转 FP16。四 VRU 是本平台资源缩放，论文为 32 VRU。NPU 不参与。
- 原直接索引版本 HLS 循环 II=72。强制消除依赖的候选虽然显示 cosim Pass，却产生大量依赖检查错误，已拒绝并冻结证据，不能上板。当前改成显式 16 像素工作集，载入→计算→提交，未使用依赖豁免；计算循环综合 II=1，六模式真实 C 参考保持一致，RTL 联合仿真仍在运行。II=1 不是每条 Gaussian 每拍完成，载入/排空仍需实测。
- 两个完整视角六模式的 HLS C 参考已经生成、哈希冻结。关闭 AABB/CAT 时逐位匹配 FLK0。Dense 相对官方图 PSNR 45.451/47.924 dB，Sparse 30.977/32.143 dB；均不是实板测量，也不能用改善的 PSNR 推导 CAT 提升真实质量，误差可能抵消。完整质量文件位于 `pipeline_full_v0/quality.json` 和 `pipeline_full_v10/quality.json`。
- 新板端程序 `stagecat_20260927T122707` 已编译；FLK1 尚未安装，当前板仍为 FLK0。后续必须通过六模式真实 DMA/CDC 集成、整板布线和新路径时序，再网络安装、回归、完整场景比较。原 FLK0 平台和 BOOT 保留。
- `register_measurement.py` 已区分 FLK0、FLK1 六模式和 CPU Dense 的测试范围，并要求硬件逐位通过各自 FP16 参考，防止版本表误写“无 CAT”。

2026-09-26 夜间，用户授权先修正三个架构问题，然后沿单一 FLICKER 论文继续。NPU 暂停。

已完成且有证据：

- 旧 Q16 渲染器的末项 R/G 重复累加与周期漏计已修复；64 像素 / 512 条输入回归逐字段、周期通过。仅修复旧契约，不作为本轮数值基线。
- 原 GSB1 位流上测试原厂 SDK 的 PL-DDR 块传输，1 KiB–8 MiB；8 MiB 五次写 7.65–12.01 ms、读 7.87–7.97 ms，逐字节一致。旧加法器 DMA 被证实 count+1，新接口已重新实现精确计数。`evidence/block_20260926T235754`。
- 新全 FP16 完整参数输入渲染器 C/RTL cosim 通过，3 Tile / 768 像素。HLS 估算 45 DSP、14,591 LUT、9,498 FF、7 BRAM18；估计周期 5.292 ns，不能当成 200 MHz 时序已通过。
- 新 512 位 DMA 预取、跨时钟描述符、一运行一排队任务、完成 ACK、写后读屏障、地址拒绝、DDR 停顿、保护区联合仿真通过。`evidence/integration_20260926T235843`。
- 真实场景 768 Gaussian / 768 pixel 的 HLS+DMA+CDC 输出逐位匹配 HLS C 参考。`evidence/integration_20260926T234812`。
- 两完整模型视角各 1 次预热 + 3 次测量，CPU 单核/四核严格官方精度通过。主视角 19.458 / 5.829 s；第二视角 17.089 / 4.670 s。`evidence/measure_20260926T235156` 与 `versions.csv`。这不是新增 FPGA 性能。

实现中的功能：

- `hls/renderer.cpp`：每条高斯到 FPGA 内部计算 power/exp/alpha、终止、合成和背景。输入每个 Gaussian 64 B，在 Tile 内复用，不展开像素命令。mode=0，仅完整 FP16 基线；CAT 不存在，不能声称有 CTU。
- `rtl/flicker_dma.sv`：512 位 PL-DDR，一未完成读取 + 32 深度预取队列，输出独立，末地址读屏障后完成。
- `rtl/flicker_regs.sv`：0x400c0200 窗口，capability FLK0，一运行一排队描述符，toggle CDC，完成快照保留至 ACK。
- `board/render.cpp`：同一输入纯 CPU 基线；serial/pipeline 两缓冲批处理；整帧回读与标签校验；raw 输入模式用于 PC HLS golden 精确对照；记录阶段时间及上传前后硬件 busy 状态。

当前进行中：

1. `platform/flicker_fpga` 是独立拷贝，保留旧 GSC1/GSB1 和加法器；首轮完整厂商实现 `build/board_20260926_233846` 在 bitgen 前被 LUTLP-1 拒绝。根因是 HLS WRITE/FULL_N 与厂商 DDR VALID/READY 组合反馈。已加四条输出 FIFO，修复前/后同一 reactive-ready 回归红→绿；真实输入回归及更强的满队列暂停检查继续中。详细证据见 `HANDSHAKE_FIX.md`。失败 DCP 与报告已保留，禁止豁免组合环路。新 BOOT 尚未生成、未安装，原板仍为 GSB1。
2. 完整 559,263 模型两个视角 HLS C golden 已在 `evidence/full_reference_v0` / `full_reference_v10` 完成；主视角输入 1,072,038 条 64 B 记录（包括头），68,610,432 B。未优化 CSim 输入装载慢，已停止该进程保留日志，改为 64 bit 装载与 `csim_design -O` 重跑。数值内核不变。第二视角通过四个独立 CSim 进程按 Tile 分片生成并合并，执行记录和输出哈希在 `execution.json`，这不是 CPU 性能测试。
3. 两视角 FP16 参考对官方 FP32 PSNR=43.5383/46.3670 dB，RGB max=0.249753/0.0444612、mean=0.00251789/0.00253952，T max=0.0444297/0.0260626，last 不同=8043/7975，均未通过严格门槛。无非有限输入或半精度量化后非正定 conic；误差不能简单归因于这两项。`evidence/fp16_quality_audit` 保存误差分布与可视化。硬件必须先逐位匹配其 FP16 参考，随后单独评价近似质量。
4. 板端最新 `stage_20260927T003123` 已编译 raw CLI、只读 status、输出前后保护区；含串行/流水、busy 时间线，CSV 的 `unserved_kernel_read_requests` 保留真实含义。`board/validate_raw.py` 只上传输入并回读，与电脑上的 HLS golden 逐字节比较；`board/measure.py` 对万点及两个完整视角都核对 FP16 参考。`register_measurement.py` 维护版本 CSV、分阶段指标及保守的上传重叠证据。以后用最新已编译且 source hash 匹配的 stage。

接续顺序：

1. 等待综合实现并审查资源、整板时序与新模块路径。若新渲染路径失败先修复；不能靠统计 HLS 周期宣称板频已达成。
2. 用 `evaluate_reference.py evidence/full_reference_v0` 把完整 HLS raw 转为 GSSOUT01 并对官方精度比较。保留严格失败；FP16 与官方 FP32 不是同一数值契约。
3. 包装 `board/package.ps1`、检查 PL/非 PL 分区一致性；`board/install_boot.py` 使用当前 GSB1 SHA256 437f6894…，保存旧 BOOT 和回滚命令。新路径时序未验收禁止安装。
4. 网络重启后先 cap/read-only/原 SDK 回归，再上传真实 input.flk（不上传 Golden）运行 raw，PC 回读逐位比较；通过后跑万点和两个完整模型 serial/pipeline，对纯 CPU。
5. 完成三项实板验收后，在同一数据通路加入 FLICKER 的 8×8 AABB、4×4 CAT、PR 共享、adaptive、FIFO 背压和精度消融，再补论文前处理/排序与模型准备。还没有完整论文复现，不能提前宣称完成。

JTAG 保持断开；SSH 用已固定的主机公钥及环境变量密码，不在文件中保存口令。不要重跑旧直接 `/dev/mem` 写实验。

## 2026-09-27 00:55 接续状态

- 输出握手修复的真实输入回归 `integration_20260927T003713` 通过，768 Gaussian / 768 pixel 全部原始位一致，预取交叠 770 次。
- 强化暂停/满队列/输出保持回归 `integration_20260927T004154` 通过。`platform_sync_20260927T004326` 已按哈希同步唯一修改的 DMA 源码，并保存原 manifest。
- 第二轮板级构建目标 `build/board_20260927_004328`，正在布局后优化。仍未生成或安装新 BOOT。
- 独立 `ctu/` 已实现全 FP16 的共享阈值、双矩形、Dense/Sparse/两种 adaptive、级间 FIFO 与 mask merge，尚未接入当前平台。120 输入 × 5 模式 + 空输入 C/RTL 通过，SRL 版本 HLS 估计 14,032 LUT / 22,447 FF / 135 DSP / 34 BRAM18；目标 5 ns、估计 5.292 ns，未做物理时序验收。证据 `ctu_fp16_srl_20260927T005305`。
- CTU 真实属性测试正在运行：768 Gaussian 展开为 2,560 个有效 8×8 sub-tile 输入，各跑五个模式。输入、属性、标志生成规则在 `evidence/ctu_real`，日志 `build/ctu_real_20260927.log`。不能将此模块通过表述为完成三个实板问题或完整 FLICKER。

## 2026-09-27 01:22 实板基线验收

第二轮 bitgen 与厂商转换通过；新路径 setup/hold +0.111/+0.055 ns，整板 +0.030/+0.029 ns，仍有原厂 AI 时钟 WPWS -0.409 ns。完整限制写入 timing_review.json；BOOT 分区一致性通过后已网络安装并重启，旧 BOOT 保留。旧加法器与块传输回归通过，3 Tile 和两个 240 Tile 完整视角实际 FPGA 原始输出逐字节一致。12 组同输入 CPU/serial/pipeline 结果已追加到 versions.csv；详细指标及误差见 VALIDATION.md。当前另在执行两个完整视角各 10 次 pipeline 连续运行稳定性检查。之后继续论文 CTU 集成，不能把本基线声称完整论文。
