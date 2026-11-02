# 两个局部预测头的 NPU 迁移与实板验收（2026-11-02）

## 结论与边界

按“运行可靠性 → to_gaussians → to_disparity”的顺序实施。两个真实模型预测头已经在
悟净 30TAI Lite / ICraft 3.36.1 上执行，通过固定输入 CPU/NPU 三留出视角的原生 FPGA
图像资格、完整网络重复一致、资格前后输入切换和有限值检查；四子图公开视频入口已到首帧。
这不是全部 MVSplat 迁入 NPU，也不是板端反向训练。没有改权重、图像分辨率、高斯数量、
FPGA RTL、BOOT、四业务模块或冻结渲染包；没有放宽既有图像资格门槛。

严格张量门槛 `rtol=2e-3, atol=2e-4` **仍失败**。SDK 权重预检与跨请求状态问题的根因
**仍未关闭**。本轮实现并验收的是有限形状、固定模型和图像政策下的运行规避，不应声称
SDK 已完全修复、任意视频均可靠或没有精度损失。短耗时失败项不计加速。

## 结构与分工

```text
initialize：预先加载固定权重、NPU 会话和渲染环境，合成预热、检查，发布 READY
video_input / CPU：视频完成后解码、选帧
pose_estimation / CPU：SIFT、匹配、上下文位姿及目标 PnP
gaussian_generation：
  NPU：backbone_cnn → depth_head → to_gaussians / to_disparity（按网络实际依赖调用）
  CPU：跨视图 Transformer、代价体重采样、残差/升采样/投影、细化 UNet、世界坐标适配
rendering：CPU 投影/分组/排序 → 冻结四路 FPGA 求值和合成 → 网页图像
```

NPU 只在生成/更新场景时工作，已有场景更换观察视角不会重新执行网络。
代码仍以 `video_input/pose_estimation/gaussian_generation/rendering` 四模块组织；
本轮候选位于 `gaussian_generation/npu_branch`，没有新增平行主线。

## 运行可靠性：先诊断再接入

| 对照 | 实板观察 | 结论 |
|---|---|---|
| 原完整 128×128 单视图 head | 出现非有限输出，预检报权重错误 | 原大块执行不能接入 |
| 已知权重 identity16 / affine17 / spatial16 | 各 6 次 A,A,B,B,A,A，均逐元素完全相等且有限；后两者预检仍报错 | 预检错误不能单独等同计算失败，也不能直接忽略 |
| SDK 分配区读写 | 64 B～20 MiB，2 图样×2 回读，共 24 检查零错误字节 | 未复现通用内存搬运损坏；不等于证明所有 SDK 权重路径正确 |

真实 head 的单视图 36×36 图能得到有限、接近参考的结果，batch=2 的小图仍曾非有限。
仅分块、仅 fresh input、额外等待、重新分配 OCM 等候选未通过完整交错检查，均保留失败证据。
fresh input 单次曾通过，但后续控制再次失败，因此未将其单独当成修复。

当前受验收约束的组合是：单视图小块 + 每次新 SDK input Tensor + 每块执行两次 forward、
取第二次结果。backbone 继续使用原有两次策略。两次真实执行的输入写入、提交、等待和输出
转换均计入，不读缓存伪造第二次，不把 CPU head 计算冒充 NPU。
单 worker 串行管理会话；完整 head 拼回过程持同一事务锁，防止不同请求交错覆盖。

完整 oracle 在资格前后复查，同时检查输入 A/B 切换、有限值、重复结果；任一检查失败就不发布
READY 或场景。运行身份绑定 fresh input、head 两次执行、SDK/桥接库、模型与编译资产。
SDK 权重预检报错仍保存在日志中，不能据本轮结果宣布其根因关闭。
独立 `transport_probe.cpp` 只能在所有 NPU worker 停止后运行；SDK 分配器的进程本地状态
不能当成跨进程设备内存隔离。该诊断没有接入正式请求入口。

## 分块为何保持原来的计算含义

两个头均为局部 Conv/GELU/Conv，两层 3×3、步长 1、原始层内 padding 不变。
每块输入 36×36、有效区最多 32×32，带两像素边界；两张 128×128 图共 32 次请求/头。
图像最右/最下块钳制原图中的窗口起点，再截取相应有效区，保留原逐层零填充。
没有先在整个输入外补零再执行两层：有 bias 时该做法会改变边缘结果。

`prepare_spatial_bundle.py` 校验 ONNX 除执行 batch/H/W 和中间 shape 信息外逐字节不变，
绑定原 checkpoint、完整 graph 和 oracle；检查两层卷积属性与半径，拒绝不匹配网络。
组包时用 ORT 对完整两视图及全部接缝、边缘比较：gaussians 最大绝对误差
`6.86646e-5`、RMSE `1.42099e-6`；disparity 分别 `4.29153e-6`、`3.75665e-7`。
这是 CPU 导出/拼回检查，与以下真实 NPU 精度结果分开。

完整 head 的板端 oracle 误差：to_gaussians 最大绝对误差 `0.139107`、RMSE `0.00786184`；
to_disparity 分别 `0.0272698`、`0.00455878`。两者严格逐元素门槛仍未通过。

## 固定已准备输入：完整网络前向

均为两上下文、128×128、32768 高斯、4 CPU 线程，同输入 CPU/NPU；不是单卷积耗时。

| 配置 | CPU 前向 | NPU 混合前向 / 第二次 | 验收 |
|---|---:|---:|---|
| 三子图：backbone + depth + gaussians | 16.737 s | 11.925 / 11.220 s | 图像、重复及输入切换通过 |
| 四子图：再加 disparity | 16.735 s | 11.771 / 11.180 s | 图像、重复及输入切换通过 |

四子图第一遍相对同轮 CPU 减少 4.965 s，约 29.66%（1.422 倍），不能把重复遍最佳值
冒充平均。每配置仅两次前向，没有足够样本作 P95 或微小差异显著性结论。
三、四子图首次差约 0.154 s；不足以证明 disparity 能稳定带来这一幅度收益。

四子图第一遍调用端计时：backbone 38.45 ms、depth 33.42 ms、gaussians 1022.50 ms、
disparity 208.70 ms，总计 1.3031 s，已包含 Python IPC、布局整理和两次 SDK 成本。
主机完整前向 11.771 s 还包含大量 CPU 工作，不能将 1.3031 s 称整个模型推理时间。

三子图 34 个 worker 请求/67 次 SDK forward，逻辑 IPC I/O 42.4512 MiB；
四子图 66 请求/131 次 SDK forward，逻辑 IPC I/O 47.8301 MiB。
边界重叠增加数据量；这些数字不是实测 DDR 流量，也没有包含所有 SDK 内部复制次数。

## 三视角画质

固定资格图像政策保持：NPU 对 CPU PSNR 至少 40 dB、SSIM 至少 0.99；相对真实图像的
PSNR 下降最多 0.2 dB、SSIM 下降最多 0.005。参考图像与目标相机哈希均绑定资格记录。

| 四子图目标帧 | NPU 对 CPU PSNR / SSIM | NPU 对留出实拍 PSNR / SSIM |
|---|---|---|
| 7 | 44.321 dB / 0.995568 | 20.097 dB / 0.801559 |
| 15 | 43.613 dB / 0.995337 | 20.245 dB / 0.834049 |
| 22 | 42.748 dB / 0.995606 | 24.188 dB / 0.844375 |

最差真实图像 PSNR 下降 0.005975 dB、SSIM 下降 0.000318。
三子图对 CPU 为 52.713～53.454 dB、SSIM 0.998769～0.998862，比四子图更接近 CPU。
因此 disparity 会增加数值差异；真实图像指标变化较小，不能据个别上升声称画质提高。
重建本身仍有模糊/遮挡问题，CPU 参考相似度很高不等于绝对重建质量很高。

## 视频首帧与网页

固定公开视频 `nyu_snippet_curl.mp4`，30 帧供几何准备，网络只用上下文 0/29，目标帧 7；
资格另用 15/22。不是 30 帧都进网络，也不是全段空间已完整覆盖。
预热先完成，再模拟接收已结束视频。CPU 推理 3 线程、准备 1 线程，保持原重叠调度。

四子图 `VIDEO_COMPLETE → FRAME_COMPLETE` **21.244528 s**，其中 Gaussian scene
21.106975 s、之后首帧 0.137553 s。完整前向 12.194532 s；上下文就绪 6.426109 s，
首目标位姿 7.658976 s，全部评测位姿 9.882548 s。解码 4.020284 s，目标 PnP 未删除。
预热 105.081077 s 只记录，不计系统场景准备性能；**20 秒尚未达到**。

同轮两子图控制在归档不用的候选 oracle 后通过：首帧 **22.802695 s**、前向 **14.348699 s**、
预热 79.262472 s；四子图相对其首帧减少 **1.558166 s（6.83%）**，前向减少
**2.154168 s（15.01%）**。均一次，且开始时 RAM 驻留文件不同，不作为稳定分布或微差显著性。
两条成功路径使用同一视频、调度、SDK 和 96 MiB 全局内存门槛，未改变目标 PnP。
两子图采样 PSS 峰值 609.188 MiB、最低 MemAvailable 113.293 MiB。
历史 22.393 s 仅作此前记录，不冒充本轮对照；网页结果见配套 JSON。
网页恢复后核验四子图来源标志、FPGA 后端、三个视角各两次、相同视角 hash 一致及不同视角差异。
HTTP 时间为电脑发请求至收到 PNG，不含浏览器实际显示；首帧与常驻换视角时间分开。
六次板端渲染 40.68～54.49 ms，平均 43.308 ms、中位 41.065 ms；HTTP 51.65～70.15 ms，
平均 55.791 ms、中位 52.327 ms。三种视角分别重复相同，三种之间图像不同；不是六张缓存同图。
这些是新四子图场景的本轮实测，不表示 FPGA 内核本轮又获得了加速。

## 资源与失败项

四子图视频采样 PSS 峰值 544.792 MiB、系统最低 MemAvailable 113.320 MiB；同一进程树
RSS 550.184 MiB。PSS 将共享映射按占用者分摊，RSS 会重复累加共享页。
原视频试跑因共享 RSS 累加超过 1 GiB 中止，不算完成；后改为显式 PSS 700 MiB 预算，
仍保留全局 96 MiB 可用内存门槛。两个两子图控制因全局余量不足中止，也不是有效时间成绩。
候选新文件放 tmpfs 会占 RAM；历史自有测试文件逐文件下载、SHA 校验后才回收。

FPGA 逻辑与位流未改：沿用历史资源 LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM 82.45%、
DSP 63%，200 MHz；本轮未重新综合/布线，既有时序限制不变，不据此宣称新的时序通过。
NPU 利用率百分比、真实 DDR 流量、功耗均未测。Linux RAM 和设备 DDR guard 分开统计。

板端根文件系统 100% 使用，新结果/候选暂存在 `/dev/shm/fpga43dgs_npu_head_20261102`，
重启会丢失；这限制持久部署，不影响已归档的测量真实性。未清理用户录像或 SD 系统资产。

## 复跑与回退

在 mvsplat 目录，先对原始七分区 bundle 导出两个局部头，再用既有编译器编译单视图块：

```powershell
python -m gaussian_generation.npu_branch.head_probe export-spatial --graphs BASE_GRAPHS --partition to_gaussians --tile-size 36 --out GAUSS_TILE_GRAPHS
python -m gaussian_generation.npu_branch.head_probe export-spatial --graphs BASE_GRAPHS --partition to_disparity --tile-size 36 --out DISP_TILE_GRAPHS
python -m gaussian_generation.npu_branch.compile_graphs --graphs GAUSS_TILE_GRAPHS --compiler ICRAFT_BIN --partitions to_gaussians_tile36_b1 --qdtype tf32 --out GAUSS_TILE_COMPILED
python -m gaussian_generation.npu_branch.compile_graphs --graphs DISP_TILE_GRAPHS --compiler ICRAFT_BIN --partitions to_disparity_tile36_b1 --qdtype tf32 --out DISP_TILE_COMPILED
python -m gaussian_generation.npu_branch.prepare_spatial_bundle --graphs BASE_GRAPHS --compiled BASE_COMPILED --head-graphs GAUSS_TILE_GRAPHS DISP_TILE_GRAPHS --head-compiled GAUSS_TILE_COMPILED DISP_TILE_COMPILED --partitions backbone_cnn depth_head to_gaussians to_disparity --core-size 32 --out FOUR_PART_BUNDLE
```

占位路径需替换为本地目录；实际导出 case 名须以各 graph manifest 为准，不能猜测。
组包只证明形状/权重/放置/参考拼回，不代替板端 `qualify.py`。先使用固定权重、SDK 与
BOOT 跑三目标原生资格，得到四子图 receipt，然后在 initialize 原入口显式指定：

```text
--backend npu --partitions backbone_cnn depth_head to_gaussians to_disparity
--partition-bundle FOUR_PART_BUNDLE/compiled --oracle-graphs FOUR_PART_BUNDLE/graphs
--npu-acceptance CURRENT_BOARD_ACCEPTANCE.json --npu-library CURRENT_BRIDGE.so
```

fresh input 由 tiled runtime 自动要求；显式关闭会拒绝运行。head 两次执行绑定身份，
旧两子图 receipt 不能当作四子图资格使用。打包纯 relocation 保持 manifest 身份，
子图集合/顺序或内容改变则重新绑定并资格验收。

已组包和 preflight 核对的四子图 compiled manifest SHA256 为
`749f59b1b13cffc7220e07b1f17bcddc86920d034fd8f4bc14aeffe12bfa49f0`，
graphs manifest 为 `102796c85b107ed3e99eaf57dee3a2e75da4a2ed887f37561f2e0f2e71626c59`。
板端桥接库 SHA256 `3d5c11b4249e142c8a816eb89c7fdf1bb501e4b18bdea4d51578a8055ccd6ffe`。
preflight 校验 26 个文件、共享 IPC arena 2 MiB；它本身不执行 NPU，不代替资格记录。

回退保留原 `backbone_cnn depth_head` 与对应 bundle/receipt，或 `--backend cpu`。
不改所有入口的默认配置；新四子图候选需明确选择，不能把未经资格的七子图全开。

## 证据与后续

代码分支 `feat/npu-gaussian-head-20261102`，基点 `dfb73a1`。
电脑原始证据在 `D:/ADProjects/FPGA for 3DGS/outputs/npu_head_reliability_20261102/`：
`three_part_gaussians_clean/`、`four_part_final/`、`video_four_part_pss/`、各失败控制、
`transport.log`、独立卷积日志、归档 SHA 清单与 `reproducible_candidate/`。
无私人视频，权重/SDK/大型输出不提交 Git。报告配套 JSON 保存完整数字、资产哈希及网页结果。
原生三、四子图资格的 Gaussian NPZ 及视频全部文件已下载并逐文件 SHA 校验；四子图视频和
同轮两子图控制共归档 328 文件、47156018 字节。为两子图控制临时移走的候选 oracle 已恢复并
复核哈希。原两子图最近有效场景仍在，旧测试副本归档后仅回收本轮自有 tmpfs 文件。
软件回归共 182 项：178 通过、4 跳过（本地缺 FFmpeg 的 3 项及 Windows 创建符号链接的 1 项）；
真实板端视频另已执行成功。132 Python 文件语法通过；冻结 render_main 151 文件和
render_branch 197 文件哈希、95 个默认隔离源检查通过。`git diff --check` 通过。

下一步优先解决 SDK 状态/权重预检根因，再评估去掉重复执行和合并局部边界；未经 A/B 切换
与原生图像检查，不删除两次执行。较大剩余瓶颈仍是 CPU 细化/注意力/重采样与视频解码，
而不是 NPU head 核心运算。完整前向和首帧均需重新测，不能只优化微基准宣称系统提速。
