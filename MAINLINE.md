# 当前主线和验收边界

更新：2026-11-02。架构见 [代码结构](docs/CODE_STRUCTURE.md)，
数据交接见 [四模块接口](examples/3dgs_reconstruction/modules/INTERFACES.md)。

## 推荐生成与渲染配置

- 生成：显式 --backend npu --partitions backbone_cnn depth_head，使用 ICraft 3.36.1；
  其余网络、注意力、代价体及位姿仍在 CPU，并非全网 NPU。
- NPU worker 默认预留 64 MiB PL DDR 工作区，并为 backbone 请求执行两次 SDK forward，
  丢弃首份输出、计入两次成本，避免已发现的上一输入残留；SDK 根因尚未关闭。
- 就绪：交错 oracle、B,B,A,A 输入切换、输出有限性及实际运行身份检查；
  TF32 严格 tensor 门槛不通过时，须使用同资产原生 FPGA 三留出视角生成的图像 receipt。
  不允许旧 receipt 绕过输入切换，重新编译桥接库或改变策略须重新验收。
- 渲染：rendering/mainline.json、rendering/render_main/，常驻 C++ 投影/稳定深度基数排序，
  四路 200 MHz、Dense mode 2、精确指数 ROM、有序合成、不额外抽点。
- render_branch/ 保留 PipeGS HGR v4 候选，默认不调用；CPU 网络保留显式对照与回退。

NPU 接入、共存保护和网页源码归属原模块，未引入第五个业务模块。
initialize 只负责视频前预热和生命周期；scene_preparation.py / warm_pipeline.py 协调视频后计算。
通用 CLI 默认 cpu 仅为兼容旧调用；推荐 NPU 命令见根 README，不会在 NPU 失败后静默回退。

## 时间与画质

预热及视频输入只记录。场景准备从 VIDEO_COMPLETE 到首个留出视角 FRAME_COMPLETE；
换视角从已有场景的相机提交到完整 RGB 返回，浏览器封装/网络/绘制另计。

2026-11-02 同一公开视频、128×128、32768 高斯，推理 3 线程、准备 1 线程重叠：

| 配置 | 场景准备 | 网络前向 | 进程树峰值 RSS |
|---|---:|---:|---:|
| CPU 对照 | 24.512 s | 16.414 s | 604.14 MiB |
| NPU 双子图＋backbone 两次执行 | 24.353 s | 16.019 s | 591.75 MiB |
| 同上＋完整合成前向预热候选 | 23.933 s | 14.508 s | 584.36 MiB |

各一轮，没有统计稳定性结论，不能称稳定加速；20 s 目标未达到。
额外预热候选将视频前预热增至 72.964 s，默认不启用 --warm-kernels。
NPU 无/有额外预热的 Gaussian 哈希一致。
修正候选首帧对 CPU 为 55.332 dB / 0.999232；
对真实留出图为 19.450 dB / 0.804061。严格 tensor 门槛仍失败，
图像策略仅证明固定测试场景相对 CPU 变化很小，不代表所有场景质量。

此前场景六次原生换视角 40.63–45.95 ms，浏览器 57.5–67.2 ms。
修正双子图全视频场景的网页恢复和最后原始证据下载因断网待补；
不能将前述网页时间标作修正新场景已复测。
LPIPS、真实 DDR 流量、功耗及能量/帧未测。
[详细实板结果、负结果与证据边界](examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch/NPU_DEPTH_VALIDATION_20261102.md)。

## 冻结硬件与历史对照

冻结 BOOT SHA256：
b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6。
全板历史资源：LUT 77.94%、FF 62.58%、Slice 99.84%、BRAM36 等效 82.45%、DSP 63.00%。
既有 AI 脉宽及 GT/CDC 限制仍在，不能称全板无条件时序签核；本次未重新综合或布线。

2026-10-08：固定 32768 高斯、SH3、128×128、相机 7/15/22，60 帧，
四路平均 43.575 ms、中位数 42.183 ms、P95/P99 52.893/62.080 ms；
HGR v4 两路 53.302 ms，因此继续保留四路主线。
这些与上述整链、网页测量分别记录，不能混为同一次实验。
[冻结包](examples/3dgs_reconstruction/mvsplat/rendering/render_main/README.md)；
[同轮候选比较](examples/3dgs_reconstruction/mvsplat/rendering/render_branch/validation/RESULTS.md)；
[历史索引](archive/history_20261009/index.json)。

## 当前合并的范围

合入已实现的 NPU 入口、验收、共存保护、输入切换规避、网页和已存在相机功能。
保留 CPU 回退、四模块接口与所有冻结渲染资产；私人视频与生成结果不进入提交。
本次只做软件回归、源码包完整性和冻结包核验，未新增板端性能、质量或时序结果。

## 主线清理

2026-11-02 从活动树删除 250 个旧实验文件，257 项历史/入口快照已逐项归档校验。
核心四模块、CPU 对照、NPU 正式验收与冻结渲染包保持。
旧训练/导出公共命令拒绝调用；实际依赖的 SfM、合同、SSH 工具保留。
这不是新的硬件或速度优化，测量数据仍按前述范围阅读。
[具体删除范围与核验](docs/MAINLINE_CLEANUP_20261102.md)。
