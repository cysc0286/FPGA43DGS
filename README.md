# HeteroGS：板端视频重建与 NPU＋FPGA 渲染

推荐主线：**视频输入 → CPU 位姿 → MVSplat（NPU backbone＋depth_head，CPU 其余网络）→ CPU＋FPGA 渲染 → 网页交互**。
平台为悟净 30TAI Lite；使用固定预训练权重，无须为每段视频重新训练。
NPU 生成高斯后，换视角直接调用常驻渲染器，不重复网络推理。

先读 [项目要求](docs/PROJECT_REQUIREMENTS.md)、[代码结构](docs/CODE_STRUCTURE.md)
和 [四模块接口](examples/3dgs_reconstruction/modules/INTERFACES.md)。
当前配置及测量边界见 [MAINLINE.md](MAINLINE.md)；
部署、验收与网页操作见 [联调说明](examples/3dgs_reconstruction/mvsplat/FINAL_INTEGRATION_20261102.md)。

~~~text
examples/3dgs_reconstruction/
├── pipeline.py                  唯一整链 CLI：initialize / reconstruct
├── mvsplat/
│   ├── initialize/              视频前预热、生命周期和部署协调
│   ├── video_input/             文件/摄像头输入、解码、选帧、标定
│   ├── pose_estimation/         两视图几何、目标 PnP
│   ├── gaussian_generation/     常驻 MVSplat、Gaussian → 渲染 ABI
│   │   └── npu_branch/          推荐 NPU 后端、编译、验收与诊断
│   ├── rendering/               常驻 CPU＋FPGA 后端及网页
│   │   ├── render_main/         四路主线冻结包，保留回退
│   │   └── render_branch/       PipeGS HGR v4 独立候选，默认不调用
│   ├── scene_preparation.py     视频与位姿交接协调
│   └── warm_pipeline.py         高斯生成与首帧协调
└── modules/                     完整 SfM 冷对照依赖与共用合同
archive/history_20261009/         历史实验及哈希索引
releases/                        较早冻结后端回退包
~~~

四个业务模块和数据接口保持原结构；新增独立模块须先取得用户明确同意。
目录名称 npu_branch 沿用既有架构，不表示它仍与正式入口脱离。

## 运行推荐主线

先在板端按联调说明编译桥接库，针对实际权重、图、SDK、渲染器和运行策略完成
qualify，取得匹配的 acceptance.json。从重建目录运行：

~~~sh
python pipeline.py initialize --video INPUT.mp4 --out NEW_RUN \
  --weights weights/re10k.ckpt --vendor vendor/MVSplat_reference \
  --renderer /path/to/3dgs_renderer_v1_20260928 \
  --live-renderer /path/to/accepted/live_exact --render-profile mainline \
  --backend npu --partitions backbone_cnn depth_head \
  --partition-bundle /path/to/compiled --oracle-graphs /path/to/graphs \
  --npu-library /path/to/libmgs_npu.so --worker-python /path/to/arm/python \
  --npu-acceptance /path/to/qualification/acceptance.json \
  --threads 3 --prepare-threads 1 --overlap-prepare --interactive
~~~

上述依赖、权重和硬件必须自行配置，源码提交不会安装 SDK 或烧录固件。
NPU worker 默认启用已测 backbone 两次 SDK 执行规避策略，累计全部成本；
启动检查必须通过输入切换和当前资产绑定，旧单次执行 receipt 不能复用。
显式关闭策略仅用于诊断，不能跳过 READY 检查。

CPU 对照/回退用 --backend cpu，并去掉所有 NPU 专用参数。
通用 CLI 省略后端时仍默认为 CPU，以兼容旧命令；**推荐 NPU 主线必须显式使用上面的参数**。
NPU 错误会报错，不会偷偷换 CPU 再标作 NPU 成绩。
左侧网页选择视角，点击确认后才向板卡提交；右侧显示返回画面。

## 已有实测范围

2026-11-02，128×128、32768 高斯、固定公开视频，同调度 CPU 首帧 **24.512 s**，
修正后的双子图 NPU **24.353 s**；额外完整前向预热候选 **23.933 s**。
各一次，不足以证明稳定加速，20 s 目标尚未达到；预热只记录，不计入上述时间。
修正候选首帧相对 CPU：PSNR **55.332 dB**、SSIM **0.999232**；
相对真实留出图：**19.450 dB / 0.804061**，重建质量仍有提升空间。
TF32 严格逐元素门槛仍失败，当前采用资产绑定的原生 FPGA 三视角图像验收策略。

已加载场景的六次网页换视角，在此前场景实测板端 **40.63–45.95 ms**，
浏览器请求到绘制 **57.5–67.2 ms**；修正后的新场景网页恢复因断网待补。
这些是历史实板记录，本次合并未新增板测。
详见 [NPU 实板增量](examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch/NPU_DEPTH_VALIDATION_20261102.md)。

冻结 render_main 和 BOOT 保持原样；全网 NPU、长期连续 NPU 更新及新场景泛化仍未验收。
私人视频、原始帧、权重、SDK 和运行结果不上传。

## 旧实验与重复入口

OpenSplat 训练、旧特征 NPU、独立诊断包和早期渲染实验已从活动树删除。
公共 CLI 不再提供 gaussian/export 旧训练与导出命令；完整 SfM CPU 对照仍保留。
需要历史源码时从 [清理归档](archive/history_20261102_cleanup/README.md) 恢复到新目录。
该清理只减少维护和误调用风险，不构成新的推理或渲染加速。
