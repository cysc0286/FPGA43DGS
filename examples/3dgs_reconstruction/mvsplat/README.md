# MVSplat 板端主线

六部分分别为 initialize、video_input、pose_estimation、gaussian_generation、rendering、web。
从上一级 `pipeline.py` 启动；模块内部工具从本目录使用 `python -m 所属模块.工具`。
结构、接口和 CLI 说明统一见 [CODE_STRUCTURE.md](../../../docs/CODE_STRUCTURE.md)，当前实测见 [MAINLINE.md](../../../MAINLINE.md)。
历史验证记录分别归 initialize/docs/history 与 gaussian_generation/npu_branch/docs/history，不是当前操作手册。
固定权重推理、CPU 回退、NPU 资格、内存和设备隔离规则继续执行；冻结 render_main 不改。
