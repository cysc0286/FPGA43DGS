# 预热与生命周期

`run.py` 是统一 initialize 入口；`session.py` 在视频前打开渲染设备、NPU worker 和常驻模型，检查通过后才发布 READY。
`scene_preparation.py` 只调用输入和几何模块；`pipeline.py` 协调视频后的推理、内存交接、场景装载和首帧。
`events.py` 明确四事件；`measure.py` 监测进程树／PSS／系统余量；`transport.py` 和 `board_sync.py` 管理显式部署。
网页生命周期入口已移至 `web/camera_ui.py`；渲染 C++、回退和验证工具归 rendering。
预热不读测试视频；视频后计算不能冒称预热。新部署必须重新核验 NPU 身份，不能复用旧源码 receipt。
相关历史记录在 docs/history，本次结构与接口以根 docs/CODE_STRUCTURE.md 为准。
