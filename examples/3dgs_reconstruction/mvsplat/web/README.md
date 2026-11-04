# 网页与 HTTP

统一入口：上一级 `pipeline.py web --help`；模块入口为 `python -m web.camera_ui`。
`recording_server.py` 服务 `/capture` 和 `/upload`，分别调用 video_input 的录像与文件接收；两端共用一个初始化／重建所有者。
`camera_ui.py` 管理页面状态及重建进程，实际预热／推理由 initialize 和 gaussian_generation 执行。
`viewer.py` 服务常驻场景视角请求，静态页面是 viewer.html、viewer.css、viewer.js。
相机数学归 rendering/camera.py；网页不实现另一份投影、排序或生成算法。
确认、拖动及已有自动更新行为沿用原页面；请求按现有合并规则送给同一渲染器。
输入令牌、上传数据、浏览器测试输出和历史场景永不提交。合成浏览器测试不代表 NPU 或 FPGA 实测。
