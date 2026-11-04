# 六部分代码与数据接口

更新：2026-11-04。代码根为 `examples/3dgs_reconstruction/mvsplat/`。
六部分是原五部分加独立网页；`common.py` 只放共用哈希、有限值和 Gaussian 文件校验，不另建业务模块。

```text
examples/3dgs_reconstruction/
├── pipeline.py                         统一 CLI：initialize / web / view
└── mvsplat/
    ├── common.py                       共用内容身份与数值校验
    ├── initialize/
    │   ├── run.py / session.py          入口与视频前 READY
    │   ├── scene_preparation.py         调用视频和几何，发布上下文／目标位姿
    │   ├── pipeline.py / ring.py        视频后推理、交接和首帧协调
    │   ├── events.py / measure.py       事件计时、内存保护
    │   └── board_sync.py / transport.py 部署及 SSH 传输
    ├── video_input/
    │   ├── capture.py / recording.py    摄像头录像、同源预览和封装
    │   ├── upload.py / receipt.py       已完成文件接收、来源身份
    │   ├── decode.py / calibration.py   解码、裁剪变换和相机参数
    │   └── ring.py / streaming.py       已有有界环视／连续输入内部实现
    ├── pose_estimation/
    │   ├── geometry.py                 两视图几何及目标 PnP
    │   └── rotation*.py / streaming.py 固定中心旋转估计和局部跟踪
    ├── gaussian_generation/
    │   ├── runtime.py / upstream.py    常驻 MVSplat 与原网络加载
    │   ├── memory.py                   代价体与中间内存调度
    │   ├── adapter.py                  Gaussian／相机到渲染 ABI
    │   └── npu_branch/                 NPU 编译、运行、资格、相关诊断及测试
    ├── rendering/
    │   ├── runtime.py / pipeline_adapter.py  原生常驻渲染与首帧交接
    │   ├── camera.py / contracts.py    相机运算与 PLY／帧缓冲合同
    │   ├── live_renderer.cpp           常驻场景、投影、排序和 FPGA 提交
    │   ├── resident_runtime.py / *_resident.cpp  保留的渲染回退
    │   └── render_main/                原四路冻结代码、硬件、资源与实板证据
    └── web/
        ├── camera_ui.py               页面到初始化／重建进程的生命周期桥接
        ├── recording_server.py        /capture、/upload、状态与预览 HTTP
        ├── viewer.py                  相机请求到渲染结果的 HTTP
        └── *.html / viewer.js / viewer.css  页面及浏览器交互
```

## 六部分怎样交接

1. **预热 → 输入**：`WarmSession` 先打开渲染器，再建立 NPU worker 和模型。资格检查完成才发布 `READY`；先打开共享设备的顺序必须保留。
2. **输入 → 位姿**：已封装视频及来源 receipt；解码后保留原帧索引、尺寸变换和内参。USB 标定与上传视频的估算内参分别处理。
3. **位姿 → 生成**：带哈希的上下文图像，以及归一化内参和 `c2w`。上下文就绪与首目标位姿分别发布；不得删去新视角估计来缩短首帧。
4. **生成 → 渲染**：世界坐标高斯的 means、covariances、harmonics、opacities，经过有限值／形状／内容身份检查。`adapter.gaussian_rows` 转成连续 `N×62` float32，场景只装载一次；相机为 `FLCAM001`、136 字节。
5. **渲染 → 网页**：`LiveFrame` 携带完整 RGB 和本次板端耗时。网页返回 PNG、请求编号与场景版本；相机数学留在 rendering，不依赖 HTTP。
6. **网页 → 渲染**：页面形成视角控制请求，经校验后交给同一个常驻渲染所有者；不为每次换视角重新跑视频、位姿或 NPU。

关键路径保持内存交接；PLY、NPZ 和审计资料继续按现有流程保留，不借整理改变数值或质量门槛。
`rendering/contracts.py` 保留文件交换时的 PLY、相机和帧缓冲验证。

## 计时合同

`READY → VIDEO_COMPLETE → SCENE_READY → FRAME_COMPLETE`。
预热及视频输入只记录；场景准备为 `VIDEO_COMPLETE` 到第一张目标视角完整帧。
之后换视角单列板端相机到 RGB 的耗时；浏览器“请求到画面就绪”不等于物理显示延迟。
恢复历史场景不能发布新一次零秒重建成绩。

## 目录纪律

实现、测试、构建脚本随所属模块放置。测试直接调用真实接口；合成 HTTP／浏览器测试不冒充实板。
`npu_branch` 名字沿用，但包含正式后端；其中连续 U-Net 等失败候选只用于资格诊断，显式开启且仍受原门槛，不是默认已采用方案。
冻结 `render_main` 全部字节保持不变。旧独立候选、归档 ZIP、旧冷训练链和别名入口已从当前树删除。
整理前已有未提交改动已在仓库外逐文件备份；私人数据和其他工作树未动。
