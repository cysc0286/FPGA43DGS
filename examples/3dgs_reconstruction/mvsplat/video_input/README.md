# 视频输入

只负责视频数据，不承载网页：capture／recording 接收 UVC；upload 接收文件；receipt 记录完整性和来源。
decode 保留原帧索引并进行等比处理；calibration 管理内参和裁剪变换；ring／streaming 是现有有界采样实现。
固定公开视频获取工具为 `python -m video_input.fetch_video`，校验原 SHA256 后写入被忽略的 reconstruction/data。
HTTP 和页面已移至 web；USB 标定不套用到上传片段，未标定内参必须显式标明估计值。
真实摄像头、环拍覆盖等验收范围见各日期验证记录，不由本机接口测试替代。
