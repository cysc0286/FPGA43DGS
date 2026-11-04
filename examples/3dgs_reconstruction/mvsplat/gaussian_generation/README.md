# 高斯生成

runtime 常驻固定 MVSplat；upstream 加载原始网络；memory 管理已有代价体调度；adapter 校验并转换 Gaussian／相机 ABI。
npu_branch 包含正式 ICraft 运行、图构建、资格和定位数值问题所需工具，测试集中于其 tests。
主线为四个既有局部子图加六个 Transformer FFN；其余计算仍在 CPU，CPU 显式回退继续保留。
拒绝的 corr/refine 等候选仅供相关诊断，不是默认已合格实现；严格 TF32 失败事实不因整理消失。
详细分工与数据接口见根 docs/CODE_STRUCTURE.md。历史 Windows 参考锁文件只作记录，不用于板端安装。
