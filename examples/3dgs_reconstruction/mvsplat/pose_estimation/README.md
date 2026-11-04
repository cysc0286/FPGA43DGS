# 位姿估计

geometry 完成特征、匹配、两视图几何、三角化及目标 PnP，当前由板端 CPU 执行。
rotation 和 rotation_graph 服务固定中心旋转候选，不虚构平移基线；streaming 是现有局部跟踪接口。
输入来自 video_input；输出是带来源哈希、内参、c2w 的上下文与目标位姿，交给 initialize 协调和生成模块。
首目标位姿不能为追求速度被省略；纯旋转模式的深度／覆盖局限仍必须在元数据中说明。
