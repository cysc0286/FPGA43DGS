# 高斯生成

输入 PoseSet；training.py 执行有资源控制的 OpenSplat CPU 优化；export(a) 导出 model.ply 与相机。独立命令 pipeline.py gaussian / export。导出是此模块的适配步骤。

共用字段与运行命令见 [接口文档](../INTERFACES.md)。

反向传播属于本模块内部的训练循环：OpenSplat CPU 正向渲染、图像损失、`mainLoss.backward()`、参数更新。当前冻结 FPGA 渲染包只做正向渲染，没有接入该训练循环。源码位置、可复用的 MVSplat 候选及预训练推理与训练的区别见 [仓库与反向传播说明](../../REPOSITORIES_AND_BACKPROP.md)。
