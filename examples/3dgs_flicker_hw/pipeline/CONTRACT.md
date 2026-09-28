# FLICKER CTU 与渲染集成合同

2026-09-27：FLK1 lean 已通过各自 FP16 契约下的仿真、物理实现及实板验收，详见 [VALIDATION.md](VALIDATION.md)。官方 FP32 严格精度失败和整板原厂时序限制仍保留。本合同描述已安装 ABI；后续优化必须保留本版本和 Golden，不能将近似契约通过改写成官方严格通过。

本目录保留 FLK0 基线，独立建立论文 §IV-B/C 的层次结构。输入仍为已投影、已排序的 16×16 Tile 列表；下一阶段的 3D 预处理和排序尚未迁入 FPGA。

- 一条 64 B Gaussian 改为九个连续 IEEE FP32：mean_x, mean_y, conic_a, conic_b, conic_c, opacity, R, G, B；头和输出与 FLK0 相同。FPGA 内转换为 FP16，避免 CPU 逐帧做半精度转换。该输入 ABI 不兼容 FLK0，必须用新 capability 标识，不能给旧内核送新输入。
- 上游硬件从 FP32 conic 恢复二维协方差，采用 vanilla 3DGS 的 `ceil(3*sqrt(lambda_max))` 半径做 8×8 AABB；分类用长短轴比 3。这里的 AABB 数值细节是明确记录的独立实现，论文未公开作者 RTL。
- 一条 Gaussian 在片上最多展开为四个 8×8 sub-tile。每个 CTU 请求计算两个共享矩形，再由 mask 分发到四个 mini-tile FIFO。当前容量候选使用 II=8 的跨周期算术共享，不等于论文两套 PRTU 每拍处理两个矩形；该 FPGA 资源缩放需实际面积、时序和完整帧时延验收。被拒绝的 mini-tile 不向 VRU 发送任务，实际缩短循环。
- 四个 VRU 通道各保存四个 sub-tile 的对应 4×4 mini-tile 状态。一条任务计算 16 像素，保持每个像素的原深度顺序。此处按 FPGA 资源缩放为四个 VRU，不等于作者 ASIC 的 32 VRU。
- 模式 0：AABB/CAT 均关闭；1：仅 AABB；2：AABB+Dense；3：AABB+Sparse；4：AABB+Smooth-Focused；5：AABB+Spiky-Focused。同一硬件对照模式 0/1/2，独立计量并行 VRU、AABB、CAT 的收益。
- CTU 仍使用已验证的全 FP16 分支，FP8 后续独立消融。端点 markers 通过同一有序背压通道，不能绕过尚未完成的贡献。

验收先要求 mode 0 对 FLK0 HLS golden 逐位一致，再比较 AABB/CAT 的独立软件掩码和完整画质；仿真、布线、上板三类结果分别登记。不能把本目录存在源文件写成已经部署。
