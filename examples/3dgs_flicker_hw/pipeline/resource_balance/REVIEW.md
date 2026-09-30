# 四路换视角渲染：资源重分配与物理实现审查

目标是固定场景已加载后的相机请求到完整 RGB 帧返回。保留四路 evaluator、
200 MHz、FP16 算术和像素深度顺序，允许充分使用 BRAM/DSP。起初板卡不可用，
用户恢复网线后 SSH 已确认，已补做本轮原位流的 60 帧基线。
本目录记录审查与物理对照；最终实板验收见 [BOARD_VALIDATION.md](BOARD_VALIDATION.md)。
已安装共享参数四路版本：完整点集 120 帧均值 43.786 ms，硬件周期减少
8.68%，输出逐位一致；16k 有损预览 60 帧均值 27.337 ms。

## 审查入口和工具边界

使用 readable-verilog-generator 的 dependency check 与只读 review，检查
`rtl/platform_adder_top.v` 和 HLS 的 `render_lane_0u_s_Ffa.v` 参数缓存。
该技能 formatter 对前者的组合端口声明、后者的生成式声明解析失败。
八项公开门禁：compile、ast、readability、comment、naming、profile 为
failed；testbench、toolchain 为 not_requested。这是技能报告的状态，
不是 Vivado 编译失败，也不能据此声称风格检查通过。报告保存在
`build/resource_review/`。技能缺少 erie-remote-ssh 和 WaveDrom 依赖，
只使用本地静态入口；外部验证由 xilinx-suite 指导的项目既有 Vivado
2018.3/xsim 流程承担。未安装或调用远程依赖，未手工重写 HLS 生成 RTL。

## 修改前的结构审查

| 项目 | 证据和约束 |
|---|---|
| 接口 | 外部 512-bit 原厂 DDR 接口、FLK1 寄存器和 128-bit 像素结构保持；只改 dispatch 到 lane 的内部 FIFO |
| 复位 | GP/HP 两时钟域同步由现有 regs 实现；保留 DMA、FIFO 指针和有效标志复位，不直接删 HLS 控制寄存器复位 |
| 流水 | 四路 evaluate_group 保持 II=1；每个高斯完成后才更新同一像素的下一贡献，不加 DEPENDENCE 豁免 |
| 握手 | 继续使用阻塞 FIFO 读写；header/end/frame-end 按同一 FIFO 广播，DDR 背压传播不变 |
| 位宽 | render_lane 仅读 [223:0] 参数/背景/ordinal、[229:228] 子块、[511:510] kind；内部压缩为低 230 位加 2 位 kind |
| 存储 | 对照紧凑缓存 LUTRAM；新增 232-bit SRL 队列对照和 232-bit BRAM 队列候选；像素反馈 BRAM 不分区、不改端口 |
| 验证 | 36 Tile 六模式 C 参考；HLS C/RTL；真实 DMA/CDC 六模式，含背压、哨兵、完成所有权、非法模式 |
| 风险 | BRAM 增加与同步读延迟、物理列位置、控制信号扇出可能抵消 LUT 节省；必须看全板布局/布线和原时钟时序 |

## 资源选择依据

已布通旧 ROM 设计 Slice 19484/19650、LUT 60398/78600、FF 97523/157200、
BRAM36 等效 226.5/265、DSP 252/400。寄存器分组版本布局失败，剩余实例
需要 14534 Slice、可用 13805，控制集 2624。不能把空闲 FF 当可用 Slice。

紧凑缓存版本的 HLS 顶层队列估计消耗 3818 LUT，其中四路 512-bit 待处理
队列各 526 LUT。优先缩窄队列，再把这四条队列映射至 BRAM，避免将全部
512-bit FIFO 无区别搬入 BRAM。`FLK_LANE_FIFO_STORAGE`：0 是旧 512-bit
SRL；1 是 232-bit SRL；2 是 232-bit BRAM。默认 0 保持既有路径。

缓存报告中 `records_V_U` 的 448/14 列分别是 **HLS FF/LUT 估计**，不能
颠倒；真实 RTL 含 ram_style=distributed，最终资源以 Vivado 映射为准。
总资源中原厂后插入网表必须计入，不能拿前插入综合报告替代整板占用。

## 物理优化顺序

1. 完成紧凑缓存版本全板实现，作为真实资源映射对照。
2. 验证两个缩窄队列候选，仅对计算流水和输出通过者准备独立平台。
3. 按全板 Slice/BRAM、控制集、setup/hold 与拥塞选择映射。
4. 以固定原始时钟对合格候选进行物理优化；仅保留实际改善的独立 checkpoint。

不移动原厂保护区域，不新增时序豁免，不把未布线频率估计当可运行频率。
复现入口和实板指标、回退方式已在 BOARD_VALIDATION.md 补齐。

## 已完成的候选验证

| 候选 | BRAM18 (HLS) | DSP (HLS) | FF (HLS) | LUT (HLS) | mode 2 RTL 周期 |
|---|---:|---:|---:|---:|---:|
| 先前紧凑缓存 | 203 | 244 | 70003 | 37907 | 3414 |
| 232-bit SRL 队列 | 203 | 244 | 68603 | 36787 | 3414 |
| 232-bit BRAM 队列 | 255 | 244 | 69555 | 36627 | 3413 |
| 公共参数循环外复用 + 232-bit SRL | 203 | 244 | 65163 | 33947 | 3409 |

后面三组本轮均通过：HLS C/RTL、36 Tile 六模式 C 模型参考（55296 像素
记录）、三 Tile 六模式实际 DMA/CDC（4608 像素），输出逐字节一致。
每路 evaluator II=1。公共参数版本每次调用 135–183 拍，比紧凑缓存多一拍
公共参数读取；三 Tile mode 0/1 比紧凑缓存略慢，mode 2 略快，不能只选快的
模式概括所有负载。逐模式结果在各自 `result.json`。

复现主候选：

```powershell
& examples/3dgs_flicker_hw/pipeline/lane_workset/build_variant.ps1 `
  -Variant grouped-shared -Project hls_pipeline_groupshared_new
python examples/3dgs_flicker_hw/pipeline/run_integration.py `
  --project hls_pipeline_groupshared_new --profile-split
```

已验收的平台为 `platform/flicker_groupshared_fpga`。独立紧凑缓存对照为
`platform/flicker_grouplutram_fpga`。后者另从相同 post-opt checkpoint
尝试 `AltSpreadLogic_high` 放置指令，与原 `ExtraNetDelay_high` 比较；不改时钟、
器件、原厂保护约束或数据通路。`physical_opt/full_place.tcl` 新增可选第四个
参数选择这项对照，默认行为不变，输出目录必须尚不存在。

## 继续审查发现

失败版本 post-opt checkpoint 的实际四条输入 FIFO 各为 233 LUT，其中
222 SRL；Vivado 已经裁掉了许多无用位。HLS 报告中 512→232 位所减少的
1120 LUT 不能直接当全板节省。该 checkpoint 没有 pblock，固定位置原语
747 个；不能把旧工具包中仅适用于另一型号的 pblock 范围认作本机保留区。
post-opt 控制集 2024 与失败布局日志的 2624 属于不同优化阶段，分别记录。

新增 `grouped-shared`：同一 Gaussian ordinal 的参数和颜色在四个子块中
相同，先从第一个有效子块取出，作为整个像素循环的不变量。origin/extent
仍逐子块取；可以处理第零子块无效的情况。避免让相同颜色/透明度/ordinal
反复经过逐像素流水，不改 FP16 运算或裁剪顺序。`FLK_GROUP_SHARED_ATTR=1`
显式开启，默认 0；与 232-bit SRL 队列配合。HLS 综合估算 FF=65163、
LUT=33947、BRAM18=203、DSP=244，四路 II=1；已完成 C/RTL、物理和实板验收。

`grouped-bram-fifo` 的实际 HLS 映射为每队列 13 个 BRAM18，共 52 个，
并非最初按 36-bit 端口估计的 28 个。相对 packed SRL 只减少 160 LUT、
反而增加 952 FF，因此优先尝试循环外公共参数版本，保留 BRAM 实验结果。

恢复后的现位流 BOOT 哈希与历史 ROM 一致：
`0528e019a5fb19c7f281b773ddf5ab5a8a524ddfebdc4fc9acf74273db31d048`。
相同 native binary、32768 点、128×128、三视角两轮 60 帧，平均 46.930 ms，
P95 48.028 ms，FPGA engine 26.682 ms、硬件周期 5215402。与历史输出完全
一致。CPU 分组/散布比历史慢，故硬件前后对照采用本轮基线，不选历史更快值。

## 全板物理阶段对照

下面是放置后、尚未布线的结果，**不是可安装位流的时序结论**。同器件、
同 200 MHz 目标、四路计算，不增加时序豁免。

| 候选/放置策略 | Slice | LUT | FF | 控制集 | 估计 WNS (ns) |
|---|---:|---:|---:|---:|---:|
| 紧凑缓存 / ExtraNetDelay_high | 19650 | 62697 | 99903 | 2762 | -1.315 |
| 紧凑缓存 / AltSpreadLogic_high | 19644 | 62655 | 99932 | 2736 | -0.430 |
| 公共参数复用 / ExtraNetDelay_high | 19617 | 61199 | 98364 | 2656 | -0.200 |
| 公共参数复用 / AltSpreadLogic_high | 19496 | 61211 | 98346 | 2789 | -0.236 |

四者 BRAM36 等效均为 218.5、DSP 均为 252。公共参数版本比紧凑缓存的
同策略放置减少 1498 LUT、1539 FF、106 控制集，但仅释放 33 Slice。
这说明约束、控制集和可用布线共同限制布局，LUT/FF 的百分比不能代替
Slice 可容纳性。新版本仍需布线；紧凑缓存的路由器已经报告拥塞。
公共参数版本的 AltSpreadLogic_high 又减少 121 Slice；保持其独立 checkpoint，
用 `route_placed.tcl` 在原 NoTimingRelaxation 指令下物理优化并布线，不覆盖
平台默认路径。放置 WNS 没有比原策略更好，最终仍由布线后的结果决定。

原始报告在 `results/20260930/*_placed/`；同阶段 post-opt 审计则显示
公共参数版本比紧凑缓存减少 1626 LUT、1560 FF。两个阶段的计数不可混用。
