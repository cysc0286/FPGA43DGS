# 换视角渲染：像素工作集候选

## 最新候选：紧凑记录与分布式 RAM

针对寄存器版的全板布局失败，新增 `grouped-lutram`：只缓存 evaluator
实际读取的 224 位（0–223 位），用分布式 RAM 保存四条记录，保留尾部
空块裁剪。外部 512-bit ABI 不变，控制标签在缓存前已处理，不丢弃任何
渲染参数。此处调整存储是为了使更快的计算路径装得下，不是预留前端空间。

| 候选 | mode 2 RTL 周期 | HLS FF | HLS LUT | HLS BRAM_18K | HLS DSP |
|---|---:|---:|---:|---:|---:|
| 原已安装 RTL 对照 | 4861 | 64227 | 34927 | 219 | 244 |
| 上轮寄存器合并 | 3442 | 92935 | 38011 | 203 | 244 |
| 本轮尾部裁剪＋寄存器 | 3403 | 92975 | 38191 | 203 | 244 |
| 本轮尾部裁剪＋紧凑 LUTRAM | **3414** | **70003** | **37907** | **203** | **244** |

同输入三 Tile 的 mode 2 比原版减少 29.77%，比上一轮减少 0.81%；相比
本轮最快寄存器裁剪版多 0.32% 周期、HLS FF 少 22,972。四个计算流水
均 II=1，单次调用 134–182 拍。36 Tile 六模式 C 对照、HLS C/RTL、
DMA/CDC 六模式集成均通过，输出与冻结参考逐字节相同；仿真输入规模
和既有测试相同，不把周期收益当作整场景毫秒收益。

证据：`results/20260930/grouped_lutram/result.json` 及
`group_lutram_c_reference/result.json`。平台 `platform/flicker_grouplutram_fpga`
已经独立准备，其 HLS RTL 哈希与集成测试一致，**该版全板布局布线仍待做**。
本轮无 SSH、烧录或上板测速；已有板端 32k 的 45.212 ms、历史 16k 的
31.932 ms 不能被这些仿真数据替换。功耗、全视频首帧、本候选实板画质未测。

下一步是该候选的全板物理验证，再于设备恢复后进行 16k/32k 配对测试。
可用的新构建入口：

```powershell
& examples/3dgs_flicker_hw/pipeline/lane_workset/build_variant.ps1 `
  -Variant grouped-lutram -Project hls_pipeline_grouplutram_new
```

## 后续离线增量：裁剪尾部空块

用户现明确允许充分利用资源、不预留前端空间；暂时没有板卡，本次不访问
设备。资源政策与论文借鉴顺序见 [PAPER_GUIDED_NEXT.md](PAPER_GUIDED_NEXT.md)。

新增显式 `grouped-trimmed` 候选：在原寄存器合并版本上，固定从像素 0
遍历，处理到最后一个有效子块即结束。内部空洞仍按原顺序经过；不改变
数学和像素贡献顺序。四路 evaluator 均 II=1，延迟由固定 181 改为
133–181 拍。HLS 顶层估算 BRAM_18K=203、DSP=244、FF=92975、LUT=38191，
相比寄存器合并版只多 40 FF 和 180 LUT。

同一组 51 条记录、三 Tile、含 DMA/CDC 的 RTL 六模式均通过，4608 像素
逐字节相同；mode 2 周期 **3442→3403，再减 1.13%**，相对原 4861
累计减少 **29.99%**。不是新实板成绩。另 36 Tile、六模式 C 参考共
55,296 像素记录精确一致；C/RTL 联合仿真通过。

第一个动态起点版本虽然 C 仿真通过，但 HLS II=60、调用 984–3864 拍，
已否决并保存 `results/20260930/rejected_trim_dynamic_begin/`。随后通过
固定零起点和显式 64 上限让 HLS 证明地址不重复；没有增加 DEPENDENCE
豁免。新候选来源和完整哈希见 `results/20260930/grouped_trimmed/result.json`。

```powershell
& examples/3dgs_flicker_hw/pipeline/lane_workset/build_variant.ps1 `
  -Variant grouped-trimmed -Project hls_pipeline_grouptrim_new
python examples/3dgs_flicker_hw/pipeline/run_integration.py `
  --project hls_pipeline_grouptrim_new --profile-split
```

下文是上一轮冻结对照。未裁剪寄存器版的全板综合通过，但布局失败：
剩余实例需要 14534 Slice、可用位置 13805；总器件有 19650 Slice，
控制集/原厂已固定逻辑也限制打包。不能用 HLS 资源总数推断必定能放下。
该候选未完成布线、没有可验收的新位流；证据在
`results/20260930/physical_grouped_registers/`。裁剪版本也需自己的物理
验收；紧凑记录与分布式 RAM 版本的已完成检查见本文顶部。

## 上一轮冻结对照

本轮只优化已加载场景后的相机到完整 RGB 帧时间。现有 FPGA 位流和默认
计算路径保持不变；全部新增选项均显式启用，尚未做新位流布线或实板测速。

## 主耗时与实际改动

最近验收的 32,768 点、128×128、60 帧平均为 45.211598 ms。其中 FPGA
服务阶段为 26.671604 ms（58.99%），纯排序为 2.419983 ms（5.35%）。
详见 MVSplat rendering/HOT_PATH_20260930.md；不要把 16k 预览的历史
31.93 ms 与完整点集混为同一配置。

旧 VRU 每次处理 16 个像素：装入状态、完成 133 周期计算、写回状态，
同一个高斯的不同子块会重复这套过程。新增 `FLK_GROUP_SUBTILES` 候选
按高斯序号收集最多四个子块，在一个单调的 64 像素循环中直接更新状态。
每个像素的贡献顺序、FP16 运算、透明度裁剪、提前终止和 ABI 不变。
没有用 DEPENDENCE false 掩盖依赖。

`grouped-registers` 另将四条很宽、很浅的参数记录放到寄存器中。
原始 grouped-bram 映射仅四条 512-bit 记录就消耗每路 15 个 BRAM_18K；
寄存器方案消除这项浪费，但明显增加 FF，不能只报道 BRAM 节省。

## 本轮验证结果

表内周期来自同一组 51 条输入、三 Tile 的实际 DMA/CDC RTL 测试，包含
DDR 停顿、回写和完成握手。**这是有限输入的 RTL 仿真，绝不是整场景实板帧时间。**
正式程序使用的筛选模式是 mode 2；不拿收益更高的 mode 0 替代它。

| 候选 | mode 2 周期 | 相对原版减少 | BRAM_18K | DSP | FF | LUT |
|---|---:|---:|---:|---:|---:|---:|
| 已安装设计的冻结 RTL 对照 | 4861 | — | 219 | 244 | 64227 | 34927 |
| 双端口搬运 | 4737 | 2.55% | 235 | 244 | 64799 | 35067 |
| 合并子块，BRAM 记录 | 3454 | 28.94% | 263 | 244 | 70655 | 37675 |
| 合并子块，寄存器记录 | 3442 | 29.19% | 203 | 244 | 92935 | 38011 |

资源是 **HLS 顶层估算，不是全板综合后利用率**；时序和布局尚未验收。
寄存器合并方案每次 64 像素调用为 181 周期，仍为 II=1；旧每次 16 像素
调用为 133 周期。仅命中一个子块的高斯可能因为较长遍历而吃亏，完整
场景收益还取决于子块覆盖分布、CTU 负载和 DDR，不能按表线性外推。

验证均已完成：

- HLS C 参考、综合、C/RTL 联合仿真；计算流水 II=1。
- 两个合并版本在 36 个稀疏、密集、边缘 Tile、六种模式下逐字节相同，
  共比较 55,296 个像素记录。输入包含不同颜色、各向异性、裁剪和空块。
- 双端口和两个合并版本的 DMA/CDC RTL 集成全部六模式通过；各次
  4,608 个像素记录与旧参考逐位一致，检查输出停顿、边界和结果所有权。
- 三个候选使用的 DMA/寄存器/顶层包装源文件哈希均与原对照一致。
- 新的构建入口拒绝覆盖同名工程；普通 build.ps1 增加 C/RTL Pass 报告
  核验，因为 Vivado HLS 2018.3 遇 Tcl 错误有时仍返回进程退出码 0。

没有新板端图片或画质数值；已有图像指标不能冒充这些候选的上板验证。
板卡 SSH 在本轮超时，本机以太网显示 Disconnected，未执行安装或重启。

## 复跑

始终使用未存在的新工程名。下面命令选择最后一行候选，原默认不变：

```powershell
& examples/3dgs_flicker_hw/pipeline/lane_workset/build_variant.ps1 `
  -Variant grouped-registers -Project hls_pipeline_group64reg_new
python examples/3dgs_flicker_hw/pipeline/run_integration.py `
  --project hls_pipeline_group64reg_new --profile-split
```

其他 Variant：`control`、`dual-port`、`grouped-bram`。
`FLK_GROUP_SUBTILES` 是编译选项，和板端 `mode 2` 筛选模式不是同一参数。

下一关：优先以寄存器合并方案进行全板布局布线，检查 200 MHz 的真实
setup/hold、拥塞和 FF 代价；通过后保留旧 BOOT 做配对实板比较，16k 预览
为速度主线、32k 为质量对照，同一视角各十次、两轮，报告相机到完整 RGB
平均/P95/P99及逐项耗时。未通过物理或实板验收之前不修改默认后端。

## 证据与失败记录

- `results/20260930/summary.json`：综合资源、全部六模式周期、来源和哈希。
- `results/20260930/group64reg_c_reference/result.json`：36 Tile 数值比较。
- `results/20260930/grouped_registers/`：编译选项、HLS 和 RTL 摘要。
- `verify_reference.py`：可重放的确定性数值测试；不报告本机速度。
- `results/20260930/rejected_banks4/`：四分银行负结果。搬运 17→4 周期，
  但计算 II=60、133→984 周期，已在综合阶段否决，没有上板。
- `hls_pipeline_dualport_20260930` 是默认配置控制实验，不是双端口结果；
  准备命令曾因元数据文件名错误中断，实际编译选项仅有精确指数 ROM。
  已根据 vivado_hls.app 识别，真实双端口使用 dualport2 名称，没有混算。
- 早期 C 模型独立调用的三个失败目录仅为缺少 DLL 搜索路径；修复运行时
  路径后 v4 和 grouped-registers 才是完整通过的数值记录。
