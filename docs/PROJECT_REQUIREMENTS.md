# 项目要求：模块归属与目录变更

生效：2026-10-09，依据用户本轮要求。执行入口同步写入仓库根目录 `AGENTS.md`。

## 固定的模块边界

业务代码统一位于 `examples/3dgs_reconstruction/mvsplat/`，保留四个业务模块：

| 模块 | 代码归属 |
|---|---|
| `video_input/` | 视频接收、解码、选帧及相关实现和测试 |
| `pose_estimation/` | 位姿、特征、匹配、几何及相关实现和测试 |
| `gaussian_generation/` | MVSplat 生成、Gaussian 数据适配及相关实现和测试；NPU 候选放在 `npu_branch/` |
| `rendering/` | 相机投影、排序、CPU/FPGA 渲染及相关实现和测试；冻结主线为 `render_main/`，独立候选为 `render_branch/` |

已有 `initialize/` 只负责视频前预热、资源生命周期与整链部署协调；
`scene_preparation.py`、`warm_pipeline.py`、`pipeline.py` 负责已有流程的调度。
算法代码不得以“协调”为名脱离对应业务模块。

## 新增代码的强制要求

1. 今后新增实现、编译/部署脚本、测试和模块说明，必须放到其所属现有模块下。
   全项目共享的现有文档/工具继续使用 `docs/`、`tools/`；不在根目录散落实验脚本。
2. **禁止未经用户明确批注/同意，额外创建独立业务模块或同级模块目录。**
   若现有边界确实无法容纳，应先说明新模块的名称、位置、用途、输入输出和为何不能放入现有模块，
   得到用户明确批注/同意后再创建。不能把沉默、时间经过或一般性的“继续优化”视为批准。
3. 在现有模块内增加合理的实现文件、测试文件或内部辅助目录，不等于新增独立业务模块；
   但不得用 `misc`、`experimental`、`tmp` 等目录绕过归属规则或复制第二条业务主线。
4. 各模块通过 [既定数据接口](../examples/3dgs_reconstruction/modules/INTERFACES.md) 交接。
   改路径时同步改导入、子进程入口、部署清单和说明，不留下指向旧位置的活动调用。
5. `render_main/` 冻结基线必须可回退；`render_branch/` 默认不调用。
   `gaussian_generation/npu_branch/` 默认不启用，仅显式选择 NPU 且通过既有检查后才能进入整链。
6. 历史实验使用既有归档体系，保留原始路径、哈希和测量口径；不改写历史证据来适应新目录。

## 本轮完成与验证边界

- 原 `mvsplat/npu/` 整体迁至 `mvsplat/gaussian_generation/npu_branch/`，不增加第五个业务阶段。
- 用户已要求板卡离线时不测试；本轮仅整理源码、文档，进行静态检查并更新 GitHub。
- 不把目录搬迁当作 NPU 数值通过、性能提升或整链验收；功能与板端结果待恢复测试后记录。

完整当前目录见 [代码结构](CODE_STRUCTURE.md)，默认配置和历史测量范围见 [主线说明](../MAINLINE.md)。
