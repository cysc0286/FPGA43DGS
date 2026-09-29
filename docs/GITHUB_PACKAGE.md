# 核心源码交付（2026-09-28）

这是按模块组织的核心源码交付。当前主线为视频 → ARM 位姿估计 → MVSplat 固定权重前馈 → CPU＋FPGA 渲染。入口为 `python examples/3dgs_reconstruction/pipeline.py reconstruct --video INPUT.mp4 --out NEW_RUN`。2026-09-29 的实板精简结果已随仓库保存；前馈 NPU 高斯预测仍未部署。OpenSplat 训练仅保留历史对照。

## 下载后先做什么

需要 Python 3.10+。在仓库根目录建立独立环境并安装轻量检查依赖：

```text
python -m venv .venv
```

Windows 使用 `.venv\Scripts\python.exe`，Linux 使用 `.venv/bin/python` 执行下面命令；或先激活环境后使用 `python`：

```text
python -m pip install -r requirements-core.txt
python tools/check_core.py
python examples/3dgs_reconstruction/pipeline.py --help
```

检查使用程序生成的合成文件接口样例，不需要下载视频、模型、OpenSplat、Vivado 或连接板卡。它验证16项接口/错误输入、8项资源控制和3项主入口兼容行为，不等价于完成重建或上板。已有真实30帧样例的本地回归可通过 `HGS_TEST_REFERENCE_RUN` 指定独立运行目录；其格式和分辨率需符合测试约定。

开发入口：[接口](../examples/3dgs_reconstruction/modules/INTERFACES.md)、[反向传播位置](../examples/3dgs_reconstruction/REPOSITORIES_AND_BACKPROP.md)。模块通过文件接口交接；接口变更同时验证生产方和消费方。

## 发布范围

| 内容 | 本仓库提供什么 |
|---|---|
| 重建主线 | 四模块、独立 CLI、接口校验、资源监控、构建脚本、补丁、评价代码 |
| MVSplat 主线 | 板端位姿、固定权重前馈、导出、渲染调度与画质评价源码；精简实板指标和效果图在 `examples/3dgs_reconstruction/mvsplat/results/20260929` |
| 渲染主线 | CPU 投影/SH/分组排序，CAT 参考，HLS/RTL、DMA/流水、仿真及构建脚本 |
| NPU | 前端匹配候选及 CPU/NPU 比较入口，当前不作为默认加速后端 |
| 冻结四单元回退 | `releases/3dgs_renderer_v1_20260928` 中源码、接口、原始清单；完整本地运行包继续保留 |
| 共享依赖 | 仍被引用的 compositor 板卡工具及 scene/CAT 检查器；不是另起一条旧实验主线 |
| 基础示例 | 基础 ALU 源码和测试入口；加法器 smoke test 不随源码发布 |
| 历史结果 | 报告和小型汇总；大部分逐次日志、图像、原始回读不随源码发布 |

**GitHub 源码包不是可直接烧录的板卡镜像或完整训练环境。** SDK、厂家平台工程、BOOT/bitstream、已编译程序、视频、场景模型、Python 环境和大体积原始证据不在本次源码包中。历史文档中的绝对本地路径、`runs/`、`evidence/` 和平台目录属于原实验环境，不能理解为 clone 后自动存在；本页和主线接口是新人入口。

冻结目录的原 manifest 保留完整运行包哈希。因为本次只发布源码子集，直接运行冻结包的完整哈希检查会因缺少二进制/模型/固件而失败，这是发布边界，不要删除校验来绕过。真实运行需补齐团队原有完整包及匹配 SDK/位流；来源和历史验证由原清单核对。FPGA 厂商工程和工具的独立依赖同样需要配置。

源码包不存储设备密码、私钥或下载令牌。现有 SSH 工具读取 `FPGA_BOARD_PASSWORD`，使用固定主机公钥核验；换板需核对实际地址和公钥。`tools/check_core.py` 不访问设备，也不运行安装或烧录脚本。

## 第三方代码如何准备

本次仓库保留上游链接、固定 revision、补丁和许可证；OpenSplat/MVSplat 完整工作副本在本地 `vendor/`，不重复塞入主仓库。版本见 [third_party/sources.json](../third_party/sources.json)。

仅回放旧 OpenSplat 训练对照时，用以下命令在新的工作目录准备，已有目录请先核对版本，不覆盖：

```text
git clone https://github.com/pierotofy/OpenSplat.git examples/3dgs_reconstruction/vendor/OpenSplat
git -C examples/3dgs_reconstruction/vendor/OpenSplat checkout 62fd86fe3b62644b65890557ad9bf397abdc7cfb
python examples/3dgs_reconstruction/bounded/prepare_source.py
```

随后依照 `bounded/README.md` 准备 LibTorch/OpenCV 和本机工具链并构建。电脑CPU链有历史验证，ARM完整高斯训练尚未验收，不能把源码准备成功称为已上板训练。

MVSplat 主线固定到 `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`，可 clone 作者仓库并 checkout 该版本。这个 revision 从已下载 GitHub ZIP 的归档注释读出；归档完整性见 sources.json。2026-09-29 已用锁定权重在 ARM CPU 上完成视频到 FPGA 图像的实板验证，依赖、权重哈希、运行命令和画质/速度限制见 [MVSplat 说明](../examples/3dgs_reconstruction/mvsplat/README.md)。源码包仍不包含权重、测试视频、ARM 二进制依赖或 NPU 图。

第三方源代码、派生实现和补丁保留对应许可；没有给整个仓库擅自添加一个覆盖所有来源的新许可证。

## 可复现打包与验收范围

```text
python tools/package_core.py --output core-source.zip
```

脚本使用显式目录/文件清单，拒绝覆盖同名包，包内包含逐文件 SHA256。只读打包，不执行 Git 推送或板端动作。GitHub 的 Code → Download ZIP 或某次提交的 archive 链接也能下载已提交源码。

原核心源码包验收覆盖干净解压目录下的 CLI、Python 语法和24项软件回归，见 [CORE_VALIDATION.md](CORE_VALIDATION.md)；本次主线切换又增加3项入口回归，本机共通过27项。MVSplat 的实板结果单独记录在 [FAST_VALIDATION.md](../examples/3dgs_reconstruction/mvsplat/FAST_VALIDATION.md)；本次没有重新综合、布线或测功耗。`package_core.py` 使用显式源码清单并排除 `results/`，所以 Git 仓库中的精简结果与效果图不在该 source-only ZIP 中；需要结果时以仓库提交为准。
