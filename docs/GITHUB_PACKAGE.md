# 核心源码交付（2026-09-28）

这是供三人分模块开发的源码交付。主线为视频 → 位姿与稀疏点 → 高斯生成 → CPU＋FPGA 渲染；前馈 NPU 高斯预测仍是候选，不是本次发布的新能力。

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

检查使用程序生成的合成文件接口样例，不需要下载视频、模型、OpenSplat、Vivado 或连接板卡。它验证16项接口/错误输入以及8项资源控制行为，不等价于完成重建或上板。已有真实30帧样例的本地回归可通过 `HGS_TEST_REFERENCE_RUN` 指定独立运行目录；其格式和分辨率需符合测试约定。

三人入口：[工作分工](../TEAM_MODULES.md)、[接口](../examples/3dgs_reconstruction/modules/INTERFACES.md)、[反向传播位置](../examples/3dgs_reconstruction/REPOSITORIES_AND_BACKPROP.md)。A 主责 video_input/pose_estimation，B 主责 gaussian_generation/bounded，C 主责 CPU＋FPGA 后端。创建各自分支提交，接口变更同时验证生产方和消费方。

## 发布范围

| 内容 | 本仓库提供什么 |
|---|---|
| 重建主线 | 四模块、独立 CLI、接口校验、资源监控、构建脚本、补丁、评价代码 |
| 渲染主线 | CPU 投影/SH/分组排序，CAT 参考，HLS/RTL、DMA/流水、仿真及构建脚本 |
| NPU | 前端匹配候选及 CPU/NPU 比较入口，当前不作为默认加速后端 |
| 冻结四单元回退 | `releases/3dgs_renderer_v1_20260928` 中源码、接口、原始清单；完整本地运行包继续保留 |
| 共享依赖 | 仍被引用的 compositor 板卡工具及 scene/CAT 检查器；不是另起一条旧实验主线 |
| 基础示例 | 原加法器移动到 archive；附基础 ALU 源码和测试入口 |
| 历史结果 | 报告和小型汇总；大部分逐次日志、图像、原始回读不随源码发布 |

**GitHub 源码包不是可直接烧录的板卡镜像或完整训练环境。** SDK、厂家平台工程、BOOT/bitstream、已编译程序、视频、场景模型、Python 环境和大体积原始证据不在本次源码包中。历史文档中的绝对本地路径、`runs/`、`evidence/` 和平台目录属于原实验环境，不能理解为 clone 后自动存在；本页和主线接口是新人入口。

冻结目录的原 manifest 保留完整运行包哈希。因为本次只发布源码子集，直接运行冻结包的完整哈希检查会因缺少二进制/模型/固件而失败，这是发布边界，不要删除校验来绕过。真实运行需补齐团队原有完整包及匹配 SDK/位流；来源和历史验证由原清单核对。FPGA 厂商工程和工具的独立依赖同样需要配置。

源码包不存储设备密码、私钥或下载令牌。现有 SSH 工具读取 `FPGA_BOARD_PASSWORD`，使用固定主机公钥核验；换板需核对实际地址和公钥。`tools/check_core.py` 不访问设备，也不运行安装或烧录脚本。

## 第三方代码如何准备

本次仓库保留上游链接、固定 revision、补丁和许可证；OpenSplat/MVSplat 完整工作副本在本地 `vendor/`，不重复塞入主仓库。版本见 [third_party/sources.json](../third_party/sources.json)。

OpenSplat 用以下命令在新的工作目录准备，已有目录请先核对版本，不覆盖：

```text
git clone https://github.com/pierotofy/OpenSplat.git examples/3dgs_reconstruction/vendor/OpenSplat
git -C examples/3dgs_reconstruction/vendor/OpenSplat checkout 62fd86fe3b62644b65890557ad9bf397abdc7cfb
python examples/3dgs_reconstruction/bounded/prepare_source.py
```

随后依照 `bounded/README.md` 准备 LibTorch/OpenCV 和本机工具链并构建。电脑CPU链有历史验证，ARM完整高斯训练尚未验收，不能把源码准备成功称为已上板训练。

MVSplat 作为独立候选固定到 `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`，可 clone 作者仓库并 checkout 该版本。这个 revision 从已下载 GitHub ZIP 的归档注释读出；归档完整性见 sources.json。其权重、数据和 CUDA 默认环境需另行准备，当前还没有 NPU 图或新候选速度结果。

第三方源代码、派生实现和补丁保留对应许可；没有给整个仓库擅自添加一个覆盖所有来源的新许可证。

## 可复现打包与验收范围

```text
python tools/package_core.py --output core-source.zip
```

脚本使用显式目录/文件清单，拒绝覆盖同名包，包内包含逐文件 SHA256。只读打包，不执行 Git 推送或板端动作。GitHub 的 Code → Download ZIP 或某次提交的 archive 链接也能下载已提交源码。

本次验收覆盖干净解压目录下的 CLI、Python 语法和24项软件回归；没有新增训练画质、板端 FPS、资源/时序或功耗测量。收益是队友能下载同一核心代码和独立验收，代价是环境、真实数据和运行包仍须单独准备。详细发布核验结果见 [CORE_VALIDATION.md](CORE_VALIDATION.md)。
