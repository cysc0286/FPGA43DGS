# 2026-10-09 历史包

来自整理前提交 `b55b95085a1eef965c951b3b73a4f2fa2daf02c1` 的已跟踪文件。
总共 823 个文件条目；其中 814 个未修改的历史文件已从活动源码树移除，
1 个未提交硬件脚本原地保留，8 个当前文档另存整理前快照后重写。

| 包 | 内容 |
|---|---|
| `frontend_history.zip` | MVSplat、预热、NPU 和早期换视角的原始计时、图片、诊断与源码快照 |
| `fpga_history.zip` | 资源重分配、布局布线、ROM、lane、PipeGS 等历史源码、成功/失败证据 |
| `documentation_history.zip` | 整理前 README、主线说明、架构说明及旧四模块接口文档 |
| `index.json` | 原始路径、文件大小、SHA256、包 SHA256、未提交工作保留标识 |

包保持原始字节，不能把旧记录当作新板测。`lane_workset/build_variant.ps1`
在归档时已有未提交修改：包保存 HEAD 中的已提交版本，索引标记活动副本需保留；
活动工作副本没有覆盖，其未提交内容也不包含在本次整理提交中。

从仓库根目录查找某段历史：

```powershell
Select-String -Path archive/history_20261009/index.json -Pattern 'pipegs|attempts_audit|npu_step1'
```

需要复看时恢复到一个**不存在的新目录**，不会覆盖当前主线：

```text
python tools/archive_history.py restore --archive archive/history_20261009 --output archive/restored_history
```

恢复器先核对每个 ZIP、内部清单与每个文件的长度和 SHA256，再按原相对路径解包。
历史源码若使用跨目录相对依赖，应将它与上述提交的独立 checkout 配套使用。
日常主线不加载这些包。若只运行或复建当前渲染器，直接用 `rendering/render_main/`，
不必恢复实验工作树。
