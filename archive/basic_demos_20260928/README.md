# 基础示例归档

2026-09-28：从 `examples/` 移入 `adder_smoke/` 和 `basic_alu/`，不属于当前视频到高斯渲染的默认入口。

- `*_original.tar.gz`：移动前完整快照，含当时的源码、说明、历史仿真与验证记录。
- `manifest.json`：原路径、新路径、各文件 SHA256 和压缩包 SHA256。
- 展开目录保留可用代码。`basic_alu/board/prepare_project.py` 代码迁移修正为向上查找仓库根，避免移动后错误指向 archive；原代码可从原始包恢复。两个示例 README 的可执行命令已更新；这些变更另记 `migration_changes.json`。
- 当前渲染平台、BOOT、冻结渲染包和厂商 SDK 未迁移。

加法器仿真入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\archive\basic_demos_20260928\adder_smoke\run_sim.ps1
```

如需恢复旧目录，在确认 `examples/adder_smoke` 或 `examples/basic_alu` 不存在后，将对应原始 tar.gz 解压到 `examples/`。不要覆盖已有目录。历史日志里的旧绝对路径是原实验记录，不批量改写。
