# 2026-11-02 主线清理归档

来源为 NPU 主线合并提交 f9af617a4103a2aa7d72e7818318212564dfacb5 的干净工作树。
257 个文件条目保存原始路径、字节长度和 SHA256；其中 250 个旧文件从活动树删除，
7 个入口/说明保留修改前快照。归档共 590537 字节，解压数据 1195945 字节。

| 包 | 内容 |
|---|---|
| training_history.zip | OpenSplat 训练、bounded、旧 SIFT/NPU 前端及旧入口 |
| npu_diagnostic_history.zip | 独立 NPU deploy_20261021 报告、源码和证据 |
| renderer_development_history.zip | 已被冻结包替代的早期渲染、合成和硬件实验 |
| entrypoint_snapshots.zip | 公共 CLI、接口测试与说明的修改前快照 |

归档保留失败结果与第三方声明；不是新的板测，也不作为活动模块加载。
当前渲染使用 mvsplat/rendering/render_main，回退包和在研工作目录未删除。
恢复到新目录：

~~~sh
python tools/archive_history.py verify --archive archive/history_20261102_cleanup
python tools/archive_history.py restore --archive archive/history_20261102_cleanup --output /path/to/NEW_HISTORY
~~~

恢复旧工程时配套使用来源提交或相应历史 checkout；不要覆盖现有主线。
index.json 中 retain_in_worktree 为 true 的 7 项是旧入口快照，不能覆盖当前版本。
