# YSU-keg

多模态推荐实验的代码、预注册协议、审计证据和结果记录。使用
[Karpathy autoresearch](https://github.com/karpathy/autoresearch) 的实验循环，
并适配现有推荐任务：先冻结假设、代码和判定标准，再执行实验，记录成功、失败与技术故障。

## 当前结果（2026-10-10）

- **尚无足以支持论文的新推荐性能证据。** 完整状态见
  [INNOVATION_STATUS.md](result/INNOVATION_STATUS.md)。
- F27 评论残差与其他作者评论库匹配的训练内诊断失败：AUC 0.588003，
  内容基线 0.588482、原始评论库 0.670312、流行度 0.757632；该方案冻结。
- F26 发现同作者评论表达的一致性，但不能据此声称用户偏好识别或推荐提升。
- Electronics 原始基线在 SVD 反向传播中出现非有限梯度，五个种子完成数为 0/5。
  数值导数修复通过 CPU 及同输入 GPU 探针审计；完整模型 GPU 检查因频带差异
  超过固定门槛而失败。进程已退出，恢复训练尚未启动；正在定位原模型重复前向差异。
- GPU 评估缓存未通过等价性检查：最大分数误差 0.00026226，六批中三批
  Top-50 排名不同；恢复训练采用原始评估路径。

## 目录

| 路径 | 内容 |
| --- | --- |
| `result/` | 当前结论、失败台账、验证及论文证据标准 |
| `research/` | 各轮假设、实现、必要测试和可分享的审计报告 |
| `baselines/BRIDGE-official/` | BRIDGE 实验代码及原有许可证 |
| `PRAVI/`、`SiMGR2025/` | 当前工作区相关实现及原有许可证 |
| `SOURCE_MANIFEST.json` | 快照文件 SHA256、来源及局部实验仓库版本 |

数据集、原始评论、模型权重、服务器访问凭据和私人飞书材料保留在实验环境中。
远程路径及源版本是原始审计证据，复现时需配置自己的数据和运行环境。
历史探索数字与正式多种子推荐指标的解释范围不同；以各轮协议和报告为准。

## 更新与复现

每个已确认的实验结果或实质实现进展提交一次 Git 更新，失败结果同样保留。
提交说明列出变化及验证；未完成任务如实标为进行中。
源工作区不会因快照发布而修改其冻结实验文件。

从实验工作区生成快照：

```powershell
python YSU-keg/tools/sync_snapshot.py --workspace .
git -C YSU-keg diff --stat
git -C YSU-keg add .
git -C YSU-keg commit -m "Record experiment milestone"
git -C YSU-keg push origin main
```

执行具体实验前先阅读相应目录的 `PROTOCOL.md`、`program.md` 和依赖清单。
正式论文证据要求见 [PUBLICATION_PROTOCOL.md](result/PUBLICATION_PROTOCOL.md)。
