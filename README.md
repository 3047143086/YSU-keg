# YSU-keg

多模态推荐实验的代码、预注册协议、审计证据和结果记录。使用
[Karpathy autoresearch](https://github.com/karpathy/autoresearch) 的实验循环，
并适配现有推荐任务：先冻结假设、代码和判定标准，再执行实验，记录成功、失败与技术故障。

## 当前结果（2026-10-10）

- 关机前已在服务器后台启动 Baby 官方 MSCA 均匀基线复核，seed2026，最多8小时；启动快照确认50次有限优化更新、GPU1利用率99%，监督器已脱离SSH。此任务是复现记录，不计作创新收益。见 [启动状态](research/20261010_baseline_confirmation/current_status.json)。
- 图像特征扰动预检查最终 `blocked`：首个固定TRAIN批次选定2,858行，其中26行BPR梯度为零，按预注册非零条件停止，尚未评估攻击或随机对照。见 [回执与历史故障](research/20261010_feature_perturbation_precheck/PRECHECK_REPORT.md)。

- 兴趣组负采样三臂实验已完成：Baby 均匀基线 R@20/N@20 = 0.102898/0.045759，兴趣组候选 0.103790/0.045870，固定同池乱组对照 0.103472/0.045755。候选 N@20 收益不足且 R@10 回退，未通过冻结门，记为 F28/discard；按协议没有进入 Sports。见 [实验协议](research/20261010_interest_negative/PROTOCOL.md) 和 [独立结果复核](research/20261010_interest_negative/status/FINAL_AUDIT.md)。
- 品牌/类别关系辅助监督的 CPU 资格审计已完成但不合格：Baby 品牌覆盖 69.8595% 低于 70% 门槛，类别只有一个签名，两个关系类没有可用配对；未启动 GPU，Sports 未继续审计。见 [资格结果](research/20261010_catalog_relation_assessment/RESULTS.md)。
- MASAM-motivated 固定参数块曲率诊断已完成：48/48 探针有效、170 项保护哈希一致，但未满足跨批次/方向/状态的稳定模态主导门，状态为 `diagnostic_inconclusive`。仅计算 TRAIN 批次的局部曲率，无优化更新或验证/测试评估，未启动后续 SAM 训练。见 [诊断结果](research/20261010_curvature_diagnostic/result.json) 与 [预注册协议](research/20261010_curvature_diagnostic/PROTOCOL.md)。
- **尚无足以支持论文的新推荐性能证据。** 完整状态见
  [INNOVATION_STATUS.md](result/INNOVATION_STATUS.md)。
- F27 评论残差与其他作者评论库匹配的训练内诊断失败：AUC 0.588003，
  内容基线 0.588482、原始评论库 0.670312、流行度 0.757632；该方案冻结。
- F26 发现同作者评论表达的一致性，但不能据此声称用户偏好识别或推荐提升。
- Electronics 原始基线在 SVD 反向传播中出现非有限梯度，五个种子完成数为 0/5。
  数值导数修复通过 CPU 及同输入 GPU 探针审计；完整模型 GPU 检查因频带差异
  超过固定门槛而失败。后续两次原模型前向比较也出现差异：36 个频带中 27 个不同，
  13 个超过原界限；状态、RNG 和输入文件不变。新的同输入集成审计通过：一次真实
  损失前向的 36 个频带精确相同，一次反向的 27 项参与参数梯度均有限。
  [原始回执](research/20261010_same_input_integration_audit/archives/v1/receipts/integration_receipt.json)
  与旧失败共同保留；此结果限于该观察批次。独立恢复 v3 已通过实际 CPU 50/50 测试
  并冻结，部署后的实际检查也 50/50 通过。客户端 13:08（北京时间）快照确认队列已启动，
  首个种子 2026 完成 4 次有限更新后，在 epoch 0 / batch 4 的前向中触发索引 43 处的
  谱边界检查，已按协议停止并保存失败输入。未完成任何轮次或种子，队列已退出。
  [启动及失败记录](research/20261010_electronics_recovery_status/RUN_STATUS.md)
  与冻结源码分开归档。原定 24 小时/种子、72 小时总队列，实际子进程 107.984 秒
  后退出，完成种子 0/5；客户端与服务器事件时钟未同步校准。
  [只读诊断](research/20261010_recovery_boundary_diagnostic/AUDIT.md)发现保存的边界间隔
  0.744223 低于固定阈值 0.757424，奇异值未严格重合；没有放宽门槛或重启该版本。
  失败不会计入正式基线结果或新推荐收益。
- GPU 评估缓存未通过等价性检查：最大分数误差 0.00026226，六批中三批
  Top-50 排名不同；恢复方案保留原始评估路径。

## 目录

| 路径 | 内容 |
| --- | --- |
| `result/` | 当前结论、失败台账、验证及论文证据标准 |
| `research/` | 各轮假设、实现、必要测试和可分享的审计报告 |
| `research/20261010_frozen_carrier/` | 原 Electronics v1 实际部署的 35 份源码及配置、SHA256 和许可证 |
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
