# 运行状态

更新时间：2026-09-07。本记录为启动核验快照；服务器上的进度和结果会继续自动更新。

## 已启动

- 服务器：213，环境 `/data/sunzixuan/miniconda3/envs/MMRec`。
- 实验根目录：`/data/sunzixuan/rave_20260906_stage1`。
- 后台流水线 PID：`2645820`，与当前会话断开后仍继续运行。
- `global_s2026`：GPU 1，PID `2646468`，已观察到第 13 个 epoch。
- `local_s2026`：GPU 0，PID `2646526`，已观察到第 10 个 epoch。
- 剩余七次运行由队列自动接续，不需要再次手动启动。

任务为 Baby 上的全局、局部、打乱局部证据三组对照，各三个种子
2026/2027/2028。它是 RAVE 的第一阶段信息增益筛选，不是完整的主动获取策略。

## 已验证

- 7,037 张原始图片已下载；全目录保留 7,050 个商品及缺失掩码。
- 冻结 CLIP ViT-B/16 的五种图像视图与最多八段文本已提取，约耗时 110.6 秒。
- 全局组池化相同的图文内容，不把额外输入或编码器差异作为收益来源。
- 五项单元测试通过；三组真实 GPU 冒烟测试全部通过。
- 冒烟测试确认新增参数确实更新，推理确实调用新增模块。
- 训练 loss、开发集 Recall/NDCG、最佳 checkpoint 正常落盘。
- 从原始训练集内部划分 99,106 条支持交互和 19,445 条开发交互。
- 官方验证集、测试集均不用于本轮训练或评估。
- 原有数据、代码和 MMRec 环境未覆盖；新增依赖位于隔离目录 `vendor/`。

## 查看结果

服务器上优先查看：

```text
/data/sunzixuan/rave_20260906_stage1/RESULTS.md
/data/sunzixuan/rave_20260906_stage1/summary.json
/data/sunzixuan/rave_20260906_stage1/pipeline_status.json
```

每次运行的 `runs/<组名>_s<种子>/` 包含 `progress.json`、`epochs.jsonl`、
`best.json`、`best.pth` 和最终的 `result.json`。错误写入 `failure.json`
及 `stdout.log`，汇总不会把失败或未完成运行当成成功结果。

在当前本地工作目录刷新状态：

```powershell
python research\20260906_stage1\manage.py status --brief
```

该命令将完整状态保存为本地 `latest_status.json`。本地文件不会在会话关闭后
自动同步；服务器的汇总表会持续更新。不要重复执行 `start`。

最多每组 200 epoch，连续八次开发集评估无提升则早停，每次运行上限两小时，
整条流水线上限十八小时。按目前每 epoch 约七秒的训练速度，正常情况下九次
训练可在数小时内完成；实际受评估时间、早停和服务器负载影响。

目前不能报告新方法提升。需要等三种子完整对照结束；即使开发集筛选通过，
也仍需独立确认和后续策略实验，不能直接作为论文测试结果。
