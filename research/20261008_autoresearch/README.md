# 当前实验的 autoresearch 配置与恢复

使用你指定的 karpathy/autoresearch，固定提交 228791f。
原版以 program.md 为入口；本地补充 Codex 技能 `$auto-research`。
完整配置见 [program.md](program.md)，科学协议见 [PROTOCOL.md](PROTOCOL.md)。

2026-10-08 查询确认原软隔离流水线因 baseline 非有限损失停止。
Baby soft-edge 已完成；baseline 未完成，history 未运行。
本次在独立目录校验并复用完整 soft-edge 结果，原配置从头补跑两组对照。
若再次出现数值异常则停止。固定 alpha=0.5、seed=2026；不使用测试集。

从工作区根目录查看：

```powershell
python research/20261008_autoresearch/manage.py status --brief
python research/20261008_autoresearch/verify.py --report
python research/20261008_autoresearch/karpathy_results.py
```

远程目录：/data/sunzixuan/target_isolated_20261008_autoresearch_v1。
latest_status.json 是带查询时间的快照；技术恢复成功与方法通过分别记录。
服务器后台只执行这条固定队列，不在会话结束后自动提出新研究方向。
