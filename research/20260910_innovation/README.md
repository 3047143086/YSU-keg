# 目标交互隔离实验

本轮研究预计算协同行为证据中的训练目标自支持：训练正例参与构建自己的物品共现证据，
预测未观察物品时则没有该贡献。候选方法在训练期间逐目标扣除这一贡献，检验多模态
偏好学习是否因此改善。尚未取得性能结论。

具体公式、已有工作的边界和预设继续/停止条件见 [PROTOCOL.md](PROTOCOL.md)。
本次不是重命名已失败的图增边、模态路由或冻结特征重排序方案。

## 文件

- `evidence.py`：固定 KNN 支持下的逐目标行为证据解析计算。
- `prepare.py`：只用训练交互做机制诊断、与原 BRIDGE 行为分数对照、生成缓存。
- `proposal.py`：只替换训练正负样本的行为证据，保留模型参数和推理路径。
- `train.py`：从头训练的独立入口，仅根据验证集选择 checkpoint。
- `run_pipeline.py`：三组对照的自动顺序运行、失败停止及结果汇总。
- `manage.py`：部署、测试、启动与拉取远程状态。

远程目录：`/data/sunzixuan/target_isolated_20260910_v2`。
模型代码复制在该目录 `carrier`；数据仍使用既有 LOBSTER 数据。

从本地工作区检查：

```powershell
python research/20260910_innovation/manage.py status
```

状态拉取会保存 `latest_status.json` 和 `pipeline_status.json`。
服务器的 `runs/*/progress.jsonl` 记录每轮验证指标，`runs/*/result.json`
仅在训练、最佳权重复算和输入检查全部成功后生成；`baby_screen.json`、
`sports_screen.json` 给出同协议下三组的差值和预设判定。

运行状态需要区分：

- `running`：查看 stage 判断是准备、冒烟、正式训练还是等待资源。
- `closed_mechanism_gate`：训练数据诊断未支持假设，未继续该数据集训练。
- `closed_validation_gate`：正式筛选未满足预设提升条件，不继续后续数据集。
- `technical_failure`：代码、资源或输入检查失败，不能作为方法无效结论。
- `pilot_passed_needs_multiseed_and_multimodal_attribution`：单种子两数据集筛选通过，
  仍需要多种子、第三数据集和多模态机制归因，不能直接当论文结论。

## 开发核对

第一轮远程测试中，20/21 项通过；稀疏/稠密实现对比发现一个近零行标准化的数值问题。
原 float32 求和与减法的舍入残差被小方差分母放大，错误达到 0.0149。
远程逐字段诊断进一步确认：旧 SciPy 的稀疏求和即使传入 dtype=float64，
仍可先按 float32 累积。修正为先转换矩阵精度，再求和，并用存储相似度除以
实际共现计数计算自投票。这是实现精度问题，不放宽测试或筛选门槛。
修正后服务器 25 项测试全部通过，见 `verify_receipt.txt` / 服务器 `test_receipt.json`。

v1 目录保留：Baby 诊断已通过，原始正例行为证据均值相对下降 86.04%，
99.93% 的正例在完整扣除后的标准分比仅删历史低至少 0.1；原始证据与 BRIDGE
最大差 0.00001145。正式训练尚未运行时，冒烟遇到 Torch 1.12 显存限制 API
要求显式设备索引的问题。v2 改用已选 GPU 的逻辑索引 0，保持显存上限、
所有模型配置、种子与筛选规则不变，重做诊断和冒烟。
