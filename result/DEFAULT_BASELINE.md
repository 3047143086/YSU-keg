# 后续多模态推荐实验的默认对照（2026-09-10）

默认实验对照选用 **当前 BRIDGE 实现，关闭新增实验分支**。这是用于后续方法筛选的工程基线；既有单种子权重不能直接作为完成论文级独立复现的证明。SonicRec 的命令行默认值不是本次选型依据，已失败的 DyPER/HARM 不再作为默认方案。

本文件适用于后续 BRIDGE 相关实验。旧 [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) 中 MSCA 主对照的规定属于此前研究方案；保留 MSCA 作为独立比较模型，不把两个协议下的数字混为一谈。

## 已核对的结果与证据

2026-09-10 已连接服务器，核对 Baby、Sports 权重、交互划分、核心模型代码和模型配置的 SHA256，均与 2026-09-05 记录一致。图像/文本特征及图缓存的当前哈希也已记录。

| 数据集 | 种子 | 本次验证 R@20 | 本次验证 N@20 |
|---|---:|---:|---:|
| Baby | 2026 | 0.10555933096 | 0.04923659660 |
| Sports | 2026 | 0.11984492318 | 0.05666584268 |

本次已以 CPU、四线程重新调用 `BRIDGE.full_sort_predict` 完成两项验证集复算，退出码均为 0。Baby 用时 162.38 秒，Sports 用时 579.09 秒，合计约 12.4 分钟。两项的 R@20、N@20 与历史记录完全相同（差值为 0）；较大 K 的少数 NDCG 存在小于 0.0000001 的浮点/排名差异。所有追踪输入在运行前后哈希一致。本次未训练、未评测试集。

完整结果：[完成收据](../research/20260910_baseline/completion.json)、[Baby 本次复算](../research/20260910_baseline/baby_confirmation.json)、[Sports 本次复算](../research/20260910_baseline/sports_confirmation.json)。每份结果包含有效配置、运行环境、数据和代码哈希、CPU 兼容键及逐项指标差值。

证据位置：

- 历史复算：[Baby](../research/20260905/baby_baseline.json)、[Sports](../research/20260905/sports_baseline.json)。
- 当前文件审计：[remote_audit.txt](../research/20260910_baseline/remote_audit.txt)。
- 原始训练日志审计：[provenance_audit.txt](../research/20260910_baseline/provenance_audit.txt)。
- 本次入口：[confirm.py](../research/20260910_baseline/confirm.py)、[manage.py](../research/20260910_baseline/manage.py)。
- CPU 首次加载兼容记录：[V1_DIAGNOSTIC.md](../research/20260910_baseline/V1_DIAGNOSTIC.md)。
- 远程复算目录：`/data/sunzixuan/bridge_baseline_confirm_20260910_v2`。

## 固定的数据和模型

- 代码目录：`/data/sunzixuan/bridge_official`。Git HEAD 为 `4ee976f5f1715ff9c186aac8a8ddd44a74a41ece`，但工作区有本地修改，必须同时使用审计记录中的文件哈希，不能仅凭提交号宣称是未修改的官方实现。
- 使用该目录 `data/{baby,sports}` 下现有 `.inter`、`image_feat.npy`、`text_feat.npy` 与 `mm_adj_10.pt`。沿用 `x_label` 的训练/验证/测试划分；不下载数据，不替换成 CLIP 特征，也不混入此前内部留出或时间重划分。
- 保持全部图像和文本特征、SVD 三频带、64 维嵌入、两层编码器、物品图 K=10。
- 保持 BRIDGE 原生行为校正和门控：`use_behavior_correction/use_topk_correction/use_behavior_gate/use_calibration_gate=true`；行为相似度 TopK=1000、候选 TopK=200、权重 0.4。
- 关闭 `use_activity_simplex`、行为原型、私有残差及残差门；`msr_weight=0`。可选用户图因缺少 `user_emb.npy` 未启用，复算入口检查它保持关闭。
- 两个既有检查点分别为 `src/saved_models/BRIDGE/baby/bridge_baby_seed2026_screen/model.pth` 与 `src/saved_models/BRIDGE/sports/bridge_sports_s2026_confirm/model.pth`。

## 本次确认协议

只加载 seed=2026 的现有权重，不训练、不调参、不评测试集。模型仅依据同一训练划分重建行为图与历史统计；验证时全物品评分，屏蔽训练正例，逐用户计算多正例 Recall/NDCG，再对用户取平均。评估批大小 64；实际评分均走原始 `full_sort_predict`，不使用缓存替代评分。

主核对指标为 R@20 和 N@20；预先沿用旧 preflight 的绝对差阈值 0.00015，容纳 CPU/GPU 浮点及边界排序差异，同时完整记录实际差值。输入哈希在复算前后均检查。旧权重缺少的 `user_activity_z` 不参与关闭后的分支；CPU 额外注册的 `result_embed` 经确认仅声明而未使用，兼容例外在收据中明示。其余不匹配键仍报错。

## 论文实验仍需补充的证据

已找到 `src/log/bridge_baby_seed2026_screen.log` 与 `src/log/bridge_sports_s2026_confirm.log`。两份日志都明确只有 `1/1` 个参数组合，验证最优指标与本次复算相符；因此没有证据表明这两次运行内部进行了多参数的测试集选型。Baby 日志报告最优 epoch 89（上限 120，stopping_step=12），Sports 报告 epoch 108（上限 300，stopping_step=20）；epoch 按原日志编号记录。日志中的路径和指标对应加强了来源证据，但日志本身没有绑定 checkpoint 哈希。

原始 `quick_start.py` 的通用逻辑会把测试集传给训练器，并按测试指标挑选超参数行；原 `Trainer.fit` 每轮评测试集，两份日志也都有测试结果标记。单次训练的 checkpoint 保存依据是验证指标，但这不足以证明跨实验的历史选型过程完全未使用测试结果。因此旧权重用于工程对照和诊断，后续论文主表须通过独立的验证集选型入口重新训练。

新训练入口必须绕过上述测试集评估/选型逻辑，保存有效配置、训练命令、数据与代码哈希、最优 epoch 和验证指标。后续统一预算为最多 300 epoch、stopping_step=20；Baby 的新训练不能称为按旧 120/12 预算逐字节复现。固定配置先运行预定种子 `2026, 2027, 2028, 2029, 2030`，不选最好种子；相同协议下报告均值、离散程度和成对差值，再扩展第三数据集。沿用旧早停的四位小数 R@20、严格大于才更新、平局保留较早 checkpoint 规则；连续未改善次数大于 stopping_step 才停止。这些重训尚未在本次执行。

模型结构、超参数和报告指标冻结后再进行正式测试集评估。此前 BRIDGE 验证结果与 RCMQR/HPAIR/时间重划分的内部验证结果不可直接比较。本次基线确认不代表新方法有提升，也不代表论文投稿条件已经满足。
