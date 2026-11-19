# 六分类模型结构说明

本次检查的旧文件是 `model_benchmark_all(8).py`、`train_all(7).py`、`config_6_all_6(2).py`、`IterDataset(2).py` 和 `run_all_4_models(5).py`。Figure 3 入口为 `train_all.py`；请将整个包放在同一目录，避免导入 Figure 2 的同名模块。

| 部分 | 旧实现 | 本版默认 |
|---|---|---|
| 任务 | 六类输出，但批量入口仍指向五分类 Y 文件且没有 validation | A/T/C/G/X/Y 六类，明确 train/validation/test |
| 上下文 | 取决于旧 CSV 是否已经遮蔽 | 数据入口统一将第 4 位 one-hot 设为 N；不遮蔽目标信号与响应 |
| 信号和特征融合 | 每碱基电流 CNN 加统计量/序列/响应融合 | 保留该总体组织及 6 个碱基 token |
| LiteMamba mixing | 门控深度卷积，无状态空间递推 | 输入依赖的 Δ/B/C、可学习负 A、显式状态递推；双向共享参数 |
| 读出 | 全局 attention 加第 6 位表示 | 全局 attention 加第 4 位目标表示，四种架构统一 |
| 结构对照 | 未明确区分 | 可指定 `legacy_gatedconv`、`legacy_last_attention`；配置与名称完整留档 |

重新绘制 Figure 1 时，整体仍可按“特征输入 → 双分支融合 → mixing blocks → 目标/全局读出 → classifier”组织。应更新 LiteMamba block 内部为 selective SSM，并把目标读出标为 **fourth-position target token (index 3)**。不能把新增状态递推写成旧代码原本就已实现的内容。

这是基于选择性状态空间思想的轻量实现，采用纯 PyTorch 循环，非官方优化 CUDA kernel。代码中明确了离散化与双向计算，便于审阅；需要实际消融验证改动收益。仅有 6 个碱基 token，不据此提出长序列效率结论。

默认配置的参数量（全部特征、model_dim=96、blocks=4、state_dim=16）：

| 架构 | 参数量 |
|---|---:|
| BiGRU | 187,495 |
| Transformer | 188,551 |
| ConvMixer | 78,727 |
| LiteMamba | 325,639 |

以上四模型使用相同输入和读出，但并非参数量匹配。LiteMamba 在这里并非参数最少；应使用实际参数量、独立推理基准和真实测试结果描述取舍。若另做参数量匹配实验，应预先固定各模型宽度，作为不同对照队列保存。

原先独立 config 的 raw length=64、dwell/quality divisor=1 与旧批量入口的实际参数 30/60/50 不一致。本版采用旧批量入口的 **30/60/50**，全部写入实际配置。请结合原提取代码核对二次缩放含义；若确需修改，建立独立预声明方案，并对所有模型一致使用，不能只给某一模型换输入。

参考：

- Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)。
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)：allocated、reserved 与 GPU 进程总占用含义不同，因此调度保留额外上下文与安全余量。
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)：多分类概率、OvR ROC/PR 与损失指标。
