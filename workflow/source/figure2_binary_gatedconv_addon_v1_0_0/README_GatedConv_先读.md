# Figure 2：额外补充最终 GatedConv 二分类模型

原压缩包的四模型代码和说明保留原样。新增入口只训练当前最终 GatedConv，不会重新运行已经完成的 BiGRU、Transformer、ConvMixer 或 selective SSM 训练，也不覆盖这些旧结果。

GatedConv 使用 `legacy_gatedconv + target_attention`：96维、4个门控卷积块、第四位表示与全局注意力池化拼接，6mer第四位统一N，二分类输出。内部兼容键仍为 `model_type=litemamba`，报告名称为 **GatedConv**，不将其称为选择性状态空间模型。这里从独立随机初始化开始训练，不载入六分类权重。

## 直接运行

将本包解压至 `paper` 目录的新代码文件夹，使用原来成功运行二分类实验的环境：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code/figure2_binary_gatedconv_addon_v1_0_0
bash launch_gatedconv_binary.sh
```

默认执行 **pos、neg两方向 × 原二分类默认种子42、43、44＝6次新增训练**。最多200 epoch，patience30，AdamW学习率0.001，batch256。原二分类默认用的是42、43、44；它与六分类的种子列表不同。如果你原来的二分类运行时显式指定了别的种子，应传入同一完整列表，例如 `--seeds 42 2024 2025`，不能依据测试成绩挑种子。

启动器会打印本次独立日志及对应的 `tail -F` 指令，提交后台后可以关闭终端，无需额外 `nohup`。启动成功不等于训练完成，以日志中的 `[FULL PASS] Figure2 GatedConv phase=all` 为准。同一结果目录有排他锁，重复启动请求不会再启动另一组训练。

默认输入直接使用已经制备并运行过的二分类数据：

```text
/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721
```

默认新增结果目录：

```text
/path/to/revision-results/figure2_binary_gatedconv_results_v1_0_0
```

不需要重新提取特征或重新划分数据。不要将 `--output_dir` 指向原四模型的结果目录；新增入口会拒绝使用非本组拥有的已有实验目录。结果和输入、代码、旧基线目录必须相互独立。

如需修改路径：

```bash
bash launch_gatedconv_binary.sh \
  --data_dir /实际路径/binary_read_split_721 \
  --output_dir /path/to/revision-results/另一个新的GatedConv结果目录
```

前台运行或只检查计划：

```bash
bash run_gatedconv_binary.sh --dry_run
bash run_gatedconv_binary.sh
```

## 训练、测试与恢复

新增任务沿用已经成功运行的原二分类训练器和队列，默认串行。`workers=0`、每进程4个CPU线程；本次不改写为新的并行调度器，避免在导出、分析与训练交错的旧流程中同时累积大内存占用。

全部新增任务先训练并用验证集选模及拟合温度，全部完成后固定这组模型的最佳检查点，再统一运行测试集。测试集不参与学习率调整、早停、温度拟合或检查点选择。pos/neg分开汇总，不将两个方向当作重复实验平均。

只训练或集中评价时可使用 `--phase train` / `--phase evaluate`；所有其他参数必须保持相同。默认同命令自动启用原队列的 `--resume_failed`：已完成且配置相同的任务跳过，有完整last检查点的失败任务从其恢复；首次检查点前失败不会被静默当作新训练重启，需要先查看对应子任务日志。不要删除检查点、伪造完成状态或改写方案文件。

## 保存内容

为控制磁盘占用，本组默认关闭逐epoch逐样本预测、embedding、attention数组和周期检查点副本，仍保留：

- 每个epoch的完整训练/验证指标与资源记录。
- best、last检查点、最佳state_dict，以及优化器、调度器、AMP和RNG恢复状态。
- 最佳模型对train、val、test的完整逐样本CSV、float32 logits、labels及统计分析。
- 原始数据SHA、read审计、配置、环境、源码快照、验证集温度和整组测试冻结记录。

缓存统一保存在新增结果目录的 `shared_feature_cache`，相同输入由本组的不同种子共享，不写入原四模型结果目录。该入口保留原训练器的train/val/test完整预测，区别于最近五分类精简版省略train预测的策略。

完成后生成 `figure2_gatedconv_results_core_v1_0_0.tar.gz` 小包，用于上传分析；完整逐样本预测和模型检查点仍保留在服务器。本组主要性能图、ROC/PR、校准及参考序列/motif分析可直接利用保存预测重算。以后若需要嵌入或注意力图，可用最佳检查点另做推理。

如果希望同时只读汇总已经完成的原四模型结果，追加：

```bash
bash launch_gatedconv_binary.sh --baseline_dir /原二分类四模型结果目录
```

`--baseline_dir` 仅交给新增汇总程序读取，不会重新训练或预测旧模型。汇总按实际数据SHA、方向、seed和配置标识区分结果；不同输入或无法核对的旧结果不能冒充同一数据上的配对对照。
