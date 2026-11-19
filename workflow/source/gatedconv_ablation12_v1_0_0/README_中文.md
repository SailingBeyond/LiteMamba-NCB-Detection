# 最终 GatedConv：12次去除组消融 v1.0.0

新增 `no_raw`、`no_context`、`no_responses`、`no_statistics` 各3个种子（42、2024、2025），共12次训练。维持最终GatedConv、第四位目标读出及第四位序列置N，所有新增任务从各自种子的随机初始化开始。已有三个完整输入GatedConv只用于只读比较，不重新训练，也不作为初始化。

本包完成12次训练、固定内部测试、汇总和核心结果打包。**不会自动启动Figure4外部预测。** 后续可直接复用这12个检查点，不必重新训练。

## 1. 直接运行

使用原来的 `pytorch2.0.1` 环境，解压到独立新目录：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip gatedconv_ablation12_v1_0_0.zip
cd gatedconv_ablation12_v1_0_0

# 前台核验环境、输入SHA、完整输入对照和空间；不训练。
bash run_ablation12.sh --check_only

# 预检通过后后台执行全部12项。
bash run_ablation12.sh
```

预检会读取六份输入及已有三个full检查点进行内容核验，出现 `[VERIFY SHA]` 时仍在工作。脚本先打印本次独立启动日志；后台返回PID仅表示已提交请求，是否通过预检以日志为准。

默认正式队列日志：

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/gatedconv_ablation12_v1_0_0
tail -F "$RESULTS/logs/queue_latest.log"
```

`queue_latest.log`在正式队列启动后生成；此前查看启动器打印的 `launcher_logs/launch_*.log`。

## 2. 四种去除方式

| 方案 | 去掉的输入 | 保留的输入 |
|---|---|---|
| `no_raw` | 原始信号片段CNN分支 | 序列上下文、均值/标准差/中位数/持续时间、测序响应 |
| `no_context` | 显式ACTGN序列编码（列0:5） | 原始信号、统计量、测序响应 |
| `no_responses` | quality/mismatch/insertion/deletion（列9:13） | 原始信号、序列上下文、统计量 |
| `no_statistics` | mean/std/median/dwell（列5:9） | 原始信号、序列上下文、测序响应 |

这些方案沿用已冻结模型代码的输入屏蔽，发生在线性投影和归一化之前。`no_raw`停用原始信号编码器，并禁止该分支的BatchNorm更新；其余三组继续使用原始信号分支。

**解释时需要区分：**`no_raw`仍含电流统计摘要，因此它检验的是波形分支在其他输入存在时的增量作用；不能把较小下降解读为电流信息无用。`no_context`去掉显式编码，不会消除电流中的序列信息；`no_statistics`去掉显式四项统计量，不会消除波形中的幅值和长度信息。

网络宽度96、4个门控卷积块、第四位目标注意力读出、分类头、信号长度30和全部训练参数保持一致。总模型参数仍为301,831；`no_raw`的原始分支被停用/冻结，可训练参数量相应减少。每个方案重新训练，不是对Full权重临时遮挡后测试。

目录中的 `litemamba`是兼容引擎键，实际结构为 `litemamba_variant=legacy_gatedconv`、`readout=target_attention`，没有选择性状态空间递推。

## 3. 默认数据、缓存和结果

数据目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721
```

核验 `train.csv`、`validation.csv`、`test.csv`和对应三个 `_meta.tsv`，内容必须与已经完成的Full模型一致。样本数分别为1,751,316、500,376、250,188。程序不重新划分、不重新下采样，也不重新提取特征。

默认输出为上述数据目录下：

```text
sixclass_auto_v1_0_4_results/gatedconv_ablation12_v1_0_0
```

共享只读特征缓存沿用：

```text
sixclass_auto_v1_0_4_results/feature_cache
```

12个新任务各自输出到：

```text
OUTPUT_DIR/<no_raw|no_context|no_responses|no_statistics>/litemamba/seed_<42|2024|2025>
```

完整输入对照默认来自：

- `litemamba_structure_seed42_v1_0_0/full/gatedconv_target/seed_42`
- `litemamba_structure_confirm3_v1_1_0/full/gatedconv_target/seed_2024`和`seed_2025`
- 独立内部test记录：`gatedconv_internal_test_v1_0_0/runs/gatedconv_target/seed_<seed>`

三个Full必须全部通过检查点、数据、结构、validation温度、val-loss选模及内部test身份核验；缺少某种子时明确停止。不会拿旧SSM Full或验证集指标替代。

自定义目录时：

```bash
bash run_ablation12.sh \
  --data_dir /实际路径/sixclass_overlap_read_split_721 \
  --output_dir /有足够空间的新路径/gatedconv_ablation12_v1_0_0 \
  --cache_dir /实际路径/feature_cache \
  --baseline_dir /实际路径/litemamba_structure_confirm3_v1_1_0 \
  --baseline_test_dir /实际路径/gatedconv_internal_test_v1_0_0
```

旧结构计划引用的seed42路径仍需可访问。输入内容可以按参数搬迁，不能改变其SHA。新输出与代码、旧训练、旧Figure4、基线和缓存目录互不包含。

## 4. 优先顺序和训练协议

固定任务顺序：先声明 `no_raw`的42、2024、2025，然后依次为 `no_context`、`no_responses`、`no_statistics`。资源调度优先尝试前面的任务；若后面的任务更适合当前可用资源，也可能先启动，不强制所有进程等待某一个任务。

执行顺序：

1. 核验数据、三个Full对照、冻结源码及空间，串行准备共享缓存和数据审计。
2. 四种去除方式分别实测8次AdamW更新与单批验证，估计资源；不做test推理。
3. 根据实测预算并行完成全部12项权重训练。
4. 全部权重成功后，串行导出train/val，并仅在validation拟合各自温度。
5. 冻结全部12个验证集选定的检查点SHA。
6. 串行完成固定内部test，随后与三个Full汇总和打包。任一训练失败时不进入整队test。

固定科学设置：batch256、最多200轮、patience30、lr0.001、AdamW weight_decay0.0001、label_smoothing0、梯度裁剪1、accum_steps1，以 `val_loss`同时用于选模和训练控制。保存原有AMP策略、随机种子和模型初始化方式。命令行只允许改路径、资源预算、并发等运行设置，不提供改变科学参数的入口。

内部test沿用已冻结的固定集合，属于后续预设对照评估；不额外声称是一套从未看过的新盲测。test指标不进入学习率调度、early stopping或检查点选择。

## 5. 并行资源

默认并发硬上限4，workers0，每进程CPU线程4。程序按实测显存/RAM/CPU及未兑现的进程预约动态准入，实际并发可能少于4。默认保留GPU 2 GiB、RAM 3 GiB和10%比例余量，实测需求乘1.35，并保留训练RAM额外余量1 GiB。已确认OOM时有限重试并降低并发，不改变batch或梯度累积。

```bash
# 首次运行或恢复时都可以降低并发。
bash run_ablation12.sh --max_parallel 2
```

默认 `--device auto`要求当前环境有可用CUDA，不会静默开始CPU长训练；仅合成验证可以显式 `--device cpu`。GPU UUID探测沿用已成功运行的修复版本。本轮不会重复正式架构效率benchmark；资源探测和并发总耗时不能当作公平效率比较。

## 6. 磁盘和恢复

保持每轮验证数组、完整最终导出和检查点时，按最多200轮计算，**12项新结果的保守新增预算约116.1 GiB，另保留8 GiB**。如果共享缓存都缺失，另预算约10.1 GiB。实际启动按未完成的轮次、缺失导出及缓存重新计算；完成任务不会重复计入。

输出和缓存位于同一磁盘时合并预算，不重复扣保留量；不同磁盘分别检查。运行期间持续监测剩余空间，触及保留线时停止本队列。估算不是对最终占用的承诺，也不能防止其他进程抢占空间；程序不自动删除旧结果。

清理空间或处理错误后，在同一包中恢复：

```bash
bash run_ablation12.sh --resume

# 同时降低并发。
bash run_ablation12.sh --resume --max_parallel 2
```

保留首次使用的自定义路径参数。已经完整的任务跳过；中断任务从 `last_checkpoint.pt`恢复模型、Adam、调度器、GradScaler、随机数和历史。未完整提交的那一轮重新执行，已经提交的轮次保留。若首个检查点之前失败，仅在引擎核实中断记录后才允许以同一种子重新初始化。

不要删除任务目录、修改冻结输入或替换运行中的代码。输出锁防止重复启动；旧Full来源持有共享锁并核对内容。程序只终止自己启动的队列及子进程。

## 7. 保存的信息

每个任务保留：

- 请求/实际配置、模型配置、结构文本、环境、源码快照、数据指纹和审计、参数量。
- 每轮train/val损失、指标、学习率、选模状态和验证logits/labels数组。
- 梯度范数有限值统计、NaN/Inf次数、更新尝试/成功及AMP跳步计数。跳步来自GradScaler缩放变化推断，相关说明保留。
- 最佳及最后检查点、优化器/调度器/GradScaler/RNG状态、周期检查点。
- 最佳检查点的train/val/test逐窗口logits、原始/校准六类概率、标签、预测、float32嵌入、注意力与逐行元数据。
- 每类与各可用来源/参考/motif分层的计数、混淆矩阵、precision/recall/specificity/FPR/FDR、校准、ROC/PR与分析图。

`ablation_summary/`另外保存15组（12新增+3Full）的逐种子/均值样本SD、同种子Full减去除组差值、逐类比较、学习曲线、数值诊断及资源计时表。差值的率单位是百分点，NLL/Brier使用原指标单位。无分母项留空，不用0伪装缺失。SD描述训练重复变化，不是read级置信区间。

只重做汇总，不训练、不推理：

```bash
bash run_ablation12.sh --summary_only
```

只检查计划和预算而不读取大数据或启动训练：

```bash
bash run_ablation12.sh --dry_run
```

## 8. 完成后上传什么

看到以下完成信息才代表12项训练和内部评估全部完成：

```text
[FULL PASS] 12次去除组训练、内部冻结测试与汇总完成；Figure4外测尚未启动。
[RESULT PACKAGE] .../gatedconv_ablation12_results_core_v1_0_0.tar.gz
```

**上传 `gatedconv_ablation12_results_core_v1_0_0.tar.gz` 即可继续分析。**

核心包先收齐12项新训练和3个Full的配置、历史、校准、val/test计数及整队冻结记录，源码按SHA去重，再收录单独限额的motif大表。不会因总量上限遗漏最后一个任务。缺核心记录时明确失败，不发布不完整结果包；归档成员逐个核对SHA后原子发布。

权重、逐窗口概率/嵌入/注意力和大型明细仍保留在服务器，不塞进上传包。它们足够后续重画和Figure4预测，无需重训。可以单独重新打包：

```bash
python3 summarize_ablation.py --output_dir "$RESULTS" --package_only
```

## 9. 本地验证与适用范围

本地核验详见 `VALIDATION.md`。测试包括真实96维4块模型的屏蔽/梯度/保留输入响应，以及注入磁盘满后与未中断训练完全一致的恢复验证；服务器CUDA显存与实际并发在启动时实测。本包沿用现有环境，不要求升级PyTorch/CUDA。

PyTorch官方说明，不同版本、平台或CPU/GPU之间即使种子相同也不能保证完全一致。因此保留服务器环境、种子和源码记录，而不把本地CPU测试当作GPU逐位复现证明：
https://docs.pytorch.org/docs/2.0/notes/randomness.html

本轮只回答去掉某组输入后、其余输入仍保留时的效果；单输入实验与去除组实验是互补对照。它不单独排除来源/链向混杂，也不自动完成审稿人其他实验要求。四种去除方式及三个种子预先固定，不根据内部test或Figure4成绩选取报告子集。
