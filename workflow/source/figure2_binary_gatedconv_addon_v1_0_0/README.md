# Figure 2 二分类训练与审稿修订代码

本目录只处理 Figure 2 的 PC/XNA 二分类。`pos` 表示正链，`neg` 表示负链；两种链各自训练二分类模型。标签 `0=PC`、`1=XNA`。本运行器只调用同目录的 `train.py`。

这一版需要在新的 read 隔离划分上从头训练。原始特征计算正确、CSV 与 metadata 逐行对应时，可以直接复用已提取的特征，不必重新提取原始电流。

## 1. 安装与放置文件

将整个代码包解压到同一个目录，保留 `train.py`、`IterDataset.py`、`model_benchmark.py`、`config_19.py`、`analyze_results.py` 和本运行器。

建议 Python >= 3.9；CPU 或 GPU 版 PyTorch 均可。GPU 服务器请先按实际 CUDA 环境安装合适的 PyTorch，再安装其余依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --help
python run_all_4_models.py --help
```

纯 PyTorch selective SSM 不需要安装 `mamba-ssm` 或编译官方 selective-scan CUDA 扩展。

## 2. 输入数据：使用已通过检查的 7:2:1 目录

默认数据目录是：

```text
/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721
```

可通过 `--data_dir` 改为实际目录。默认寻找 `xna_pc_6mer_features_pos_balanced_train.csv`、`xna_pc_6mer_features_pos_balanced_val.csv`、`xna_pc_6mer_features_pos_balanced_test.csv`，以及相同规则命名的 `neg` 三个 CSV；metadata 默认为各文件名去掉 `.csv` 后加 `_meta.tsv`，也接受 `.meta.tsv`。单次运行可以用 `--train_data`／`--val_data`／`--test_data` 和相应 `--train_meta`／`--val_meta`／`--test_meta` 显式指定。批量运行器使用标准目录命名。这些文件应来自新的均衡划分脚本，不能再使用旧的按窗口随机划分结果。

特征 CSV 的 11 列为：

```text
kmer,mean,std,median,dwell,quality,mismatch,insertion,deletion,signal,label
```

metadata 至少需保留可靠的 `read_id`、参考序列身份（如 `chrom=PC01/XNA01`）、方向、kmer 和 label。模板、测序 run、来源字段若有，应一并保留，便于后续分层统计和外部留出审计。

数据模块会核对 CSV/meta 对应关系、参考名与标签、标签范围、序列长度、数值和信号长度，以及 train/val/test 的 read ID 交集。审计失败就停止。数据缓存采用确定性预处理并保存原文件摘要；正式训练前确认划分脚本的 `summary` 已成功。

## 3. 统一输入与模型含义

* 所有模型都把参考序列的第 4 位替换为 N：Python 索引为 3。模型读到 ATCGN 五维编码，目标位点的信号、信号统计量及实际 basecalling 响应仍保留。
* 每个碱基的 raw signal 默认固定到 30 个采样点。过长时按原顺序均匀选取；过短时对称 padding。不插值，评价时也不随机采样。
* `dwell_divisor=60`、`quality_divisor=50` 来自旧批量命令的实际二次缩放参数。请核对之前提取时的缩放定义，防止重复缩放；它们是明确的配置常数，不是从所有数据拟合的归一化参数。
* 四种架构共享输入分支和默认读出：第 4 位的表示与全局 attention pooling 拼接，再分类。序列编码长度是 6 个碱基级 token。
* `litemamba_variant=selective_ssm` 是具有输入依赖参数与状态递推的轻量双向 SSM，采用纯 PyTorch 实现。它不是官方 Mamba CUDA 实现。
* `litemamba_variant=legacy_gatedconv` 保留旧门控卷积模块作结构对照；该变体不应称为 selective SSM。`readout=legacy_last_attention` 保留旧第 6 位加全局池化的读出作对照。

九种特征方案：

| 参数 | 保留的信息 |
|---|---|
| `full` | raw signal、context、statistics、responses 全部 |
| `raw_only` | 仅 raw signal |
| `context_only` | 仅第 4 位已遮蔽的序列背景 |
| `responses_only` | quality、mismatch、insertion、deletion |
| `statistics_only` | mean、std、median、dwell |
| `no_raw` | 除 raw signal 外全部 |
| `no_context` | 除序列背景外全部 |
| `no_responses` | 除 basecalling 响应外全部 |
| `no_statistics` | 除信号统计量外全部 |

禁用分支会在模型中屏蔽对应输入，方案写入运行配置。不要混用不同含义的消融名称。

## 4. 先跑一个试验，测试集暂不评估

下面是一条完整的正链 LiteMamba 试跑命令。训练更新只使用 train；调学习率、早停和 best checkpoint 选择只使用 val。

```bash
python train.py \
  --mode train \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --strand pos \
  --model_type litemamba \
  --seed 42 \
  --output_dir ./pilot_pos_litemamba_seed42 \
  --epochs 200 \
  --patience 30 \
  --lr 0.001 \
  --batch_size 256 \
  --workers 4 \
  --raw_signal_len 30 \
  --dwell_divisor 60 \
  --quality_divisor 50 \
  --accum_steps 1 \
  --device auto \
  --model_dim 96 \
  --num_blocks 4 \
  --state_dim 16 \
  --litemamba_variant selective_ssm \
  --readout target_attention \
  --feature_scheme full \
  --embedding_dtype float32 \
  --bootstrap 200 \
  --defer_test
```

检查数据审计、训练曲线、val 表现、预测导出与资源需求。这个阶段不要用测试集选择架构或参数。数据审计可读取 test 的身份／标签做完整性检查，但测试性能不参与训练选择。

## 5. 正式批量队列

确认输入和结构设置后，默认运行四架构 × 三 seeds × 两个方向 = **24 次训练**，只运行 `full` 主对比，不会默认展开全部消融。

```bash
python run_all_4_models.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir ./binary_main_comparison \
  --models bigru transformer convmixer litemamba \
  --seeds 42 43 44 \
  --strands pos neg \
  --feature_schemes full \
  --phase all
```

`--phase all` 先顺序训练全部实验，均传入 `--defer_test`；只有全部训练成功，才写入固定的 `evaluation_plan.json` 并统一评估测试集。结果聚合报告全部 seeds 的均值、标准差，不挑选测试分数最高的 seed。

如需先训练、之后集中评估，用完全相同的参数分别运行以下两个完整命令：

```bash
python run_all_4_models.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir ./binary_main_comparison \
  --models bigru transformer convmixer litemamba \
  --seeds 42 43 44 --strands pos neg --feature_schemes full --phase train

python run_all_4_models.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir ./binary_main_comparison \
  --models bigru transformer convmixer litemamba \
  --seeds 42 43 44 --strands pos neg --feature_schemes full --phase evaluate
```

`--dry_run` 只生成命令计划。默认遇到失败立即停止。`--continue_on_error` 允许继续训练其他实验，但只要有一个训练失败，整队测试仍不启动。

同一运行目录只有在 `queue_requested_config.json` 完全相同、`results.json` 状态为 `train_complete`／`complete` 且 best checkpoint 存在时才跳过。改变模型、数据路径、缩放、seed、源代码等应使用新输出目录。中断后配置不变且存在 `last_checkpoint.pt` 时，可在原批量命令增加 `--resume_failed`。不要手工改状态文件伪装完成。

单次训练中断的恢复示例：

```bash
python train.py \
  --mode train --output_dir ./pilot_pos_litemamba_seed42 \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --strand pos --model_type litemamba --seed 42 \
  --epochs 200 --patience 30 --lr 0.001 --batch_size 256 --workers 4 \
  --raw_signal_len 30 --dwell_divisor 60 --quality_divisor 50 --accum_steps 1 \
  --device auto --model_dim 96 --num_blocks 4 --state_dim 16 \
  --litemamba_variant selective_ssm --readout target_attention --feature_scheme full \
  --embedding_dtype float32 --bootstrap 200 --defer_test --resume
```

## 6. 需要时再展开消融

以下队列只运行 LiteMamba 的四项单类特征对照；这是用户明确要求后才展开的额外训练。

```bash
python run_all_4_models.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir ./binary_feature_only \
  --models litemamba --seeds 42 43 44 --strands pos neg \
  --feature_schemes raw_only context_only responses_only statistics_only --phase all
```

结构对照分别用新的输出目录和 `--litemamba_variant legacy_gatedconv` 或 `--readout legacy_last_attention`，同时保持其他训练设置相同。先固定需要比较的队列，再统一测试；模型是否改进以实际结果为准。

## 7. 输出及复用方式

每次实验保留实际解析的配置、环境与版本、源代码快照／摘要、输入数据 manifest 与审计、完整训练 checkpoint（含优化器、调度器和 RNG）、best checkpoint 和可移植 state dict、每 epoch 指标和资源测量。

默认还保存每个 epoch 的紧凑验证集输出 `epoch_predictions/val_epoch_XXXX.npz`，包含 float32 logits、labels 和 sample_index；通过 val manifest 可追溯对应的 read、motif 和参考信息。因此，后续分析各 motif／参考指标随 epoch 的变化无需重新训练。`--no_epoch_logits` 可显式关闭该留档；`--save_epoch_predictions` 可额外保存逐 epoch 的完整 CSV。

`--defer_test` 完成后写 `status=train_complete`，保存 best checkpoint 对 train/val 的预测；最终评估完成后为 `status=complete`。每个 split 的输出位于运行目录的 `train/`、`val/`、`test/`：

| 文件／目录 | 用途 |
|---|---|
| `predictions.csv.gz` | 逐样本身份、read／参考／motif metadata、真实标签、预测、logits、原始概率及校准概率 |
| `logits.npy`、`labels.npy` | 无需重训即可重算指标与图 |
| `embeddings.npy`、`attention.npy` | 表征与注意力；默认 embedding 存 float32，显式 `--embedding_dtype float16` 可节省空间，`--no_embeddings` 可关闭 |
| `analysis/` | 总体、参考对、单参考、motif、来源、链方向、run 的可用分层指标与区间、混淆矩阵、校准、阈值及阳性比例情景分析 |
| `export_complete.json` | 当前导出完成标记 |

不具有对应 metadata 的分层不能凭空恢复。单个 PC 或 XNA 参考只有一个类别，不能把其 precision 当作完整二分类表现；优先看 `PCxx/XNAxx` 参考对和两类齐全的分组。

温度缩放的温度 T 只由 val 拟合，之后固定应用到 test 和外部数据。阈值分析保留概率分布与覆盖率，不能根据 test 曲线挑阈值后再把同一 test 当作独立证据。阳性比例情景基于条件性能保持稳定的假设，不等于真正进行了低丰度混合实验。

批量目录额外保存 `queue_manifest.json`、`commands.jsonl`、逐 run 的 stdout 日志、`evaluation_plan.json` 与 `aggregate/`。测试每个预定模型／seed 评估一次；后续改图、重算置信区间和汇总应读取已保存预测，通过 `analyze_results.py` 处理，不要重新训练或借重复测试挑最优结果。

每个 run 的源代码快照与 SHA256 用于核对实际实现，数据 manifest 保存输入内容摘要和样本身份。队列配置还记录关键源文件摘要和输入文件状态；`evaluation_plan.json` 固定全部实验配置、源文件摘要及各 best checkpoint 的内容摘要。恢复和复用必须通过一致性检查，避免改动源码或输入后沿用旧完成状态。计划固定后不能通过编辑 JSON 把新增设置伪装成原测试计划。

下面三个命令使用 `RUN_DIR` 环境变量指定一条已完成实验的实际目录。先把变量值改成真实路径，再整体执行。第一条重新计算已保存预测的分析，第二条只重绘已有分析表，第三条汇总指定运行目录；多个 seeds 时在 `--runs` 后逐一列出所有目录。

```bash
RUN_DIR=/absolute/path/to/completed_run
python analyze_results.py analyze --predictions "$RUN_DIR/test/predictions.csv.gz" --output_dir "$RUN_DIR/test/analysis" --bootstrap 200 --seed 42
python analyze_results.py plots --output_dir "$RUN_DIR/test/analysis"
python analyze_results.py aggregate --runs "$RUN_DIR" --output_dir ./recomputed_aggregate --split test
```

## 8. 外部模板或 run 留出评价

外部清单为 CSV，列名为：

```text
name,features,metadata,require_template_disjoint,require_run_disjoint,strand
```

每行提供一个外部集合的实际路径；两个 `require_...` 标记取 true/false。`strand` 为可选列，取 `pos`／`neg`，用于把清单中的外部集合分配给相应方向的模型；同一个清单可同时包含两类方向。trainer 还会核验 metadata 中的链方向与当前模型一致，避免正负链模型误用同一外部集合。只在已取得对应外部数据时加入清单。外部 read 不得与训练／验证共享；要求模板或 run 不重叠时，相关 metadata 必须可验证，未知不能按独立处理。

使用固定 checkpoint 做外部推理，不训练：

```bash
python train.py \
  --mode evaluate \
  --output_dir ./pilot_pos_litemamba_seed42 \
  --external_manifest /absolute/path/external_manifest.csv
```

上述命令也会评估该运行尚未导出的内部 test，必须在模型方案固定后使用。外部清单中的数据需与模型链方向和 PC/XNA 标签定义相容。源文件保存于同一 run 中，结果应明确区分内部 read 留出、模板留出与测序 run 留出。

## 9. 可选的来源／链方向诊断

`train_metadata_probe.py` 使用固定的现有 read 划分，合并 pos/neg，并用线性 `SGDClassifier(log_loss)` 预测 metadata 中的链方向、来源或测序 run。这是诊断基线，不参加 LiteMamba 的 PC/XNA 主任务性能排名。

```bash
python train_metadata_probe.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir ./diagnostic_strand_full \
  --target strand --feature_scheme full --seeds 42 43 44 --phase all
```

`--target source` 默认要求可靠的 `source_dataset`，`--target run` 默认要求真实 `run_id`；未知或只有一种来源／run 时会失败。若原 metadata 中有另一个经过核实的字段，可显式用 `--target_column` 指定，代码不会从文件名虚构来源。

可选输入为 `full/raw/context/responses/statistics`；context 包含第 4 位遮蔽。跨六个输入文件统一审计 read，同一 read 在同一集合只保留第一条记录，标签／参考／诊断目标矛盾或跨集合就失败。StandardScaler 只拟合 train；每个 seed 只用 val 选择迭代次数。所有 seeds 完成训练和 val 后统一固定模型摘要，才开始 test。输出包含 scaler、模型 joblib、类名与索引映射、逐样本概率、balanced accuracy、macro-F1、混淆矩阵和输入／源码摘要；`environment.json` 保存 Python、numpy、scikit-learn、joblib、torch 版本及数据预处理版本。

`--phase train` 仅完成拟合和 val；之后可用 `python train_metadata_probe.py --output_dir ./diagnostic_strand_full --phase evaluate` 加载固定模型测试。只加载自己信任来源的 joblib 文件。诊断模型预测率低不能证明没有混杂；预测率高提示输入中包含与该 metadata 相关的可分辨信号，需要进一步用匹配数据或独立留出实验验证。

## 10. 模板／run 留出划分入口

`prepare_group_holdout.py` 可以把现有六个特征／metadata 文件重新组织成模板或 run 留出的标准输入目录，保留全部样本，让同一个 group 完整进入同一集合。7:2:1 是 group 层面的近似比例，实际样本／read 比例会报告。模板或 run 组数量及双类覆盖不足时会失败；run 身份未知时不能构造 run 留出。

以下命令先建立模板留出数据，再使用新的输出目录训练。划分 seed 应预先固定，不能根据 test 表现挑选模板划分。

```bash
python prepare_group_holdout.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_read_split_721 \
  --output_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_template_holdout_721 \
  --group_by template --strands pos neg --seed 42

python run_all_4_models.py \
  --data_dir /path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/binary_template_holdout_721 \
  --output_dir ./binary_template_holdout_models \
  --models bigru transformer convmixer litemamba \
  --seeds 42 43 44 --strands pos neg --feature_schemes full --phase all
```

run 留出时改用 `--group_by run`，并为数据和实验设置新的输出目录。重新组织划分只重用原始特征，不会自动产生新的独立测序 run。准备脚本的输出目录不能已存在，也不能与输入目录互相嵌套。

## 11. 审稿回复范围

见 `reviewer_coverage.md`。代码为可复现训练、诊断和后处理提供工具；它不能替代尚未采集的外部数据、诊断与模板／run 留出的实际运行、公开数据与权重、文献比较及正文修改。目前未在用户真实数据上完成训练，不能填写未经测量的准确率或声称已经解决全部审稿意见。
