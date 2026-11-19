# Figure 4 外部六分类验证 v1.0.0

本程序读取已经完成的 `sixclass_training_auto_v1_0_4` 训练结果，不重新训练。默认对 LiteMamba、Transformer、ConvMixer、BiGRU 的全部 5 个随机种子（42、2024、2025、2026、2027）进行外部预测，共 20 组权重；随后离线统计并生成 Figure 4A–D 的 CSV、PNG 和 PDF。模型间公开比较保留全部预定种子，不根据外部测试结果挑选种子、模型、阈值或架构。

当前仅支持新版第四位掩码为 N 的六分类权重，类别顺序固定为 `0=A, 1=T, 2=C, 3=G, 4=X, 5=Y`。旧版五分类 5X/5Y 权重不兼容；不能通过删除六分类模型的一个输出冒充独立训练的五分类实验。

## 1. 在服务器上一键运行

解压后进入本目录，继续使用原来训练成功的 PyTorch 环境。例如：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code/figure4_revision_v1_0_0
python3 -c 'import torch,numpy,pandas,matplotlib; print(torch.__version__,numpy.__version__)'
bash run_figure4_auto.sh
```

`run_figure4_auto.sh` 已经使用 `nohup` 在后台启动，不需要再额外添加 `nohup` 或 `&`。脚本会打印本次独立日志的绝对路径；只有成功取得输出目录的队列锁后，才更新正式 PID 和最新日志。提交 PID 不等于已经完成。

默认训练目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/main_20
```

默认输出目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_external_v1_0_0
```

查看日志及磁盘：

```bash
OUT=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_external_v1_0_0
tail -F "$OUT/logs/figure4.log"
df -h "$OUT"
df -i "$OUT"
```

日志最后出现 `[FULL PASS] Figure 4 all 阶段全部完成`，且 `queue_state.json` 的 `status` 为 `complete`，才表示本次完整流程结束。每个阶段、每个模型的详细日志路径都会写入主日志。

如果需要更换输入或输出位置：

```bash
bash run_figure4_auto.sh \
  --training_results /实际路径/sixclass_auto_v1_0_4_results/main_20 \
  --output_root /有足够空间的磁盘/figure4_external_v1_0_0 \
  --max_parallel 4
```

`--training_results` 也接受包含 `main_20` 的上一层结果目录。输入训练队列必须已完整结束，checkpoint、结果记录、冻结测试计划和 validation 校准记录必须相互吻合；程序会检查模型源代码快照，避免用错误架构加载权重。请保留原训练结果的目录结构和相关 JSON、数据库及 checkpoint。

## 2. 中断、空间不足与恢复

先清理或扩容**对应输出分区**，确认容量和 inode 均充足；随后在同一代码目录使用原参数，并增加 `--resume`：

```bash
bash run_figure4_auto.sh --resume
```

如果首次运行指定过路径、batch、CPU 或导出选项，恢复时也需保持相同设置：

```bash
bash run_figure4_auto.sh \
  --training_results /实际路径/sixclass_auto_v1_0_4_results/main_20 \
  --output_root /有足够空间的磁盘/figure4_external_v1_0_0 \
  --max_parallel 4 --batch_size 256 --cpu_threads 4 --workers 0 \
  --resume
```

恢复以“模型/seed × 外部文件”为单位：已完成且大小、SHA256、冻结计划与导出设置均匹配的结果跳过；当前未完成的外部文件从头导出，不从其某个 batch 接续。完整缓存先做内容校验；损坏缓存重新生成，内容相同不会改变科学评估计划。`.building`、`.partial` 是未完成的临时目录，不是完整结果。新缓存完整校验并替换成功后，程序会删除本次产生的损坏派生缓存备份；替换失败则保留旧缓存，不删除用户源文件，也不扫描清理历史无关目录。重建期间需同时容纳旧缓存和新缓存。

队列使用文件锁并核实存活子进程，避免重复启动同一输出目录。若提示仍有旧进程，先根据提示 PID 检查实际运行状态，不要同时启动第二套队列。出现资源 OOM 时会有限重试并降低并行上限；其他错误保留结果并报告对应日志，不会无限重跑。恢复使用旧容量、已降档上限和本次上限中的较小值。

冻结后改变外部 CSV/metadata、checkpoint、模型代码、模型/seed 列表或不兼容的预测设置，需要使用**新的输出目录**，不能混合到原结果中。`--resume` 不会把不同实验强行合并。

## 3. 默认外部数据与显式清单

默认搜索以下目录中的 CSV（只搜索目录本层）：

| 数据组 | 实际搜索目录 |
|---|---|
| XNA1–12 / 对照 | `/path/to/staged-storage2/analysis/paper/xna_pc_6mer_features_for_model_test_X_to_T/ATCGXY_6class` |
| XNA13–16 / 对照 | `/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/fast5_single_guppy/xna_pc_13_16_6` |
| XNA17–20 | `/path/to/staged-storage2/analysis/paper/POC-XNA-17_20/fast5_single_guppy/xna_6mer_features_17_20_split_by_ref_strand` |

路径可分别通过 `--dir_01_12_root`、`--dir_13_16_6`、`--dir_17_20_6` 修改。第一项传父目录，程序会自动追加 `ATCGXY_6class`。

自动识别的文件名结尾为 `_<reference>_pos.csv`、`_<reference>_neg.csv`，或 `_pos_<reference>.csv`、`_neg_<reference>.csv`。`reference` 支持 `XNA01`/`XNA1`、`DNA01`、`PC01` 等；DNA/PC 表示 control，XNA 表示 target。别名 `84Ds4-AA/AB/AC/AD` 依次对应 XNA17/18/19/20 的 target。文件名无法可靠解析时会报错，不能凭空猜测来源。

默认严格要求 XNA1–16 各自同时存在 control 和 target，XNA17–20 各自至少存在 target；这项检查是模板/角色覆盖，不证明每种实际测序方向都齐全。局部调试须显式添加 `--allow_partial_templates`，缺少模板会保留为缺失/NA，不补成零。正式 Figure 4 应使用完整预定集合。

若文件名不同或需要提供来源身份，使用随包示例清单，传入 CSV 或 TSV：

```bash
bash run_figure4_auto.sh --data_manifest /实际路径/figure4_inputs.tsv
```

| 清单字段 | 要求及含义 |
|---|---|
| `feature_csv` | 必需；原始 11 列特征 CSV，路径可相对清单文件所在目录 |
| `template` | 必需；配对设计编号，如 `XNA1`。对应 DNA1 的 control 也写 `XNA1` |
| `role` | 必需；`control` 或 `target` |
| `metadata_tsv` | 可选；同次提取且逐行对应的 metadata；缺失时留空 |
| `reference` | 原始参考名称；未知留空/`unknown`，不是人为生成的新参考身份 |
| `file_strand` | 文件分组，通常 `pos`/`neg`/`unknown`；不等于实际 BAM 的 `+/-` |
| `cohort` | 可省略；由模板编号推导 `XNA1-12`、`XNA13-16`、`XNA17-20`；提供时必须一致 |
| `source_dataset`、`run_id` | 可选；真实数据来源和测序 run。未知保持未知 |
| `template_group`、`pair_id` | 可选；真实模板分组和配对身份，不应仅为了得到分组统计而编造 |

同一特征文件不能重复列入。metadata 自动匹配的候选为 `stem_meta.tsv`、`filename.csv.meta.tsv`、`stem.meta.tsv`；发现多个候选会要求通过清单明确指定。没有 metadata 可以运行窗口级分析，但无法据此验证 read/run 独立性。

metadata 有表头，至少包含 `read_id`，且与特征 CSV 逐行对应。支持的其他字段包括 `original_read_id`、`read_id_kind`、`kmer`、`label`、`chrom`、`reference`、`bam_strand`、`position`/`pos`，以及清单中的来源/分组字段。已提供的标签和 kmer 必须与特征对应；双方第四位均为 A/T/C/G 时还核对完整原始 kmer，任一侧第四位为 N 时核对掩码后的 motif。能解析的 metadata 参考名若与清单 template/role 冲突会拒绝；无法解析的真实参考名照实保存，不臆测其模板身份。

真实 ID 才可用于 read 审计。`read_id_kind` 为 unknown、unavailable、proxy、synthetic、row 等，或 `positive-row:*` 这类代理编号，会保持未知，不能升级为真实 read。`sample_id` 是文件 SHA256 与行号组成的可追溯行身份，**不是**分子/read 身份。原 metadata 全部保存在共享缓存中。

## 4. 输入处理与冻结规则

原始特征 CSV 可有或没有表头，列顺序为：

```text
kmer,mean,std,median,dwell,quality,mismatch,insertion,deletion,signal,label
```

每个 kmer 为大写 6mer（A/C/T/G/N）；8 项逐位标量各有 6 个有限数值。真值只读取 CSV 的最后一列，接受 `1`、`1.0` 这种整数表示，拒绝 `1.2` 或超出 0–5 的值。control 不能含 X/Y 真值；天然类别在未掩码时的第四位必须与标签吻合。文件名 pos/neg 不会覆盖 CSV 真值。

所有输入在缓存中把**第 4 位（索引 3）改为 N**；原 CSV 不改写，输出保留 `original_kmer` 和掩码 `motif`。one-hot 顺序为 A/C/T/G/N；类别标签顺序为 A/T/C/G/X/Y，两者不要混淆。dwell、quality 缩放和信号长度直接沿用每个训练配置，所有待比较权重的预处理必须一致。原始信号按训练版规则确定性零填充或等距索引采样并保留 4 位小数，不再使用旧脚本的随机 `random.sample`。

模型采用训练时冻结的最佳 checkpoint 和仅由 validation 拟合的正温度。外部集不拟合温度、不调阈值、不更新参数。程序检查外部文件是否与内部 train/val/test 文件完全相同；若原训练 `read_membership.sqlite` 与外部真实 read ID 均可用，还检查 read 重叠，发现重叠直接停止。

## 5. 并行预测与磁盘预算

程序先串行建立共享只读 mmap 缓存，再用实际模型 forward 实测资源；不同模型配置分别探测。探测包括 warmup、真实 batch 计算和输出数组处理，记录 CPU RAM、GPU 峰值、环境和参数量等。它用于安排并发容量，**不根据探测准确率或 Figure 4 指标选择模型**。

默认并行硬上限为 4；实际同时运行数量取决于可核实的 GPU UUID、显存余量、主存、CPU 配额和安全系数。GPU 资源不足时等待，CPU 模式默认最多 2 个任务；不保证 4 个任务能同时运行，也不保证并行一定比串行快。共享输入缓存减少重复预处理；每个进程只流式处理一个文件的 batch。`--workers` 必须为 0，此处不需要 DataLoader 子进程。

GPU UUID 必须与遥测设备对应，不能直接猜测逻辑 CUDA 编号与物理编号。若 UUID 检查失败，按报错检查驱动或使用服务器上真实 GPU UUID 设置 `CUDA_VISIBLE_DEVICES=GPU-...`，再恢复。未核实的 MIG 切片不支持本调度器。

空间需要按**所有外部窗口 × 20 次模型/seed 导出**估算。信号长度 30 时，共享 tensor 缓存约为 1,040 字节/窗口，另有 metadata 文本；预检为 metadata 额外估计约 1,800 字节/窗口。默认每组预测基础预算 1,536 字节/窗口，另外加入 embedding 和 attention 预算。若 `model_dim=96`，默认 float16 embedding 加 384 字节/窗口，attention 加 24 字节/窗口；100 万个外部窗口、20 组预测加缓存的规划量约 39 GiB，此外还需要默认 5 GiB 余量、离线分析的临时 SQLite/区间数据、日志和历史分析目录。

这些是预检估算，不是容量上界或压缩率保证。长文件名、额外 metadata、异常备份、重复离线分析及其他进程写盘都会增加占用。程序在准备/入队前检查容量并在预测队列中继续检查；提前停止不能完全防止并发写盘造成的 ENOSPC。默认保存 gzip 逐窗预测、logits、embedding 和 attention，是为了以后统计/画图复用，不保存每层全部隐藏状态。

若初次启动前确定不需要 embedding/attention，可用 `--no_embeddings --no_attention` 节约空间；这不会删除逐窗预测和 logits。已经开始的原输出目录不能通过恢复时改变这些选项混用结果。若需要更高精度的 embedding，用新输出目录设置 `--embedding_dtype float32`。

## 6. 无需重新预测的离线分析与重绘

完成预测后，下面的分析程序只读取已保存结果，不加载模型、不调用 GPU：

```bash
python3 figure4_analyze.py --output_root "$OUT" --bootstrap 200 --analysis_seed 42
python3 plot_figure4_final_finetune.py \
  --data_dir "$OUT/analysis_data" --out_dir "$OUT/figures" \
  --formats png pdf --dpi 300
```

只重绘图用第二条命令即可。`--formats` 后用空格分隔格式。也可通过队列入口运行 `--phase analyze`（分析并画图）或 `--phase plot`（只画图）。离线重分析使用新的 `.analysis-working-*` 临时目录；完整生成后才替换正式目录，原完整分析保存在 `analysis_data.previous_*`。反复分析前应考虑这些备份的容量。

默认 bootstrap=200 便于先完整跑通。最终论文如使用 bootstrap 区间，可在预测完成后离线增加到 2,000 次再重绘，无需模型重新推理；这是一项分析建议，不是声称审稿人明确要求 2,000 次。增加次数无法补救缺失的真实 read 身份，身份不完整仍不会产生 read 聚类区间。

```bash
python3 figure4_analyze.py --output_root "$OUT" --bootstrap 2000 --analysis_seed 42
python3 plot_figure4_final_finetune.py \
  --data_dir "$OUT/analysis_data" --out_dir "$OUT/figures" \
  --formats png pdf --dpi 300
```

默认 Figure 4A 为 LiteMamba 各模板天然对照的 X/Y 误报热图；B 为 XNA1–16 的 X/Y 正确类别恢复率；C 为 XNA17–20 的残余误差及分解；D 为四模型的透明外部验证指标对比。A–C 使用全部预定 LiteMamba 种子，D 使用全部预定模型/种子。图中的种子误差条为 seed 间 SD，不是 read 聚类置信区间；没有真实身份或缺少模板的数据不会生成虚假的区间或零值。

## 7. 输出文件和关键字段

| 路径 | 用途 |
|---|---|
| `evaluation_plan.json` | 冻结模型、checkpoint/源代码/输入 hash、温度、预处理、模板覆盖、评估计划 ID |
| `discovered_inputs.csv` | 实际读取的文件、角色、样本数和可用真实 read 数 |
| `external_identity_audit.json`、`external_read_audit.sqlite` | 可用真实 read 与内部划分重叠审计；身份限制如实记录 |
| `cache/<dataset_id>/` | 共享 `merged.npy`、`raw.npy`、`labels.npy`、完整 `metadata.jsonl`；`cache_manifest.json` 保存每个缓存文件大小/mtime/hash |
| `predictions/full/<model>/seed_<seed>/<dataset_id>/` | 单个模型、种子、外部文件的完整预测和完成记录 |
| `queue_state.json`、`commands.jsonl`、`resource_events.jsonl` | 队列状态、实际命令、调度与失败事件 |
| `resource_profiles/`、`logs/` | 串行资源探测、参数规模、环境、各任务独立日志 |
| `analysis_data/` | 全部统计、Figure 4 绘图输入、覆盖及审稿证据范围记录 |
| `figures/` | Figure 4A–D 的 PNG/PDF 及绘图追溯记录 |

每个预测文件夹默认保存：

- `predictions.csv.gz`：逐窗口身份、真值及全部 6 类 logits、原概率和校准概率。
- `logits.npy`：`N×6` float32；`labels.npy`：`N` int64。
- `embeddings.npy`：`N×(2×model_dim)`，默认 float16；`attention.npy`：逐窗口读出注意力，float32。
- `completion.json`：完成状态、样本数、冻结身份、全部导出文件大小/hash、环境和计时；其所属模型/seed 还有 `run_complete.json`。

逐窗 CSV 的类别概率字段为 `logit_0..5`、`prob_0..5`、`calibrated_prob_0..5`；`prediction` 为类别整数，`confidence`/`calibrated_confidence` 为最大概率。身份字段包括 `sample_id,source_file,source_row,source_sha256,read_id,read_id_kind,original_kmer,motif,reference,template,cohort,role,file_strand,bam_strand,template_group,pair_id,run_id,source_dataset,position,reference_identity_basis`，另有 `label,true_class,model,seed,feature_scheme`。预测 CSV 的 `run_id` 是样本测序 run，训练任务身份在目录和完成记录中，二者不要混淆。

计时分别记录 forward（包含 batch 传输）和端到端导出（含 CSV/array 写入、hash）耗时、吞吐及 GPU 峰值；资源探测还记录包含 warmup 和传输的串行吞吐。它们依赖 batch、并发负载、磁盘和硬件，不能直接宣称为标准化部署单样本 latency；不要用并发导出的耗时与另一篇论文不同硬件上的数字直接比较。

主要离线统计包括 overall/cohort/template/template_role/template_strand/reference/source/run/strand/motif 分组的指标 CSV、`per_class_metrics.csv.gz`、混淆矩阵、`confidence_thresholds.csv`、`prevalence_scenarios.csv`、`seed_summary.csv`、`metadata_coverage.csv`、`template_coverage.csv`、read 聚类重采样数据，以及 `reviewer_coverage.json`、`coverage_report.json`、`analysis_manifest.json`。提供来源、run、实际方向和 pair_id 时还生成相应联合/配对分层表；字段未知不会被文件名替代为真实身份。

Figure 4 绘图输入固定为：

```text
figure4A_dna01_16_false_call_heatmap_data_6class.csv
figure4B_xna01_16_target_recall_lollipop_data_6class.csv
figure4C_xna17_20_residual_error_detail_6class.csv
figure4C_xna17_20_residual_error_summary_6class.csv
figure4D_external_validation_profile_data_6class.csv
```

## 8. 统计解释和审稿证据边界

逐类输出 TP/FP/FN/TN、support、precision、recall、specificity、FPR、FDR、F1；同时保存固定六类 macro-F1、混淆矩阵、NLL、Brier 与 15 个固定等宽 bin 的 ECE。模板可能只含一个或两个真值类别，此时不存在的类别分母为未定义，固定六类 macro-F1 也不能单独作为该模板的主要解释指标。

`target_recall_micro` 表示 X/Y **具体类别**正确恢复；`ncb_recall` 表示被判成任意 X/Y 的检测率，X→Y 对前者为错、对后者仍是检出，两者不能混写。天然背景误报分母为真实天然窗口。阈值扫描使用固定的 0、0.5、0.7、0.8、0.9、0.95、0.99、0.995、0.999；低于阈值为弃权，计算 recall 时不把弃权真阳性从分母删掉。外部结果用于报告固定阈值表现，不用于挑一个最有利阈值后冒充预定性能。

全部预定种子取 mean 与样本 SD（ddof=1）。read 聚类 bootstrap 仅在相关组的真实 read 身份完整时可用，保留同 read 的全部窗口及各自真值；默认重采样 200 次，提供 overall/template/template_role 及 X/Y 相关区间。正温度校准不改变 argmax，raw/calibrated 复用同一组 read 重采样。身份缺失时明确报告不可用；窗口 Wilson 区间仅作描述，不能当作相关窗口独立时的 read 级推断。校准与未校准是同一训练 run 的两种概率表示，不是额外重复实验。

低丰度表是固定观测 TPR/FPR 和背景组成假设下的 PPV/FDR 情景推算，**不是**实测稀释梯度或真实低丰度检出限。文件声明的 DNA/XNA 模板配对也不证明来自同一实验 run 或真正匹配的分子对照。

本包能提供可复算的外部预测、全部种子比较、误差/校准/阈值/分层/不确定性及资源记录；不能仅靠代码保证覆盖审稿人的所有要求。缺少真实阳性 read UUID、来源、run、真实方向或训练参考信息时，不能证明 read/template/run 完全独立，也不能排除来源混杂。来源/方向预测负对照、严格留出模板/批次重训、源匹配的新实验、消融和五分类训练仍需相应独立设计与数据。本包不会补造这些实验结果。

## 9. 常用参数

完整选项以 `python3 run_figure4.py --help` 为准，版本可用 `python3 run_figure4.py --version` 检查。保留了旧名 `plot_figure4_full_external_validation.py` 作为新预测入口的兼容转发；建议新任务直接使用本 README 的队列命令。

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--phase` | `all` | `all/predict/analyze/plot`；predict 只预测，analyze 统计后画图 |
| `--models` | 4 模型 | 可调试子集；正式比较保留全部预定模型 |
| `--seeds` | `42 2024 2025 2026 2027` | 可调试子集；不按外部性能筛选 |
| `--feature_schemes` | `full` | 只接受原训练结果中实际完成且兼容的方案 |
| `--max_parallel` | 4 | 并行硬上限，受实测资源进一步约束 |
| `--device` | `auto` | `auto/cpu/cuda/cuda:N` |
| `--batch_size` | 256 | 每个预测任务 batch；OOM 后需改此项时使用新输出目录 |
| `--workers` | 0 | 必须为 0 |
| `--cpu_threads` | 4 | 每任务 PyTorch/数学库线程数 |
| `--cpu_parallel_limit` | 2 | CPU 模式并行上限 |
| `--cpu_budget` | 自动 | 可用 CPU 线程总预算 |
| `--gpu_reserve_gib` / `--ram_reserve_gib` | 2 / 3 | GPU/主存至少保留量 |
| `--reserve_fraction` | 0.10 | 按总容量保留比例，与绝对保留量共同约束 |
| `--memory_safety_factor` | 1.35 | 实测资源申请安全系数 |
| `--profile_batches` / `--profile_min_seconds` | 8 / 5 | 资源探测至少 batch 数/运行秒数 |
| `--resource_wait_seconds` | 3600 | 无法容纳任务时的等待上限 |
| `--poll_seconds` / `--oom_retries` | 2 / 1 | 轮询间隔/确认 OOM 后有限重试数 |
| `--disk_reserve_gib` | 5 | 输出分区额外保留空间 |
| `--disk_bytes_per_sample` | 1536 | 每模型/seed/窗口的基础导出预算，另加 embedding/attention |
| `--bootstrap` / `--analysis_seed` | 200 / 42 | read 聚类重采样数及可复算随机种子；0 可禁用重采样 |
| `--no_embeddings` / `--no_attention` | 不启用 | 初次运行时可关闭相应导出 |
| `--embedding_dtype` | `float16` | 可选 float32，影响磁盘占用 |
| `--allow_partial_templates` | 不启用 | 显式允许局部调试并保留缺失覆盖记录 |

## 10. 环境和测试范围

建议沿用服务器已完成训练的 PyTorch 2.0.1 环境，不为运行本包升级 PyTorch。Python 依赖为 numpy、pandas、matplotlib 和 PyTorch；若需要补依赖，先检查现有版本，参考 `requirements.txt`，其中 numpy 限制为 `<2` 以兼容现有 PyTorch 2.0.1。GPU 版 PyTorch/CUDA 应使用服务器原有匹配环境，不由本包重新安装。

提供合成数据及随机权重的 CPU 回归测试，覆盖预处理、四模型加载与概率、缓存/预测损坏恢复、冻结身份、metadata 对齐、真实 ID 处理、离线指标和绘图，以及调度资源不足/锁/失败情景。合成测试仅验证代码行为，不能验证论文真实准确率。当前验证环境为 CPU PyTorch 2.5 系列；**未在用户服务器的 GPU / PyTorch 2.0.1 上实测**，完整外部真实数据也未在本地重跑。测试 NumPy 2 环境可与新 CPU PyTorch 配合，不代表旧 PyTorch 2.0.1 应升级到 NumPy 2。

```bash
python3 -m unittest discover -s tests -v
bash -n run_figure4_auto.sh
python3 -m py_compile run_figure4.py prepare_figure4.py figure4_worker.py figure4_analyze.py plot_figure4_final_finetune.py
```

为后续协助分析，可先提供 `analysis_data/`、`figures/`、`evaluation_plan.json`、`external_identity_audit.json`、`discovered_inputs.csv`、`queue_state.json` 及主日志；这些通常比全部逐窗数据小得多。保留服务器上的 `predictions/` 和共享缓存，后续改变统计口径或重绘时可以复用。
