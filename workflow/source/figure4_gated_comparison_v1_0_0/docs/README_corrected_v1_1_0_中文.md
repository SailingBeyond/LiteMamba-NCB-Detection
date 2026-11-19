# Figure 4：修正参考后的 XNA13–16 重评估（v1.1.0）

本版接入你重新生成的 `/path/to/staged-storage2/analysis/paper/xna_pc_13_16_6_X_to_T`，默认只替换 XNA13–16 的 target 正、负链文件。原先这些 target 使用错误参考，其旧预测和包含它们的总体统计应由新结果替代；结果变化属于参考/数据修正，不能解释为模型架构改进。

## 1. 本次处理范围

| 项目 | 默认处理 |
|---|---|
| XNA13–16 target | 用新提取数据替换，4个模板 × pos/neg = 8个文件 |
| DNA/PC13–16 control | 复用旧冻结预测；新提取版本未被假定与旧版完全相同 |
| XNA1–12及其对照、XNA17–20 | 复用旧冻结预测 |
| 模型与种子 | 完整继承旧计划：通常4模型 × 5种子，不挑选子集 |
| 权重、输入预处理、校准 | 原训练最佳checkpoint、第四位N、validation温度，不更新参数 |
| 全部统计和图 | 汇合新旧有效预测后，完整重算全部标准表和Figure 4图 |

原计划72个文件、20组权重时，本次新增 **160组逐文件预测**，复用其他 **1,280组**；文件数不代表窗口数，新旧样本数可以不同。原来所有目标与对照仍同时进入总体、模板、类别、motif、来源、方向、置信度等统计，不把旧summary直接拼成新总体。

如果明确希望DNA/PC13–16也全部换为本次重新提取的数据，首次启动时加 `--replace_controls`。这会变成16个新文件 × 20组权重，且新目录必须具备对应control文件及metadata。已开始后不能改变替换范围；改变范围应使用另一个输出目录。

## 2. 直接启动

解压并进入本包目录，在原 `pytorch2.0.1` 环境运行：

```bash
unzip figure4_revision_v1_1_0.zip
cd figure4_revision_v1_1_0
bash run_figure4_xna13_16_corrected.sh
```

不需要重新训练，不要为本包升级原PyTorch/CUDA环境。依赖沿用上一版；若需检查缺失依赖，参见 `requirements.txt`，PyTorch2.0.x应使用NumPy1.x。

默认旧结果：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_external_v1_0_0
```

默认新结果为其旁边的独立目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_external_xna13_16_corrected_v1_1_0
```

入口后台启动，终端会打印本次独立日志和 `tail -F` 命令。可直接采用打印的命令查看。旧目录保留，不要把新CSV覆盖进旧目录后使用旧版 `--resume`。

路径不同时显式指定：

```bash
bash run_figure4_xna13_16_corrected.sh \
  --previous_output /实际路径/figure4_external_v1_0_0 \
  --corrected_13_16 /实际路径/xna_pc_13_16_6_X_to_T \
  --output_root /同一文件系统/figure4_external_xna13_16_corrected_v1_1_0 \
  --max_parallel 4
```

程序自动继承旧模型/种子列表、batch、CPU线程、embedding精度和辅助输出设置。显式传入与旧设置不同的值会报错，不混合导出设置。`--max_parallel` 是上限，可依据资源调节，不影响科学评估设置。

## 3. 严格预检所需文件

新拆分文件名支持你脚本生成的形式：

```text
xna_pc_13_16_pos_XNA13.csv
xna_pc_13_16_pos_XNA13.meta.tsv
xna_pc_13_16_neg_XNA13.csv
xna_pc_13_16_neg_XNA13.meta.tsv
```

XNA13、14、15、16均须有pos/neg及逐行metadata。天然类别顺序为A/T/C/G=0/1/2/3，X/Y=4/5；第4位为N。重复匹配同一模板/角色/方向时直接报错，不任意挑一个文件。新目录里的PC/DNA文件默认不进入替换范围。

新拆分CSV本身已经掩码，因此预测里的 `original_kmer` 保存的是该输入文件的序列，第四位仍为N，不会凭空恢复原参考碱基。未掩码序列保留在原始未拆分CSV/metadata中，其路径和内容指纹进入审计，请一并保存。

上传的拆分脚本使用 `zip` 对齐原特征与metadata，部分错位仅警告继续写；两边改写kmer/label后，下游仅检查拆分文件不能发现原始错位。因此本版**必须读取原始未拆分的4个输入**，默认就是你提供的：

```text
/path/to/staged-storage2/analysis/paper/xna_pc_13_16_pos.csv
/path/to/staged-storage2/analysis/paper/xna_pc_13_16_neg.csv
/path/to/staged-storage2/analysis/paper/xna_pc_13_16_pos.meta.tsv
/path/to/staged-storage2/analysis/paper/xna_pc_13_16_neg.meta.tsv
```

这是只读核验，不重新提取特征、不重写原文件。检查内容包括：

- 原始CSV与metadata以 `zip_longest` 严格对齐，包括恰好多/少末尾一行。
- 原始完整kmer、精确0/1标签、chrom模板/角色、提供的实际BAM方向；拒绝把1.9等小数截断为1。
- 按上传脚本规则转换后，每个待替换文件的**全11列、完整metadata、顺序、行数及逐行流SHA256**均与预期相同。
- 后续缓存逐位标量/信号有效性、标签与metadata一致性，以及可用真实read与训练划分的重叠。

原文件不在默认路径时，通过 `--original_pos_csv`、`--original_neg_csv`、`--original_pos_meta`、`--original_neg_meta` 提供实际路径。默认由新拆分目录的父目录推导。

若预检失败，先按具体报错核对原始提取文件配对、拆分日志和文件是否完整；不要修改真值或删行来绕过检查。若原输入有错位，需要修复来源对应关系后重新拆分。这个检查证明拆分与输入转换一致；**它不能凭CSV独立证明提取时使用的FASTA/比对参考正确**，请保留本次实际参考文件与提取命令/日志。

## 4. 复用、磁盘与并行

复用的每个旧文件会核验大小、完整SHA256、旧plan ID、模型/种子、样本数、导出设置、checkpoint SHA和validation温度。旧的待淘汰XNA13–16 target逐窗文件无需重新核验或复制，因为不再用于新评估。

默认新旧结果位于同一文件系统，使用**硬链接**复用不变的预测CSV、logits、embedding等，不额外复制大块数据；新completion和新plan独立写入，旧JSON和旧文件内容保持原样。请把复用数据视为只读，不直接编辑任何一份硬链接数据。旧目录和原训练权重应保留，以支持恢复和来源追溯。

跨文件系统不能硬链接时，显式选择：

```bash
# 复用绝对路径软链接，整个使用期间必须保留旧预测目录及目标文件
bash run_figure4_xna13_16_corrected.sh --output_root /另一个盘/新目录 --reuse_mode symlink

# 复制全部不变产物，需自行准备相应空间
bash run_figure4_xna13_16_corrected.sh --output_root /另一个盘/新目录 --reuse_mode copy
```

已有旧缓存可保留，也可因此前空间清理而缺失：**复用完整预测不要求旧缓存存在**。本次只建立8个新文件的共享缓存，资源探测也只读取新数据；不变文件的预测若缺失/损坏会停止并指出具体文件，不偷偷重新提取旧数据或使用错误参考。

程序先串行准备新缓存，按各模型架构进行实际forward资源实测，再依据显存、RAM、CPU配额和安全余量调度，默认并行上限4。GPU逻辑编号用原版已验证的UUID机制对应到实测设备；资源不足会等待，OOM有限重试并降并行度。CPU模式默认最多2个并行进程。

新增空间主要是新缓存及160组新预测，不是复制全部旧结果。仍有完整校验的读盘成本、离线分析临时空间和新图表/压缩包空间，且其他进程写盘会影响余量。默认保留5GiB磁盘余量；提前预算不能绝对避免ENOSPC。新旧目录的实际空间和inode均应充足。

## 5. 中断或空间不足后继续

确认对应分区空间/inode已恢复，旧任务退出后，原参数加 `--resume`：

```bash
bash run_figure4_xna13_16_corrected.sh --resume
```

使用自定义路径时恢复须带回相同路径/替换范围；资源并行上限可降低。复用迁移和预测都支持断点：完成且校验通过的外部文件跳过，未完成文件从头导出，不从文件内某个batch继续。所有8个新文件及原始未拆分文件均冻结指纹；冻结后改动数据、参考拆分范围、权重或校准，需要新的输出目录。

`[FULL PASS] Figure 4 all 阶段全部完成` 表示预测、完整分析、绘图和结果打包结束。`--phase predict` 只完成预测，不表示最终图表已生成。

## 6. 统计格式与离线操作

保留上一版的表名和列含义，主要包括：

| 输出 | 内容 |
|---|---|
| `analysis_data/overall_metrics.csv` | 全部有效外部数据的窗口加权指标 |
| `analysis_data/per_template*_metrics.csv` | 模板、角色、文件方向、来源/run/实际方向、已知pair等分层 |
| `analysis_data/per_class_metrics.csv.gz` | 六类precision/recall/F1、误报、支持数、区间及分母 |
| `analysis_data/confusion_matrices.csv.gz` | 完整6×6混淆计数及行比例 |
| `analysis_data/calibration_bins.csv.gz` | raw与validation温度校准概率的固定15bin统计 |
| `analysis_data/confidence_thresholds.csv` | 原先固定阈值下的覆盖、错误与拒判统计 |
| `analysis_data/seed_summary.csv` | 全部预先指定种子的均值、SD和可用数 |
| `analysis_data/figure4A...csv` 至 `figure4D...csv` | 原格式A–D画图输入 |
| `analysis_data/coverage_report.json`、`reviewer_coverage.json` | 覆盖范围、指标定义与证据限制 |
| `figures/` | Figure 4A–D及补充阈值/校准图PNG/PDF |

还保留motif、cohort、来源、run、实际方向、对照假阳性、X/Y正确类别恢复、prevalence情景等原版统计。原概率和校准概率属于同一次训练，不能当作两次独立实验；不根据外部结果重新拟合温度、阈值或选择种子。

Figure 4A仍使用天然对照；B的13–16更新；C的17–20数据保留；D及包含13–16的总体、cohort和motif等指标全部重新计算。A/C原数据未变，但会用同一版绘图程序重新输出。

你提供的“每条read只有一个6mer”实验设计说明会单独记录。真实read ID可用且覆盖完整时按read聚类bootstrap；未知/代理ID保持未知，不合成真实ID。种子SD、窗口Wilson区间与read聚类区间有各自含义；文件名不能代替实际run或分子配对身份。

预测完成后可单独重分析/重绘，不调用GPU：

```bash
OUT=/实际路径/figure4_external_xna13_16_corrected_v1_1_0
python3 figure4_analyze.py --output_root "$OUT" --bootstrap 200 --analysis_seed 42
python3 plot_figure4_final_finetune.py \
  --data_dir "$OUT/analysis_data" --out_dir "$OUT/figures" --formats png pdf --dpi 300
python3 package_figure4_results.py --output_root "$OUT"
```

也可 `bash run_figure4_xna13_16_corrected.sh --phase analyze` 串行执行分析、绘图、打包。重新分析会保存旧analysis_data备份，反复执行需注意磁盘。原逐窗分析引擎保留为 `--engine legacy`，用于复核；新版引擎的验收和速度记录见 `TEST_REPORT.md`。

默认使用分块引擎（每块50,000行），减少Python逐窗口统计开销；内存较紧张时可单独运行 `figure4_analyze.py --output_root "$OUT" --chunk_size 10000` 后重画并打包。所有输出统计仍从完整逐窗预测重新汇总，没有使用旧错误参考的summary作增量修补。

## 7. 新增追溯与上传分析

| 路径 | 用途 |
|---|---|
| `corrected_split_audit.json` | 原输入及新拆分文件SHA、全行转换核验、行数和排除范围 |
| `reference_correction_provenance.json` | 原计划指纹、替换映射、control保留策略、旧错误参考作用范围 |
| `reuse_manifest.json` | 未变文件集合、重算集合及复用方式 |
| `evaluation_plan.json` | 完整新72文件计划、原全部权重/温度和导出设置 |
| `predictions/.../completion.json` | 新预测或继承预测的来源、文件hash与完成状态 |
| `logs/`、`resource_profiles/` | 唯一日志、实际命令、资源实测和恢复记录 |
| `figure4_results_for_analysis_v1_1_0.tar.gz` | 自动生成的统计、图、审计和完成记录压缩包 |

跑完后上传 **`figure4_results_for_analysis_v1_1_0.tar.gz`** 即可按原格式分析新结果。该包包含所有标准统计表、图、计划、参考修正审计、全部完成记录及可用训练JSON元数据；不装入巨大的逐窗CSV、数组、缓存或checkpoint。完整逐窗信息继续留在服务器，可供之后重分析和画图，不需要重新预测。归档内有每个成员文件的SHA清单。

旧参考结果可以保留作纠错追溯，正式论文及审稿回复应使用这次核验完成的新结果。新的性能数值必须等服务器实际跑完后判断。

## 8. 验证与适用范围

本包包含原有回归测试和新参考替换测试。详见 `TEST_REPORT.md`。本地使用独立合成数据与随机权重进行CPU流程验证；它验证程序行为，不代表论文真实准确率，也不等于已在你的GPU服务器实测。运行时还会核验服务器上的真实checkpoint、校准、训练代码快照和输入文件。

本版覆盖当前六分类Figure 4流程可产生的统计和追溯证据；未提供的审稿意见全文、实验批次或参考文件内容，不能被程序自动补足或宣称全部验证。原版详细参数说明保存在 `docs/README_original_v1_0_0_中文.md`，其中旧启动路径仅供版本追溯，本次以本文入口为准。
