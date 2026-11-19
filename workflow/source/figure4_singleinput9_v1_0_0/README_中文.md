# Figure 4 九个单输入模型外部评价 v1.0.0

本包直接加载已经完成训练的 **raw_only、context_only、responses_only × 种子42、2024、2025** 九个检查点，对修正后的 Figure 4 全部72份文件预测。已有 **full GatedConv 三种子** 的逐窗口结果经 SHA256 核验后只读复用，共比较四种输入方案、12个模型运行。**本包不训练、不重新选模型、不在外部数据上拟合温度或阈值。**

## 1. 直接运行

在之前的 `pytorch2.0.1` 环境中，解压到新的代码目录；不要覆盖任何训练代码目录。

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip figure4_singleinput9_v1_0_0.zip
cd figure4_singleinput9_v1_0_0

# 前台只读预检：核验检查点、旧结果、外部源文件及缓存、磁盘预算。
bash run_figure4_singleinput9.sh --check_only

# 预检通过后，后台执行预测、分析、作图和核心打包。
bash run_figure4_singleinput9.sh
```

`--check_only` 不创建正式评价计划，也不预测；之后正常启动不用加 `--resume`。内容 SHA 核验需要读取较多文件，出现 `[VERIFY EXTERNAL]` 日志时仍在工作。若缓存已删除但源数据仍在，预检会提示缓存需重建；正式启动可尝试按旧冻结内容重建并核对四个缓存文件哈希。

后台提交返回 PID 只代表启动请求已提交。先查看启动脚本打印的本次独立日志；取得队列锁后，也可以用：

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/figure4_singleinput9_v1_0_0
tail -F "$RESULTS/logs/singleinput_external_latest.log"
```

## 2. 默认来源和新输出

以下 `RESULTS_ROOT` 为：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results
```

| 用途 | 默认位置 |
|---|---|
| 九个已训练模型 | `RESULTS_ROOT/gatedconv_singleinput9_v1_0_0/<scheme>/litemamba/seed_<seed>` |
| 修正后的72份数据、已有 full GatedConv 预测 | `RESULTS_ROOT/figure4_gated_comparison_v1_0_0` |
| 本次新输出 | `RESULTS_ROOT/figure4_singleinput9_v1_0_0` |

完整输入只复用旧计划中 `model=gatedconv_target` 的三个模型，**不会把旧 SSM 的 `model=litemamba` 结果当作完整输入对照**。单输入训练目录中的 `litemamba` 是兼容模型键；实际结构固定为 `legacy_gatedconv`、`target_attention`，第四位为目标位置，序列第四位编码为 N。新评价计划统一记录 `actual_model_type=litemamba`、展示模型键 `gatedconv_target`，同时保留原训练运行 ID。

所需服务器文件包括：九个 `best_checkpoint.pt`、对应配置/温度/完成记录/验证历史与来源快照；原单输入训练目录的计划及冻结记录；旧 Figure 4 的计划、缓存、216份 full 逐文件预测，以及旧计划引用的模型元数据。**不需要把模型权重上传给 ChatGPT；它们直接在服务器加载。**

若目录移动过，可显式指定：

```bash
bash run_figure4_singleinput9.sh \
  --training_dir /实际路径/gatedconv_singleinput9_v1_0_0 \
  --previous_output /实际路径/figure4_gated_comparison_v1_0_0 \
  --output_dir /有足够空间的新路径/figure4_singleinput9_v1_0_0
```

旧计划内的数据和模型绝对路径仍需可访问。新输出必须与旧结果、训练目录和代码目录互不包含。

## 3. 并发、资源和磁盘

默认并发硬上限 **4**；程序先对三种输入方案分别实测预测资源，再根据可用显存、RAM、CPU及预约余额动态并发。实际并发可能低于4。默认每进程4个CPU线程、workers=0，保留GPU 2 GiB、RAM 3 GiB及10%比例余量，资源需求乘1.35安全系数；CPU回退默认最多并行2项。没有用训练时并发数推断预测并发，也不修改 batch 来让更多进程挤进显存。

默认 `--device auto` 使用当前可用CUDA设备；确实没有CUDA时使用CPU。显式 `--device cuda:0` 则不可静默回退CPU。GPU映射沿用已修复的真实UUID探测；资源探测使用独立子进程，已修复其模块导入问题。探测耗时是资源准入依据，不是发表用的公平推理效率基准。

```bash
# 资源紧张时：保留全部科学设置，降低并发。
bash run_figure4_singleinput9.sh --max_parallel 2
```

预测的 batch、CPU线程及嵌入导出精度沿用旧 Figure 4 计划，保证复用和新预测设置一致。资源余量、并发硬上限可以调整；试图更改被冻结的预测设置会明确停止。

保留所有逐窗口数据的默认预算较大。按已有每模型7,258,270个窗口及旧float16嵌入设置，九个新增模型的**保守预测输出预算约118.3 GiB，另保留8 GiB**；这不是对最终实际占用的承诺。预检和正式启动均按尚未完成文件计算需求并检查输出盘；正式队列持续监测剩余空间。默认full采用符号链接复用，不再复制三套大数组。

请保留原full输出，不要清理其链接目标。若选择 `--reuse_mode copy`，复制完整输入结果还会额外占空间，并逐文件检查。磁盘估算不能阻止其他进程占用空间；失败后保留已有文件再续跑，不自动删除任何结果。

## 4. 中断后继续

修复磁盘或资源问题后，用相同代码、来源和输出路径执行：

```bash
bash run_figure4_singleinput9.sh --resume

# 可以同时降低资源上限。
bash run_figure4_singleinput9.sh --resume --max_parallel 2
```

已完整提交、身份和所有文件 SHA 一致的逐文件预测会跳过。未完整提交的单个数据文件重新预测，不必重跑该模型已经完成的全部数据。输出以临时目录写入，再原子提交。来源、模型或代码发生变化时会拒绝混入旧结果。确认OOM时有限重试并降低并发；其他错误不会盲目重跑。

已经预测完成后只重做离线分析/图表：

```bash
bash run_figure4_singleinput9.sh --phase analyze
bash run_figure4_singleinput9.sh --phase plot
```

这两项不加载模型。`analyze`会重算统计并保留旧analysis目录备份，可能增加磁盘占用；`plot`复核已有分析并重画比较图。不要在同一个输出目录同时启动两个入口。程序只清理本次拥有的子进程，不终止其他任务。

## 5. 保留哪些信息

- `evaluation_plan.json`：全部12运行、72数据、检查点/源码/配置/预处理/温度SHA及来源。
- `predictions/<scheme>/gatedconv_target/seed_<seed>/<dataset_id>/`：float32 logits、int64 labels、包含原始与温度校准六类概率的逐窗口CSV、沿用旧精度的embedding、float32 attention、元数据/环境/计时/完成标记。真实read与原窗口记录保留；不会伪造生物学read ID。
- `analysis_data/`：原始和校准概率的整体、三组cohort、模板、目标/天然对照、方向、来源/测序run、参考、motif、已知配对分层；六类混淆矩阵、逐类指标、固定置信阈值表、患病率情景、校准指标、元数据覆盖。
- `singleinput_data/`：四方案逐种子、均值与样本SD；同种子full减单输入差值；X/Y正确身份召回与任一X/Y检出分开统计；天然→X/Y误报；类别分布及多数类基线；完整性核对和HTML报告。
- `figures_singleinput/`：四方案的整体、模板、召回/误报比较图，PNG和SVG及对应数值表；混淆矩阵保存在分析表中。
- `source_provenance/`、`resource_profiles/`、`commands.jsonl`、`resource_events.jsonl`、`queue_state.json`：来源、资源实测、实际命令和运行过程。

默认完整真实read信息可用的分层采用200次read聚类bootstrap，并保留重复值；同一read的2/4窗口留在同一个聚类中。标准bootstrap覆盖准确率、检出及X/Y逐类指标等；**Macro-F1表中的种子SD不是read聚类CI**。缺真实ID时仅保留可核验范围说明，不把行号当作独立read。

继承既有窗口规则：XNA/DNA1–12最多1窗/read；13–16最多2窗/read；XNA17–20最多4窗/read；13号模板及对照允许1bp交叠。本次输入内容不变，核验全体源文件和缓存后继承已完成的身份/窗口审计，不再次构建数百万行审计数据库，也不丢弃重叠窗口。

兼容格式的旧 `figure4A/B/C` CSV现在明确标注full GatedConv，供原图流程使用；四方案比较请使用 `singleinput_data/` 和 `figures_singleinput/`。

## 6. 完成后上传什么

看到以下两行才表示默认 `all` 的完整流程完成：

```text
[RESULT PACKAGE] .../figure4_singleinput9_results_core_v1_0_0.tar.gz
[FULL PASS] Figure4九个单输入模型外部评价、四方案比较及打包完成：...
```

**只上传 `figure4_singleinput9_results_core_v1_0_0.tar.gz` 即可分析。**

打包先收齐全部12运行、864份逐文件完成记录、模型核心元数据、分析表和比较图，再考虑可选motif大表；不会为了大小上限遗漏最后一个模型。归档每个成员均重新读取并核对SHA后原子发布。逐窗口概率、数组、缓存、权重留在服务器，核心包不包含这些大文件。核心文件异常超出预算时会报错并保留全部结果，而不会悄悄截断。可单独重新打包，无需预测：

```bash
python3 summarize_singleinput_external.py --output_root "$RESULTS" --package_only
```

## 7. 评价解释边界

本实验回答三个单输入模型在既定外部序列上的表现与完整输入的差异。全部三个种子预先固定，原始与温度校准概率属于同一个模型，不重复算作独立实验。默认分类采用六类argmax；固定阈值表只作描述，不依据外部数据挑阈值。

外部数据类别比例与内部均衡测试不同，不能把内部context_only的1/6准确率直接搬到Figure4解释；本包额外提供实际类别支持和多数类基线。各模板可能只含部分真值类别，固定六类Macro-F1须结合支持和X/Y身份召回解读。默认整体指标按窗口合并，也提供cohort及模板分层，不把某一大模板组的权重当作普遍优势。

单输入对照不能单独排除训练来源与标签关联，也不能替代另外12次去除输入组的消融、来源匹配对照或公平效率测量。本包不保证覆盖审稿人的全部实验要求；它完成已确定优先开展的Figure4单输入外部评价。

## 8. 本地核验与环境

沿用服务器现有PyTorch2.0.1环境；不要为本包升级CUDA/PyTorch。额外Python依赖列在 `requirements.txt`，如果原Figure4运行环境完整，通常不需安装。Python最低3.9。

本地验证说明见 `VALIDATION.md`。本地CPU合成数据测试验证代码衔接；真实GPU显存、速度和最终数据结果以服务器日志为准。运行本地测试：

```bash
python3 -m unittest discover -s tests -v
```
