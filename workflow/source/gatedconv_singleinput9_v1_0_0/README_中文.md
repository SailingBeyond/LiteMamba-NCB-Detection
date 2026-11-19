# GatedConv 三种单输入、九次训练启动包 v1.0.0

本包新增 **raw_only、context_only、responses_only × 种子 42、2024、2025，共九次训练**。使用已经确认的最终 GatedConv 架构，从各自种子的随机初始化开始训练。

**本包完成训练、内部冻结测试、汇总和核心结果打包，不自动运行 Figure 4 外部预测。** 完成日志中的 `[FULL PASS]` 仅表示本包这九次训练及内部评价完成。外部评价后续加载本次保存的检查点执行。

## 1. 首次运行

在原 `pytorch2.0.1` 环境中，首次将压缩包解压到独立目录：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip gatedconv_singleinput9_v1_0_0.zip
cd gatedconv_singleinput9_v1_0_0

# 前台核验环境、输入内容 SHA256、冻结源码及剩余空间；不会训练。
bash run_singleinput9.sh --check_only

# 预检通过后提交后台任务；使用当前 conda 环境。
bash run_singleinput9.sh
```

脚本会打印本次独立的启动日志。首次运行需要读取六份输入文件进行内容核验，出现 `[VERIFY SHA]` 时仍在正常预检。随后会生成正式队列日志。

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/gatedconv_singleinput9_v1_0_0
tail -F "$RESULTS/logs/queue_latest.log"
```

`queue_latest.log` 在正式队列启动后生成；此前查看脚本打印的 `launcher_logs/launch_*.log`。后台提交返回 PID 只表示已提交请求，是否通过预检应以日志为准。同一输出目录只允许一个启动器运行。

## 2. 默认输入与输出

数据目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721
```

所需文件为 `train.csv`、`validation.csv`、`test.csv` 及对应的 `train_meta.tsv`、`validation_meta.tsv`、`test_meta.tsv`。本包冻结了已经完成的 GatedConv 完整输入模型所用文件的内容 SHA256 和大小；数据不同会在训练开始前停止，不会重新划分或替换输入。

输出目录：

```text
DATA_DIR/sixclass_auto_v1_0_4_results/gatedconv_singleinput9_v1_0_0
```

共享只读特征缓存默认放在：

```text
DATA_DIR/sixclass_auto_v1_0_4_results/feature_cache
```

每次训练分别保存到：

```text
OUTPUT_DIR/raw_only/litemamba/seed_42
OUTPUT_DIR/raw_only/litemamba/seed_2024
OUTPUT_DIR/raw_only/litemamba/seed_2025
OUTPUT_DIR/context_only/litemamba/seed_42
OUTPUT_DIR/context_only/litemamba/seed_2024
OUTPUT_DIR/context_only/litemamba/seed_2025
OUTPUT_DIR/responses_only/litemamba/seed_42
OUTPUT_DIR/responses_only/litemamba/seed_2024
OUTPUT_DIR/responses_only/litemamba/seed_2025
```

目录中的 `litemamba` 是兼容既有引擎的模型键。实际模型配置固定为 `litemamba_variant=legacy_gatedconv`、`readout=target_attention`，没有选择性状态空间递推。模型宽度 96、四个门控卷积块，目标为第4位；第4位序列编码仍统一为 N。

## 3. 三种单输入的确切范围

| 方案 | 保留的输入 | 排除的样本输入 |
|---|---|---|
| `raw_only` | 六个位点各自的固定长度原始信号片段 | 序列上下文、统计量、测序响应 |
| `context_only` | ACTGN 序列编码；第4位为 N | 原始信号、统计量、测序响应 |
| `responses_only` | quality、mismatch、insertion、deletion | 原始信号、序列上下文、统计量 |

输入屏蔽沿用已经冻结的模型代码，不改变 GatedConv 的网络宽度、读出或分类头。本包不是把完整输入模型的权重屏蔽后直接测试：九次均重新初始化并完整训练。

已有 full 的三个检查点只用于只读来源核对及结果比较，不作为本次初始化，也不重新训练。默认会查找：

- `DATA_DIR/sixclass_auto_v1_0_4_results/litemamba_structure_seed42_v1_0_0` 中的 seed42。
- `DATA_DIR/sixclass_auto_v1_0_4_results/litemamba_structure_confirm3_v1_1_0` 中的 seed2024、2025及复用记录。
- `DATA_DIR/sixclass_auto_v1_0_4_results/gatedconv_internal_test_v1_0_0` 中的 full 内部测试补评估。

完整输入的有效内部 test 记录缺失或身份不匹配时，汇总会说明原因，并保留相应差值为空，不用验证集数值代替 test。

## 4. 训练、冻结与测试顺序

1. 核验运行环境、输入内容及模型/预处理源码；准备共享缓存并审计数据。
2. 每种单输入在设备上串行执行 **8 次真实 AdamW 更新及验证推理**，测量资源。共三种资源探测，不保存为正式模型结果，不进行 test 推理。
3. 按实测显存、RAM、CPU 预算并发完成九次正式权重训练。
4. 九次权重训练全部成功后，串行导出 train/val 预测，并仅在 validation 上拟合温度参数。
5. 冻结九个验证集选出的最佳 checkpoint 哈希；随后串行评估固定内部 test。
6. 汇总所有任务，生成核心结果压缩包。任一训练失败时，不启动整队 test。

训练设置与完整输入对照保持一致：batch=256、epochs 上限200、patience=30、lr=0.001、AdamW weight_decay=0.0001、以 `val_loss` 选择检查点。workers 默认0，每进程 CPU 线程4。没有为了省显存自动降低 batch 或改变梯度累积；GPU环境沿用原引擎的 AMP 策略。

默认并发硬上限为4，**实际并发不保证达到4**，由资源预约、保留余量和探测结果决定。确认 OOM 或疑似资源性终止时，会降低并发并在保持科学设置不变的情况下有限重试。资源探测不是正式推理效率基准，本包不把队列并发耗时当作模型速度比较。

## 5. 磁盘检查和断点继续

启动前按尚未完成的训练轮数、每轮验证数组、最终导出、检查点及缺失缓存估算剩余新增空间，叠加安全系数，并默认另保留 **8 GiB**。同一磁盘上的输出和缓存合并计算；不同磁盘分别检查。实际预算以 `[DISK]` 日志为准，不使用一个固定空间数字代替检查。

正式队列运行期间继续监测剩余空间。触及保留线时停止本队列；不会自动删除历史、概率、嵌入或检查点。磁盘估算不能保证其他进程永不抢占空间，因此仍保留每轮原子提交及恢复机制。

清理空间或处理错误后，用同一包、同一输入和首次运行的路径参数恢复：

```bash
cd /path/to/revision-code/gatedconv_singleinput9_v1_0_0
bash run_singleinput9.sh --resume
```

已经完整的任务会跳过。有 `last_checkpoint.pt` 的中断任务会恢复模型、优化器、调度器、GradScaler、随机数状态和历史，从已提交轮次继续。若某轮输出未完整提交，则重新执行该未提交轮次。若第一次 checkpoint 之前失败，只有符合引擎中断证据检查的任务才能以相同种子从头重启。

不要删除旧任务目录、编辑冻结输入或替换正在运行队列的源码。若首次使用了 `--data_dir`、`--output_dir`、`--cache_dir` 等非默认路径，恢复时保留这些参数。可在不修改科学配置的前提下调整资源预算，例如降低并发上限。

```bash
# 示例：恢复时把并发硬上限降至2。
bash run_singleinput9.sh --resume --max_parallel 2
```

仅重新汇总已有结果、不训练也不推理：

```bash
bash run_singleinput9.sh --summary_only
```

`--dry_run` 只打印九项计划及命令，未核验实际运行环境、输入 SHA 或缓存；不能视为预检通过。`--summary_only` 在任务不完整时会报告缺失/失败，不能把部分结果标为全部完成。

## 6. 保存哪些记录

服务器完整目录保留：

- 每次训练的解析配置、架构、环境、数据审计、冻结源码、输入指纹、命令和日志。
- 每轮训练/验证指标、学习率、早停状态；每轮验证 logits/labels/sample_index 的压缩 NPZ。
- `last_checkpoint.pt`、最佳完整 checkpoint、最佳 state_dict及周期检查点。
- train/val/test 逐窗口 logits、原始及校准概率、预测与身份字段，以及嵌入、注意力和分组分析。
- 资源探测、预约、并发变化、失败和重试记录；整队冻结评价计划及最佳权重哈希。

本轮额外记录每轮更新尝试次数、推断的 AMP 跳步数、scale起止/最小/最大值，以及有限、NaN、正/负无穷梯度范数的计数。旧 `optimizer_steps` 字段仍表示尝试次数；新增 `optimizer_update_count_method` 明确说明：**对标准 GradScaler，以更新后 scale 低于更新前 scale 推断该次跳步**，不是通过优化器 hook 独立计数。原始梯度范数均值/最大值保留，同时新增只基于有限值的统计，避免非有限值转空值后失去解释依据。这些增强只增加记录，没有修改损失、梯度归一化、裁剪、AdamW、GradScaler 或模型选择规则。

## 7. 跑完后上传哪个文件

看到以下日志表示九次训练和内部评价完成：

```text
[FULL PASS] 9次单输入训练、内部冻结测试与汇总完成；Figure4外测尚未启动。
```

上传自动生成的：

```text
OUTPUT_DIR/gatedconv_singleinput9_results_for_analysis_v1_0_0.tar.gz
```

阅读入口在 `OUTPUT_DIR/singleinput_summary/`，包括：

- `run_status.csv`：九次任务的完整状态。
- `metrics_per_seed.csv`、`mean_sd.csv`：各种子指标、均值和样本标准差。
- `paired_single_vs_full.csv`、`paired_mean_sd.csv`：可验证 full 对照的同种子配对差值。
- `summary.json` 及 HTML 报告：可比性、缺失项和来源说明。

核心上传包包含配置、历史、指标、分组统计及包清单，**不包含权重、完整逐窗口预测、NPY/NPZ概率数组、嵌入或特征缓存**。这些大文件仍保存在服务器，后续 Figure 4 推理和重画图继续使用，不会因生成轻量包被删除。

## 8. 环境与验证范围

优先复用原来的 `pytorch2.0.1` 环境；本包不会自动安装或升级依赖。需要 Python≥3.9，服务器 PyTorch2.0.1 保持 NumPy<2。`requirements.txt` 供新建环境或补齐缺失依赖参考，已能运行原训练时无需重新安装，更不要为启动本包直接升级现有 CUDA/PyTorch。

本地验证包含 CPU Torch2.5.1 的模型前向、梯度和训练/导出小数据检查、输入屏蔽契约、原子写入故障注入及增强记录检查；未在你的服务器完成真实 CUDA 长训练。实际服务器会先执行环境、数据、资源和磁盘检查。本包完成并不等于全部审稿问题已解决：单输入在留出模板上的评价、来源/方向匹配控制等仍需对应的后续证据。
