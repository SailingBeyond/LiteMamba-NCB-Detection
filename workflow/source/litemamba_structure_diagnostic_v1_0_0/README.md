# LiteMamba 两项结构诊断 v1.0.0

本包只新增两次固定seed42的结构对照，已有full/seed42结果保持原位并被复用。两个新模型均从头训练，保留full输入、原数据划分和原科学训练参数，按val_loss选检查点。此轮用于定位结构贡献，不自动宣称提分、结构创新或跨模型优越性。

## 直接启动

将压缩包传到服务器的paper目录，使用原环境运行：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip litemamba_structure_diagnostic_v1_0_0.zip
cd litemamba_structure_diagnostic_v1_0_0
bash run_structure.sh
```

启动脚本已使用nohup后台提交，无需再套一层nohup。控制台打印PID、独立启动日志和查看命令；PID只表示已提交，预检通过后才会开始训练。

## 三个比较对象

| experiment | SSM/混合模块 | 读出 | 执行 |
| --- | --- | --- | --- |
| full | selective_ssm | 全局attention池化与第四位表示拼接 | 复用旧seed42 |
| gatedconv_target | 原有legacy_gatedconv模块 | 与full相同 | 新训练seed42 |
| ssm_global_only | 与full相同 | 仅全局attention，直接目标分支置零 | 新训练seed42 |

门控卷积和selective SSM模块的参数与运算不同。这是模块整体的结构对照，不是严格等参数、只改变一个递推算子的干预。

global_attention_only输出为`[attention_pool, zeros]`，保持原分类器192维输入及相同的初始化路径。后96列输入恒为零，对应第一层128×96=12288个权重没有数据梯度（AdamW权重衰减仍可更新它们）。名义参数量相同不代表有效容量完全相同；CSV记录名义参数、实际接收数据的输入列数和无数据梯度的输入权重数量。

第四位token仍参与上下文编码和全局attention，本对照只去除其直接读出分支。不会改为错误的第六位目标。

## 旧文件夹与新文件夹

旧引擎保持原样：

```text
/path/to/revision-code/sixclass_training_auto_v1_0_4
```

本包包含一个有明确源码指纹的独立engine副本。只扩展模型的新读出模式、对应CLI/元数据和两项结构队列；数据处理、优化器、调度、AMP、checkpoint、概率及嵌入导出算法沿用原实现。

默认full基线：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/main_20/full/litemamba/seed_42
```

默认输出：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/litemamba_structure_seed42_v1_0_0
```

新任务位于`full/gatedconv_target/seed_42`及`full/ssm_global_only/seed_42`。结构身份独立记录，两个任务不会被错误合并为同一个litemamba/full/42重复实验。

如需调整路径，支持：

```bash
bash run_structure.sh \
  --engine_dir /原始未修改引擎目录 \
  --data_dir /包含train和validation及test文件的数据目录 \
  --baseline_run /旧full的seed_42目录 \
  --output_dir /新的独立结果目录
```

第一次启动后路径、数据、源码和科学设置冻结；续跑保留相同参数。输入应使用已完成前述实验的同一份六分类数据，旧结果无需搬动。

## 自动预检

真实启动时依次执行：

1. 核对旧full的配置、验证历史、原始指标、已有校准记录及best checkpoint的SHA。
2. 核对原引擎8份源码SHA与旧full一致；核对本包engine只有预定的3份源码发生变化。
3. 核验train/validation/test特征及metadata共六个文件的内容SHA。test仅用于固定缓存、指纹和read隔离审计，不做模型推理。
4. 在当前PyTorch环境运行CPU等价性检查：原/新full相同seed初始化的state_dict、eval输出、固定dropout随机状态下train输出、梯度、BN状态以及真实旧checkpoint的strict加载与前向计算逐张量相等。
5. 验证全局读出只含池化与零分支、零分支/对应权重无数据梯度，验证门控卷积候选不含SSM；记录三个模型的参数统计。
6. 通过后启动单共享资源队列，串行实测两项资源需求，再按统一GPU/RAM/CPU预约预算训练，最后串行导出train/val结果。

日志先出现`[EQUIVALENCE PASS]`才会启动训练。检查使用的旧full权重只用于验证兼容性，**不会拿来初始化新对照模型**。

也可先检查路径和计划，不训练：

```bash
python3 run_structure.py --dry_run
```

dry_run核验小元数据、checkpoint和源码SHA，生成独立dry-run计划；不读取六个大输入，不运行PyTorch等价性检查或训练，不覆盖正式计划。

## 训练设置与资源

科学参数直接从旧full请求配置读取，不提供随意改seed、学习率、epoch或结构任务列表的入口。应对应原实验：seed42、full输入、batch256、最大200epoch、patience30、lr0.001、model_dim96、4blocks、state16、第四位索引3。

并发硬上限2，并读取原容量/降档记录取更低值。两个任务共用一个ResourceController，已预约但尚未实际分配的资源也计入预算；不会并行启动两个互相看不到预约的独立队列。可用`--max_parallel 1`降低并发。

训练权重阶段并行，train/val导出串行。逐epoch记录、logits、checkpoint、最终逐窗口概率、嵌入、注意力、温度校准和分析设置全部沿用基线。是否保存逐epoch预测CSV也沿用基线，通常原设置为不保存该重复CSV；不关闭原来已开启的导出。

启动时输出/cache所在各文件系统默认至少需30 GiB空闲；运行中每2秒检查，低于8 GiB时停止本队列并保留现场。阈值是保护线，不保证覆盖整个实验、其他作业写入或inode/用户配额限制。支持`--startup_free_gib`和`--disk_reserve_gib`调整。建议先准备足够空间，保留原及本次checkpoint、配置和训练记录。

## 看日志、续跑和只重做汇总

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/litemamba_structure_seed42_v1_0_0
tail -F "$RESULTS/logs/structure_latest.log"
```

预检阶段出错时，查看启动器打印的独立启动日志；正式日志可能尚未建立。

中断后，确认旧进程已停止，清理空间并保留全部现场：

```bash
bash run_structure.sh --resume
```

已`train_complete`的任务跳过；`weights_complete`只补导出；有完整last checkpoint的任务恢复模型、优化器、调度器、AMP scaler、历史和随机状态，从最后成功保存的下一epoch继续。中断的未提交epoch重做。没有保存过checkpoint的任务仅在原引擎确认中断状态允许时重新开始。

如果训练和导出已完成，仅汇总失败：

```bash
python3 run_structure.py --summary_only
```

summary_only不会创建训练进程，会核对已有冻结计划、完整的3项结果及已通过的等价性报告。自定义路径参数需照常提供。

## 完成后上传什么

成功日志：

```text
[FULL PASS] 2项seed42结构诊断及验证汇总全部完成，无新增test推理。
[RESULT PACKAGE] .../structure_results_for_analysis_v1_0_0.tar.gz
```

上传 **structure_results_for_analysis_v1_0_0.tar.gz** 即可。包内是3项验证指标、每类结果、完整学习曲线、参数统计、等价性报告、计划与元数据；不复制庞大的权重/概率/embedding，这些保留在各run_dir。

`structure_summary/structure_report.html`可直接查看结果；打包的状态文件是生成汇总之前的快照，显示training_validation_complete是正常的。

## 结论范围

本轮只有固定seed42，属于结构诊断，不自动计算多种子SD或显著性P值。若后续作为正式结构证据，需要两项统一补固定种子，不根据结果方向选择性补跑。本包不开展新的test或Figure4模型推理，也不读取旧full的test指标。

本地验证范围见`VALIDATION.md`。CPU逐张量等价检查不代表所有GPU内核或跨版本/平台训练必定逐位复现；继续使用原pytorch2.0.1环境。参考[PyTorch官方可复现性说明](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md)和[nn.Module的state_dict加载说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。
