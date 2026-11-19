# Figure 3 六分类训练 v1.0.4

本包根据当前服务器资源，先实测四种模型，再验证 4→3→2→1 路并发，随后自动运行四模型 × 五个 seed 的 **20 次主比较实验**。训练权重阶段并行，完整预测、embedding 和统计分析阶段串行。输出统一保存在原数据目录中新建的子文件夹中。

## 直接运行

使用现有 `pytorch2.0.1` 环境，将本包解压到新的代码目录，完整保留同级文件：

```bash
cd /path/to/revision-code
unzip sixclass_training_auto_v1_0_4.zip
cd sixclass_training_auto_v1_0_4
bash run_auto_max4_v1_0_4.sh
```

脚本内部已使用 `nohup ... &`，不需要再套一次 nohup。依赖见 `requirements.txt`；已有依赖满足时无需重新安装。支持 PyTorch 2.0.1，NumPy 应保持 <2；无需安装 `mamba-ssm`。

默认输入目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721
```

默认新建输出目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results
```

缓存、容量探测、训练结果和日志都放在这个输出目录中。原来的六个输入文件不修改。指定其他 `--data_dir` 后，默认输出也跟随该数据目录；只有显式传 `--output_root` 才覆盖默认位置。

查看主日志：

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results
tail -f "$RESULTS/logs/auto_max4_v1_0_4.log"
```

脚本会先打印本次独立日志文件。程序取得输出目录锁后才更新正式 PID 和上面的最新日志链接；如果重复启动，原运行的 PID 和日志不被覆盖。正式训练队列的每次调用另存为 `logs/main_20_<时间>_<PID>.log`。

## 本版修复与并发规则

- 并发探测前，先串行测量四种模型的真实前向、反向、AdamW 更新和单批 validation 推理。根据测量值、安全系数、RAM/GPU 预留与 CPU 预算筛选候选组合；共享只读缓存只计一次，每个作业的私有内存单独预约。
- 每次启动并发探针前重新检查资源，并立即登记尚未实际分配的预约。所有探针首批数据进入设备、模型和优化器就绪后，同步放行；每路至少完成 8 次有效优化器更新，且默认累计至少 15 秒有效更新计算时间。AMP 跳过的更新不计数。
- 4 路预算不够或探测未通过，依次试 3、2、1 路。只有同步训练时段和 GPU 自有进程重叠有证据，才接受多路档位。容量报告保存测量值、尝试、失败分类与资源轨迹。
- 测得的档位是进程硬上限。正式队列仍按每个作业的实测预算动态准入，实际并发可以更低。不会通过减小 batch、少跑 seed 或缩短正式训练来增加并发。
- 恢复参数正确传入训练子进程；有 checkpoint 时恢复模型、优化器、调度器、AMP 和 RNG。首次 checkpoint 前中断，只有匹配配置的队列中断记录等有效证据才允许重启，保留原尝试日志。
- OOM/资源不足采用有界降并发重试；SIGKILL 原因标为未知，不直接认定为 OOM。模型或数据异常会停止流程。恢复上限取历史通过档位、已降低档位和当前 `--max_parallel` 的最小值。

默认参数：batch=256、epochs=200、patience=30、lr=0.001、workers=0、每作业 CPU threads=4；GPU 预留为 2 GiB 与总量 10% 的较大值，RAM 预留为 3 GiB 与总量 10% 的较大值，资源安全系数 1.35，另加每作业 1 GiB 训练 RAM 余量。默认使用逻辑设备 `cuda:0`。GPU UUID 沿用驱动上下文核实方式，兼容 PyTorch 2.0.1 缺少属性 UUID 的情况。

短程探测不能保证后续永不 OOM，也不保证四路比两路更快。其他用户作业、I/O、CPU 与 GPU 竞争都会影响实际吞吐。程序只清理自己启动的进程；不处理其他用户的训练任务。运行环境没有真实 CUDA，本版验证范围见 `VALIDATION.md`。

## 只探测、恢复与停止

只验证资源，不开始 20 次训练：

```bash
bash run_auto_max4_v1_0_4.sh --probe_only
```

探测通过后开始正式训练，或恢复本版中断的正式队列：

```bash
bash run_auto_max4_v1_0_4.sh --resume_main
```

恢复时进一步降低并发硬上限：

```bash
bash run_auto_max4_v1_0_4.sh --resume_main --max_parallel 2
```

探测中断、容量报告没有通过，或设备/运行时资源设置改变后重新测量：

```bash
bash run_auto_max4_v1_0_4.sh --reprobe
```

已有正式队列时用 `--resume_main --reprobe`。每次探测使用独立目录；旧尝试和容量报告会保留。源代码、数据或科学训练设置改变时，重新探测不代表允许混用旧训练结果，正式队列仍会拒绝不兼容恢复，应使用新的输出子目录。

正常停止当前自动程序：

```bash
kill -TERM "$(cat "$RESULTS/logs/auto_max4_v1_0_4.pid")"
```

请先在当前 shell 设置上面的 `RESULTS`。有效 checkpoint 按完整 epoch 写入，尚未写入的进度需要重算。不要使用旧 v1.0.3 结果目录直接续训本版：源码检查会拒绝混用。

## 数据与实验固定规则

输入目录需要 `train.csv`、`validation.csv`、`test.csv` 及逐行对应的 `train_meta.tsv`、`validation_meta.tsv`、`test_meta.tsv`。内部将 validation 简称为 val。这里继续使用已完成的 7:2:1 划分，不再次下采样或移动样本。

标签固定为 **0=A、1=T、2=C、3=G、4=X、5=Y**；上下文 one-hot 通道顺序为 A/C/T/G/N。目标为第 4 位（Python index=3），该位序列统一改为 N，目标电流、统计量及 basecalling 响应保留。默认 raw length=30、dwell divisor=60、quality divisor=50，沿用前版实际训练参数。

支持均衡划分输出中明确标记的阳性 row proxy，不再寻找原始 `positive_meta`。这是基于“一条阳性 read 对应一个提取窗口”的实验假设；不能据此声称独立核验了缺失的原始 UUID、模板或 run。阴性同一 read 可含多个窗口、多个真实类别；各窗口真值保留，统计使用 read cluster。已知跨集合 read/proxy 冲突会拒绝训练。

默认模型为 Transformer、LiteMamba、ConvMixer、BiGRU；seed 为 42、2024、2025、2026、2027，特征方案为 full。四模型的实现和数据预处理相对上传的 v1.0.3 保持一致，本次改动集中在调度、恢复与记录。不会预设模型排名。LiteMamba 使用已有的 pure-PyTorch selective SSM，第 4 位目标读出加全局注意力，架构说明见 `MODEL_CHANGES.md`。

## 测试集使用与保存内容

所有预先列出的训练先完成权重训练，再串行导出 train/val；checkpoint 选择、early stopping、调度器及温度缩放只用 validation。全队列最佳 checkpoint 的 SHA256 冻结后，才串行进行 test 推理和独立效率基准。任一训练失败会阻止全队列测试。

每次实验保存在 `main_20/full/<model>/seed_<seed>/`，主要包括：

| 文件或目录 | 用途 |
|---|---|
| `resolved_config.json`、环境记录、源码快照、输入 SHA256 和数据审计 | 复现输入、设置、身份假设与代码 |
| `best_checkpoint.pt`、`last_checkpoint.pt`、`best_state_dict.pt` | best/last 完整状态、恢复训练和冻结核验 |
| `metrics.csv`、`epoch_metrics.jsonl`、`epoch_predictions/val_epoch_*.npz` | 每轮训练/验证曲线和六类验证 logits |
| `train/`、`val/`、`test/` | labels、logits、原始/校准概率、预测、embedding、attention 和 metadata 对齐导出 |
| 各 split 的 `predictions.csv.gz`、`analysis/` | 每窗口六类概率及来源字段，混淆矩阵、ROC/PR、校准、分层分析 |
| 队列的 `commands.jsonl`、`resource_events.jsonl`、`resource_profiles/` | 实际命令、PID、尝试、资源预约与测量 |
| `evaluation_plan.json`、`queue_manifest.json`、`aggregate/` | 测试前冻结、完成状态和跨 seed 汇总 |

详细文件名以实际生成目录为准。默认保存 embedding、attention 和每轮 val logits，磁盘使用随样本数和 epoch 增加。完整分析会用到主机内存，采用串行导出以减少并行占用。

后续重新画图无需训练，可直接读已保存的预测：

```bash
python3 analyze_results.py analyze \
  --predictions "$RESULTS/main_20/full/litemamba/seed_42/test/predictions.csv.gz" \
  --output_dir "$RESULTS/redraw_litemamba_seed42_test" \
  --bootstrap 1000 --subgroup_bootstrap 200 --seed 42
```

置信区间按 read cluster 重采样；逐窗口 Wilson 区间仅作描述。未知模板/run 保持未知，不能通过重绘补出真实来源。代码覆盖范围与仍需真实实验验证的审稿事项见 `reviewer_coverage.md`，不将合成测试当作真实准确率或泛化证据。

## 单独运行 45 次 LiteMamba 特征消融

主比较默认不包含下面队列。需要时单独运行，并使用同一个数据目录下的另一个新输出子目录；请避免在自动主队列还运行时另开独立调度器争抢资源。

```bash
python3 -u run_all_4_models.py \
  --data_dir /path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721 \
  --output_dir /path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_litemamba_ablation_v1_0_4_results \
  --models litemamba --seeds 42 2024 2025 2026 2027 \
  --feature_schemes full raw_only context_only responses_only statistics_only no_raw no_context no_responses no_statistics \
  --phase all --staged_exports --device auto --max_parallel 4 \
  --workers 0 --cpu_threads 4 --batch_size 256 --epochs 200 --patience 30 --lr 0.001 \
  --gpu_reserve_gib 2 --ram_reserve_gib 3 --reserve_fraction 0.10 \
  --memory_safety_factor 1.35 --train_ram_extra_gib 1.0
```

独立 runner 同样先逐模型/方案实测，再动态准入；恢复时原命令追加 `--resume_queue`。它是前台命令，后台运行可自行使用独立 nohup 日志。`context_only` 的目标字符仍为 N；保存所有方案和 seed，不能只保留表现最佳项。

历史说明收在 `docs_previous_versions/`，其中的旧路径和启动命令不适用于本版。完整包文件校验见 `MANIFEST_SHA256.txt`。
