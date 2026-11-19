# LiteMamba 三种子结构确认 v1.1.0

本包继续已完成的两项 seed42 结构诊断，只新增 **4 次从头训练**：两项对照分别补种子 2024、2025。复用已有 3 个 full 及 2 个 seed42 对照，最后统一汇总 **3 个结构 × 3 个种子 = 9 个结果**。不预设 SSM 或门控卷积必胜，不挑选有利种子。

## 直接启动

将压缩包放到服务器原来的 `paper` 目录，使用原环境运行：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip litemamba_structure_confirm3_v1_1_0.zip
cd litemamba_structure_confirm3_v1_1_0
bash run_structure.sh
```

脚本已通过 `nohup` 后台提交，无需再套 `nohup`。控制台输出 PID、独立启动日志和查看命令；PID 表示启动请求已提交，预检通过后才开始训练。保留原来的 `sixclass_training_auto_v1_0_4` 代码目录和全部已有结果，不要将本包覆盖解压到旧目录。

| 结构 | seed42 | seed2024 | seed2025 |
| --- | --- | --- | --- |
| full：selective SSM＋第四位直接读出 | 复用 | 复用 | 复用 |
| gatedconv_target：门控卷积＋第四位直接读出 | 复用 | 新训练 | 新训练 |
| ssm_global_only：selective SSM＋仅全局读出 | 复用 | 新训练 | 新训练 |

## 默认输入与输出

数据目录保持为：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721
```

在该目录下的 `sixclass_auto_v1_0_4_results` 中，程序默认读取：

- `main_20/full/litemamba/seed_42`、`seed_2024`、`seed_2025`：3 个原 full。
- `litemamba_structure_seed42_v1_0_0`：上轮结构计划、等价性报告、比较记录及 2 个 seed42 对照。

新输出目录：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/litemamba_structure_confirm3_v1_1_0
```

4 个新运行分别位于新输出目录的 `full/gatedconv_target/seed_2024`、`seed_2025` 和 `full/ssm_global_only/seed_2024`、`seed_2025`。复用结果保持原位，只读校验，不复制庞大权重或重新导出旧运行。

自定义路径时使用：

```bash
bash run_structure.sh \
  --engine_dir /原始sixclass_training_auto_v1_0_4目录 \
  --data_dir /原六分类数据目录 \
  --baseline_root /原main_20/full/litemamba目录 \
  --previous_structure_dir /原litemamba_structure_seed42_v1_0_0结果目录 \
  --output_dir /新的独立确认结果目录
```

后续续跑或只重做汇总时继续提供相同自定义路径。

## 预检、训练设置和信息保存

程序先核对 5 个复用结果的配置、数据指纹、源码版本、验证指标和检查点来源，并核对上轮冻结计划与等价性报告；不兼容时停止，避免混合不同实验。正式运行还核验 6 个大输入文件的内容 SHA。数据处理保留第 4 位 N、原 read 划分、类别顺序和全量特征。

原始引擎、上轮结构引擎和本次确认引擎分别记录 SHA。相比上轮，本包扩展队列对新增固定种子 2024/2025 的准入，并修复逐轮文件与检查点的提交顺序；模型计算、优化器、调度及随机状态计算沿用上轮。服务器预检再次运行 CPU 等价检查，确认原/新 full 初始化、前向、梯度、BN 状态和旧检查点兼容，检查通过后才训练。

学习率、batch size、优化器、AMP、调度、早停和导出设置从原 full 读取并冻结；仍按 `val_loss` 选检查点，最大 200 epoch、patience 30 等参数保持原实验口径，不事后改挑 F1 峰值。新增任务从头训练，旧权重仅供兼容性检查。保存 checkpoint、逐 epoch 指标和 logits、最终逐窗口概率、embedding、attention、校准、数据与源码指纹等原引擎信息；是否保存重复的逐 epoch 预测 CSV 也沿用原设置。

本轮只训练及验证，不新增 test 或 Figure4 模型推理。原引擎预检仍可读取 test 输入用于缓存、内容指纹和 read 隔离审计，这与执行 test 模型预测不同。

## 并行与空间

并发硬上限为 **2**，结合旧容量/降档记录以及当前 GPU、RAM、CPU 实测预约预算自动准入，资源不足会降为串行。4 个任务使用一个共享队列；训练权重阶段可并行，最终 train/val 导出串行。可加 `--max_parallel 1` 主动降低上限。保留完整输出时，磁盘写入也可能成为耗时来源。

启动时输出/cache 所在文件系统默认至少需要 **30 GiB 空闲**；运行中每 2 秒检查，低于 **8 GiB** 停止本队列并保留现场。阈值只是保护线，不能保证剩余全部任务、其他作业写入、inode 或用户配额不会再次耗尽空间。程序不会自动删除旧结果、checkpoint 或概率文件。可用 `--startup_free_gib`、`--disk_reserve_gib` 调整保护线。

逐轮历史、NPZ 和可选预测 CSV 先完整写入临时文件并发布，成功后才提交 last checkpoint；写入失败的未提交轮次由续跑重做。若首次 checkpoint 前磁盘已经满到无法保存任何中断记录，程序会拒绝无法核验的自动重启，需要保留现场排查。

## 查看日志与恢复

```bash
RESULTS=/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/litemamba_structure_confirm3_v1_1_0
tail -F "$RESULTS/logs/structure_latest.log"
```

预检阶段失败时先看启动器打印的独立启动日志。空间不足或其他中断后，确认旧队列已结束、清理空间，保留现场再运行：

```bash
bash run_structure.sh --resume
```

`train_complete` 的新任务跳过；`weights_complete` 只补导出；有完整 last checkpoint 的任务恢复模型、优化器、调度器、AMP scaler、历史与随机状态，从最后成功提交的下一 epoch 继续，未提交的 epoch 重做。尚未保存 checkpoint 的任务仅在原引擎允许的中断状态下重新开始。

如果 4 次训练及导出已经完成，仅汇总失败：

```bash
python3 run_structure.py --summary_only
```

只查看计划、检查小元数据与源码指纹而不训练：

```bash
python3 run_structure.py --dry_run
```

`dry_run` 不读取 6 个大输入，不运行 PyTorch 等价检查，也不覆盖正式冻结计划。

## 完成后上传什么

成功后日志显示 `[FULL PASS]` 和 `[RESULT PACKAGE]`。上传新结果目录中的：

```text
structure_confirm3_results_for_analysis_v1_1_0.tar.gz
```

小结果包包含 9 个运行的 raw 验证指标、同种子配对差值、3 种子均值及样本标准差、每类指标、完整学习曲线、参数/校准记录、计划与来源元数据。大权重、逐窗口概率和 embedding 保留在各运行目录，不会为上传再次复制。汇总 HTML 可直接用浏览器打开。

## 解释边界

3 个种子是同一数据划分上的训练随机重复，不是 3 个独立数据集。样本标准差用于描述种子波动，不自动产生显著性或外部泛化结论。主比较为每个种子对齐后的“对照 − full”差值，9 个结果全部保留。

门控卷积替换整个混合模块，参数与运算不同，不是严格等参数的单项递推删除。global-only 将第四位直接分支置零，第四位仍参与全局编码和 attention；保留名义分类器宽度时有 12,288 个权重没有数据梯度，因此名义参数相同不等于有效输入容量相同。

本轮不保证提分，也不证明结构新颖性或 LiteMamba 优于 Transformer。若门控卷积优势重复出现，论文应据实报告，不能将不含 SSM 的门控卷积对照解释为 SSM 机制创新。

本地验证范围见 `VALIDATION.md`。继续使用原 `pytorch2.0.1` 环境；CPU 等价检查不保证所有 GPU 内核跨平台或跨版本逐位复现。参见 [PyTorch 可复现性说明](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md)与[通用检查点恢复说明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)。
