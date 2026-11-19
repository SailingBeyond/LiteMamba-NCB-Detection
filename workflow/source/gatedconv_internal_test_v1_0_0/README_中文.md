# GatedConv 固定内部测试补评估 1.0.0

本包只评估现有三个 GatedConv 最佳检查点，种子固定为 **42、2024、2025**。不训练，不改变结构，不搜索检查点，不在测试集拟合温度或阈值。原模型目录和 Figure 4 结果只读使用。请在原来的 `pytorch2.0.1` 环境运行。

## 运行

把压缩包上传到 `/path/to/revision-code/` 并解压，然后执行：

```bash
cd /path/to/revision-code/gatedconv_internal_test_v1_0_0
bash run_internal_test.sh
```

默认输出：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results/gatedconv_internal_test_v1_0_0
```

脚本会打印本次独立日志路径，复制其 `tail -F` 指令即可查看。`[FULL PASS]` 才表示三个种子的预测、分析、比较和打包全部完成。刚启动打印的 PID 只表示请求已提交。

## 输入与冻结规则

- 测试文件固定为原 `sixclass_overlap_read_split_721/test.csv` 与 `test_meta.tsv`，默认应有 250,188 个窗口、每类 41,698 个。实际内容 SHA256 必须与三个门控模型以及比较模型的训练记录完全一致；不重新划分、下采样或换数据。
- 种子 42 来自 `sixclass_auto_v1_0_4_results/litemamba_structure_seed42_v1_0_0/full/gatedconv_target/seed_42/best_checkpoint.pt`。
- 种子 2024、2025 来自 `sixclass_auto_v1_0_4_results/litemamba_structure_confirm3_v1_1_0/full/gatedconv_target/seed_2024`、`seed_2025` 的 `best_checkpoint.pt`。
- 比较模型的既有内部测试来自 `main_20/full/{transformer,convmixer,bigru,litemamba}/seed_N/results.json`，不重新推理这些模型。主要比较统一用相同三个种子；原四模型的完整五种子结果另列，不能与门控三种子直接混合排序。
- 保持训练时第四位 N、目标读出 `target_index=3`、类别顺序 A/T/C/G/X/Y、每碱基 30 个采样点、dwell÷60、quality÷50；具体参数以通过指纹校验的保存配置为准。沿用各自验证集拟合的正温度参数，决策为固定六类 argmax。
- 随包模型、数据和配置代码必须逐字节匹配训练快照。只读取 `best_checkpoint.pt`，校验其 SHA256 与结果、验证集校准记录一致，严格加载 state dict。
- 评估计划在正式预测前冻结；修改输入、权重、代码或科学设置后不能继续冒用旧计划。
- 初始结构比较和 Figure 4 已被观察，本次是返修中的冻结内部测试补评估；不能称作新采集的盲测实验。结果不用于继续搜索同一测试集上的更好结构。

## 并行与空间

默认先实测完整批大小的推理资源，最多同时运行三个预测进程；GPU、RAM、CPU 预算不足时少开。GPU 监测直接使用 PyTorch 逻辑设备的 `mem_get_info`，无需猜测物理卡编号或强制要求 UUID。预测完成后逐个进行 CPU 分析，减少分析内存叠加。

每批 256，DataLoader workers=0，每进程 CPU 线程=4，保持训练保存的 AMP 策略，默认保存 float32 嵌入和注意力。资源探测不是标准速度比较，不能用它宣称模型更快。默认另外保留 GPU 2 GiB、RAM 3 GiB、磁盘 8 GiB；首次额外预留 4 GiB 给测试缓存、完整预测和分析产物。程序运行期间继续检查磁盘。

可限制并行：

```bash
bash run_internal_test.sh --max_parallel 1
```

前台运行并直接看到错误：

```bash
python3 -u run_internal_test.py --max_parallel 3
```

## 中断续跑与离线分析

清理空间或释放资源，确认原进程已结束后：

```bash
bash run_internal_test.sh --resume
```

完整的预测经哈希校验后复用；未完成的种子重新预测，完成预测而未完成分析的只补分析。不会重新训练。单个种子中途留下的 `.partial` 数据不能当作完整预测；如恰在写完完整推理标记后中断，则续跑可以直接补统计。

只补离线分析和打包：

```bash
bash run_internal_test.sh --phase analyze --resume
```

若需要再次绘图或计算其他统计，可直接读取服务器保存的逐样本概率和元数据，不必重新推理。

## 产物与上传

- `analysis/report.html`：总体结论、同三种子比较、逐类结果和限制说明。
- `analysis/*.csv`：每次运行、平均值和样本标准差、配对差值、逐类结果、原四模型五种子附表。
- `runs/gatedconv_target/seed_N/test/`：logits、真实标签、原始/校准概率、预测类别、192 维嵌入、六位注意力、带 read/来源/序列元数据的 `predictions.csv.gz` 及完整分析表。
- `source_metadata/`：原始训练、验证集校准和内部划分审计元数据副本。原目录不改写。
- `evaluation_plan.json`、完成记录、文件指纹、环境和资源日志：核实版本与输入来源。
- `gatedconv_internal_test_results_for_analysis_v1_0_0.tar.gz`：自动整理的小型分析包，完成后上传这个文件即可。庞大的逐样本数组和预测 CSV 留在服务器，不放入小包，但包含它们的哈希与数量记录。

不把每条窗口都当成独立 read；每个种子的离线分析沿用原分析器的 read-cluster bootstrap，阳性 CSV 行代理仍受原实验一条 read 一个阳性窗口的假设约束。数据来源与链方向混杂等原有限制仍需在论文中说明。三种子平均值排序和标准差不自动等于统计显著性。

## 路径发生变化

可显式指定根目录或个别种子的原运行目录。例如：

```bash
bash run_internal_test.sh \
  --data_dir /原固定划分目录 \
  --results_dir /原训练结果根目录 \
  --output_dir /新输出目录 \
  --seed42_dir /种子42原运行目录 \
  --seed2024_dir /种子2024原运行目录 \
  --seed2025_dir /种子2025原运行目录
```

这些参数只允许定位原文件；内容仍必须匹配原指纹，不能借路径参数替换测试数据或权重。
