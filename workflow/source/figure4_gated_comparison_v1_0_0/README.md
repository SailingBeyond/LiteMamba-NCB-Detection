# Figure4：门控卷积与SSM版full的三种子冻结比较

本包使用已经训练好的42、2024、2025三个种子。复用原full的Figure4预测，新增门控卷积预测，再按相同数据、预处理和指标统一分析。没有训练、微调、温度重拟合或按外部得分挑种子。

## 直接运行

将压缩包上传到服务器原来的paper目录，在原环境执行：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code
unzip figure4_gated_comparison_v1_0_0.zip
cd figure4_gated_comparison_v1_0_0
bash run_gated_comparison.sh
```

启动器自动使用nohup后台运行，无需再套nohup。终端会打印独立日志、PID和查看命令。旧代码、旧权重及旧结果均须保留，勿将本包覆盖解压到旧代码目录。

本包的启动入口为`run_gated_comparison.sh`；继承的Python文件供内部兼容调用。旧版两个shell入口已停止使用，其原文保存在`docs/historical_entrypoints/`。

## 处理范围

| 模型 | seed42 | seed2024 | seed2025 |
| --- | --- | --- | --- |
| full：selective SSM＋第4位直接读出 | 复用旧Figure4 | 复用旧Figure4 | 复用旧Figure4 |
| gatedconv_target：门控卷积＋第4位直接读出 | 新预测 | 新预测 | 新预测 |

数据清单完整继承修正参考后的Figure4计划，包含72个文件、7,258,270个窗口。full共216组逐文件预测经核验后复用；门控卷积新增216组，共21,774,810条窗口预测。每个窗口仍有自己的真值，不能把三个种子的重复评价当成三倍独立样本。

full和门控卷积分别使用与其训练记录匹配的模型源码；门控卷积按真实的`legacy_gatedconv`结构严格加载，不会因模型工厂名称同为litemamba而误加载SSM。类别仍为A/T/C/G/X/Y，第4位仍改为N，dwell与quality缩放沿用原值，使用各自validation拟合的温度。

## 默认路径

默认结果根目录为：

```text
/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721/sixclass_auto_v1_0_4_results
```

该根目录下读取：

- `figure4_external_xna13_16_corrected_v1_1_0`：有效的修正后Figure4计划和full预测。
- `litemamba_structure_confirm3_v1_1_0`：三种子结构确认的冻结计划、指标与模型来源。
- 原训练计划定位的full权重及`litemamba_structure_seed42_v1_0_0`中的门控seed42权重。
- 结构确认目录中的门控seed2024、2025权重。

新输出写入根目录下新建的`figure4_gated_comparison_v1_0_0`文件夹。

路径不同时使用：

```bash
bash run_gated_comparison.sh \
  --previous_output /实际修正后Figure4结果目录 \
  --confirmation_dir /实际三种子结构确认结果目录 \
  --output_root /新的独立比较结果目录
```

实际输入特征和metadata路径来自旧冻结计划，不能改为之前参考错误的XNA13–16文件。脚本核对它们的内容指纹；不会重新发现另一套输入，也不会将旧汇总表直接拼为新总体。

## 复用与信息保存

旧full的权重、配置、校准、数据处理、样本数及逐文件完成记录须一致；预测CSV、logits、labels、embedding、attention按大小和完整SHA核验。默认使用只读用途的绝对软链接引用这些大文件，新completion独立保存，因此必须保留旧结果目录。也可用`--reuse_mode hardlink`或`--reuse_mode copy`；硬链接应视为只读，复制需要额外空间。

旧full预测若缺失或损坏，会明确指出文件并停止，不自动变成一次新的full预测。恢复完整旧文件后同命令续跑即可。

门控预测保存逐窗口CSV、原始logits、labels、embedding、attention、raw与校准概率、文件/样本索引、可用真实read及窗口metadata、完成清单和环境记录。缺少原始字段时保持unknown，不合成真实read ID或窗口坐标。后续分析和绘图使用这些结果，不需要重新预测。

## 多窗口read规则

继承用户的实验设计：XNA1–12及DNA1–12每条read最多1个目标窗口；XNA13–16及DNA13–16最多2个，XNA13/DNA13允许两个6mer有1个碱基重叠；XNA17–20最多4个。少于预期可来自过滤或未覆盖位点，不要求每条read恰好达到上限。

同一read的多个窗口不会因共用read ID而被误判为训练泄漏。能取得真实ID时，按read聚类处理相关性并核对与训练划分的重叠；缺少ID或明确坐标时保留设计声明和可核验范围，继续窗口级预测，不伪称已经独立验证read/坐标。

## 并行、磁盘与恢复

程序先准备共享只读缓存，再串行实测门控结构的forward资源需求。三个门控模型由一个ResourceController调度，默认并发上限3，仍受当前GPU/RAM/CPU预算约束；资源不足时等待或降并发，OOM仅有限重试。可加`--max_parallel 2`或`--max_parallel 1`降低上限。

按当前冻结数据及默认完整导出设置，新增预测的保守预算约39.4 GiB；若旧缓存全部需要重建，约再增加12.8 GiB，分析临时文件及图表还需额外空间。建议开始前准备至少70–80 GiB空闲量，并避免其他作业大量写入。实际预算以启动日志为准，估算不代表写入上限保证。默认保留8 GiB；准备、预测、分析阶段都会检查磁盘，空间过低时停止本队列并保留现场。

完成的逐文件结果只在校验通过后跳过；中断文件从头导出，不从文件内的某个batch继续。清理空间、确认旧队列退出后，在同一代码目录运行：

```bash
bash run_gated_comparison.sh --resume
```

自定义路径时仍提供同一组路径。不要删除权重、配置、完成记录或冻结计划来绕过恢复检查。程序不自动删除旧结果。

只重新分析、绘图、打包：

```bash
bash run_gated_comparison.sh --phase analyze
```

只从已有比较数据重新画图并打包：

```bash
bash run_gated_comparison.sh --phase plot
```

`--phase predict`仅完成准备和预测，不代表完整分析完成。分析和绘图阶段不启动模型预测。

## 输出与上传

原格式的完整统计表保存在`analysis_data/`，包括总体、模板、cohort、类别、motif、来源、方向、阈值、校准和可用read聚类区间。旧Figure4A/B/C兼容CSV继承原版full口径；新双结构比较从通用表和整数混淆矩阵生成，以新比较报告和图为准。

新增`comparison_data/`保存全部三个种子的原值、同seed配对差值、均值及样本SD、模板/角色/cohort与错误组成；打开其中的`comparison_report.html`查看比较报告，新增图在`figures_comparison/`。raw与校准结果分开，主比较保留raw argmax规则，不根据外部数据重新选阈值。

正常完成显示`[FULL PASS]`和`[RESULT PACKAGE]`。上传新输出目录中的：

```text
figure4_gated_comparison_results_for_analysis_v1_0_0.tar.gz
```

该小包包含标准统计、新比较表和图、评估计划、复用/窗口审计、完成记录及训练小元数据；不复制庞大逐窗口CSV、数组、缓存或权重。这些大文件继续留在服务器供后续分析。

## 论文解释范围

这是一轮固定模型的后续外部比较，已有Figure4数据先前已经被观察过，不能重新称为全新盲测。脚本保留全部3个种子，不自动选择赢家、不修改模型或训练设置。门控卷积不含SSM；即使外部结果较好，也不能继续将其解释为SSM机制优势。是否更换主模型，需要结合外部取舍和补充消融的成本判断。

当前验证范围见`TEST_REPORT.md`。继续使用原pytorch2.0.1环境；本包不要求升级PyTorch/CUDA。推理采用模型eval模式及关闭梯度的inference_mode，二者作用不同，参见[PyTorch官方说明](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)。
