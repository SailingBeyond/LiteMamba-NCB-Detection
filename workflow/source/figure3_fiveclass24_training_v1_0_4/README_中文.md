# 五分类24次训练 1.0.4：完整FP32计算恢复版

## 为什么更换1.0.3

1.0.3仅把最后一个分类Linear改为FP32。你的服务器已证实：前向loss有限，但8次独立诊断的反向梯度均非有限，因此程序在任何检查点迁移之前停止。这说明末层修复不足。1.0.4将整个网络的前向和反向改为FP32，四个模型统一执行；不依赖继续降低FP16缩放因子来绕过问题。

模型结构、参数数量/形状/顺序、初始化规则、数据、第四位N、batch size、学习率规则和种子保持不变。保存的Adam、scheduler、GradScaler和RNG状态均继续恢复。原`amp=True`作为已保存的GradScaler启用标志保留，但autocast在所有训练/验证/推理路径明确关闭；配置和历史会记录actual_autocast_enabled=false、compute_precision=float32和`fp32_compute_preserved_scaler_v1`，不会把它描述成仍在进行FP16混合精度。

## 直接运行

原1.0.3队列已因gate失败退出后，将新包解压到paper目录，保留之前所有代码、结果和故障诊断目录，然后运行：

```bash
conda activate pytorch2.0.1
cd /path/to/revision-code/figure3_fiveclass24_training_v1_0_4
bash launch_all.sh --max_parallel 1 --workers 0 --cpu_threads 4
```

启动器自动带`--resume`并后台运行，打印独立日志和`tail -F`命令；无需另加nohup或重新运行诊断包。不要从旧目录启动1.0.3，也不用手动改原脚本的AMP/学习率/batch。

结果仍写入：

```text
/path/to/revision-results/figure3_fiveclass24_results_v1_0_0
```

## 自动执行顺序

1. 取得原队列锁，防止与其他原队列并发。
2. 归档原1.0.3失败的gate报告，读取已捕获且SHA一致的同一故障批次；不重放整轮。
3. 在服务器检查完整FP32前向、所有观测到的叶模块浮点输出dtype、反向梯度、梯度裁剪及一次独立Adam更新。只有全部通过才继续。
4. 验证失败时停止，原检查点不会迁移。通过后备份原计划、检查点、配置及源码，审计迁移原1.0.2状态到1.0.4。你的1.0.3未通过gate，因此原计划和正式检查点仍是1.0.2，正是本次支持的状态。
5. X/GatedConv/seed42从第72轮继续，学习率仍从优化器保存的0.000125延续；其他已有检查点同样从下一轮继续。未开始的模型按原种子启动，仍为两个任务×四模型×三种子共24次。
6. 按原compact方案完成训练、串行导出、所有模型统一冻结后的test和汇总。

同样的FP32策略覆盖GatedConv、Transformer、ConvMixer和BiGRU。完整FP32可能改变单轮用时；容量探测会重新测量，默认单并发以避免之前的随机读取争用。

## 原数据和成果

请保留原CSV、feature_cache及last/best，不用重新提取特征或重跑前71轮。计算精度改变会影响之后的数值轨迹，原历史按原精度保留，新轮次与最终val/test预测使用完整FP32。

迁移只改经过记录的版本/源码/计算精度元数据，所有权重、Adam、scheduler、GradScaler、RNG、epoch、早停状态和history值均逐项核对不变。原始检查点字节先备份，best_state_dict.pt字节保持。原结果目录中的位置为：

```text
precision_fix_validation/gate.json
precision_fix_validation/previous_gates/...
migration/v1_0_2_to_v1_0_4_fp32_compute/backup/
migration/v1_0_2_to_v1_0_4_fp32_compute/staged/
migration/v1_0_2_to_v1_0_4_fp32_compute/transaction.json
```

迁移最后提交experiment_plan。中断后仍使用本版同一命令继续，不能以半迁移状态启动模型。已冻结或test开始、已有完成权重/校准/预测导出、或实际上已经迁移过1.0.3的情况，会拒绝自动迁移，避免混用来源；请保留现场提供日志。

历史最佳验证分数按当时的计算精度保存，明确标记，不冒称已重算；最终导出的验证/测试指标由新FP32模型计算。整个数据划分和模型选择规则不改变。

## 故障批次定位

默认自动读取：

```text
/path/to/revision-results/figure3_fiveclass_nonfinite_probe_v1_0_0/attempt_*
```

若移动过文件，可指定同时含fault_state.pt和failed_batch.npz的原attempt目录或core解压目录：

```bash
bash launch_all.sh --max_parallel 1 --workers 0 --cpu_threads 4 \
  --precision_probe_dir /你的故障材料目录
```

## 看什么输出

先出现：

```text
[FAULT BATCH FP32] ... finite_loss=True finite_grad=True AMP_skip=False
[PRECISION GATE PASS] ...
[MIGRATE] ...
[VERSION] Figure3 fiveclass24 1.0.4 ...
[RUNNING] ... Resume ... 下一 epoch=72
```

日志中的AMP_skip是为保留历史接口而记录的GradScaler更新状态，不表示新版仍有FP16前向。报告中actual_repaired_autocast_enabled=false，修复批次activation_dtypes应仅有torch.float32。

只有全部训练、导出、冻结测试及汇总结束后，`[FULL PASS] phase=all`才表示整个任务完成。若gate仍失败，上传`precision_fix_validation/gate.json`和本次启动日志，此时检查点尚未迁移。完整FP32排除了本次剩余的FP16运算路径，但不保证未来永远不会有其他数值或系统问题。

compact输出仍保留逐轮指标、best/last恢复状态和最终val/test完整预测，不新增逐轮预测、embedding或attention。新增备份仅涉及小型检查点与源码，不复制大型特征或缓存。不要删除原结果。

本地验证范围见VALIDATION.md；真实GPU批次的完整FP32反向由服务器启动时验证，验证通过后才自动恢复。
