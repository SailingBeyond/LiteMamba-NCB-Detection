#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-disjoint linear diagnostics for measured strand/source/run identities.

This is a deliberately simple diagnostic baseline, not a competitor selected for
the PC/XNA task. Failure to predict a nuisance variable does not exclude confounding.
"""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sqlite3
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from IterDataset import prepare_dataset


DEFAULT_DATA_DIR = (
    "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/"
    "fast5_single_guppy/binary_read_split_721"
)
TARGET_COLUMNS = {"strand": "bam_strand", "source": "source_dataset", "run": "run_id"}
SPLITS = ("train", "val", "test")
UNKNOWN = {"", "unknown", "na", "n/a", "none", "null", "nan"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target", choices=TARGET_COLUMNS, default="strand")
    p.add_argument("--target_column", help="显式可靠 metadata 字段；默认 strand/source/run 对应 bam_strand/source_dataset/run_id")
    p.add_argument("--feature_scheme", choices=("full", "raw", "context", "responses", "statistics"), default="full")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--raw_signal_len", type=int, default=30)
    p.add_argument("--dwell_divisor", type=float, default=60.0)
    p.add_argument("--quality_divisor", type=float, default=50.0)
    p.add_argument("--cache_dir")
    return p


def feature_matrix(dataset, indices, scheme):
    merged = np.asarray(dataset._merged[indices], dtype=np.float32)
    if scheme == "context":
        return merged[:, :, :5].reshape(len(indices), -1)
    if scheme == "statistics":
        return merged[:, :, 5:9].reshape(len(indices), -1)
    if scheme == "responses":
        return merged[:, :, 9:13].reshape(len(indices), -1)
    raw = np.asarray(dataset._raw[indices], dtype=np.float32).reshape(len(indices), -1)
    return raw if scheme == "raw" else np.concatenate([merged.reshape(len(indices), -1), raw], axis=1)


def prepare_inputs(args, output):
    """Prepare deterministic features, audit six inputs, retain one row per read."""
    column = args.target_column or TARGET_COLUMNS[args.target]
    root = Path(args.data_dir).expanduser().resolve()
    connection = sqlite3.connect(str(output / "read_membership.sqlite"))
    connection.execute("CREATE TABLE reads (read_id TEXT PRIMARY KEY, split TEXT, target TEXT, binary_label INTEGER, chrom TEXT)")
    datasets, selected, fingerprints = {}, {}, {}
    counts = {name: Counter() for name in SPLITS}
    duplicates = {name: 0 for name in SPLITS}
    try:
        for strand in ("pos", "neg"):
            for split in SPLITS:
                feature = root / f"xna_pc_6mer_features_{strand}_balanced_{split}.csv"
                meta = feature.with_name(feature.stem + "_meta.tsv")
                if not meta.is_file():
                    alternate = feature.with_suffix(".meta.tsv")
                    meta = alternate if alternate.is_file() else meta
                prepared = prepare_dataset(
                    str(feature), str(meta), cache_dir=args.cache_dir,
                    raw_signal_len=args.raw_signal_len, dwell_divisor=args.dwell_divisor,
                    quality_divisor=args.quality_divisor, split_name=f"probe_{strand}_{split}",
                )
                dataset = prepared["dataset"]
                dataset._open_arrays()
                key = (strand, split)
                datasets[key], selected[key] = dataset, []
                fingerprints[f"{strand}_{split}"] = prepared["input_fingerprints"]
                for index, metadata in enumerate(dataset.iter_metadata()):
                    read_id = str(metadata["read_id"]).strip()
                    target = str(metadata.get(column, "")).strip()
                    if target.lower() in UNKNOWN:
                        raise ValueError(f"{feature}: read {read_id} 缺少可靠的 {column}；不从文件名推断来源或 run。")
                    if args.target == "strand" and target not in ("+", "-"):
                        raise ValueError("strand 目标必须为真实 metadata 的 + 或 -。")
                    binary_label = int(metadata["label"])
                    chrom = str(metadata["chrom"])
                    previous = connection.execute("SELECT split,target,binary_label,chrom FROM reads WHERE read_id=?", (read_id,)).fetchone()
                    if previous:
                        if previous[0] != split:
                            raise ValueError(f"read 跨集合：{read_id}: {previous[0]} 与 {split}")
                        if previous[1] != target:
                            raise ValueError(f"同一 read 的目标 metadata 冲突：{read_id}: {previous[1]} 与 {target}")
                        if previous[2] != binary_label or previous[3] != chrom:
                            raise ValueError(f"同一 read 的二分类标签或参考身份冲突：{read_id}: {previous[2:]} 与 {(binary_label, chrom)}")
                        duplicates[split] += 1
                        continue
                    connection.execute("INSERT INTO reads VALUES (?,?,?,?,?)", (read_id, split, target, binary_label, chrom))
                    selected[key].append(index)
                    counts[split][target] += 1
                connection.commit()
        classes = sorted(counts["train"])
        if len(classes) < 2:
            raise ValueError(f"目标 {column} 的训练集只有 {len(classes)} 类，不能训练来源／方向诊断。")
        for split in SPLITS:
            if set(counts[split]) != set(classes):
                raise ValueError(f"{split} 的目标类别与 train 不一致：{dict(counts[split])}；请提供各集合类别可评价的数据。")
        mapping = {name: index for index, name in enumerate(classes)}
        dimension = {"raw": 6 * args.raw_signal_len, "context": 30, "statistics": 24,
                     "responses": 24, "full": 78 + 6 * args.raw_signal_len}[args.feature_scheme]
        data = {}
        for split in SPLITS:
            n = sum(counts[split].values())
            x = np.lib.format.open_memmap(output / f"{split}_features.npy", mode="w+", dtype="float32", shape=(n, dimension))
            y = np.lib.format.open_memmap(output / f"{split}_labels.npy", mode="w+", dtype="int64", shape=(n,))
            cursor = 0
            with (output / f"{split}_metadata.jsonl").open("w", encoding="utf-8") as handle:
                for strand in ("pos", "neg"):
                    dataset, indices = datasets[strand, split], selected[strand, split]
                    for start in range(0, len(indices), args.batch_size):
                        batch = indices[start:start + args.batch_size]
                        values = feature_matrix(dataset, batch, args.feature_scheme)
                        x[cursor:cursor + len(batch)] = values
                        for offset, index in enumerate(batch):
                            metadata = dataset.get_metadata(index)
                            metadata["probe_target"] = str(metadata[column]).strip()
                            y[cursor + offset] = mapping[metadata["probe_target"]]
                            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                        cursor += len(batch)
            x.flush(); y.flush()
            data[split] = (x, y)
        audit = {
            "target_column": column, "class_mapping": mapping,
            "unique_read_counts": {split: dict(values) for split, values in counts.items()},
            "duplicate_rows_removed": duplicates, "read_overlap_across_splits": 0,
            "sampling_unit": "one deterministic first record per read across pos/neg inputs",
            "input_fingerprints": fingerprints,
            "caution": "Low probe accuracy does not rule out confounding; target metadata must be independently reliable.",
        }
        write_json(output / "input_audit.json", audit)
        write_json(output / "class_mapping.json", mapping)
        return data, audit
    finally:
        connection.close()
        for dataset in datasets.values():
            dataset.close()


def metrics(y, probability, classes):
    prediction = np.argmax(probability, axis=1)
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probability, labels=np.arange(len(classes)))),
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(len(classes))).tolist(),
        "class_order": classes,
    }


def predict(clf, scaler, x, batch_size):
    result = np.empty((len(x), len(clf.classes_)), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        values = scaler.transform(np.asarray(x[start:start + batch_size]))
        result[start:start + len(values)] = clf.predict_proba(values)
    if not np.isfinite(result).all():
        raise ValueError("SGD 概率出现非有限值；请检查数据与正则化参数。")
    row_sums = result.sum(axis=1, keepdims=True)
    if (result < 0).any() or (row_sums <= 0).any():
        raise ValueError("SGD 返回无效概率。")
    result /= row_sums
    return result


def export_predictions(output, split, seed, data, model, scaler, classes, batch_size):
    x, y = data[split]
    probability = predict(model, scaler, x, batch_size)
    directory = output / f"seed_{seed}"
    columns = ["read_id", "chrom", "bam_strand", "motif", "run_id", "source_dataset",
               "binary_label", "true_class_index", "true_class", "predicted_class_index", "predicted_class"]
    columns += [f"probability_class_{index}" for index in range(len(classes))]
    with gzip.open(directory / f"{split}_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        with (output / f"{split}_metadata.jsonl").open(encoding="utf-8") as metadata_file:
            for index, line in enumerate(metadata_file):
                metadata = json.loads(line)
                predicted = int(np.argmax(probability[index]))
                record = {key: metadata.get(key, "unknown") for key in columns[:6]}
                record.update(binary_label=metadata.get("label"), true_class_index=int(y[index]),
                              true_class=classes[int(y[index])], predicted_class_index=predicted,
                              predicted_class=classes[predicted])
                record.update({f"probability_class_{c}": float(probability[index, c]) for c in range(len(classes))})
                writer.writerow(record)
    result = metrics(y, probability, classes)
    write_json(directory / f"{split}_metrics.json", result)
    return result


def main(argv=None):
    provided = {value.split("=", 1)[0][2:] for value in (argv if argv is not None else sys.argv[1:]) if value.startswith("--")}
    args = parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    config_path = output / "resolved_config.json"
    if args.phase == "evaluate":
        saved = read_json(config_path)
        for key, value in saved["arguments"].items():
            if key not in provided and key not in ("phase", "output_dir"):
                setattr(args, key, value)
        for key in provided - {"phase", "output_dir"}:
            if saved["arguments"].get(key) != getattr(args, key):
                raise ValueError("评价不得改变固定 probe 设置：" + key)
        if saved["script_sha256"] != sha256(__file__):
            raise ValueError("Probe 源码已改变；请使用训练时源码评价。")
        if saved["data_loader_sha256"] != sha256(sys.modules["IterDataset"].__file__):
            raise ValueError("数据预处理源码已改变；请使用训练时源码评价。")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("输出目录已有内容；请使用新目录，或 --phase evaluate 评价已固定模型。")
    if min(args.epochs, args.patience, args.batch_size, args.raw_signal_len) < 1 or args.alpha <= 0:
        raise ValueError("epochs/patience/batch_size/raw_signal_len/alpha 必须为正数。")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds 不能重复。")
    output.mkdir(parents=True, exist_ok=True)
    if args.phase != "evaluate":
        write_json(config_path, {"arguments": vars(args), "script_sha256": sha256(__file__),
                                "data_loader_sha256": sha256(sys.modules["IterDataset"].__file__),
                                "purpose": "linear metadata diagnostic; not the PC/XNA model ranking"})
        versions = {}
        for name in ("numpy", "scikit-learn", "joblib", "torch"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        write_json(output / "environment.json", {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions,
            "data_preprocessing_version": getattr(sys.modules["IterDataset"], "CACHE_VERSION", "unknown"),
        })
        data, audit = prepare_inputs(args, output)
    else:
        audit = read_json(output / "input_audit.json")
        data = {split: (np.load(output / f"{split}_features.npy", mmap_mode="r"),
                        np.load(output / f"{split}_labels.npy", mmap_mode="r")) for split in SPLITS}
    classes = [name for name, _ in sorted(audit["class_mapping"].items(), key=lambda item: item[1])]
    all_metrics = {}
    if args.phase != "evaluate":
        scaler = StandardScaler()
        x_train, y_train = data["train"]
        for start in range(0, len(x_train), args.batch_size):
            scaler.partial_fit(np.asarray(x_train[start:start + args.batch_size]))
        joblib.dump(scaler, output / "scaler.joblib")
        weights = len(y_train) / (len(classes) * np.bincount(y_train, minlength=len(classes)))
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed)
            clf = SGDClassifier(loss="log_loss", alpha=args.alpha, random_state=seed,
                                learning_rate="optimal", fit_intercept=True, average=True)
            directory = output / f"seed_{seed}"
            directory.mkdir()
            history, best_score, patience = [], None, 0
            for epoch in range(1, args.epochs + 1):
                order = rng.permutation(len(x_train))
                for start in range(0, len(order), args.batch_size):
                    indices = order[start:start + args.batch_size]
                    x = scaler.transform(np.asarray(x_train[indices]))
                    y = np.asarray(y_train[indices])
                    clf.partial_fit(x, y, classes=np.arange(len(classes)), sample_weight=weights[y])
                val_probability = predict(clf, scaler, data["val"][0], args.batch_size)
                val = metrics(data["val"][1], val_probability, classes)
                score = (val["macro_f1"], -val["log_loss"])
                history.append({"epoch": epoch, "val": val})
                if best_score is None or score > best_score:
                    best_score, patience = score, 0
                    joblib.dump({"classifier": clf, "scaler": scaler, "class_order": classes}, directory / "probe.joblib")
                else:
                    patience += 1
                if patience >= args.patience:
                    break
            write_json(directory / "history.json", history)
            fitted = joblib.load(directory / "probe.joblib")
            all_metrics[str(seed)] = {"val": export_predictions(output, "val", seed, data,
                fitted["classifier"], fitted["scaler"], classes, args.batch_size)}
        plan = {"seeds": args.seeds, "target": args.target, "class_order": classes,
                "config_sha256": sha256(config_path), "input_audit_sha256": sha256(output / "input_audit.json"),
                "models": {str(seed): sha256(output / f"seed_{seed}" / "probe.joblib") for seed in args.seeds},
                "feature_cache_sha256": {split: {name: sha256(output / f"{split}_{name}.npy")
                                                for name in ("features", "labels")} for split in SPLITS},
                "metadata_sha256": {split: sha256(output / f"{split}_metadata.jsonl") for split in SPLITS},
                "policy": "all seeds fitted and validation-selected before any test inference"}
        write_json(output / "evaluation_plan.json", plan)
        write_json(output / "results.json", {"status": "train_complete", "metrics_by_seed": all_metrics})
    if args.phase in ("all", "evaluate"):
        plan = read_json(output / "evaluation_plan.json")
        if plan["config_sha256"] != sha256(config_path) or plan["input_audit_sha256"] != sha256(output / "input_audit.json"):
            raise ValueError("固定计划的配置或输入审计已改变。")
        all_metrics = read_json(output / "results.json")["metrics_by_seed"]
        for split in SPLITS:
            if plan["metadata_sha256"][split] != sha256(output / f"{split}_metadata.jsonl"):
                raise ValueError("固定计划的 metadata 缓存已改变。")
            for name in ("features", "labels"):
                if plan["feature_cache_sha256"][split][name] != sha256(output / f"{split}_{name}.npy"):
                    raise ValueError("固定计划的数据缓存已改变。")
        for seed in args.seeds:
            model_path = output / f"seed_{seed}" / "probe.joblib"
            if plan["models"][str(seed)] != sha256(model_path):
                raise ValueError("固定模型已改变：" + str(model_path))
            fitted = joblib.load(model_path)
            all_metrics[str(seed)]["test"] = export_predictions(output, "test", seed, data,
                fitted["classifier"], fitted["scaler"], classes, args.batch_size)
        write_json(output / "results.json", {"status": "complete", "metrics_by_seed": all_metrics})
    summary = {}
    for split in ("val", "test"):
        if all(split in result for result in all_metrics.values()):
            summary[split] = {metric: {"mean": float(np.mean([result[split][metric] for result in all_metrics.values()])),
                                      "std": float(np.std([result[split][metric] for result in all_metrics.values()], ddof=1))
                                      if len(all_metrics) > 1 else None}
                              for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")}
    write_json(output / "aggregate.json", summary)
    print(json.dumps({"output_dir": str(output), "target": args.target, "summary": summary,
                      "caution": "弱诊断模型不能凭低预测率排除混杂；高预测率提示存在可分辨的 metadata 相关信号。"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
