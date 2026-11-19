#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed six-class metadata-only source/confounding diagnostic.

Predict A/T/C/G/X/Y from one-hot metadata categories, without signals. Encoder,
multinomial logistic regression and prior-only dummy are fitted on train only.
All declared seeds are frozen before test prediction. This diagnostic exposes
available associations; it cannot establish that the neural models are free of
confounding, nor turn CSV-row positive proxies into real read/run identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

from prepare_group_holdout import SPLITS, LABEL_NAMES, discover_sources, known, paired_rows

VERSION = "sixclass-metadata-only-probe-v1.0.0"
DEFAULT_DATA_DIR = "/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721"
FEATURE_GROUPS = {"source": "source_dataset", "run": "run_id", "strand": "bam_strand", "reference": "chrom", "template": "template_group", "motif": "masked_motif"}
MISSING = "__MISSING_METADATA__"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_groups", nargs="+", choices=tuple(FEATURE_GROUPS), default=["source", "run", "strand", "reference"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    p.add_argument("--regularization_C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--class_weight", choices=("none", "balanced"), default="none")
    p.add_argument("--batch_size", type=int, default=65536, help="Prediction/export chunk size; does not change the fixed training fit")
    return p


def prepare_inputs(args, output):
    sources, _ = discover_sources(Path(args.data_dir).expanduser().resolve())
    columns = [FEATURE_GROUPS[group] for group in args.feature_groups]
    connection = sqlite3.connect(str(output / "read_membership.sqlite"))
    connection.execute("CREATE TABLE reads(read_id TEXT PRIMARY KEY, split TEXT, windows INTEGER, proxy INTEGER, conflict INTEGER)")
    arrays, fingerprints, reports = {}, {}, {}
    try:
        for source in sources:
            split = source["previous_split"]
            categories, labels = [], []
            class_counts, missing = Counter(), Counter()
            category_classes = {column: {} for column in columns}
            identity = Counter()
            with (output / f"{split}_metadata.jsonl").open("w", encoding="utf-8") as handle:
                batch = []
                for index, (line, row, metadata, info) in enumerate(paired_rows(source)):
                    values = []
                    for column in columns:
                        raw = info["motif"] if column == "masked_motif" else known(metadata.get(column))
                        value = raw if raw is not None else MISSING
                        # Intern repeated categorical strings to keep metadata
                        # storage small without retaining signal feature rows.
                        values.append(sys.intern(value))
                        missing[column] += int(value == MISSING)
                        counts = category_classes[column].setdefault(value, [0] * 6)
                        counts[info["label"]] += 1
                    categories.append(tuple(values))
                    labels.append(info["label"])
                    class_counts[str(info["label"])] += 1
                    identity["positive_row_proxy" if info["proxy"] else "sequencing_read_id"] += 1
                    record = dict(metadata)
                    record.update(sample_index=index, label=info["label"], masked_motif=info["motif"], is_positive_row_proxy=info["proxy"])
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    batch.append((info["read_id"], split, 1, int(info["proxy"]), 0))
                    if len(batch) >= 5000:
                        _store_reads(connection, batch)
                        batch.clear()
                _store_reads(connection, batch)
            if not labels or set(labels) != set(range(6)):
                raise ValueError(f"{split}: all six classes are required for this fixed six-class diagnostic; counts={dict(class_counts)}")
            arrays[split] = (np.asarray(categories, dtype=object), np.asarray(labels, dtype=np.int64))
            fingerprints[split] = source.pop("last_fingerprints")
            reports[split] = {"sample_count": len(labels), "class_counts": dict(class_counts), "missing_feature_counts": dict(missing), "category_class_counts": category_classes, "read_id_kind_counts": dict(identity)}
        connection.commit()
        conflicts = connection.execute("SELECT read_id,split,windows,proxy FROM reads WHERE conflict=1 OR (proxy=1 AND windows>1) LIMIT 20").fetchall()
        if conflicts:
            raise ValueError(f"Cross-split read overlap or repeated positive row proxy: {conflicts}. Multiple labels/windows on a negative read within one split are permitted.")
        for split in SPLITS:
            reports[split]["unique_read_or_proxy_count"] = connection.execute("SELECT COUNT(*) FROM reads WHERE split=?", (split,)).fetchone()[0]
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        x_train, _ = arrays["train"]
        encoder.fit(x_train)
        joblib.dump(encoder, output / "encoder.joblib")
        data = {}
        for split in SPLITS:
            x, y = arrays[split]
            reports[split]["unseen_feature_counts_vs_train"] = {column: int(np.count_nonzero(~np.isin(x[:, i], encoder.categories_[i]))) for i, column in enumerate(columns)}
            encoded = encoder.transform(x).tocsr()
            sparse.save_npz(output / f"{split}_features.npz", encoded)
            np.save(output / f"{split}_labels.npy", y, allow_pickle=False)
            data[split] = encoded, y
        audit = {"version": VERSION, "feature_groups": args.feature_groups, "feature_columns": columns, "encoded_feature_count": len(encoder.get_feature_names_out(columns)), "onehot_feature_names": encoder.get_feature_names_out(columns).tolist(), "label_mapping": dict(enumerate(LABEL_NAMES)), "split_summary": reports, "input_fingerprints": fingerprints, "read_cross_split_overlap_count": 0, "sampling_unit": "one CSV window/target base; repeated real negative reads may have multiple true classes", "read_identity_scope": "real negative/provided extraction IDs and fully disclosed positive row proxies", "positive_read_identity_note": "One positive CSV row per positive read is the user-provided experimental assumption; original positive UUID independence is not independently verified.", "diagnostic_scope": "Predict the six task labels from metadata categories only. Unknown markers and source membership are allowed diagnostic predictors, never newly inferred physical identities.", "scientific_limitations": ["High metadata-only accuracy exposes available task/source associations. Low accuracy does not rule out hidden confounding.", "A/T/C/G are E. coli windows and X/Y are synthetic positive targets in these inputs; code cannot replace a source/run-matched validation experiment.", "Metrics are window-level descriptive diagnostics. Repeated windows on one negative read are correlated; these reports do not claim independent-window confidence intervals.", "Motif, if requested, is the six-base context with position four masked to N. No original target base, label, read ID or signal is a predictor."]}
        write_json(output / "input_audit.json", audit)
        return data, audit
    finally:
        connection.close()


def _store_reads(connection, batch):
    if batch:
        connection.executemany("INSERT INTO reads VALUES (?,?,?,?,?) ON CONFLICT(read_id) DO UPDATE SET windows=windows+1, conflict=conflict OR split!=excluded.split OR proxy!=excluded.proxy", batch)


def metrics(y, probability):
    prediction = np.argmax(probability, axis=1)
    cm = confusion_matrix(y, prediction, labels=np.arange(6))
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=np.arange(6), zero_division=0)
    per_class = []
    for label in range(6):
        tp = int(cm[label, label]); fn = int(cm[label].sum() - tp); fp = int(cm[:, label].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
        per_class.append({"label": label, "base": LABEL_NAMES[label], "support": int(support[label]), "precision": float(precision[label]), "recall": float(recall[label]), "f1": float(f1[label]), "specificity": tn / (tn + fp) if tn + fp else None, "false_positive_rate": fp / (tn + fp) if tn + fp else None, "tp": tp, "fn": fn, "fp": fp, "tn": tn})
    return {"n": len(y), "accuracy": float(accuracy_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "macro_f1": float(f1_score(y, prediction, labels=np.arange(6), average="macro", zero_division=0)), "log_loss": float(log_loss(y, probability, labels=np.arange(6))), "multiclass_brier_sum": float(np.mean(np.sum((probability - np.eye(6)[y]) ** 2, axis=1))), "confusion_matrix": cm.tolist(), "per_class": per_class, "class_order": list(LABEL_NAMES), "classification_report": classification_report(y, prediction, labels=np.arange(6), target_names=LABEL_NAMES, output_dict=True, zero_division=0)}


def predict(model, x, batch_size):
    probability = np.empty((x.shape[0], 6), dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        values = model.predict_proba(x[start:start + batch_size])
        if not np.array_equal(model.classes_, np.arange(6)):
            raise ValueError("Saved model class order is not the fixed 0..5 six-class mapping")
        probability[start:start + len(values)] = values
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1.0):
        raise ValueError("Diagnostic model produced invalid probabilities")
    return probability


def export_predictions(output, split, model, batch_size, destination):
    x = sparse.load_npz(output / f"{split}_features.npz")
    y = np.load(output / f"{split}_labels.npy", allow_pickle=False)
    probability = predict(model, x, batch_size)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / f"{split}_predictions.npz", probabilities=probability, labels=y, sample_indices=np.arange(len(y), dtype=np.int64))
    columns = ["sample_index", "read_id", "read_id_kind", "read_identity_basis", "original_read_id", "reference_identity_known", "chrom", "bam_strand", "source_dataset", "run_id", "masked_motif", "true_label", "true_base", "predicted_label", "predicted_base", "confidence"] + [f"probability_class_{label}" for label in range(6)]
    with gzip.open(destination / f"{split}_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        with (output / f"{split}_metadata.jsonl").open(encoding="utf-8") as metadata_handle:
            count = 0
            for index, line in enumerate(metadata_handle):
                metadata = json.loads(line)
                predicted = int(probability[index].argmax())
                record = {key: metadata.get(key, "unknown") for key in columns[:11]}
                record.update(sample_index=index, true_label=int(y[index]), true_base=LABEL_NAMES[int(y[index])], predicted_label=predicted, predicted_base=LABEL_NAMES[predicted], confidence=float(probability[index, predicted]))
                record.update({f"probability_class_{label}": float(probability[index, label]) for label in range(6)})
                writer.writerow(record)
                count += 1
        if count != len(y):
            raise ValueError("Prediction/metadata row count mismatch")
    result = metrics(y, probability)
    write_json(destination / f"{split}_metrics.json", result)
    with (destination / f"{split}_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true/predicted"] + list(LABEL_NAMES))
        writer.writerows([LABEL_NAMES[label]] + row for label, row in enumerate(result["confusion_matrix"]))
    return result


def artifact_hashes(output, seeds):
    names = ["resolved_config.json", "input_audit.json", "encoder.joblib", "dummy.joblib"]
    names += [f"seed_{seed}/metadata_probe.joblib" for seed in seeds]
    names += [f"{split}_{suffix}" for split in SPLITS for suffix in ("features.npz", "labels.npy", "metadata.jsonl")]
    return {name: sha256(output / name) for name in names}


def evaluate(output, args):
    plan = read_json(output / "evaluation_plan.json")
    actual = artifact_hashes(output, args.seeds)
    if actual != plan["frozen_artifact_sha256"]:
        raise ValueError("A frozen model/config/input cache changed after the evaluation plan was written")
    if plan["script_sha256"] != sha256(__file__) or plan["input_helper_sha256"] != sha256(Path(__file__).with_name("prepare_group_holdout.py")):
        raise ValueError("Use the saved training source version to evaluate this diagnostic")
    test_started = output / "test_started.json"
    if test_started.exists() and read_json(test_started).get("evaluation_plan_sha256") != sha256(output / "evaluation_plan.json"):
        raise ValueError("Evaluation plan changed after test prediction began")
    if not test_started.exists():
        write_json(test_started, {"started_utc_unix": time.time(), "evaluation_plan_sha256": sha256(output / "evaluation_plan.json"), "policy": "models/settings frozen before first test prediction"})
    results = read_json(output / "results.json")
    for seed in args.seeds:
        model = joblib.load(output / f"seed_{seed}" / "metadata_probe.joblib")
        results["metrics_by_seed"][str(seed)]["test"] = export_predictions(output, "test", model, args.batch_size, output / f"seed_{seed}")
    dummy = joblib.load(output / "dummy.joblib")
    results["dummy_metrics"]["test"] = export_predictions(output, "test", dummy, args.batch_size, output / "dummy_prior")
    results["status"] = "evaluation_complete"
    results["repeated_seed_summary"] = {}
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier_sum"):
        values = [results["metrics_by_seed"][str(seed)]["test"][name] for seed in args.seeds]
        results["repeated_seed_summary"][name] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n_seeds": len(values), "values": values}
    write_json(output / "results.json", results)
    return results


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    provided = {value.split("=", 1)[0][2:] for value in argv if value.startswith("--")}
    args = parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if args.phase == "evaluate":
        saved = read_json(output / "resolved_config.json")
        for key, value in saved["arguments"].items():
            if key not in provided and key not in {"phase", "output_dir"}:
                setattr(args, key, value)
        for key in provided - {"phase", "output_dir", "batch_size"}:
            if getattr(args, key) != saved["arguments"].get(key):
                raise ValueError(f"Evaluation cannot change the predeclared diagnostic setting: {key}")
        return evaluate(output, args)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use a new directory or --phase evaluate for frozen models")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or len(set(args.feature_groups)) != len(args.feature_groups):
        raise ValueError("seeds and feature_groups must be nonempty and distinct")
    if min(args.max_iter, args.batch_size) < 1 or args.regularization_C <= 0 or args.tolerance <= 0:
        raise ValueError("Iteration/batch/regularization/tolerance settings must be positive")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", {"version": VERSION, "arguments": vars(args), "purpose": "six-class task-label prediction from metadata only; fixed diagnostic, not neural model selection", "solver": "saga multinomial logistic regression; train-only one-hot encoder; no hyperparameter search or test-based selection"})
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions})
    data, audit = prepare_inputs(args, output)
    x_train, y_train = data["train"]
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(x_train, y_train)
    joblib.dump(dummy, output / "dummy.joblib")
    results = {"status": "train_complete", "metrics_by_seed": {}, "dummy_metrics": {}, "feature_groups": args.feature_groups, "read_identity_scope": audit["read_identity_scope"], "interpretation": "High diagnostic accuracy supports a metadata/task association. It does not quantify how much the neural classifier uses that association."}
    for seed in args.seeds:
        directory = output / f"seed_{seed}"
        directory.mkdir()
        model = LogisticRegression(C=args.regularization_C, solver="saga", random_state=seed, max_iter=args.max_iter, tol=args.tolerance, class_weight=None if args.class_weight == "none" else "balanced")
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        joblib.dump(model, directory / "metadata_probe.joblib")
        fit = {"seed": seed, "fit_seconds": time.perf_counter() - started, "n_iter": model.n_iter_.tolist(), "classes": model.classes_.tolist(), "convergence_warnings": [str(item.message) for item in caught], "train_rows": int(x_train.shape[0]), "encoded_features": int(x_train.shape[1]), "train_only_fit": True, "selection_policy": "single fixed model; validation/test never change C/solver/features/iterations"}
        write_json(directory / "fit_report.json", fit)
        np.savez_compressed(directory / "coefficients.npz", coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
        results["metrics_by_seed"][str(seed)] = {split: export_predictions(output, split, model, args.batch_size, directory) for split in ("train", "validation")}
    for split in ("train", "validation"):
        results["dummy_metrics"][split] = export_predictions(output, split, dummy, args.batch_size, output / "dummy_prior")
    write_json(output / "results.json", results)
    plan = {"version": VERSION, "seeds": args.seeds, "class_order": list(LABEL_NAMES), "script_sha256": sha256(__file__), "input_helper_sha256": sha256(Path(__file__).with_name("prepare_group_holdout.py")), "frozen_artifact_sha256": artifact_hashes(output, args.seeds), "policy": "all declared models, feature groups and preprocessing frozen before first test prediction", "created_utc_unix": time.time()}
    write_json(output / "evaluation_plan.json", plan)
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    for name in ("train_metadata_probe.py", "prepare_group_holdout.py", "IterDataset.py"):
        shutil.copy2(Path(__file__).with_name(name), source_dir / name)
    write_json(source_dir / "sha256.json", {name: sha256(source_dir / name) for name in ("train_metadata_probe.py", "prepare_group_holdout.py", "IterDataset.py")})
    if args.phase == "all":
        results = evaluate(output, args)
    print(f"[PASS] metadata-only six-class diagnostic: {output}; status={results['status']}", flush=True)
    return results


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"[REJECTED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
