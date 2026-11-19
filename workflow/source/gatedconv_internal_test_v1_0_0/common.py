"""Frozen identities and preparation for the three existing GatedConv models.

This module never trains a model or fits a calibration parameter.  Importing it
does not import PyTorch; only ``prepare_plan`` needs the frozen dataset runtime.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
SEEDS = (42, 2024, 2025)
ALL_MAIN_SEEDS = SEEDS + (2026, 2027)
CLASSES = ("A", "T", "C", "G", "X", "Y")
DEFAULT_DATA_DIR = "/path/to/revision-inputs/CPLX-XNA-replenishment/CPLX-XNA-replenishment/fast5_single_guppy/sixclass_overlap_read_split_721"
DEFAULT_RESULTS_DIR = DEFAULT_DATA_DIR + "/sixclass_auto_v1_0_4_results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR + "/gatedconv_internal_test_v1_0_0"
METADATA_FILES = ("resolved_config.json", "model_config.json", "results.json",
                  "calibration.json", "data_audit.json")
RUNTIME_FILES = ("model_benchmark_all.py", "IterDataset.py", "config_6_all_6.py")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("File changed while hashing: " + str(path))
    return {"path": str(path), "sha256": digest, "size": after.st_size,
            "mtime_ns": after.st_mtime_ns}


@contextlib.contextmanager
def file_lock(path):
    """Nonblocking Linux advisory lock; stale files do not block a restart."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation owns this output directory: " + str(path)) from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _content(value):
    """Compare content, not server path spelling or modification times."""
    return {split: {kind: item["sha256"] for kind, item in pair.items()}
            for split, pair in value.items()}


def _classes(value):
    if isinstance(value, dict):
        return tuple(value[str(i)] for i in range(6))
    return tuple(value)


def code_manifest():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + list(VENDOR.glob("*.py"))
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}


def _check_test_identity(value, actual, where):
    _require(isinstance(value, dict) and set(value) >= {"train", "val", "test"},
             "Missing original train/val/test fingerprints: " + where)
    for kind in ("feature", "metadata"):
        _require(value["test"][kind]["sha256"] == actual[kind]["sha256"],
                 "Fixed test content SHA256 differs from training record: " + where + "/" + kind)


def _archive(directory, output, relative, names=METADATA_FILES):
    target = output / "source_metadata" / relative
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in names:
        source, destination = directory / name, target / name
        _require(source.is_file(), "Missing source metadata: " + str(source))
        digest = sha256(source)
        if destination.exists():
            _require(sha256(destination) == digest, "Frozen metadata changed: " + str(source))
        else:
            shutil.copyfile(source, destination)
        hashes[name] = digest
    return str(target), hashes


def _gated_source(results, seed, args):
    override = getattr(args, "seed" + str(seed) + "_dir", None)
    if override:
        return Path(override).expanduser().resolve()
    folder = "litemamba_structure_seed42_v1_0_0" if seed == 42 else "litemamba_structure_confirm3_v1_1_0"
    return results / folder / "full" / "gatedconv_target" / ("seed_" + str(seed))


def _validate_model(config, model, seed):
    expected = {"model_type": "litemamba", "seed": seed, "feature_scheme": "full",
                "litemamba_variant": "legacy_gatedconv", "readout": "target_attention",
                "target_index": 3, "monitor": "val_loss"}
    for key, value in expected.items():
        _require(config.get(key) == value, "Unexpected frozen GatedConv setting: " + key)
    _require(not config.get("external_manifest"), "Source training configuration includes external evaluation")
    _require(_classes(config["class_names"]) == CLASSES and _classes(model["class_names"]) == CLASSES,
             "Class order must be A,T,C,G,X,Y")
    for key in ("model_dim", "target_index", "num_blocks", "state_dim", "dropout",
                "classifier_dropout", "readout", "feature_scheme", "litemamba_variant"):
        _require(model.get(key) == config["model_kwargs"].get(key), "Model/config mismatch: " + key)
    _require(model.get("raw_signal_len") == config["raw_signal_len"] and
             model.get("seq_len") == 6 and model.get("num_classes") == 6 and
             model.get("one_hot_base_order") == ["A", "C", "T", "G", "N"],
             "Frozen model input shape/order mismatch")


def collect_plan(args):
    """Validate source identities and checkpoint bytes without importing torch."""
    output = Path(args.output_dir).expanduser().resolve()
    data = Path(args.data_dir).expanduser().resolve()
    results = Path(args.results_dir).expanduser().resolve()
    _require(output != results and not _inside(output, results / "main_20"),
             "Output must be a new directory, never a training run")
    _require(not _inside(data, output) and not _inside(results, output),
             "Output must not be an ancestor of original data or training results")
    output.mkdir(parents=True, exist_ok=True)
    actual = {"feature": fingerprint(data / "test.csv"),
              "metadata": fingerprint(data / "test_meta.tsv")}
    runs, original, inference_reference = [], None, None
    for seed in SEEDS:
        directory = _gated_source(results, seed, args)
        _require(not _inside(output, directory) and not _inside(directory, output),
                 "Evaluation output must not modify an original training directory")
        config = read_json(directory / "resolved_config.json")
        model = read_json(directory / "model_config.json")
        record = read_json(directory / "results.json")
        calibration = read_json(directory / "calibration.json")
        audit = read_json(directory / "data_audit.json")
        _validate_model(config, model, seed)
        inference = {key: config[key] for key in (
            "raw_signal_len", "dwell_divisor", "quality_divisor", "amp", "model_kwargs")}
        if inference_reference is None:
            inference_reference = inference
        _require(inference == inference_reference, "GatedConv models use different preprocessing or inference settings")
        _require(record.get("status") == "train_complete" and record.get("test_evaluated") is False,
                 "Expected a completed, not-yet-test-evaluated source GatedConv run: " + str(directory))
        _require((record.get("model_type"), record.get("seed"), record.get("feature_scheme"),
                  record.get("litemamba_variant"), record.get("readout")) ==
                 ("litemamba", seed, "full", "legacy_gatedconv", "target_attention"),
                 "Source completion record identifies a different model")
        _require(set(record.get("metrics_by_split", {})) == {"train", "val"} and
                 set(record.get("exports_by_split", {})) == {"train", "val"},
                 "Source run is missing train/val export or already contains test results")
        _require(not (directory / "test_evaluated.json").exists() and
                 not (directory / "test_started.json").exists() and
                 not ((directory / "test").exists() and any((directory / "test").iterdir())),
                 "Source run already contains test artifacts; refusing a second independent-test claim")
        _require(audit.get("status") == "PASS", "Original split audit did not pass")
        _check_test_identity(config["input_fingerprints"], actual, str(directory))
        _require(_content(config["input_fingerprints"]) == _content(record["provenance"]["input_fingerprints"]),
                 "Source result and configuration input identities differ")
        if original is None:
            original = _content(config["input_fingerprints"])
        _require(original == _content(config["input_fingerprints"]), "GatedConv runs used different fixed splits")
        source_hashes = config["source_snapshot"]["files_sha256"]
        for name in RUNTIME_FILES:
            _require(sha256(VENDOR / name) == source_hashes.get(name),
                     "Frozen evaluation runtime differs from training: " + name)
        checkpoint = directory / "best_checkpoint.pt"
        digest = sha256(checkpoint)
        _require(digest == record["provenance"].get("checkpoint_sha256") == calibration.get("checkpoint_sha256"),
                 "Checkpoint bytes, result, and frozen validation calibration differ: " + str(directory))
        temperature = float(calibration["temperature"])
        _require(calibration.get("fitted_on") == "val" and math.isfinite(temperature) and temperature > 0,
                 "Calibration must be a positive temperature fitted only on validation")
        n = int(audit["split_summary"]["test"]["sample_count"])
        _require(n > 0, "Original test split is empty")
        archive, hashes = _archive(directory, output, Path("gatedconv_target") / ("seed_" + str(seed)))
        runs.append({"seed": seed, "run_id": "gatedconv_target/seed_" + str(seed),
                     "model": "gatedconv_target", "run_dir": str(directory),
                     "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                     "temperature": temperature, "config": config, "model_config": model,
                     "metadata_sha256": hashes, "metadata_archive": archive,
                     "expected_test_samples": n,
                     "expected_test_class_counts": audit["split_summary"]["test"]["class_counts"]})
    _require(len({run["expected_test_samples"] for run in runs}) == 1 and
             all(run["expected_test_class_counts"] == runs[0]["expected_test_class_counts"] for run in runs),
             "Source runs disagree about test size/class counts")
    baselines, missing = [], []
    for model in ("transformer", "convmixer", "bigru", "litemamba"):
        for seed in ALL_MAIN_SEEDS:
            directory = results / "main_20" / "full" / model / ("seed_" + str(seed))
            if not (directory / "results.json").is_file():
                _require(seed not in SEEDS, "Missing mandatory same-seed baseline: " + str(directory))
                missing.append({"model": model, "seed": seed, "path": str(directory)})
                continue
            record = read_json(directory / "results.json")
            config = read_json(directory / "resolved_config.json")
            _require(record.get("test_evaluated") is True and "test" in record.get("metrics_by_split", {}),
                     "Historical baseline has no completed internal test: " + str(directory))
            _require(record.get("model_type") == model and record.get("seed") == seed and
                     record.get("feature_scheme") == "full", "Historical baseline identity mismatch")
            _check_test_identity(config["input_fingerprints"], actual, str(directory))
            _require(_content(config["input_fingerprints"]) == original ==
                     _content(record["provenance"]["input_fingerprints"]),
                     "Historical baseline uses different fixed train/val/test data")
            for mode in ("raw", "calibrated"):
                metric = record["metrics_by_split"]["test"][mode]
                _require(metric.get("n") == runs[0]["expected_test_samples"], "Historical baseline test count mismatch")
            archive, hashes = _archive(directory, output, Path("historical_main20") / model / ("seed_" + str(seed)),
                                       ("resolved_config.json", "results.json", "model_config.json", "calibration.json", "data_audit.json"))
            baselines.append({"model": model, "seed": seed, "run_dir": str(directory),
                              "metrics_by_split": {name: value for name, value in record["metrics_by_split"].items() if name in ("val", "test")},
                              "checkpoint_sha256": record["provenance"]["checkpoint_sha256"],
                              "input_fingerprints": config["input_fingerprints"],
                              "metadata_archive": archive, "metadata_sha256": hashes})
    return {"version": VERSION, "purpose": "frozen GatedConv internal-test evaluation; no training, model selection, or calibration fitting",
            "fixed_seeds": list(SEEDS), "class_names": list(CLASSES),
            "data_dir": str(data), "results_dir": str(results), "output_dir": str(output),
            "test_inputs": actual, "original_split_content_sha256": original,
            "runs": runs, "historical_baselines": baselines,
            "missing_optional_historical_runs": missing,
            "prediction_settings": {"batch_size": 256, "workers": 0,
                                    "cpu_threads": int(args.cpu_threads), "embedding_dtype": "float32",
                                    "amp_policy": "training_config", "device": args.device},
            "analysis_settings": {"bootstrap": int(getattr(args, "bootstrap", 200)),
                                  "subgroup_bootstrap": 0},
            "code_manifest": code_manifest(),
            "evaluation_scope": read_json(Path(runs[0]["metadata_archive"]) / "data_audit.json").get("evaluation_scope"),
            "interpretation": "Same original internal test; architecture follow-up in a revision, not a newly collected blind external dataset. Positive read identity remains the original one-row-per-positive-read assumption."}


def verify_plan(plan):
    identity = dict(plan)
    expected = identity.pop("plan_id", None)
    identity.pop("created_at", None)
    _require(expected == identity_hash(identity), "Frozen evaluation plan was changed")
    _require(plan["code_manifest"] == code_manifest(), "Evaluation code changed after plan freeze")
    for item in plan["test_inputs"].values():
        current = fingerprint(item["path"])
        _require(current["sha256"] == item["sha256"] and current["size"] == item["size"],
                 "Fixed test inputs changed after plan freeze")
    for run in plan["runs"]:
        _require(sha256(run["checkpoint"]) == run["checkpoint_sha256"], "Frozen checkpoint changed")
        for name, digest in run["metadata_sha256"].items():
            _require(sha256(Path(run["run_dir"]) / name) == digest,
                     "Source metadata changed after plan freeze: " + name)
            _require(sha256(Path(run["metadata_archive"]) / name) == digest,
                     "Archived source metadata changed: " + name)
    return plan


def prepare_plan(args):
    """Prepare the test cache once, then freeze all three evaluations together."""
    plan = collect_plan(args)
    output = Path(args.output_dir).expanduser().resolve()
    old_path = output / "evaluation_plan.json"
    if old_path.exists():
        old = verify_plan(read_json(old_path))
        compared = {key: value for key, value in old.items() if key not in ("plan_id", "created_at", "dataset")}
        _require(compared == plan, "Arguments or frozen sources differ from the existing evaluation plan")
        _require(Path(old["dataset"]["cache_path"]).is_dir(), "Frozen test cache is missing; restore it before resuming")
        return old
    first = plan["runs"][0]["config"]
    cache = Path(getattr(args, "cache_dir", None) or output / "cache").expanduser().resolve()
    _require(not _inside(Path(plan["data_dir"]), cache) and not _inside(Path(plan["results_dir"]), cache),
             "Cache must not be an ancestor of original data or training results")
    for run in plan["runs"]:
        source = Path(run["run_dir"])
        _require(not _inside(cache, source) and not _inside(source, cache),
                 "Cache must not modify or contain an original training run")
    main20 = Path(plan["results_dir"]) / "main_20"
    _require(not _inside(cache, main20) and not _inside(main20, cache),
             "Cache must not modify or contain historical main20 runs")
    sys.path.insert(0, str(VENDOR))
    runtime = importlib.import_module("IterDataset")
    prepared = runtime.prepare_dataset(
        plan["test_inputs"]["feature"]["path"], plan["test_inputs"]["metadata"]["path"],
        cache_dir=str(cache),
        seq_len=6, raw_signal_len=int(first["raw_signal_len"]),
        dwell_divisor=float(first["dwell_divisor"]), quality_divisor=float(first["quality_divisor"]),
        mask_index=3, split_name="test", global_index_offset=0)
    dataset = prepared["dataset"]
    try:
        _require(len(dataset) == plan["runs"][0]["expected_test_samples"], "Prepared test sample count changed")
        counts = dataset.audit.get("class_counts")
        if counts is not None:
            _require({str(k): int(v) for k, v in counts.items()} == plan["runs"][0]["expected_test_class_counts"],
                     "Prepared test class counts changed")
        plan["dataset"] = {"cache_path": prepared["cache_path"], "n_samples": len(dataset),
                           "input_fingerprints": prepared["input_fingerprints"],
                           "preprocessing": dataset.audit["preprocessing"],
                           "class_counts": plan["runs"][0]["expected_test_class_counts"],
                           "class_support": [int(plan["runs"][0]["expected_test_class_counts"][str(i)]) for i in range(6)],
                           "audit": dataset.audit, "manifest_path": prepared["manifest_path"],
                           "cache_files_sha256": {name: sha256(Path(prepared["cache_path"]) / name)
                               for name in ("audit.json", "merged.npy", "raw.npy", "labels.npy",
                                            "metadata_offsets.npy", "manifest.jsonl")}}
    finally:
        dataset.close()
    plan["plan_id"] = identity_hash(plan)
    plan["created_at"] = timestamp()
    atomic_json(old_path, plan)
    return plan
