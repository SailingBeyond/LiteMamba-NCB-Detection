#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the four benchmark models sequentially using ``train_all.py``."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

AVAILABLE_MODELS = ("bigru", "transformer", "convmixer", "litemamba")


def parse_models(value: str) -> List[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model must be selected")

    unknown = sorted(set(models) - set(AVAILABLE_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown models: {unknown}. Available: {list(AVAILABLE_MODELS)}"
        )
    return models


def build_command(
    args: argparse.Namespace,
    model_name: str,
) -> Tuple[List[str], Path, Path]:
    model_output_dir = args.output_dir / model_name
    metrics_path = model_output_dir / "epoch_metrics.csv"
    model_path = model_output_dir / "best_model.pt"

    command = [
        sys.executable,
        str(args.train_script),
        "--train_data",
        str(args.train_data),
        "--test_data",
        str(args.test_data),
        "--output_dir",
        str(model_output_dir),
        "--model_type",
        model_name,
        "--num_classes",
        str(args.num_classes),
        "--seq_len",
        str(args.seq_len),
        "--raw_signal_len",
        str(args.raw_signal_len),
        "--metrics_out",
        str(metrics_path),
        "--save",
        str(model_path),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--validation_batch_size",
        str(args.validation_batch_size),
        "--workers",
        str(args.workers),
        "--prefetch",
        str(args.prefetch),
        "--accum_steps",
        str(args.accum_steps),
        "--log_every",
        str(args.log_every),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--label_smoothing",
        str(args.label_smoothing),
        "--patience",
        str(args.patience),
        "--lr_patience",
        str(args.lr_patience),
        "--lr_factor",
        str(args.lr_factor),
        "--dwell_divisor",
        str(args.dwell_divisor),
        "--quality_divisor",
        str(args.quality_divisor),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--chunk_size",
        str(args.chunk_size),
    ]

    if args.val_workers is not None:
        command.extend(["--val_workers", str(args.val_workers)])
    if not args.persistent_workers:
        command.append("--no-persistent_workers")
    if not args.drop_last:
        command.append("--no-drop-last")
    if args.pre_train is not None:
        command.extend(["--pre_train", str(args.pre_train)])
    if args.compile:
        command.extend(["--compile", "--compile_mode", args.compile_mode])
    if args.no_shuffle:
        command.append("--no_shuffle")
    if args.delete_shuffled:
        command.append("--delete_shuffled")
    if args.deterministic:
        command.append("--deterministic")
    if args.no_amp:
        command.append("--no_amp")
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))

    return command, metrics_path, model_path


def write_summary(summary_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "status",
                "returncode",
                "elapsed_sec",
                "metrics_path",
                "model_path",
                "command",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_models(args: argparse.Namespace) -> int:
    args.train_script = args.train_script.resolve()
    args.train_data = args.train_data.resolve()
    args.test_data = args.test_data.resolve()
    args.output_dir = args.output_dir.resolve()

    for path, label in (
        (args.train_script, "training script"),
        (args.train_data, "training CSV"),
        (args.test_data, "test CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"The {label} does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "benchmark_summary.csv"

    print("=" * 88)
    print("Four-model benchmark")
    print(f"Training script : {args.train_script}")
    print(f"Training CSV    : {args.train_data}")
    print(f"Test CSV        : {args.test_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models          : {args.models}")
    print(f"Number classes  : {args.num_classes}")
    print(f"Sequence length : {args.seq_len}")
    print("=" * 88)

    results: List[Dict[str, object]] = []

    for index, model_name in enumerate(args.models, start=1):
        command, metrics_path, model_path = build_command(args, model_name)
        completion_marker = args.output_dir / model_name / "completed.ok"

        if args.skip_completed and completion_marker.is_file() and model_path.is_file():
            print(f"[{index}/{len(args.models)}] SKIP {model_name}: already completed")
            results.append(
                {
                    "model": model_name,
                    "status": "SKIPPED",
                    "returncode": 0,
                    "elapsed_sec": "0.000",
                    "metrics_path": str(metrics_path),
                    "model_path": str(model_path),
                    "command": shlex.join(command),
                }
            )
            continue

        print("\n" + "-" * 88)
        print(f"[{index}/{len(args.models)}] Running: {model_name}")
        print(shlex.join(command))
        print("-" * 88)

        start_time = time.time()
        return_code = -999
        status = "FAILED"

        try:
            completed = subprocess.run(command, check=False)
            return_code = completed.returncode
            status = "OK" if return_code == 0 else "FAILED"
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            return_code = 130
        except Exception as error:
            print(f"[EXCEPTION] {type(error).__name__}: {error}")
            status = "EXCEPTION"
            return_code = -999

        elapsed = time.time() - start_time
        if return_code == 0:
            completion_marker.parent.mkdir(parents=True, exist_ok=True)
            completion_marker.write_text(
                time.strftime("%Y-%m-%d %H:%M:%S") + "\n",
                encoding="utf-8",
            )
            print(f"[OK] {model_name} completed in {elapsed:.1f} seconds")
        else:
            completion_marker.unlink(missing_ok=True)
            print(
                f"[FAIL] {model_name} returned code {return_code} "
                f"after {elapsed:.1f} seconds"
            )

        results.append(
            {
                "model": model_name,
                "status": status,
                "returncode": return_code,
                "elapsed_sec": f"{elapsed:.3f}",
                "metrics_path": str(metrics_path),
                "model_path": str(model_path),
                "command": shlex.join(command),
            }
        )
        write_summary(summary_path, results)

        if return_code != 0 and args.stop_on_error:
            break
        if return_code == 130:
            break

    write_summary(summary_path, results)

    print("\n" + "=" * 88)
    print("Benchmark summary")
    print("=" * 88)
    for row in results:
        print(
            f"{str(row['model']):12s} "
            f"{str(row['status']):12s} "
            f"time={row['elapsed_sec']}s"
        )
    print(f"Summary CSV: {summary_path}")

    failed = [row for row in results if int(row["returncode"]) != 0]
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train BiGRU, Transformer, ConvMixer and LiteMamba sequentially"
    )
    parser.add_argument(
        "--train_script",
        type=Path,
        default=script_directory / "train_all.py",
    )
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--models", type=parse_models, default=list(AVAILABLE_MODELS))

    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--raw_signal_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int)
    parser.add_argument("--prefetch", type=int, default=2)
    persistent_group = parser.add_mutually_exclusive_group()
    persistent_group.add_argument(
        "--persistent_workers",
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
    )
    persistent_group.add_argument(
        "--no_persistent_workers",
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
    )
    parser.set_defaults(persistent_workers=True)

    drop_last_group = parser.add_mutually_exclusive_group()
    drop_last_group.add_argument(
        "--drop_last",
        "--drop-last",
        dest="drop_last",
        action="store_true",
    )
    drop_last_group.add_argument(
        "--no_drop_last",
        "--no-drop-last",
        dest="drop_last",
        action="store_false",
    )
    parser.set_defaults(drop_last=True)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--dwell_divisor", type=float, default=60.0)
    parser.add_argument("--quality_divisor", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk_size", type=int, default=1_000_000)

    parser.add_argument("--pre_train", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--delete_shuffled", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument(
        "--extra_args",
        default="",
        help="Additional arguments passed verbatim to train_all.py",
    )
    return parser


def main() -> int:
    return run_models(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
