"""Defaults for the revised binary training suite.

Importing this file never creates directories or opens a log file. CLI values
in train.py override these defaults and are written into each run's JSON config.
"""

from datetime import datetime
from pathlib import Path

import torch


DEFAULT_DATA_DIR = (
    "/path/to/revision-inputs/POC-XNA_DNA-01_16/POC-XNA_DNA-01_16/"
    "fast5_single_guppy/binary_read_split_721"
)
DEFAULT_OUTPUT_DIR = "./binary_training_runs"


class ModelConfig:
    """Serializable defaults; fourth base is zero-based TARGET_INDEX=3."""

    MER_TYPE_LENGTH = 6
    NUM_CLASSES = 2
    BASE_DIM = 5
    MERGED_FEATURE_DIM = 13
    RAW_SIGNAL_LENGTH_PER_BASE = 30
    TARGET_INDEX = 3
    MODEL_DIM = 96
    NUM_BLOCKS = 4
    STATE_DIM = 16
    SSM_BIDIRECTIONAL = True
    LITEMAMBA_VARIANT = "selective_ssm"
    READOUT = "target_attention"
    FEATURE_SCHEME = "full"
    DROPOUT = 0.1
    CLASSIFIER_DROPOUT = 0.2
    NHEAD = 4
    NUM_LAYERS = 2
    GRU_HIDDEN = 64

    DWELL_SECONDARY_DIVISOR = 60.0
    QUALITY_SECONDARY_DIVISOR = 50.0
    LR = 1e-3
    epoch = 200
    PATIENCE = 30
    BENCH_SIZE = 256
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.0
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    # Compatibility names used by the former scripts. No directories are created.
    run_mode = "train"
    mer_type = "XNA_PC_6mer_read_disjoint_binary"
    datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = DEFAULT_DATA_DIR
    model_path = DEFAULT_OUTPUT_DIR
    log_dir = str(Path(DEFAULT_OUTPUT_DIR) / "logs")
    log_path = str(Path(log_dir) / f"{datetime}.log")
    train_file_path = str(Path(DEFAULT_DATA_DIR) / "train.csv")
    val_file_path = str(Path(DEFAULT_DATA_DIR) / "val.csv")
    test_file_path = str(Path(DEFAULT_DATA_DIR) / "test.csv")

    @classmethod
    def model_kwargs(cls):
        """Shared kwargs accepted by model_benchmark.build_model for any backbone."""
        return {
            "model_dim": cls.MODEL_DIM,
            "target_index": cls.TARGET_INDEX,
            "num_blocks": cls.NUM_BLOCKS,
            "state_dim": cls.STATE_DIM,
            "ssm_bidirectional": cls.SSM_BIDIRECTIONAL,
            "litemamba_variant": cls.LITEMAMBA_VARIANT,
            "readout": cls.READOUT,
            "feature_scheme": cls.FEATURE_SCHEME,
            "dropout": cls.DROPOUT,
            "classifier_dropout": cls.CLASSIFIER_DROPOUT,
            "nhead": cls.NHEAD,
            "num_layers": cls.NUM_LAYERS,
            "gru_hidden": cls.GRU_HIDDEN,
        }

    @classmethod
    def as_dict(cls):
        """Return JSON-safe defaults; paths remain strings and contain no tensors."""
        return {
            "seq_len": cls.MER_TYPE_LENGTH,
            "num_classes": cls.NUM_CLASSES,
            "raw_signal_len": cls.RAW_SIGNAL_LENGTH_PER_BASE,
            "base_dim": cls.BASE_DIM,
            "merged_feature_dim": cls.MERGED_FEATURE_DIM,
            **cls.model_kwargs(),
            "epochs": cls.epoch,
            "patience": cls.PATIENCE,
            "learning_rate": cls.LR,
            "batch_size": cls.BENCH_SIZE,
            "weight_decay": cls.WEIGHT_DECAY,
            "label_smoothing": cls.LABEL_SMOOTHING,
            "dwell_divisor": cls.DWELL_SECONDARY_DIVISOR,
            "quality_divisor": cls.QUALITY_SECONDARY_DIVISOR,
            "default_data_dir": cls.DEFAULT_DATA_DIR,
            "default_output_dir": cls.DEFAULT_OUTPUT_DIR,
        }
