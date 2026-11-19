"""Serializable defaults for the strict direction-matched five-class training suite.

Importing this module performs no CUDA discovery, logging or directory creation.
The training CLI resolves ``DEVICE='auto'`` at runtime and records actual choices.
Labels are A=0, T=1, C=2, G=3, NCB=4; NCB is X or Y by explicit scenario.
The fourth base is masked to N; sequence length remains six.
"""

from pathlib import Path
import os


VERSION = "1.0.0"
SCENARIO = os.environ.get("CPLX_FIVECLASS_SCENARIO", "")
if SCENARIO not in ("X_plus", "Y_minus"):
    raise RuntimeError("必须由运行入口设置 CPLX_FIVECLASS_SCENARIO=X_plus 或 Y_minus，不能猜测 label4 的 X/Y 身份。")
CLASS_NAMES = ("A", "T", "C", "G", "X" if SCENARIO == "X_plus" else "Y")
NUM_CLASSES = len(CLASS_NAMES)
NCB_NAME = CLASS_NAMES[-1]
MATCHED_STRAND = "+" if SCENARIO == "X_plus" else "-"
SCOPE_NOTE = ("All five classes come from CPLX-XNA-all and are matched by BAM reference assignment, direction and flank context. "
              "Natural and NCB positions still differ; this is read-held-out supplementary control, not physical-template or batch holdout.")
ONE_HOT_BASE_ORDER = ("A", "C", "T", "G", "N")
LABEL_TO_NAME = dict(enumerate(CLASS_NAMES))
DEFAULT_DATA_DIR = str(Path("cplx_all_direction_matched_5class_v1_0_0") / SCENARIO)
DEFAULT_OUTPUT_DIR = "./direction_fiveclass_training_runs"



class ModelConfig:
    """One shared comparison configuration; CLI values take precedence."""

    MER_TYPE_LENGTH = 6
    NUM_CLASSES = NUM_CLASSES
    CLASS_NAMES = CLASS_NAMES
    ONE_HOT_BASE_ORDER = ONE_HOT_BASE_ORDER
    BASE_DIM = 5
    MERGED_FEATURE_DIM = 13
    RAW_SIGNAL_LENGTH_PER_BASE = 30
    TARGET_INDEX = 3
    MODEL_DIM = 96
    NUM_BLOCKS = 4
    STATE_DIM = 16
    SSM_BIDIRECTIONAL = True
    LITEMAMBA_VARIANT = "legacy_gatedconv"
    READOUT = "target_attention"
    FEATURE_SCHEME = "full"
    DROPOUT = 0.1
    CLASSIFIER_DROPOUT = 0.2
    NHEAD = 4
    NUM_LAYERS = 2
    GRU_HIDDEN = 64

    # These are the effective settings used by the former batch runner.
    # Former standalone config defaults (64, 1, 1) disagreed with that runner.
    DWELL_SECONDARY_DIVISOR = 60.0
    QUALITY_SECONDARY_DIVISOR = 50.0
    LR = 1e-3
    epoch = 200
    PATIENCE = 30
    BENCH_SIZE = 256
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.0
    DEVICE = "auto"
    DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    # Compatibility names; no files or directories are created at import time.
    run_mode = "train"
    mer_type = "XNA_6mer_direction_fiveclass_read_grouped"
    data_path = DEFAULT_DATA_DIR
    model_path = DEFAULT_OUTPUT_DIR
    log_dir = str(Path(DEFAULT_OUTPUT_DIR) / "logs")
    train_file_path = str(Path(DEFAULT_DATA_DIR) / "train.csv")
    val_file_path = str(Path(DEFAULT_DATA_DIR) / "validation.csv")
    validation_file_path = val_file_path
    test_file_path = str(Path(DEFAULT_DATA_DIR) / "test.csv")
    train_meta_path = str(Path(DEFAULT_DATA_DIR) / "train_meta.tsv")
    val_meta_path = str(Path(DEFAULT_DATA_DIR) / "validation_meta.tsv")
    validation_meta_path = val_meta_path
    test_meta_path = str(Path(DEFAULT_DATA_DIR) / "test_meta.tsv")

    @classmethod
    def model_kwargs(cls):
        """Shared kwargs accepted by ``model_benchmark_all.build_model``."""
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
        """JSON-safe defaults, including fixed class order and preprocessing."""
        return {
            "version": VERSION,
            "seq_len": cls.MER_TYPE_LENGTH,
            "num_classes": cls.NUM_CLASSES,
            "scenario": SCENARIO,
            "class_names": list(cls.CLASS_NAMES),
            "one_hot_base_order": list(cls.ONE_HOT_BASE_ORDER),
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
            "device": cls.DEVICE,
            "default_data_dir": cls.DEFAULT_DATA_DIR,
            "default_output_dir": cls.DEFAULT_OUTPUT_DIR,
        }
