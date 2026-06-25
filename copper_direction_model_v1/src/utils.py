"""Shared utilities: project paths, config loading, logging, and seeding.

Everything else in the package imports paths from here so the project is fully
runnable from inside ``copper_direction_model_v1`` regardless of the caller's
working directory.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = the folder that contains this package's parent (i.e. the dir
# holding config.yaml, src/, data/, reports/ ...).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LOGGER_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return a package logger."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger("copper_v1")


def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config. Relative paths are resolved against the project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return cfg


def resolve_path(relative: str | os.PathLike) -> Path:
    """Resolve a config-relative path to an absolute path under the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed; return the resolved path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if present) TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow is optional; seeding it is best-effort.
        pass


SEED = 42
