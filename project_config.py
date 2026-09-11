"""Portable paths and cache settings shared by the project scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


PROJECT_DIR = _environment_path(
    "DISTILBERT_PROJECT_DIR",
    Path(__file__).resolve().parent,
)
HF_HOME = _environment_path(
    "HF_HOME",
    PROJECT_DIR / ".cache" / "huggingface",
)
DATASET_CACHE = _environment_path(
    "HF_DATASETS_CACHE",
    HF_HOME / "datasets",
)
MODEL_CACHE = _environment_path(
    "HF_HUB_CACHE",
    HF_HOME / "hub",
)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
