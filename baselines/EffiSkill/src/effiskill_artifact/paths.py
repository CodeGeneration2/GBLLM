from __future__ import annotations

from pathlib import Path


def artifact_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return artifact_root() / "data"


def outputs_root() -> Path:
    root = artifact_root() / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root
