"""Project configuration: single loader for config.yaml."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

_CONFIG_NAME = "config.yaml"


def repo_root() -> Path:
    """Directory containing config.yaml, found by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _CONFIG_NAME).is_file():
            return parent
    raise FileNotFoundError(f"Could not find {_CONFIG_NAME} above {here}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (default: the one at the repo root) into a dict."""
    cfg_path = Path(path) if path is not None else repo_root() / _CONFIG_NAME
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Absolute path for an entry of the `paths:` section of the config."""
    cfg = config if config is not None else load_config()
    try:
        rel = cfg["paths"][key]
    except KeyError:
        raise KeyError(f"Unknown paths key: {key!r}") from None
    return (repo_root() / rel).resolve()


def set_seeds(seed: int | None = None) -> int:
    """Seed random, numpy and (if importable) torch; returns the seed used."""
    if seed is None:
        seed = int(load_config()["seed"])
    seed = int(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
