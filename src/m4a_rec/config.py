"""Load the files under configs/ into one dict and resolve paths against the repo root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repo root: $M4A_ROOT if set, else the nearest parent holding configs/data.yaml."""
    if env := os.environ.get("M4A_ROOT"):
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "configs" / "data.yaml").exists():
            return p
    raise FileNotFoundError("configs/data.yaml not found; run from the repo or set M4A_ROOT")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg_path = Path(path) if path else root / "configs" / "data.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    for name in ("eval.yaml", "baselines.yaml"):
        cfg.update(yaml.safe_load((root / "configs" / name).read_text()))
    cfg["root"] = root
    cfg["paths"] = {k: root / v for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg
