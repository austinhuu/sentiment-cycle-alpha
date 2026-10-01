from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return config
