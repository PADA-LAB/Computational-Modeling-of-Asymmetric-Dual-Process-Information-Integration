# review_helpfulness_runner/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class RunnerConfig:
    experiment: Dict[str, Any]
    data: Dict[str, str]
    s1_features: Dict[str, List[str]]
    output: Dict[str, str]

    @staticmethod
    def load(config_path: Path) -> "RunnerConfig":
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in ["experiment", "data", "s1_features", "output"]:
            if key not in cfg:
                raise KeyError(f"Missing '{key}' in config: {config_path}")
        return RunnerConfig(
            experiment=cfg["experiment"],
            data=cfg["data"],
            s1_features=cfg["s1_features"],
            output=cfg["output"],
        )