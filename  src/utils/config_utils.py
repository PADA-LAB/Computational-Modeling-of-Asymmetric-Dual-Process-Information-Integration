# src/utils/config_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """
    기준 파일: src/utils/config_utils.py
    - .../<repo>/src/utils/config_utils.py
    - parents[2] == .../<repo>
    """
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_s1_feature_cols(cfg: Dict[str, Any], platform: str) -> list[str]:
    feat_map = cfg.get("s1_features", {}) or {}
    if platform in feat_map:
        return list(feat_map[platform])
    if "default" in feat_map:
        return list(feat_map["default"])
    raise KeyError("s1_features가 config에 없거나 default가 없습니다.")


def resolve_cfg_paths_abs(cfg: Dict[str, Any], project_root: Path) -> Dict[str, str]:
    """
    cfg['paths'] 아래의 모든 상대경로를 project_root 기준 절대경로로 변환.
    - paths에 어떤 키가 추가되어도 자동으로 변환됨
    """
    project_root = project_root.resolve()
    paths = cfg.get("paths", {}) or {}

    out: Dict[str, str] = {}
    for k, v in paths.items():
        out[k] = str((project_root / v).resolve())
    return out


def should_use_finetuned_t5(cfg: Dict[str, Any]) -> bool:
    fin = cfg.get("finetune", {}) or {}
    return bool(fin.get("enabled", False))