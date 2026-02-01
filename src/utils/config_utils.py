# src/utils/config_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
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


def resolve_cfg_paths_abs(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    cfg['paths'] 아래의 상대경로를 project_root 기준 절대경로로 변환.
    - 문자열 경로(str)는 abs path(str)로 변환
    - dict/list처럼 중첩된 구조도 재귀적으로 처리 (플랫폼별 경로 등)
    - 경로가 아닌 값은 그대로 둠
    """
    project_root = project_root.resolve()
    paths = cfg.get("paths", {}) or {}

    def _resolve(value: Any) -> Any:
        # string path -> absolute
        if isinstance(value, str):
            return str((project_root / value).resolve())

        # dict -> recurse
        if isinstance(value, dict):
            return {kk: _resolve(vv) for kk, vv in value.items()}

        # list -> recurse
        if isinstance(value, list):
            return [_resolve(x) for x in value]

        # otherwise 그대로 반환
        return value

    out: Dict[str, Any] = {}
    for k, v in paths.items():
        out[k] = _resolve(v)

    return out


def should_use_finetuned_t5(cfg: Dict[str, Any]) -> bool:
    fin = cfg.get("finetune", {}) or {}
    return bool(fin.get("enabled", False))