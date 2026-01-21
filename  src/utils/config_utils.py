from __future__ import annotations
from pathlib import Path


def get_s1_feature_cols(cfg: dict, platform: str) -> list[str]:
    feat_map = cfg.get("s1_features", {}) or {}
    if platform in feat_map:
        return list(feat_map[platform])
    if "default" in feat_map:
        return list(feat_map["default"])
    raise KeyError("s1_features가 config에 없거나 default가 없습니다.")


def resolve_cfg_paths_abs(cfg: dict, project_root: Path) -> dict:
    project_root = project_root.resolve()
    paths = cfg.get("paths", {}) or {}

    def _abs(p: str) -> str:
        return str((project_root / p).resolve())

    out = {
        "outputs_root": _abs(paths["outputs_root"]),
        "finetune_ckpt_dir": _abs(paths["finetune_ckpt_dir"]),
        "finetune_model_dir": _abs(paths["finetune_model_dir"]),
    }
    return out


def should_use_finetuned_t5(cfg: dict) -> bool:
    fin = cfg.get("finetune", {}) or {}
    return bool(fin.get("enabled", False))