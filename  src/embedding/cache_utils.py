# src/embedding/cache_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.utils import ensure_dir


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _finetune_enabled(cfg: Dict) -> bool:
    fin = cfg.get("finetune", {}) or {}
    return bool(fin.get("enabled", False))


def embedding_cache_dir(platform_out_dir: Path, cfg: Dict) -> Path:
    """
    outputs/<platform>/embeddings_cache/<safe_key>/X_all.npy 로 캐시를 분리한다.

    - embedding.mode == "t5":
        finetune.enabled == True  -> t5_lora__{t5-base}__{pool}
        finetune.enabled == False -> t5_base__{t5-base}__{pool}

    - embedding.mode == "sentence_transformer":
        st__{model_name}
    """
    emb = cfg.get("embedding", {}) or {}
    mode = emb.get("mode", "t5")

    if mode == "sentence_transformer":
        model_name = emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        safe_key = f"st__{_safe(model_name)}"
        return ensure_dir(platform_out_dir / "embeddings_cache" / safe_key)

    # default: t5
    base = emb.get("base_model_name", "t5-base")
    pool = emb.get("pool", "mean")

    if _finetune_enabled(cfg):
        safe_key = f"t5_lora__{_safe(base)}__{pool}"
    else:
        safe_key = f"t5_base__{_safe(base)}__{pool}"

    return ensure_dir(platform_out_dir / "embeddings_cache" / safe_key)


def embedding_cache_file(platform_out_dir: Path, cfg: Dict) -> Path:
    return embedding_cache_dir(platform_out_dir, cfg) / "X_all.npy"