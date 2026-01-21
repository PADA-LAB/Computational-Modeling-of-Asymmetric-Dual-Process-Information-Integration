# src/embedding/t5_lora_embedding.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

from src.utils import ensure_dir, log_print


def _pool_hidden(hidden: torch.Tensor, attn_mask: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "cls":
        return hidden[:, 0, :]

    mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def build_t5_lora_embeddings(
    *,
    texts: list[str],
    base_model_name: str,
    lora_dir: Path,
    out_dir: Path,
    max_length: int = 256,
    batch_size: int = 32,
    pool: str = "mean",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> np.ndarray:
    out_dir = ensure_dir(out_dir)
    cache_path = out_dir / "X_all.npy"
    if cache_path.exists():
        return np.load(cache_path)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    base = T5ForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, str(lora_dir)).to(device)
    model.eval()

    all_vecs: list[np.ndarray] = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch = texts[i: i + batch_size]
        tok = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok["attention_mask"].to(device)

        enc = model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
        hidden = enc.last_hidden_state
        vec = _pool_hidden(hidden, attn_mask, pool=pool)
        all_vecs.append(vec.detach().to("cpu").float().numpy())

    X = np.concatenate(all_vecs, axis=0)
    np.save(cache_path, X)
    log_print(f"[Emb] saved {cache_path} shape={X.shape}")
    return X