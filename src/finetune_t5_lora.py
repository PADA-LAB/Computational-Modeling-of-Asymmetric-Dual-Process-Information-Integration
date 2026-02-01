# src/finetune_t5_lora.py
from __future__ import annotations

import gc
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from .utils import ensure_dir, log_print


def build_train_idx(df, label_col: str, seed: int) -> np.ndarray:
    indices = np.arange(len(df))
    y_all = df[label_col].values
    train_idx, _ = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y_all
    )
    return train_idx


def run_lora_finetune(
    *,
    platform: str,
    df,
    text_col: str,
    label_col: str,
    model_name: str,
    max_length: int,
    target_seed_for_t5: int,
    ckpt_dir: Path,
    final_model_dir: Path,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    train_epochs: int,
    lr,  
    per_device_train_bs: int,
    per_device_eval_bs: int,
    weight_decay,  
) -> Path:
    """
    returns: platform-specific LoRA adapter dir path
    """
    ckpt_dir = ckpt_dir.resolve()
    final_model_dir = final_model_dir.resolve()

    platform_ckpt_dir = ensure_dir(ckpt_dir / platform)
    platform_final_dir = ensure_dir(final_model_dir / platform)

    # 이미 있으면 스킵
    if (platform_final_dir / "adapter_model.bin").exists() or (platform_final_dir / "adapter_model.safetensors").exists():
        log_print(f"[Skip] {platform} LoRA already exists -> {platform_final_dir}")
        return platform_final_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


    # YAML에서 str로 들어오는 하이퍼파라미터 안전 변환
    try:
        lr = float(lr)
    except Exception as e:
        raise ValueError(f"[finetune.train.lr] must be float-like, got {lr} ({type(lr)})") from e

    try:
        weight_decay = float(weight_decay)
    except Exception as e:
        raise ValueError(f"[finetune.train.weight_decay] must be float-like, got {weight_decay} ({type(weight_decay)})") from e

    try:
        max_length = int(max_length)
        train_epochs = int(train_epochs)
        per_device_train_bs = int(per_device_train_bs)
        per_device_eval_bs = int(per_device_eval_bs)
    except Exception as e:
        raise ValueError(
            f"[finetune] int-like params invalid: "
            f"max_length={max_length}, epochs={train_epochs}, "
            f"train_bs={per_device_train_bs}, eval_bs={per_device_eval_bs}"
        ) from e

    log_print(
        f"[LoRA cfg] lr={lr} ({type(lr).__name__}), "
        f"weight_decay={weight_decay} ({type(weight_decay).__name__}), "
        f"epochs={train_epochs}, max_length={max_length}, "
        f"train_bs={per_device_train_bs}, eval_bs={per_device_eval_bs}"
    )

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # leakage-free: train_idx만 사용
    train_idx = build_train_idx(df, label_col, target_seed_for_t5)
    train_df_full = df.iloc[train_idx].copy()

    train_df_full["y_true"] = train_df_full[label_col].astype(int)
    train_df_full["target_text"] = train_df_full["y_true"].map({1: "helpful", 0: "unhelpful"})
    train_df_full["input_text"] = "classify review: " + train_df_full[text_col].fillna("").astype(str)

    real_train_df, real_val_df = train_test_split(
        train_df_full,
        test_size=0.1,
        random_state=42,
        stratify=train_df_full["y_true"]
    )

    train_ds = Dataset.from_pandas(real_train_df[["input_text", "target_text"]])
    val_ds = Dataset.from_pandas(real_val_df[["input_text", "target_text"]])

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=10,
            truncation=True,
            padding="max_length",
        )

        # pad 토큰은 loss 계산에서 무시하도록 -100으로 변경
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(tid if tid != tokenizer.pad_token_id else -100) for tid in seq]
            for seq in labels_ids
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    base_model = T5ForConditionalGeneration.from_pretrained(model_name)
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(base_model, peft_cfg).to(device)

    args = Seq2SeqTrainingArguments(
        output_dir=str(platform_ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        num_train_epochs=train_epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,
        bf16=use_bf16,
        logging_dir=str(platform_ckpt_dir / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    log_print(f"[Train] LoRA fine-tune start: {platform}")
    trainer.train()

    model.save_pretrained(platform_final_dir)
    log_print(f"[Save] LoRA adapter saved: {platform_final_dir}")

    del model, trainer, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return platform_final_dir