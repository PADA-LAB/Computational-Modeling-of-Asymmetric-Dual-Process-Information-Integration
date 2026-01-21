# src/embedding/embedding_utils.py
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
from sentence_transformers import SentenceTransformer

def get_embeddings(texts, model_name, device, batch_size=32):
    print(f"Generating Embeddings with [{model_name}]...")
    embeddings = []

    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        del model

    elif "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        model.eval()
        data_loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        for batch_texts in tqdm(data_loader):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                masked_embeddings = outputs.last_hidden_state * mask
                summed = torch.sum(masked_embeddings, 1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                embeddings.append((summed / counts).cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        del model, tokenizer

    else:  # BERT 계열
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        data_loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        for batch_texts in tqdm(data_loader):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append((sum_embeddings / sum_mask).cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        del model, tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return embeddings