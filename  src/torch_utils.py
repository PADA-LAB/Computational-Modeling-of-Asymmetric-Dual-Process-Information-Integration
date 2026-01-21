# src/torch_utils.py
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, mean_absolute_error


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, n_units: int, dropout: float,
                 output_dim: int = 1, task_type: str = "classification"):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, n_units), nn.ReLU(), nn.Dropout(dropout)]
            in_d = n_units
        layers.append(nn.Linear(in_d, output_dim))
        if task_type == "classification":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    lr: float,
    epochs: int = 15,
    batch: int = 128,
    task_type: str = "classification",
):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    if task_type == "classification":
        cri = nn.BCELoss()
        metric_func = average_precision_score
        best_init = -1.0
        cmp = lambda cur, best: cur > best
        X_t, X_v, y_t, y_v = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42
        )
    else:
        cri = nn.L1Loss()
        metric_func = lambda yt, yp: -mean_absolute_error(yt, yp)
        best_init = -float("inf")
        cmp = lambda cur, best: cur > best
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.1, random_state=42)

    ds = TensorDataset(torch.FloatTensor(X_t), torch.FloatTensor(y_t).view(-1, 1))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best_score, best_w = best_init, copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for bx, by in dl:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            loss = cri(model(bx), by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_v).to(device)).cpu().numpy().flatten()
            sc = metric_func(y_v, preds)
        if cmp(sc, best_score):
            best_score, best_w = sc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_w)
    return model


def predict_torch(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()