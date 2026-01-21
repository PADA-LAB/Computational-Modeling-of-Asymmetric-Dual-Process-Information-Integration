# src/embedding/mlp_utils.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, n_layers, n_units, dropout, output_dim=1):
        super(SimpleMLP, self).__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_d, n_units), nn.ReLU(), nn.Dropout(dropout)])
            in_d = n_units
        layers.append(nn.Linear(in_d, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch(model, X_train, y_train, X_valid, y_valid, device, lr, epochs=15):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    cri = nn.BCELoss()

    ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    best_score = -1
    best_weights = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            cri(model(bx.to(device)), by.to(device)).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_valid).to(device)).cpu().numpy().flatten()
            score = average_precision_score(y_valid, preds)

        if score > best_score:
            best_score = score
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, best_score

def predict_torch(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()