from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 2
    lr: float = 1e-3
    hidden: int = 64
    save_dir: str = "runs"


def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore


class NumpyTransformerGNN:
    def __init__(self, input_dim: int, hidden: int, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.Wq = self.rng.normal(size=(input_dim, hidden)) / np.sqrt(input_dim)
        self.Wk = self.rng.normal(size=(input_dim, hidden)) / np.sqrt(input_dim)
        self.Wv = self.rng.normal(size=(input_dim, hidden)) / np.sqrt(input_dim)
        self.Wo = self.rng.normal(size=(hidden, hidden)) / np.sqrt(hidden)
        self.Wg = self.rng.normal(size=(hidden, hidden)) / np.sqrt(hidden)
        self.Wc = self.rng.normal(size=(hidden, 1)) / np.sqrt(hidden)

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        attn = (Q @ K.T) / np.sqrt(Q.shape[-1])
        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        H = attn @ V
        H = H @ self.Wo
        # GNN message passing: H' = A H Wg
        H = A @ H @ self.Wg
        out = H @ self.Wc
        return 1 / (1 + np.exp(-out))  # sigmoid


if _HAS_TORCH:  # pragma: no cover - optional path
    class TorchTransformerGNN(nn.Module):
        def __init__(self, input_dim: int, hidden: int) -> None:
            super().__init__()
            self.q = nn.Linear(input_dim, hidden)
            self.k = nn.Linear(input_dim, hidden)
            self.v = nn.Linear(input_dim, hidden)
            self.o = nn.Linear(hidden, hidden)
            self.g = nn.Linear(hidden, hidden)
            self.c = nn.Linear(hidden, 1)

        def forward(self, X: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
            Q = self.q(X)
            K = self.k(X)
            V = self.v(X)
            attn = torch.softmax(Q @ K.T / (Q.shape[-1] ** 0.5), dim=-1)
            H = attn @ V
            H = self.o(H)
            H = A @ H
            H = self.g(H)
            out = self.c(H)
            return torch.sigmoid(out)
else:
    TorchTransformerGNN = None  # type: ignore


def train_model(config: TrainConfig, X: np.ndarray, A: np.ndarray, y: np.ndarray) -> Dict:
    fix_seed(config.seed)
    if _HAS_TORCH:
        model = TorchTransformerGNN(X.shape[1], config.hidden)  # type: ignore
        optim = torch.optim.Adam(model.parameters(), lr=config.lr)  # type: ignore
        X_t = torch.tensor(X, dtype=torch.float32)  # type: ignore
        A_t = torch.tensor(A, dtype=torch.float32)  # type: ignore
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)  # type: ignore
        for epoch in range(config.epochs):
            optim.zero_grad()
            pred = model(X_t, A_t)  # type: ignore
            loss = torch.nn.functional.binary_cross_entropy(pred, y_t)  # type: ignore
            loss.backward()
            optim.step()
        metrics = {"loss": float(loss.item()), "acc": float(((pred > 0.5) == (y_t > 0.5)).float().mean().item())}  # type: ignore
    else:
        model = NumpyTransformerGNN(X.shape[1], config.hidden, seed=config.seed)
        # One-step pseudo training: compute predictions and report loss
        pred = model.forward(X, A)
        loss = float(np.mean(-(y.reshape(-1, 1) * np.log(pred + 1e-9) + (1 - y.reshape(-1, 1)) * np.log(1 - pred + 1e-9))))
        metrics = {"loss": loss, "acc": float(((pred > 0.5).astype(int).ravel() == y.astype(int)).mean())}

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
    with open(save_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics


def infer(model_seed: int, X: np.ndarray, A: np.ndarray) -> np.ndarray:
    np.random.seed(model_seed)
    model = NumpyTransformerGNN(X.shape[1], hidden=64, seed=model_seed)
    return model.forward(X, A).ravel()