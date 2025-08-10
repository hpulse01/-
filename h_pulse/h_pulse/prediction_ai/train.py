from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np

from .model import TrainConfig, train_model


def generate_synthetic_dataset(n: int = 64, d: int = 16, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    A = np.eye(n)
    A[rng.integers(0, n, size=n), rng.integers(0, n, size=n)] = 1
    weights = rng.normal(size=(d,))
    y = (X @ weights + rng.normal(scale=0.5, size=(n,)) > 0).astype(int)
    return X, A, y


def run_training(epochs: int = 2, seed: int = 42) -> Dict:
    X, A, y = generate_synthetic_dataset(seed=seed)
    cfg = TrainConfig(seed=seed, epochs=epochs)
    metrics = train_model(cfg, X, A, y)
    return {"config": asdict(cfg), "metrics": metrics}