from __future__ import annotations

from typing import Dict, List

import numpy as np

from .model import infer


def run_inference(seed: int = 42, num_events: int = 8) -> Dict:
    rng = np.random.default_rng(seed)
    n = num_events
    d = 16
    X = rng.normal(size=(n, d))
    A = np.eye(n)
    probs = infer(seed, X, A)
    # Bootstrap uncertainty
    samples = np.vstack([infer(seed + i + 1, X, A) for i in range(10)])
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return {
        "probs": probs.tolist(),
        "ci": list(map(lambda t: [float(t[0]), float(t[1])], zip(lo, hi))),
    }