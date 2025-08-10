from __future__ import annotations

from typing import Dict

import numpy as np


def process_features(eastern: Dict, western: Dict) -> Dict:
    X = np.array([len(eastern.get("da_yun", [])), len(western.get("aspects", []))], dtype=float)
    X = (X - X.mean()) / (X.std() + 1e-6)
    return {"vector": X.tolist(), "meta": {"eastern_count": int(X.shape[0])}}