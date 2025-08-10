from __future__ import annotations

from h_pulse.prediction_ai.train import run_training
from h_pulse.prediction_ai.infer import run_inference


def test_training_and_inference():
    res = run_training(epochs=1, seed=7)
    assert "metrics" in res and res["metrics"]["loss"] >= 0
    inf = run_inference(seed=7, num_events=6)
    assert len(inf["probs"]) == 6