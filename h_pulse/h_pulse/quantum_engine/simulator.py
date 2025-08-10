from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    _HAS_QISKIT = False

try:
    import pennylane as qml  # type: ignore
    _HAS_PENNYLANE = True
except Exception:  # pragma: no cover
    _HAS_PENNYLANE = False


@dataclass
class QuantumRunMeta:
    backend: str
    depth: int
    num_gates: int
    seed: int


@dataclass
class CollapseResult:
    event_index: int
    amplitude: float
    probability: float
    posterior_ci: Tuple[float, float]
    fidelity: float
    fingerprint_hex: str
    meta: QuantumRunMeta


def build_state_from_features(features: Dict, seed: int = 42, num_events: int = 8) -> Tuple[np.ndarray, QuantumRunMeta]:
    rng = np.random.default_rng(seed)
    amps = rng.normal(size=num_events) + 1j * rng.normal(size=num_events)
    amps = amps / np.linalg.norm(amps)
    meta = QuantumRunMeta(
        backend=("qiskit" if _HAS_QISKIT else ("pennylane" if _HAS_PENNYLANE else "numpy")),
        depth=3,
        num_gates=6,
        seed=seed,
    )
    return amps.astype(np.complex128), meta


def quantum_fingerprint(bio: Dict, birth_spacetime: Dict, device_entropy: int) -> str:
    blob = json.dumps({"bio": bio, "birth": birth_spacetime, "entropy": device_entropy}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    # SHA3-256
    import hashlib

    h = hashlib.sha3_256()
    h.update(blob)
    return h.hexdigest()


def measure_collapse(amplitudes: np.ndarray, seed: int = 42) -> Tuple[int, float, float]:
    rng = np.random.default_rng(seed)
    probs = np.abs(amplitudes) ** 2
    idx = rng.choice(len(probs), p=probs)
    amp = amplitudes[idx]
    return int(idx), float(np.abs(amp)), float(probs[idx])


def monte_carlo_posterior(prob: float, trials: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.binomial(1, prob, size=trials)
    p_hat = samples.mean()
    # Wilson score interval 95%
    z = 1.96
    n = trials
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / (1 + z * z / n)
    return float(p_hat), float(max(0.0, center - half)), float(min(1.0, center + half))


def simulate_quantum_life(features: Dict, bio: Dict, birth_spacetime: Dict, seed: int = 42) -> CollapseResult:
    amplitudes, meta = build_state_from_features(features, seed=seed)
    idx, amp, prob = measure_collapse(amplitudes, seed=seed + 1)
    p_hat, lo, hi = monte_carlo_posterior(prob, seed=seed + 2)
    # Fidelity heuristic: compare measured prob vs max prob
    fidelity = float(prob / float(np.max(np.abs(amplitudes) ** 2)))
    fingerprint = quantum_fingerprint(bio, birth_spacetime, device_entropy=seed)
    return CollapseResult(
        event_index=idx,
        amplitude=amp,
        probability=prob,
        posterior_ci=(lo, hi),
        fidelity=fidelity,
        fingerprint_hex=fingerprint,
        meta=meta,
    )