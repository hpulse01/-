from __future__ import annotations

from h_pulse.quantum_engine.simulator import simulate_quantum_life


def test_quantum_simulation_basic():
    res = simulate_quantum_life(features={"x": 1}, bio={"id": 1}, birth_spacetime={"t": 0}, seed=123)
    assert 0 <= res.event_index < 8
    assert 0 <= res.probability <= 1
    lo, hi = res.posterior_ci
    assert 0 <= lo <= hi <= 1