"""
H-Pulse 量子计算引擎
实现量子叠加态模拟、量子纠缠、测量塌缩、量子指纹等功能
"""

from .simulator import *

__all__ = [
    'QuantumSimulator',
    'LifeQuantumState',
    'EventAmplitude',
    'simulate_life_superposition',
    'measure_quantum_state',
    'calculate_fidelity',
]