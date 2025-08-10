"""
测试量子引擎模块
"""

import pytest
from datetime import datetime
import numpy as np

from h_pulse.quantum_engine.simulator import (
    QuantumSimulator, EventAmplitude, LifeQuantumState,
    simulate_life_superposition, measure_quantum_state
)


class TestEventAmplitude:
    """测试事件振幅"""
    
    def test_amplitude_normalization(self):
        """测试振幅归一化"""
        event = EventAmplitude(
            event_id="E001",
            event_type="test",
            description="测试事件",
            timestamp=datetime.now(),
            amplitude=complex(3, 4)  # |3+4i| = 5
        )
        
        event.normalize_amplitude()
        
        # 归一化后模应该是1
        assert abs(abs(event.amplitude) - 1.0) < 1e-10
        
        # 相位应该保持
        original_phase = np.angle(complex(3, 4))
        normalized_phase = np.angle(event.amplitude)
        assert abs(original_phase - normalized_phase) < 1e-10
    
    def test_probability_calculation(self):
        """测试概率计算"""
        event = EventAmplitude(
            event_id="E002",
            event_type="career",
            description="职业发展",
            timestamp=datetime.now(),
            amplitude=complex(0.6, 0.8)  # |0.6+0.8i|^2 = 1.0
        )
        
        prob = event.get_probability()
        expected_prob = 0.6**2 + 0.8**2  # = 1.0
        assert abs(prob - expected_prob) < 1e-10


class TestLifeQuantumState:
    """测试生命量子态"""
    
    def test_state_normalization(self):
        """测试态矢量归一化"""
        events = [
            EventAmplitude("E1", "type1", "desc1", datetime.now()),
            EventAmplitude("E2", "type2", "desc2", datetime.now()),
            EventAmplitude("E3", "type3", "desc3", datetime.now())
        ]
        
        state_vector = np.array([0.5, 0.5, 0.5], dtype=complex)
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test_fingerprint"
        )
        
        quantum_state.normalize()
        
        # 归一化后模平方和应该是1
        norm_squared = np.sum(np.abs(quantum_state.state_vector)**2)
        assert abs(norm_squared - 1.0) < 1e-10
    
    def test_measurement_probabilities(self):
        """测试测量概率"""
        events = [
            EventAmplitude("E1", "type1", "desc1", datetime.now()),
            EventAmplitude("E2", "type2", "desc2", datetime.now())
        ]
        
        # 创建一个简单的叠加态 |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
        state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test"
        )
        
        probs = quantum_state.get_measurement_probabilities()
        
        # 每个事件的概率应该是0.5
        assert len(probs) == 2
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10


class TestQuantumSimulator:
    """测试量子模拟器"""
    
    def test_create_life_superposition(self):
        """测试创建生命叠加态"""
        simulator = QuantumSimulator()
        
        birth_data = {
            'name': 'Test User',
            'birth_datetime': '1990-01-01T12:00:00',
            'longitude': 116.4074,
            'latitude': 39.9042
        }
        
        astro_features = {
            'bazi': {'rizhu': '甲子'},
            'ziwei': {'ming_gong': '紫微'},
            'natal': {'sun_sign': 'Capricorn'}
        }
        
        possible_events = [
            {"type": "career", "description": "事业发展", "timestamp": "2025-01-01T00:00:00"},
            {"type": "relationship", "description": "感情机遇", "timestamp": "2025-06-01T00:00:00"},
            {"type": "health", "description": "健康挑战", "timestamp": "2025-09-01T00:00:00"}
        ]
        
        quantum_state = simulator.create_life_superposition(
            birth_data, astro_features, possible_events
        )
        
        # 验证量子态
        assert quantum_state is not None
        assert len(quantum_state.events) == 3
        assert quantum_state.circuit_depth > 0
        assert quantum_state.gate_count > 0
        
        # 验证概率归一化
        probs = quantum_state.get_measurement_probabilities()
        assert abs(sum(probs) - 1.0) < 1e-10
    
    def test_measure_collapse(self):
        """测试测量坍缩"""
        simulator = QuantumSimulator()
        
        # 创建简单的量子态
        events = [
            EventAmplitude("E1", "type1", "desc1", datetime.now(), complex(0.8, 0)),
            EventAmplitude("E2", "type2", "desc2", datetime.now(), complex(0.6, 0))
        ]
        
        state_vector = np.array([0.8, 0.6], dtype=complex)
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test"
        )
        
        # 多次测量
        measurements = []
        for _ in range(100):
            measured_event = simulator.measure_collapse(quantum_state)
            measurements.append(measured_event.event_id)
        
        # 统计测量结果
        e1_count = measurements.count("E1")
        e2_count = measurements.count("E2")
        
        # 理论概率：P(E1) = 0.64, P(E2) = 0.36
        # 由于随机性，使用较宽的容差
        assert 40 <= e1_count <= 90  # 大致64%
        assert 10 <= e2_count <= 60  # 大致36%
    
    def test_quantum_properties(self):
        """测试量子属性计算"""
        simulator = QuantumSimulator()
        
        # 创建纠缠态
        events = [
            EventAmplitude("E1", "type1", "desc1", datetime.now()),
            EventAmplitude("E2", "type2", "desc2", datetime.now()),
            EventAmplitude("E3", "type3", "desc3", datetime.now()),
            EventAmplitude("E4", "type4", "desc4", datetime.now())
        ]
        
        # Bell态 |ψ⟩ = (1/√2)(|00⟩ + |11⟩)
        state_vector = np.zeros(4, dtype=complex)
        state_vector[0] = 1/np.sqrt(2)  # |00⟩
        state_vector[3] = 1/np.sqrt(2)  # |11⟩
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test"
        )
        
        # 计算量子属性
        simulator._calculate_quantum_properties(quantum_state)
        
        # 验证纠缠熵（Bell态的纠缠熵应该是ln(2)）
        assert quantum_state.entanglement_entropy > 0
        assert abs(quantum_state.entanglement_entropy - np.log(2)) < 0.1
        
        # 验证相干性
        assert 0 <= quantum_state.coherence <= 1
        
        # 验证保真度
        assert quantum_state.fidelity == 1.0  # 纯态的保真度是1
    
    def test_monte_carlo_refinement(self):
        """测试蒙特卡洛优化"""
        simulator = QuantumSimulator()
        
        events = [
            EventAmplitude("E1", "career", "职业发展", datetime.now()),
            EventAmplitude("E2", "health", "健康状况", datetime.now())
        ]
        
        state_vector = np.array([0.7, 0.3], dtype=complex)
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test"
        )
        
        # 进行蒙特卡洛优化
        refined_state = simulator.monte_carlo_refinement(
            quantum_state,
            n_iterations=100
        )
        
        # 验证优化后的状态
        assert refined_state is not None
        assert hasattr(refined_state, 'posterior_distribution')
        assert len(refined_state.posterior_distribution) == 2
        
        # 验证后验分布归一化
        posterior_sum = sum(refined_state.posterior_distribution.values())
        assert abs(posterior_sum - 1.0) < 1e-10


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_simulate_life_superposition_function(self):
        """测试生命叠加态模拟便捷函数"""
        birth_data = {
            'name': 'Test',
            'birth_datetime': '2000-01-01T00:00:00',
            'longitude': 0.0,
            'latitude': 0.0
        }
        
        astro_features = {'test': 'features'}
        
        possible_events = [
            {"type": "test", "description": "测试事件", "timestamp": "2025-01-01T00:00:00"}
        ]
        
        quantum_state = simulate_life_superposition(
            birth_data, astro_features, possible_events
        )
        
        assert quantum_state is not None
        assert len(quantum_state.events) > 0
    
    def test_measure_quantum_state_function(self):
        """测试量子态测量便捷函数"""
        # 创建简单量子态
        events = [EventAmplitude("E1", "type1", "desc1", datetime.now())]
        state_vector = np.array([1.0], dtype=complex)
        
        quantum_state = LifeQuantumState(
            state_vector=state_vector,
            events=events,
            birth_info={},
            quantum_fingerprint="test"
        )
        
        measured_event = measure_quantum_state(quantum_state)
        
        assert measured_event is not None
        assert measured_event.event_id == "E1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])