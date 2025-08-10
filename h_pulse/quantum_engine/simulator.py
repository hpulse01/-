"""
量子引擎模拟器
实现生命轨迹的量子叠加态模拟、测量塌缩、蒙特卡洛细化等

理论基础：
- 生命状态建模为量子叠加态 |Ψ_life⟩ = Σ α_i |Event_i⟩
- 测量导致波函数塌缩到特定事件
- 使用蒙特卡洛方法细化后验概率分布
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import structlog
from datetime import datetime, timedelta
import json
import hashlib

# 量子计算库
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit.circuit.library import QFT, HGate, RYGate, CXGate
from pennylane import numpy as pnp
import pennylane as qml

from ..utils.crypto_anchor import generate_quantum_fingerprint
from ..utils.settings import get_settings

logger = structlog.get_logger()


@dataclass
class EventAmplitude:
    """事件振幅"""
    event_id: str
    event_type: str  # 事业、感情、健康、财富等
    description: str
    timestamp: datetime
    location: Optional[Dict[str, float]] = None
    
    # 量子属性
    amplitude: complex = 0.0 + 0.0j
    probability: float = 0.0
    phase: float = 0.0
    
    # 影响因素
    influences: Dict[str, float] = field(default_factory=dict)
    
    def normalize_amplitude(self):
        """归一化振幅"""
        self.probability = abs(self.amplitude) ** 2
        self.phase = np.angle(self.amplitude)


@dataclass
class LifeQuantumState:
    """生命量子态"""
    state_vector: np.ndarray  # 状态向量
    events: List[EventAmplitude]  # 事件列表
    birth_info: Dict[str, Any]  # 出生信息
    quantum_fingerprint: str  # 量子指纹
    
    # 量子属性
    entanglement_entropy: float = 0.0
    coherence: float = 1.0
    fidelity: float = 1.0
    
    # 电路信息
    circuit_depth: int = 0
    gate_count: int = 0
    measurement_counts: Dict[str, int] = field(default_factory=dict)
    
    def get_probability_distribution(self) -> Dict[str, float]:
        """获取概率分布"""
        probs = {}
        for i, event in enumerate(self.events):
            if i < len(self.state_vector):
                prob = abs(self.state_vector[i]) ** 2
                probs[event.event_id] = prob
        return probs
    
    def get_dominant_events(self, threshold: float = 0.1) -> List[EventAmplitude]:
        """获取主要事件（概率超过阈值）"""
        dominant = []
        for i, event in enumerate(self.events):
            if i < len(self.state_vector):
                prob = abs(self.state_vector[i]) ** 2
                if prob >= threshold:
                    event.probability = prob
                    dominant.append(event)
        return sorted(dominant, key=lambda e: e.probability, reverse=True)


class QuantumSimulator:
    """量子模拟器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.backend = Aer.get_backend(self.settings.quantum_backend)
        self.shots = self.settings.quantum_shots
        self.seed = self.settings.quantum_seed
        
        # 初始化PennyLane设备
        self.qml_device = qml.device('default.qubit', wires=16)
        
    def create_life_superposition(self, birth_data: Dict[str, Any],
                                astro_features: Dict[str, Any],
                                possible_events: List[Dict[str, Any]]) -> LifeQuantumState:
        """
        创建生命叠加态
        
        Args:
            birth_data: 出生数据
            astro_features: 天文/命理特征
            possible_events: 可能的事件列表
            
        Returns:
            生命量子态
        """
        # 生成量子指纹
        quantum_fingerprint = generate_quantum_fingerprint(
            birth_data,
            {"longitude": birth_data.get("longitude", 0),
             "latitude": birth_data.get("latitude", 0),
             "timezone": birth_data.get("timezone", "UTC")}
        )
        
        # 创建事件振幅
        events = []
        n_events = min(len(possible_events), 16)  # 限制在16个量子比特
        
        for i, event_data in enumerate(possible_events[:n_events]):
            event = EventAmplitude(
                event_id=f"E{i:04d}",
                event_type=event_data.get("type", "unknown"),
                description=event_data.get("description", ""),
                timestamp=datetime.fromisoformat(event_data.get("timestamp", 
                    datetime.now().isoformat())),
                location=event_data.get("location"),
                influences=event_data.get("influences", {})
            )
            events.append(event)
        
        # 计算初始振幅（基于天文/命理特征）
        amplitudes = self._calculate_initial_amplitudes(events, astro_features)
        
        # 创建量子电路
        n_qubits = int(np.ceil(np.log2(len(events))))
        qc = self._build_life_circuit(n_qubits, amplitudes, astro_features)
        
        # 获取状态向量
        statevector = self._get_statevector(qc)
        
        # 截取到事件数量
        if len(statevector) > len(events):
            statevector = statevector[:len(events)]
        
        # 更新事件振幅
        for i, event in enumerate(events):
            if i < len(statevector):
                event.amplitude = statevector[i]
                event.normalize_amplitude()
        
        # 创建量子态
        quantum_state = LifeQuantumState(
            state_vector=statevector,
            events=events,
            birth_info=birth_data,
            quantum_fingerprint=quantum_fingerprint,
            circuit_depth=qc.depth(),
            gate_count=qc.size()
        )
        
        # 计算量子属性
        self._calculate_quantum_properties(quantum_state)
        
        logger.info("创建生命叠加态",
                   n_events=len(events),
                   n_qubits=n_qubits,
                   fingerprint=quantum_fingerprint[:16])
        
        return quantum_state
    
    def _calculate_initial_amplitudes(self, events: List[EventAmplitude],
                                    astro_features: Dict[str, Any]) -> np.ndarray:
        """计算初始振幅（基于命理/天文特征）"""
        n_events = len(events)
        amplitudes = np.zeros(n_events, dtype=complex)
        
        # 基础均匀叠加
        base_amp = 1.0 / np.sqrt(n_events)
        
        for i, event in enumerate(events):
            # 根据事件类型和天文特征调整振幅
            weight = 1.0
            
            # 事业相关
            if event.event_type == "career":
                # 查看十宫（官禄宫）状态
                career_score = astro_features.get("career_score", 0.5)
                weight *= (0.5 + career_score)
            
            # 感情相关
            elif event.event_type == "relationship":
                # 查看七宫（夫妻宫）状态
                relationship_score = astro_features.get("relationship_score", 0.5)
                weight *= (0.5 + relationship_score)
            
            # 健康相关
            elif event.event_type == "health":
                # 查看六宫（疾厄宫）状态
                health_score = astro_features.get("health_score", 0.5)
                weight *= (0.5 + health_score)
            
            # 财富相关
            elif event.event_type == "wealth":
                # 查看二宫（财帛宫）状态
                wealth_score = astro_features.get("wealth_score", 0.5)
                weight *= (0.5 + wealth_score)
            
            # 加入相位影响
            phase = 0.0
            if "phase_influence" in astro_features:
                phase = astro_features["phase_influence"].get(event.event_type, 0.0)
            
            # 设置复振幅
            amplitudes[i] = weight * base_amp * np.exp(1j * phase)
        
        # 归一化
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes /= norm
        
        return amplitudes
    
    def _build_life_circuit(self, n_qubits: int, amplitudes: np.ndarray,
                          astro_features: Dict[str, Any]) -> QuantumCircuit:
        """构建生命量子电路"""
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # 初始化到指定振幅分布
        # 使用振幅编码
        if len(amplitudes) <= 2**n_qubits:
            # 填充到2^n维
            padded_amps = np.zeros(2**n_qubits, dtype=complex)
            padded_amps[:len(amplitudes)] = amplitudes
            
            # 归一化
            norm = np.linalg.norm(padded_amps)
            if norm > 0:
                padded_amps /= norm
            
            # 初始化
            qc.initialize(padded_amps, qr)
        
        # 应用量子纠缠（模拟事件间的相互影响）
        for i in range(n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
        
        # 应用相位门（基于天文相位）
        if "planetary_phases" in astro_features:
            phases = astro_features["planetary_phases"]
            for i, phase in enumerate(phases[:n_qubits]):
                qc.rz(phase, qr[i])
        
        # 应用受控旋转（模拟条件概率）
        for i in range(n_qubits - 1):
            angle = np.pi / 4  # 可以根据特征调整
            qc.cry(angle, qr[i], qr[i + 1])
        
        # 添加一些噪声门（模拟不确定性）
        noise_level = astro_features.get("uncertainty", 0.1)
        for i in range(n_qubits):
            qc.ry(noise_level * np.pi, qr[i])
        
        return qc
    
    def _get_statevector(self, qc: QuantumCircuit) -> np.ndarray:
        """获取量子电路的状态向量"""
        # 使用状态向量模拟器
        backend = Aer.get_backend('statevector_simulator')
        
        # 执行电路
        job = execute(qc, backend, seed_simulator=self.seed)
        result = job.result()
        
        # 获取状态向量
        statevector = result.get_statevector(qc)
        
        return np.array(statevector)
    
    def _calculate_quantum_properties(self, quantum_state: LifeQuantumState):
        """计算量子属性"""
        sv = quantum_state.state_vector
        
        # 计算纠缠熵（使用约化密度矩阵）
        if len(sv) >= 4:
            # 构造密度矩阵
            rho = np.outer(sv, np.conj(sv))
            
            # 计算部分迹（简化：取前半部分）
            n = int(np.log2(len(sv)))
            n_a = n // 2
            dim_a = 2**n_a
            dim_b = 2**(n - n_a)
            
            # 重塑并计算部分迹
            rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            rho_a = np.trace(rho_reshaped, axis1=1, axis2=3)
            
            # 计算冯诺依曼熵
            eigenvalues = np.linalg.eigvalsh(rho_a)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            
            quantum_state.entanglement_entropy = entropy
        
        # 计算相干性（l1范数）
        coherence = 0.0
        for i in range(len(sv)):
            for j in range(i + 1, len(sv)):
                coherence += abs(sv[i] * np.conj(sv[j]))
        quantum_state.coherence = 2 * coherence
        
        # 保真度（与最大纠缠态的比较）
        max_entangled = np.ones(len(sv)) / np.sqrt(len(sv))
        quantum_state.fidelity = abs(np.dot(np.conj(sv), max_entangled)) ** 2
    
    def measure_quantum_state(self, quantum_state: LifeQuantumState,
                            n_shots: Optional[int] = None) -> Tuple[EventAmplitude, Dict[str, Any]]:
        """
        测量量子态（导致塌缩）
        
        Args:
            quantum_state: 生命量子态
            n_shots: 测量次数
            
        Returns:
            (测量得到的事件, 测量统计信息)
        """
        if n_shots is None:
            n_shots = self.shots
        
        # 获取概率分布
        probs = quantum_state.get_probability_distribution()
        
        # 创建测量电路
        n_qubits = int(np.ceil(np.log2(len(quantum_state.events))))
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # 初始化到当前状态
        qc.initialize(quantum_state.state_vector, qr)
        
        # 测量所有量子比特
        qc.measure(qr, cr)
        
        # 执行测量
        job = execute(qc, self.backend, shots=n_shots, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts(qc)
        
        # 更新测量计数
        quantum_state.measurement_counts = counts
        
        # 找到最可能的结果
        max_count = 0
        measured_bitstring = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                measured_bitstring = bitstring
        
        # 转换为事件索引
        if measured_bitstring:
            event_idx = int(measured_bitstring, 2)
            if event_idx < len(quantum_state.events):
                measured_event = quantum_state.events[event_idx]
            else:
                measured_event = quantum_state.events[0]
        else:
            measured_event = quantum_state.events[0]
        
        # 计算测量统计
        stats = {
            "total_shots": n_shots,
            "measurement_counts": counts,
            "probability_distribution": probs,
            "collapsed_to": measured_event.event_id,
            "confidence": max_count / n_shots if n_shots > 0 else 0,
            "entropy": self._calculate_measurement_entropy(counts, n_shots)
        }
        
        logger.info("量子态测量完成",
                   event_id=measured_event.event_id,
                   confidence=stats["confidence"])
        
        return measured_event, stats
    
    def _calculate_measurement_entropy(self, counts: Dict[str, int], 
                                     total: int) -> float:
        """计算测量熵"""
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def monte_carlo_refinement(self, quantum_state: LifeQuantumState,
                             measured_event: EventAmplitude,
                             n_samples: int = 10000) -> Dict[str, Any]:
        """
        蒙特卡洛细化（计算后验分布）
        
        Args:
            quantum_state: 原始量子态
            measured_event: 测量得到的事件
            n_samples: 蒙特卡洛采样数
            
        Returns:
            细化后的预测信息
        """
        refinements = {
            "primary_event": measured_event.event_id,
            "confidence_interval": {},
            "related_events": [],
            "timeline_distribution": {},
            "parameter_estimates": {}
        }
        
        # 基于测量结果调整概率分布
        event_idx = quantum_state.events.index(measured_event)
        
        # 贝叶斯更新
        prior_probs = quantum_state.get_probability_distribution()
        
        # 蒙特卡洛采样
        samples = []
        for _ in range(n_samples):
            # 从后验分布采样
            # 考虑测量结果的影响
            sample = {
                "timestamp": measured_event.timestamp + \
                           timedelta(days=np.random.normal(0, 30)),
                "intensity": np.random.beta(2, 5),  # 事件强度
                "duration": np.random.gamma(2, 30),  # 持续时间（天）
                "location_variance": np.random.exponential(10)  # 位置不确定性（km）
            }
            samples.append(sample)
        
        # 计算统计量
        timestamps = [s["timestamp"] for s in samples]
        intensities = [s["intensity"] for s in samples]
        durations = [s["duration"] for s in samples]
        
        # 置信区间（95%）
        refinements["confidence_interval"] = {
            "timestamp": {
                "mean": np.mean([t.timestamp() for t in timestamps]),
                "std": np.std([t.timestamp() for t in timestamps]),
                "lower": np.percentile([t.timestamp() for t in timestamps], 2.5),
                "upper": np.percentile([t.timestamp() for t in timestamps], 97.5)
            },
            "intensity": {
                "mean": np.mean(intensities),
                "std": np.std(intensities),
                "lower": np.percentile(intensities, 2.5),
                "upper": np.percentile(intensities, 97.5)
            },
            "duration_days": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "lower": np.percentile(durations, 2.5),
                "upper": np.percentile(durations, 97.5)
            }
        }
        
        # 寻找相关事件（基于量子纠缠）
        for i, event in enumerate(quantum_state.events):
            if i != event_idx:
                # 计算关联强度（使用密度矩阵元素）
                correlation = abs(quantum_state.state_vector[event_idx] * 
                                np.conj(quantum_state.state_vector[i]))
                
                if correlation > 0.1:  # 阈值
                    refinements["related_events"].append({
                        "event_id": event.event_id,
                        "correlation": float(correlation),
                        "type": event.event_type,
                        "description": event.description
                    })
        
        # 时间线分布（基于采样）
        timeline_bins = np.linspace(0, 365, 13)  # 月度分布
        hist, _ = np.histogram([(t - measured_event.timestamp).days 
                               for t in timestamps], bins=timeline_bins)
        
        refinements["timeline_distribution"] = {
            f"month_{i}": int(count) for i, count in enumerate(hist)
        }
        
        # 参数估计
        refinements["parameter_estimates"] = {
            "occurrence_probability": float(prior_probs.get(measured_event.event_id, 0)),
            "quantum_fidelity": float(quantum_state.fidelity),
            "measurement_confidence": float(measured_event.probability),
            "monte_carlo_convergence": self._check_convergence(samples)
        }
        
        logger.info("蒙特卡洛细化完成",
                   n_samples=n_samples,
                   n_related=len(refinements["related_events"]))
        
        return refinements
    
    def _check_convergence(self, samples: List[Dict[str, Any]]) -> float:
        """检查蒙特卡洛收敛性"""
        # 使用Gelman-Rubin统计量的简化版本
        n = len(samples)
        if n < 100:
            return 0.0
        
        # 分成两半
        half = n // 2
        chain1 = samples[:half]
        chain2 = samples[half:2*half]
        
        # 计算链内方差
        var1 = np.var([s["intensity"] for s in chain1])
        var2 = np.var([s["intensity"] for s in chain2])
        within_var = (var1 + var2) / 2
        
        # 计算链间方差
        mean1 = np.mean([s["intensity"] for s in chain1])
        mean2 = np.mean([s["intensity"] for s in chain2])
        between_var = half * (mean1 - mean2)**2
        
        # R统计量
        if within_var > 0:
            r_stat = np.sqrt((within_var + between_var) / within_var)
            convergence = 1.0 / r_stat if r_stat > 1 else 1.0
        else:
            convergence = 1.0
        
        return float(convergence)
    
    def calculate_fidelity(self, state1: LifeQuantumState, 
                         state2: LifeQuantumState) -> float:
        """计算两个量子态的保真度"""
        # 确保维度匹配
        min_dim = min(len(state1.state_vector), len(state2.state_vector))
        sv1 = state1.state_vector[:min_dim]
        sv2 = state2.state_vector[:min_dim]
        
        # 归一化
        sv1 = sv1 / np.linalg.norm(sv1)
        sv2 = sv2 / np.linalg.norm(sv2)
        
        # 计算保真度
        fidelity = abs(np.dot(np.conj(sv1), sv2)) ** 2
        
        return float(fidelity)
    
    def entangle_states(self, state1: LifeQuantumState, 
                       state2: LifeQuantumState) -> LifeQuantumState:
        """
        纠缠两个量子态（例如：伴侣关系）
        
        Args:
            state1: 第一个生命量子态
            state2: 第二个生命量子态
            
        Returns:
            纠缠后的联合量子态
        """
        # 张量积
        joint_state = np.kron(state1.state_vector, state2.state_vector)
        
        # 应用纠缠门（简化：使用CNOT类操作）
        dim1 = len(state1.state_vector)
        dim2 = len(state2.state_vector)
        
        # 创建纠缠
        entangling_factor = 0.7  # 纠缠强度
        for i in range(min(dim1, dim2)):
            idx1 = i * dim2 + i
            idx2 = ((i + 1) % dim1) * dim2 + ((i + 1) % dim2)
            if idx1 < len(joint_state) and idx2 < len(joint_state):
                # 交换振幅
                temp = joint_state[idx1]
                joint_state[idx1] = entangling_factor * joint_state[idx1] + \
                                   np.sqrt(1 - entangling_factor**2) * joint_state[idx2]
                joint_state[idx2] = np.sqrt(1 - entangling_factor**2) * temp + \
                                   entangling_factor * joint_state[idx2]
        
        # 归一化
        joint_state = joint_state / np.linalg.norm(joint_state)
        
        # 创建联合事件列表
        joint_events = []
        for e1 in state1.events[:4]:  # 限制数量
            for e2 in state2.events[:4]:
                joint_event = EventAmplitude(
                    event_id=f"{e1.event_id}_{e2.event_id}",
                    event_type="joint",
                    description=f"联合事件: {e1.description} & {e2.description}",
                    timestamp=max(e1.timestamp, e2.timestamp),
                    influences={**e1.influences, **e2.influences}
                )
                joint_events.append(joint_event)
        
        # 创建联合量子态
        joint_quantum_state = LifeQuantumState(
            state_vector=joint_state[:len(joint_events)],
            events=joint_events,
            birth_info={"person1": state1.birth_info, "person2": state2.birth_info},
            quantum_fingerprint=hashlib.sha256(
                (state1.quantum_fingerprint + state2.quantum_fingerprint).encode()
            ).hexdigest()
        )
        
        # 计算属性
        self._calculate_quantum_properties(joint_quantum_state)
        
        logger.info("量子态纠缠完成",
                   entanglement_entropy=joint_quantum_state.entanglement_entropy)
        
        return joint_quantum_state


# 便捷函数
def simulate_life_superposition(birth_data: Dict[str, Any],
                              astro_features: Dict[str, Any],
                              possible_events: List[Dict[str, Any]]) -> LifeQuantumState:
    """模拟生命叠加态的便捷函数"""
    simulator = QuantumSimulator()
    return simulator.create_life_superposition(birth_data, astro_features, 
                                             possible_events)


def measure_quantum_state(quantum_state: LifeQuantumState) -> Tuple[EventAmplitude, Dict[str, Any]]:
    """测量量子态的便捷函数"""
    simulator = QuantumSimulator()
    return simulator.measure_quantum_state(quantum_state)


def calculate_fidelity(state1: LifeQuantumState, state2: LifeQuantumState) -> float:
    """计算保真度的便捷函数"""
    simulator = QuantumSimulator()
    return simulator.calculate_fidelity(state1, state2)


if __name__ == "__main__":
    # 测试
    
    # 出生数据
    birth_data = {
        "name": "测试用户",
        "birth_date": "1990-01-01",
        "birth_time": "12:00:00",
        "longitude": 116.4074,
        "latitude": 39.9042,
        "timezone": "Asia/Shanghai"
    }
    
    # 天文/命理特征
    astro_features = {
        "career_score": 0.8,
        "relationship_score": 0.6,
        "health_score": 0.7,
        "wealth_score": 0.75,
        "planetary_phases": [0.1, 0.3, 0.5, 0.7],
        "uncertainty": 0.15
    }
    
    # 可能的事件
    possible_events = [
        {
            "type": "career",
            "description": "获得重要晋升机会",
            "timestamp": "2025-06-15T10:00:00",
            "influences": {"Jupiter": 0.8, "Saturn": 0.3}
        },
        {
            "type": "relationship", 
            "description": "遇见重要伴侣",
            "timestamp": "2025-09-20T18:30:00",
            "influences": {"Venus": 0.9, "Moon": 0.6}
        },
        {
            "type": "health",
            "description": "健康状况改善",
            "timestamp": "2025-03-10T08:00:00",
            "influences": {"Sun": 0.7, "Mars": 0.4}
        },
        {
            "type": "wealth",
            "description": "投资获得回报",
            "timestamp": "2025-11-30T14:00:00",
            "influences": {"Mercury": 0.6, "Pluto": 0.5}
        }
    ]
    
    # 创建叠加态
    print("=== 创建生命量子叠加态 ===")
    quantum_state = simulate_life_superposition(birth_data, astro_features, 
                                              possible_events)
    
    print(f"量子指纹: {quantum_state.quantum_fingerprint[:32]}...")
    print(f"电路深度: {quantum_state.circuit_depth}")
    print(f"门数量: {quantum_state.gate_count}")
    print(f"纠缠熵: {quantum_state.entanglement_entropy:.4f}")
    print(f"相干性: {quantum_state.coherence:.4f}")
    
    # 显示概率分布
    print("\n事件概率分布:")
    for event in quantum_state.get_dominant_events(0.05):
        print(f"  {event.event_id}: {event.description} - "
              f"概率: {event.probability:.3f}")
    
    # 测量
    print("\n=== 测量量子态 ===")
    measured_event, stats = measure_quantum_state(quantum_state)
    
    print(f"测量结果: {measured_event.event_id} - {measured_event.description}")
    print(f"置信度: {stats['confidence']:.3f}")
    print(f"测量熵: {stats['entropy']:.3f}")
    
    # 蒙特卡洛细化
    print("\n=== 蒙特卡洛细化 ===")
    simulator = QuantumSimulator()
    refinements = simulator.monte_carlo_refinement(quantum_state, measured_event)
    
    print(f"主要事件: {refinements['primary_event']}")
    print(f"时间置信区间: "
          f"{datetime.fromtimestamp(refinements['confidence_interval']['timestamp']['lower']).date()} - "
          f"{datetime.fromtimestamp(refinements['confidence_interval']['timestamp']['upper']).date()}")
    print(f"强度: {refinements['confidence_interval']['intensity']['mean']:.3f} "
          f"± {refinements['confidence_interval']['intensity']['std']:.3f}")
    print(f"相关事件数: {len(refinements['related_events'])}")