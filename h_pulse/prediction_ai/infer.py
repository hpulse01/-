"""
模型推理模块
实现模型加载、特征提取、预测等功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import structlog

from .model import LifeTrajectoryModel
from ..data_collection import calculate_bazi, calculate_ziwei, calculate_natal_chart
from ..quantum_engine import simulate_life_superposition, measure_quantum_state
from ..utils.settings import get_settings

logger = structlog.get_logger()


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def extract_features(self, birth_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        从出生数据提取所有特征
        
        Args:
            birth_data: 包含出生信息的字典
                - birth_datetime: 出生时间
                - longitude: 经度
                - latitude: 纬度
                - gender: 性别
                
        Returns:
            包含各类特征的字典
        """
        birth_dt = datetime.fromisoformat(birth_data['birth_datetime'])
        longitude = birth_data['longitude']
        latitude = birth_data['latitude']
        gender = birth_data.get('gender', '男')
        
        # 计算八字
        bazi_chart = calculate_bazi(birth_dt, longitude, latitude, gender)
        
        # 计算紫微斗数
        ziwei_chart = calculate_ziwei(birth_dt, longitude, latitude, gender)
        
        # 计算西方星盘
        natal_chart = calculate_natal_chart(birth_dt, longitude, latitude)
        
        # 提取特征
        features = {
            'astro_features': self._extract_astro_features(birth_dt, longitude, latitude),
            'eastern_features': self._extract_eastern_features(bazi_chart, ziwei_chart),
            'western_features': self._extract_western_features(natal_chart),
            'quantum_features': self._extract_quantum_features(birth_data)
        }
        
        return features
    
    def _extract_astro_features(self, birth_dt: datetime, 
                               longitude: float, latitude: float) -> np.ndarray:
        """提取天文特征"""
        # 时间特征
        year_phase = (birth_dt.year % 60) / 60.0  # 60年周期
        month_phase = birth_dt.month / 12.0
        day_phase = birth_dt.day / 30.0
        hour_phase = birth_dt.hour / 24.0
        
        # 地理特征
        lat_norm = latitude / 90.0
        lon_norm = longitude / 180.0
        
        # 季节特征
        season = self._get_season(birth_dt.month)
        
        # 太阳月亮周期
        lunar_phase = self._estimate_lunar_phase(birth_dt)
        solar_term = self._estimate_solar_term(birth_dt)
        
        # 组合特征
        features = np.array([
            year_phase, month_phase, day_phase, hour_phase,
            lat_norm, lon_norm,
            season[0], season[1], season[2], season[3],
            lunar_phase, solar_term,
            np.sin(2 * np.pi * year_phase),
            np.cos(2 * np.pi * year_phase),
            np.sin(2 * np.pi * month_phase),
            np.cos(2 * np.pi * month_phase),
        ])
        
        # 扩展到64维
        features = np.pad(features, (0, 64 - len(features)), mode='constant')
        
        # 创建序列（10个时间步）
        seq_features = np.tile(features, (10, 1))
        
        # 添加时间步编码
        for i in range(10):
            seq_features[i, -10:] = self._positional_encoding(i, 10)
        
        return seq_features.astype(np.float32)
    
    def _extract_eastern_features(self, bazi_chart, ziwei_chart) -> np.ndarray:
        """提取东方命理特征"""
        features = []
        
        # 八字特征
        # 天干地支编码
        gan_map = {g: i for i, g in enumerate("甲乙丙丁戊己庚辛壬癸")}
        zhi_map = {z: i for i, z in enumerate("子丑寅卯辰巳午未申酉戌亥")}
        
        for pillar in [bazi_chart.year, bazi_chart.month, bazi_chart.day, bazi_chart.hour]:
            features.extend([
                gan_map.get(pillar.gan, 0) / 10.0,
                zhi_map.get(pillar.zhi, 0) / 12.0
            ])
        
        # 五行特征
        wuxing_count = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
        for pillar in [bazi_chart.year, bazi_chart.month, bazi_chart.day, bazi_chart.hour]:
            wuxing_count[pillar.wuxing[0]] += 1
            wuxing_count[pillar.wuxing[1]] += 1
        
        for wx in ["木", "火", "土", "金", "水"]:
            features.append(wuxing_count[wx] / 8.0)
        
        # 紫微斗数特征
        # 命宫主星
        ming_gong = ziwei_chart.get_ming_gong()
        star_map = {s: i for i, s in enumerate(["紫微", "天机", "太阳", "武曲", "天同", 
                                               "廉贞", "天府", "太阴", "贪狼", "巨门",
                                               "天相", "天梁", "七杀", "破军"])}
        
        star_features = [0] * 14
        for star in ming_gong.zhuxing:
            if star in star_map:
                star_features[star_map[star]] = 1.0
        features.extend(star_features)
        
        # 四化特征
        sihua_features = [0] * 4
        sihua_types = ["化禄", "化权", "化科", "化忌"]
        for i, hua in enumerate(sihua_types):
            if hua in ming_gong.sihua:
                sihua_features[i] = 1.0
        features.extend(sihua_features)
        
        # 填充到128维
        features = np.array(features)
        features = np.pad(features, (0, 128 - len(features)), mode='constant')
        
        # 创建序列
        seq_features = np.tile(features, (10, 1))
        
        return seq_features.astype(np.float32)
    
    def _extract_western_features(self, natal_chart) -> np.ndarray:
        """提取西方占星特征"""
        features = []
        
        # 行星位置特征
        planets = ["Sun", "Moon", "Mercury", "Venus", "Mars", 
                  "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
        
        for planet in planets:
            if planet in natal_chart.planets:
                body = natal_chart.planets[planet]
                features.extend([
                    body.longitude / 360.0,
                    body.latitude / 90.0,
                    body.speed / 10.0,
                    1.0 if body.is_retrograde else 0.0
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # 宫位特征
        if natal_chart.houses:
            for i in range(12):
                if i < len(natal_chart.houses):
                    features.append(natal_chart.houses[i].cusp / 360.0)
                else:
                    features.append(0)
        else:
            features.extend([0] * 12)
        
        # 相位特征
        aspect_types = ["合相", "六分相", "刑相", "拱相", "冲相"]
        aspect_counts = {a: 0 for a in aspect_types}
        
        for aspect in natal_chart.aspects[:20]:  # 最多考虑20个相位
            if aspect.aspect_type in aspect_counts:
                aspect_counts[aspect.aspect_type] += 1
        
        for aspect_type in aspect_types:
            features.append(aspect_counts[aspect_type] / 10.0)
        
        # ASC/MC
        features.append(natal_chart.points.get("ASC", 0) / 360.0)
        features.append(natal_chart.points.get("MC", 0) / 360.0)
        
        # 填充到128维
        features = np.array(features)
        features = np.pad(features, (0, 128 - len(features)), mode='constant')
        
        # 创建序列
        seq_features = np.tile(features, (10, 1))
        
        return seq_features.astype(np.float32)
    
    def _extract_quantum_features(self, birth_data: Dict[str, Any]) -> np.ndarray:
        """提取量子特征（模拟）"""
        # 创建可能的事件
        possible_events = self._generate_possible_events()
        
        # 天文/命理影响分数
        astro_features = {
            'career_score': np.random.random(),
            'relationship_score': np.random.random(),
            'health_score': np.random.random(),
            'wealth_score': np.random.random(),
            'uncertainty': 0.1
        }
        
        # 创建量子叠加态
        quantum_state = simulate_life_superposition(
            birth_data, 
            astro_features, 
            possible_events
        )
        
        # 提取量子特征
        features = []
        
        # 状态向量特征
        sv = quantum_state.state_vector[:16]  # 取前16个振幅
        for amp in sv:
            features.extend([
                amp.real,
                amp.imag,
                abs(amp),
                np.angle(amp)
            ])
        
        # 量子属性
        features.extend([
            quantum_state.entanglement_entropy,
            quantum_state.coherence,
            quantum_state.fidelity,
            quantum_state.circuit_depth / 100.0,
            quantum_state.gate_count / 1000.0
        ])
        
        # 概率分布特征
        prob_dist = quantum_state.get_probability_distribution()
        for i in range(8):
            event_id = f"E{i:04d}"
            features.append(prob_dist.get(event_id, 0))
        
        # 填充到256维
        features = np.array(features)
        features = np.pad(features, (0, 256 - len(features)), mode='constant')
        
        # 创建序列
        seq_features = np.tile(features, (10, 1))
        
        return seq_features.astype(np.float32)
    
    def _get_season(self, month: int) -> List[float]:
        """获取季节编码"""
        if month in [3, 4, 5]:  # 春
            return [1.0, 0.0, 0.0, 0.0]
        elif month in [6, 7, 8]:  # 夏
            return [0.0, 1.0, 0.0, 0.0]
        elif month in [9, 10, 11]:  # 秋
            return [0.0, 0.0, 1.0, 0.0]
        else:  # 冬
            return [0.0, 0.0, 0.0, 1.0]
    
    def _estimate_lunar_phase(self, dt: datetime) -> float:
        """估算月相（简化）"""
        # 使用Metonic周期近似
        year = dt.year
        month = dt.month
        day = dt.day
        
        if month < 3:
            year -= 1
            month += 12
        
        a = year // 100
        b = a // 4
        c = 2 - a + b
        e = int(365.25 * (year + 4716))
        f = int(30.6001 * (month + 1))
        jd = c + day + e + f - 1524.5
        
        # 新月周期约29.53天
        phase = ((jd - 2451549.5) % 29.53) / 29.53
        
        return phase
    
    def _estimate_solar_term(self, dt: datetime) -> float:
        """估算节气（简化）"""
        # 24节气，每个约15天
        day_of_year = dt.timetuple().tm_yday
        return (day_of_year % 15.2) / 15.2
    
    def _positional_encoding(self, pos: int, dim: int) -> np.ndarray:
        """位置编码"""
        encoding = np.zeros(dim)
        for i in range(0, dim, 2):
            encoding[i] = np.sin(pos / (10000 ** (i / dim)))
            if i + 1 < dim:
                encoding[i + 1] = np.cos(pos / (10000 ** (i / dim)))
        return encoding
    
    def _generate_possible_events(self) -> List[Dict[str, Any]]:
        """生成可能的事件（示例）"""
        event_types = ["career", "relationship", "health", "wealth"]
        events = []
        
        for i in range(8):
            event_type = event_types[i % 4]
            events.append({
                'type': event_type,
                'description': f"事件类型{event_type}_{i}",
                'timestamp': datetime.now().isoformat(),
                'influences': {}
            })
        
        return events


class Predictor:
    """预测器"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.settings = get_settings()
        self.device = torch.device(self.settings.model_device)
        
        # 加载模型
        self.model = LifeTrajectoryModel()
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning("未指定模型路径，使用随机初始化的模型")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 事件类型映射
        self.event_types = self._load_event_types()
    
    def load_model(self, model_path: Path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已加载: {model_path}")
    
    def predict(self, birth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测生命轨迹
        
        Args:
            birth_data: 出生信息
            
        Returns:
            预测结果
        """
        # 提取特征
        features = self.feature_extractor.extract_features(birth_data)
        
        # 预测
        with torch.no_grad():
            trajectory = self.model.predict_trajectory(features, num_events=10)
        
        # 解析结果
        predictions = self._parse_predictions(trajectory, birth_data)
        
        return predictions
    
    def _parse_predictions(self, trajectory: Dict[str, Any], 
                          birth_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析预测结果"""
        birth_dt = datetime.fromisoformat(birth_data['birth_datetime'])
        
        events = []
        for event_data in trajectory['trajectory']:
            event_id = event_data['event_id']
            event_type = self.event_types.get(event_id, {})
            
            # 计算事件时间
            days_offset = event_data['expected_day']
            event_time = birth_dt.replace(tzinfo=None)
            event_time = event_time.replace(year=event_time.year + int(days_offset / 365))
            event_time = event_time.replace(day=1)  # 简化处理
            
            event = {
                'id': f"EVT_{event_id:04d}",
                'type': event_type.get('category', 'unknown'),
                'description': event_type.get('description', '未知事件'),
                'probability': event_data['probability'],
                'confidence': event_data['confidence'],
                'uncertainty': event_data['uncertainty'],
                'expected_date': event_time.isoformat(),
                'time_distribution': event_data['time_distribution'][:30],  # 前30天
                'impact': event_type.get('impact', 'medium'),
                'suggestions': event_type.get('suggestions', [])
            }
            
            events.append(event)
        
        # 生成总体预测
        prediction = {
            'user_id': birth_data.get('user_id', 'anonymous'),
            'birth_info': birth_data,
            'prediction_time': datetime.now().isoformat(),
            'life_trajectory': {
                'events': events,
                'timeline_summary': self._generate_timeline_summary(events),
                'key_periods': self._identify_key_periods(events),
                'overall_trend': self._analyze_overall_trend(events)
            },
            'feature_importance': trajectory['feature_importance'],
            'confidence_metrics': {
                'overall_confidence': np.mean([e['confidence'] for e in events]),
                'prediction_uncertainty': np.mean([e['uncertainty'] for e in events]),
                'model_version': self.settings.model_name
            }
        }
        
        return prediction
    
    def _load_event_types(self) -> Dict[int, Dict[str, Any]]:
        """加载事件类型定义"""
        # 实际应该从配置文件加载
        event_types = {}
        
        categories = ['career', 'relationship', 'health', 'wealth', 'education', 'travel']
        impacts = ['major', 'moderate', 'minor']
        
        for i in range(256):
            category = categories[i % len(categories)]
            impact = impacts[i % len(impacts)]
            
            event_types[i] = {
                'category': category,
                'description': f"{category.title()} Event {i}",
                'impact': impact,
                'suggestions': [
                    f"建议1: 关注{category}相关机会",
                    f"建议2: 提前做好准备",
                    f"建议3: 保持积极心态"
                ]
            }
        
        return event_types
    
    def _generate_timeline_summary(self, events: List[Dict[str, Any]]) -> str:
        """生成时间线摘要"""
        if not events:
            return "暂无重要事件预测"
        
        summary_parts = []
        for event in events[:3]:  # 前3个最重要的事件
            date = event['expected_date'][:10]
            desc = event['description']
            prob = event['probability']
            summary_parts.append(f"{date}: {desc} (概率{prob:.1%})")
        
        return "; ".join(summary_parts)
    
    def _identify_key_periods(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别关键时期"""
        # 按类型分组
        periods = []
        
        # 事业高峰期
        career_events = [e for e in events if e['type'] == 'career' and e['probability'] > 0.3]
        if career_events:
            periods.append({
                'type': 'career_peak',
                'description': '事业发展关键期',
                'start_date': min(e['expected_date'] for e in career_events),
                'end_date': max(e['expected_date'] for e in career_events),
                'importance': 'high'
            })
        
        # 感情机遇期
        relationship_events = [e for e in events if e['type'] == 'relationship' and e['probability'] > 0.3]
        if relationship_events:
            periods.append({
                'type': 'relationship_opportunity',
                'description': '感情机遇期',
                'start_date': min(e['expected_date'] for e in relationship_events),
                'end_date': max(e['expected_date'] for e in relationship_events),
                'importance': 'high'
            })
        
        return periods
    
    def _analyze_overall_trend(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析总体趋势"""
        if not events:
            return {'trend': 'stable', 'description': '运势平稳'}
        
        # 计算各类型事件的平均概率
        type_probs = {}
        for event in events:
            event_type = event['type']
            if event_type not in type_probs:
                type_probs[event_type] = []
            type_probs[event_type].append(event['probability'])
        
        # 找出最强势的领域
        strongest_type = max(type_probs.items(), 
                           key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
        
        trend_map = {
            'career': {'trend': 'ascending', 'description': '事业运上升'},
            'relationship': {'trend': 'harmonious', 'description': '感情和谐'},
            'health': {'trend': 'stable', 'description': '健康平稳'},
            'wealth': {'trend': 'prosperous', 'description': '财运亨通'}
        }
        
        return trend_map.get(strongest_type, {'trend': 'balanced', 'description': '整体平衡'})


def predict_life_trajectory(birth_data: Dict[str, Any], 
                          model_path: Optional[Path] = None) -> Dict[str, Any]:
    """预测生命轨迹的便捷函数"""
    predictor = Predictor(model_path)
    return predictor.predict(birth_data)


if __name__ == "__main__":
    # 测试推理
    
    birth_data = {
        'user_id': 'test_001',
        'birth_datetime': '1990-01-01T12:00:00+08:00',
        'longitude': 116.4074,
        'latitude': 39.9042,
        'gender': '男',
        'name': '测试用户'
    }
    
    # 进行预测
    prediction = predict_life_trajectory(birth_data)
    
    print("=== 生命轨迹预测 ===")
    print(f"用户: {prediction['user_id']}")
    print(f"预测时间: {prediction['prediction_time']}")
    
    print("\n主要事件:")
    for i, event in enumerate(prediction['life_trajectory']['events'][:5]):
        print(f"\n事件{i+1}:")
        print(f"  类型: {event['type']}")
        print(f"  描述: {event['description']}")
        print(f"  预期时间: {event['expected_date'][:10]}")
        print(f"  概率: {event['probability']:.1%}")
        print(f"  置信度: {event['confidence']:.2f}")
    
    print(f"\n时间线摘要: {prediction['life_trajectory']['timeline_summary']}")
    
    print("\n特征重要性:")
    for feat, imp in prediction['feature_importance'].items():
        print(f"  {feat}: {imp:.2%}")
    
    print(f"\n整体置信度: {prediction['confidence_metrics']['overall_confidence']:.2f}")