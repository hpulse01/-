#!/usr/bin/env python3
"""
H-Pulse 系统演示脚本
"""

import sys
from pathlib import Path

# 添加项目到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from h_pulse.utils.time_astro import utc_to_tt, true_solar_time
from h_pulse.data_collection.eastern import calculate_bazi
from h_pulse.data_collection.western import calculate_natal_chart
from h_pulse.quantum_engine.simulator import simulate_life_superposition
from h_pulse.utils.crypto_anchor import generate_quantum_fingerprint


def main():
    """运行演示"""
    print("=" * 60)
    print("H-Pulse Quantum Prediction System - 演示")
    print("精准 · 独特 · 不可逆")
    print("=" * 60)
    
    # 演示数据
    birth_dt = datetime(1990, 1, 1, 12, 0, 0)
    longitude = 116.4074  # 北京
    latitude = 39.9042
    
    print(f"\n演示出生数据:")
    print(f"  出生时间: {birth_dt}")
    print(f"  出生地点: 北京 ({longitude}°E, {latitude}°N)")
    
    # 1. 时间转换演示
    print("\n1. 时间天文计算:")
    tt_time = utc_to_tt(birth_dt)
    tst = true_solar_time(birth_dt, longitude)
    print(f"  UTC时间: {birth_dt}")
    print(f"  TT时间:  {tt_time}")
    print(f"  真太阳时: {tst}")
    
    # 2. 八字计算演示
    print("\n2. 四柱八字计算:")
    try:
        bazi = calculate_bazi(birth_dt, longitude, latitude, "男")
        print(f"  年柱: {bazi.year.ganzhi}")
        print(f"  月柱: {bazi.month.ganzhi}")
        print(f"  日柱: {bazi.day.ganzhi}")
        print(f"  时柱: {bazi.hour.ganzhi}")
    except Exception as e:
        print(f"  八字计算出错: {e}")
    
    # 3. 西方星盘演示
    print("\n3. 西方星盘计算:")
    try:
        natal = calculate_natal_chart(birth_dt, longitude, latitude)
        print(f"  上升点: {natal.points.get('ASC', 0):.2f}°")
        print(f"  中天: {natal.points.get('MC', 0):.2f}°")
        if natal.planets:
            sun = natal.planets[0]
            print(f"  太阳: {sun.sign} {sun.degree:.2f}°")
    except Exception as e:
        print(f"  星盘计算出错: {e}")
    
    # 4. 量子指纹演示
    print("\n4. 量子指纹生成:")
    birth_data = {
        'name': '演示用户',
        'birth_datetime': birth_dt.isoformat(),
        'longitude': longitude,
        'latitude': latitude
    }
    
    spacetime = {
        'longitude': longitude,
        'latitude': latitude,
        'timezone': 'Asia/Shanghai'
    }
    
    fingerprint = generate_quantum_fingerprint(birth_data, spacetime)
    print(f"  量子指纹: {fingerprint[:32]}...")
    
    # 5. 量子叠加态演示
    print("\n5. 量子生命叠加态:")
    try:
        possible_events = [
            {"type": "career", "description": "职业机遇", "timestamp": "2025-06-15T10:00:00"},
            {"type": "relationship", "description": "感情发展", "timestamp": "2025-09-20T14:00:00"}
        ]
        
        astro_features = {
            'demo': True,
            'bazi': {'year': '己巳'} if 'bazi' in locals() else {}
        }
        
        quantum_state = simulate_life_superposition(
            birth_data, astro_features, possible_events
        )
        
        print(f"  叠加态维度: {len(quantum_state.events)}")
        print(f"  纠缠熵: {quantum_state.entanglement_entropy:.4f}")
        print(f"  相干性: {quantum_state.coherence:.4f}")
        
        # 显示事件概率
        probs = quantum_state.get_measurement_probabilities()
        for i, (event, prob) in enumerate(zip(quantum_state.events, probs)):
            print(f"  事件{i+1} ({event.event_type}): {prob:.2%}")
            
    except Exception as e:
        print(f"  量子模拟出错: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n完整功能请运行: python main.py demo")
    print("API服务请运行: python main.py api")
    print("=" * 60)


if __name__ == "__main__":
    main()