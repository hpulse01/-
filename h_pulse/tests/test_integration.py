"""
集成测试 - 测试各模块协同工作
"""

import pytest
from datetime import datetime
import json
import tempfile
from pathlib import Path

from h_pulse.prediction_ai import predict_life_trajectory
from h_pulse.output_generation.report import generate_pdf_report
from h_pulse.utils.settings import get_settings
from h_pulse.utils.crypto_anchor import generate_quantum_fingerprint, sign_data


class TestEndToEndPrediction:
    """端到端预测测试"""
    
    def test_complete_prediction_flow(self):
        """测试完整的预测流程"""
        # 准备出生数据
        birth_data = {
            'name': '测试用户',
            'gender': '男',
            'birth_datetime': '1990-05-15T14:30:00+08:00',
            'longitude': 121.4737,  # 上海
            'latitude': 31.2304,
            'timezone': 'Asia/Shanghai',
            'user_id': 'test_user_001'
        }
        
        # 执行预测
        prediction_result = predict_life_trajectory(birth_data)
        
        # 验证预测结果结构
        assert 'user_id' in prediction_result
        assert 'prediction_time' in prediction_result
        assert 'quantum_fingerprint' in prediction_result
        assert 'life_trajectory' in prediction_result
        assert 'astrology_data' in prediction_result
        assert 'quantum_analysis' in prediction_result
        assert 'feature_importance' in prediction_result
        assert 'confidence_metrics' in prediction_result
        
        # 验证生命轨迹
        trajectory = prediction_result['life_trajectory']
        assert 'events' in trajectory
        assert 'timeline_summary' in trajectory
        assert 'key_periods' in trajectory
        assert 'overall_trend' in trajectory
        
        # 验证事件
        events = trajectory['events']
        assert len(events) > 0
        for event in events:
            assert 'id' in event
            assert 'type' in event
            assert 'description' in event
            assert 'probability' in event
            assert 'confidence' in event
            assert 0 <= event['probability'] <= 1
            assert 0 <= event['confidence'] <= 1
        
        # 验证置信度指标
        confidence = prediction_result['confidence_metrics']
        assert 'overall_confidence' in confidence
        assert 'prediction_accuracy' in confidence
        assert 'model_certainty' in confidence
        assert 0 <= confidence['overall_confidence'] <= 1
    
    def test_astrology_integration(self):
        """测试占星数据整合"""
        birth_data = {
            'name': '占星测试',
            'gender': '女',
            'birth_datetime': '1995-08-20T09:15:00+08:00',
            'longitude': 116.4074,  # 北京
            'latitude': 39.9042,
            'timezone': 'Asia/Shanghai',
            'user_id': 'astro_test_001'
        }
        
        prediction_result = predict_life_trajectory(birth_data)
        astro_data = prediction_result['astrology_data']
        
        # 验证东方占星数据
        assert 'bazi' in astro_data
        bazi = astro_data['bazi']
        assert 'year_pillar' in bazi
        assert 'month_pillar' in bazi
        assert 'day_pillar' in bazi
        assert 'hour_pillar' in bazi
        assert 'dayun' in bazi
        
        assert 'ziwei' in astro_data
        ziwei = astro_data['ziwei']
        assert 'ming_gong' in ziwei
        assert 'palaces' in ziwei
        assert 'si_hua' in ziwei
        
        # 验证西方占星数据
        assert 'natal' in astro_data
        natal = astro_data['natal']
        assert 'planets' in natal
        assert 'houses' in natal
        assert 'aspects' in natal
        assert 'asc' in natal
        assert 'mc' in natal
    
    def test_quantum_analysis_integration(self):
        """测试量子分析整合"""
        birth_data = {
            'name': '量子测试',
            'gender': '男',
            'birth_datetime': '1988-12-25T22:00:00+08:00',
            'longitude': 113.2644,  # 广州
            'latitude': 23.1291,
            'timezone': 'Asia/Shanghai',
            'user_id': 'quantum_test_001'
        }
        
        prediction_result = predict_life_trajectory(birth_data)
        quantum_analysis = prediction_result['quantum_analysis']
        
        # 验证量子分析结果
        assert 'quantum_state' in quantum_analysis
        assert 'entanglement_entropy' in quantum_analysis
        assert 'coherence' in quantum_analysis
        assert 'superposition_count' in quantum_analysis
        assert 'measurement_basis' in quantum_analysis
        
        # 验证量子属性范围
        assert quantum_analysis['entanglement_entropy'] >= 0
        assert 0 <= quantum_analysis['coherence'] <= 1
        assert quantum_analysis['superposition_count'] > 0
    
    def test_report_generation(self):
        """测试报告生成"""
        birth_data = {
            'name': '报告测试',
            'gender': '女',
            'birth_datetime': '1992-03-10T16:45:00+08:00',
            'longitude': 104.0650,  # 成都
            'latitude': 30.6595,
            'timezone': 'Asia/Shanghai',
            'user_id': 'report_test_001'
        }
        
        # 执行预测
        prediction_result = predict_life_trajectory(birth_data)
        
        # 生成报告
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.pdf"
            generated_path = generate_pdf_report(
                prediction_result,
                str(report_path)
            )
            
            # 验证报告生成
            assert generated_path.exists()
            assert generated_path.stat().st_size > 0  # 文件不为空
    
    def test_crypto_signature(self):
        """测试加密签名"""
        birth_data = {
            'name': '加密测试',
            'gender': '男',
            'birth_datetime': '1985-11-11T11:11:00+08:00',
            'longitude': 120.1551,  # 杭州
            'latitude': 30.2741,
            'timezone': 'Asia/Shanghai',
            'user_id': 'crypto_test_001'
        }
        
        # 执行预测
        prediction_result = predict_life_trajectory(birth_data)
        
        # 验证量子指纹
        assert 'quantum_fingerprint' in prediction_result
        fingerprint = prediction_result['quantum_fingerprint']
        assert len(fingerprint) == 64  # SHA3-256 hex string
        
        # 验证签名
        if 'signature' in prediction_result:
            signature = prediction_result['signature']
            assert 'data_hash' in signature
            assert 'signature' in signature
            assert 'public_key' in signature
            assert 'timestamp' in signature


class TestPerformanceAndStability:
    """性能和稳定性测试"""
    
    def test_prediction_performance(self):
        """测试预测性能"""
        import time
        
        birth_data = {
            'name': '性能测试',
            'gender': '男',
            'birth_datetime': '2000-01-01T00:00:00+08:00',
            'longitude': 116.4074,
            'latitude': 39.9042,
            'timezone': 'Asia/Shanghai',
            'user_id': 'perf_test_001'
        }
        
        start_time = time.time()
        prediction_result = predict_life_trajectory(birth_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 验证执行时间在合理范围内（应该在10秒内完成）
        assert execution_time < 10.0
        
        # 验证结果完整性
        assert prediction_result is not None
        assert len(prediction_result['life_trajectory']['events']) > 0
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试极地出生
        arctic_birth = {
            'name': '北极测试',
            'gender': '男',
            'birth_datetime': '1990-06-21T00:00:00+00:00',  # 夏至
            'longitude': 0.0,
            'latitude': 89.9,  # 接近北极
            'timezone': 'UTC',
            'user_id': 'arctic_test'
        }
        
        result1 = predict_life_trajectory(arctic_birth)
        assert result1 is not None
        
        # 测试赤道出生
        equator_birth = {
            'name': '赤道测试',
            'gender': '女',
            'birth_datetime': '1990-03-21T12:00:00+00:00',  # 春分
            'longitude': 0.0,
            'latitude': 0.0,  # 赤道
            'timezone': 'UTC',
            'user_id': 'equator_test'
        }
        
        result2 = predict_life_trajectory(equator_birth)
        assert result2 is not None
        
        # 测试日界线附近
        dateline_birth = {
            'name': '日界线测试',
            'gender': '男',
            'birth_datetime': '1990-12-31T23:59:59+12:00',
            'longitude': 179.9,
            'latitude': 0.0,
            'timezone': 'Pacific/Auckland',
            'user_id': 'dateline_test'
        }
        
        result3 = predict_life_trajectory(dateline_birth)
        assert result3 is not None
    
    def test_consistency(self):
        """测试结果一致性"""
        birth_data = {
            'name': '一致性测试',
            'gender': '男',
            'birth_datetime': '1995-05-05T05:05:05+08:00',
            'longitude': 121.4737,
            'latitude': 31.2304,
            'timezone': 'Asia/Shanghai',
            'user_id': 'consistency_test'
        }
        
        # 多次运行预测
        results = []
        for i in range(3):
            result = predict_life_trajectory(birth_data)
            results.append(result)
        
        # 验证量子指纹一致性
        fingerprints = [r['quantum_fingerprint'] for r in results]
        assert len(set(fingerprints)) == 1  # 所有指纹应该相同
        
        # 验证占星数据一致性
        for i in range(1, 3):
            assert results[0]['astrology_data']['bazi'] == results[i]['astrology_data']['bazi']
            assert results[0]['astrology_data']['natal']['asc'] == results[i]['astrology_data']['natal']['asc']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])