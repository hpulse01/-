"""
测试时间天文计算模块
"""

import pytest
from datetime import datetime
import numpy as np

from h_pulse.utils.time_astro import (
    TimeConverter, AstronomicalCalculator,
    utc_to_tt, utc_to_tdb, true_solar_time
)


class TestTimeConverter:
    """测试时间转换器"""
    
    def test_utc_to_tt(self):
        """测试UTC到TT转换"""
        converter = TimeConverter()
        utc_dt = datetime(2025, 1, 1, 0, 0, 0)
        tt_time = converter.utc_to_tt(utc_dt)
        
        # TT应该比UTC快约69秒
        delta = (tt_time - utc_dt).total_seconds()
        assert 68 <= delta <= 70
    
    def test_utc_to_tdb(self):
        """测试UTC到TDB转换"""
        converter = TimeConverter()
        utc_dt = datetime(2025, 1, 1, 0, 0, 0)
        tdb_time = converter.utc_to_tdb(utc_dt)
        
        # TDB应该与TT接近，差异小于2毫秒
        tt_time = converter.utc_to_tt(utc_dt)
        delta_ms = abs((tdb_time - tt_time).total_seconds() * 1000)
        assert delta_ms < 2
    
    def test_delta_t_calculation(self):
        """测试ΔT计算"""
        converter = TimeConverter()
        
        # 2020年的ΔT应该约为69秒
        dt_2020 = converter.get_delta_t(2020.0)
        assert 68 <= dt_2020 <= 70
        
        # 2025年的ΔT应该略大
        dt_2025 = converter.get_delta_t(2025.0)
        assert dt_2025 > dt_2020


class TestAstronomicalCalculator:
    """测试天文计算器"""
    
    def test_precession_matrix(self):
        """测试岁差矩阵"""
        calc = AstronomicalCalculator()
        
        # J2000.0时刻的岁差矩阵应该是单位矩阵
        matrix = calc.precession_matrix(2451545.0)
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-10)
        
        # 测试矩阵正交性
        jd = 2460000.0
        matrix = calc.precession_matrix(jd)
        product = matrix @ matrix.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)
    
    def test_nutation_angles(self):
        """测试章动角度"""
        calc = AstronomicalCalculator()
        
        # 测试已知时刻的章动
        jd = 2451545.0  # J2000.0
        dpsi, deps = calc.nutation_angles(jd)
        
        # 章动角度应该在合理范围内（通常小于20角秒）
        assert abs(dpsi) < 20 / 3600  # 度
        assert abs(deps) < 20 / 3600  # 度
    
    def test_aberration_correction(self):
        """测试光行差修正"""
        calc = AstronomicalCalculator()
        
        # 测试春分点附近
        jd = 2451623.0  # 约2000年春分
        ra = 0.0  # 赤经0度
        dec = 0.0  # 赤纬0度
        
        dra, ddec = calc.aberration_correction(jd, ra, dec)
        
        # 光行差修正应该在合理范围内（通常小于20角秒）
        assert abs(dra) < 20 / 3600
        assert abs(ddec) < 20 / 3600
    
    def test_true_solar_time(self):
        """测试真太阳时"""
        calc = AstronomicalCalculator()
        
        # 测试北京时间正午
        utc_dt = datetime(2025, 1, 1, 4, 0, 0)  # UTC 04:00 = 北京时间12:00
        longitude = 116.4074  # 北京经度
        
        # 不考虑均时差
        tst_mean = calc.true_solar_time(utc_dt, longitude, equation_of_time=False)
        expected_offset = longitude * 4  # 每度4分钟
        actual_offset = (tst_mean - utc_dt).total_seconds() / 60
        assert abs(actual_offset - expected_offset) < 0.1
        
        # 考虑均时差
        tst_true = calc.true_solar_time(utc_dt, longitude, equation_of_time=True)
        # 真太阳时与平太阳时的差异应该在±16分钟内
        delta_minutes = abs((tst_true - tst_mean).total_seconds() / 60)
        assert delta_minutes < 16


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_utc_to_tt_function(self):
        """测试UTC到TT便捷函数"""
        utc_dt = datetime(2025, 1, 1, 0, 0, 0)
        tt_time = utc_to_tt(utc_dt)
        
        delta = (tt_time - utc_dt).total_seconds()
        assert 68 <= delta <= 70
    
    def test_utc_to_tdb_function(self):
        """测试UTC到TDB便捷函数"""
        utc_dt = datetime(2025, 1, 1, 0, 0, 0)
        tdb_time = utc_to_tdb(utc_dt)
        
        # 应该与TT接近
        tt_time = utc_to_tt(utc_dt)
        delta_ms = abs((tdb_time - tt_time).total_seconds() * 1000)
        assert delta_ms < 2
    
    def test_true_solar_time_function(self):
        """测试真太阳时便捷函数"""
        utc_dt = datetime(2025, 1, 1, 12, 0, 0)
        longitude = 0.0  # 格林威治
        
        tst = true_solar_time(utc_dt, longitude)
        
        # 在格林威治，真太阳时应该接近UTC（差异主要是均时差）
        delta_minutes = abs((tst - utc_dt).total_seconds() / 60)
        assert delta_minutes < 16  # 均时差最大约16分钟


if __name__ == "__main__":
    pytest.main([__file__, "-v"])