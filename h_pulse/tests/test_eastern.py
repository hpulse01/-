"""
测试东方命理计算模块
"""

import pytest
from datetime import datetime

from h_pulse.data_collection.eastern import (
    BaZiCalculator, ZiWeiCalculator,
    calculate_bazi, calculate_ziwei
)


class TestBaZiCalculator:
    """测试四柱八字计算器"""
    
    def test_basic_calculation(self):
        """测试基本八字计算"""
        calc = BaZiCalculator()
        birth_dt = datetime(1990, 1, 1, 12, 0, 0)
        longitude = 116.4074  # 北京
        latitude = 39.9042
        
        chart = calc.calculate(birth_dt, longitude, latitude, "男")
        
        # 验证结果结构
        assert chart.year is not None
        assert chart.month is not None
        assert chart.day is not None
        assert chart.hour is not None
        
        # 验证干支格式
        assert len(chart.year.ganzhi) == 2
        assert len(chart.month.ganzhi) == 2
        assert len(chart.day.ganzhi) == 2
        assert len(chart.hour.ganzhi) == 2
    
    def test_gan_zhi_components(self):
        """测试天干地支分解"""
        calc = BaZiCalculator()
        birth_dt = datetime(2000, 6, 15, 10, 30, 0)
        chart = calc.calculate(birth_dt, 0.0, 0.0)
        
        # 验证天干地支
        assert chart.year.gan in "甲乙丙丁戊己庚辛壬癸"
        assert chart.year.zhi in "子丑寅卯辰巳午未申酉戌亥"
        assert chart.month.gan in "甲乙丙丁戊己庚辛壬癸"
        assert chart.month.zhi in "子丑寅卯辰巳午未申酉戌亥"
    
    def test_wuxing_calculation(self):
        """测试五行计算"""
        calc = BaZiCalculator()
        birth_dt = datetime(1985, 3, 20, 15, 45, 0)
        chart = calc.calculate(birth_dt, 121.4737, 31.2304)  # 上海
        
        # 验证五行
        wuxing_elements = ["木", "火", "土", "金", "水"]
        assert chart.year.gan_wuxing in wuxing_elements
        assert chart.year.zhi_wuxing in wuxing_elements
        assert chart.month.gan_wuxing in wuxing_elements
        assert chart.day.gan_wuxing in wuxing_elements
    
    def test_dayun_calculation(self):
        """测试大运计算"""
        calc = BaZiCalculator()
        birth_dt = datetime(1995, 7, 10, 8, 0, 0)
        chart = calc.calculate(birth_dt, 114.0579, 22.5431, "女")  # 深圳
        
        # 验证大运
        assert hasattr(chart, 'dayun')
        assert len(chart.dayun) > 0
        
        # 验证第一个大运
        first_dayun = chart.dayun[0]
        assert 'start_age' in first_dayun
        assert 'ganzhi' in first_dayun
        assert 'start_year' in first_dayun
        assert first_dayun['start_age'] >= 0
    
    def test_gender_difference(self):
        """测试性别对大运的影响"""
        calc = BaZiCalculator()
        birth_dt = datetime(1992, 4, 15, 18, 30, 0)
        
        chart_male = calc.calculate(birth_dt, 113.2644, 23.1291, "男")  # 广州
        chart_female = calc.calculate(birth_dt, 113.2644, 23.1291, "女")
        
        # 相同时间不同性别，四柱应该相同
        assert chart_male.year.ganzhi == chart_female.year.ganzhi
        assert chart_male.month.ganzhi == chart_female.month.ganzhi
        assert chart_male.day.ganzhi == chart_female.day.ganzhi
        assert chart_male.hour.ganzhi == chart_female.hour.ganzhi
        
        # 但大运可能不同（顺逆）
        # 这里简化验证，只检查都有大运
        assert len(chart_male.dayun) > 0
        assert len(chart_female.dayun) > 0


class TestZiWeiCalculator:
    """测试紫微斗数计算器"""
    
    def test_basic_calculation(self):
        """测试基本紫微斗数计算"""
        calc = ZiWeiCalculator()
        birth_dt = datetime(1988, 8, 8, 14, 0, 0)
        longitude = 106.5516  # 重庆
        latitude = 29.5630
        
        chart = calc.calculate(birth_dt, longitude, latitude, "男")
        
        # 验证十二宫
        assert len(chart.palaces) == 12
        palace_names = [p.name for p in chart.palaces]
        assert "命宫" in palace_names
        assert "财帛宫" in palace_names
        assert "官禄宫" in palace_names
    
    def test_ming_gong_calculation(self):
        """测试命宫计算"""
        calc = ZiWeiCalculator()
        birth_dt = datetime(1993, 11, 25, 6, 30, 0)
        chart = calc.calculate(birth_dt, 120.1551, 30.2741)  # 杭州
        
        # 验证命宫
        assert chart.ming_gong is not None
        assert chart.ming_gong.name == "命宫"
        assert 1 <= chart.ming_gong.position <= 12
        assert len(chart.ming_gong.main_stars) > 0
    
    def test_main_stars_placement(self):
        """测试主星安置"""
        calc = ZiWeiCalculator()
        birth_dt = datetime(1997, 2, 14, 20, 15, 0)
        chart = calc.calculate(birth_dt, 104.0650, 30.6595)  # 成都
        
        # 验证主星
        all_main_stars = []
        for palace in chart.palaces:
            all_main_stars.extend(palace.main_stars)
        
        # 应该有一些主星
        assert len(all_main_stars) > 0
        
        # 验证主星名称
        star_names = [s.name for s in all_main_stars]
        expected_stars = ["紫微", "天机", "太阳", "武曲", "天同", 
                         "廉贞", "天府", "太阴", "贪狼", "巨门",
                         "天相", "天梁", "七杀", "破军"]
        
        # 至少应该有一些预期的主星
        found_stars = [s for s in star_names if s in expected_stars]
        assert len(found_stars) > 0
    
    def test_four_transformations(self):
        """测试四化"""
        calc = ZiWeiCalculator()
        birth_dt = datetime(1986, 5, 5, 16, 45, 0)
        chart = calc.calculate(birth_dt, 117.2000, 39.1300)  # 天津
        
        # 验证四化
        assert hasattr(chart, 'si_hua')
        assert '化禄' in chart.si_hua
        assert '化权' in chart.si_hua
        assert '化科' in chart.si_hua
        assert '化忌' in chart.si_hua
    
    def test_decade_luck(self):
        """测试大限"""
        calc = ZiWeiCalculator()
        birth_dt = datetime(1991, 9, 9, 9, 9, 0)
        chart = calc.calculate(birth_dt, 119.2965, 26.0745, "女")  # 福州
        
        # 验证大限
        assert hasattr(chart, 'decade_luck')
        assert len(chart.decade_luck) > 0
        
        # 验证大限结构
        first_decade = chart.decade_luck[0]
        assert 'palace' in first_decade
        assert 'start_age' in first_decade
        assert 'end_age' in first_decade
        assert first_decade['start_age'] < first_decade['end_age']


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_calculate_bazi_function(self):
        """测试计算八字便捷函数"""
        birth_dt = datetime(2000, 1, 1, 0, 0, 0)
        chart = calculate_bazi(birth_dt, 121.4737, 31.2304)  # 上海
        
        assert chart is not None
        assert chart.year.ganzhi is not None
        assert chart.get_rizhu_gan() in "甲乙丙丁戊己庚辛壬癸"
    
    def test_calculate_ziwei_function(self):
        """测试计算紫微便捷函数"""
        birth_dt = datetime(1999, 12, 31, 23, 59, 0)
        chart = calculate_ziwei(birth_dt, 113.5439, 22.3048, "男")  # 澳门
        
        assert chart is not None
        assert len(chart.palaces) == 12
        assert chart.ming_gong is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])