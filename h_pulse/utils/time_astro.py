"""
时间与天文计算模块
处理UTC/TT/TDB时间转换、ΔT修正、岁差/章动/光行差计算、真太阳时等

References:
- IERS Conventions 2010
- IAU SOFA (Standards of Fundamental Astronomy)
- Astronomical Algorithms by Jean Meeus
"""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, Union
import pytz
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy import units as u
from skyfield.api import load, Topos
from skyfield.timelib import Time as SkyfieldTime
import structlog

logger = structlog.get_logger()


class TimeConverter:
    """高精度时间转换器，处理UTC/TT/TDB/TAI等时间系统"""
    
    def __init__(self):
        self.ts = load.timescale()
        # 加载ΔT数据
        self._load_delta_t_data()
        
    def _load_delta_t_data(self):
        """加载历史和预测的ΔT数据"""
        # ΔT = TT - UT1，这里使用IERS公布的值
        # 实际应用中应从finals2000A.all等文件加载
        self.delta_t_table = {
            2020: 69.184 + 37.0,  # TAI-UTC=37s, TT-TAI=32.184s
            2021: 69.184 + 37.0,
            2022: 69.184 + 37.0,
            2023: 69.184 + 37.0,
            2024: 69.184 + 37.0,
            2025: 69.184 + 37.0,  # 预测值
        }
    
    def utc_to_tt(self, utc_dt: datetime) -> float:
        """
        UTC转换为地球时(TT)
        
        Args:
            utc_dt: UTC时间
            
        Returns:
            TT时间的儒略日数
            
        Example:
            >>> tc = TimeConverter()
            >>> utc = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            >>> tt_jd = tc.utc_to_tt(utc)
            >>> print(f"TT JD: {tt_jd:.6f}")
        """
        t = self.ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
        return t.tt
    
    def utc_to_tdb(self, utc_dt: datetime) -> float:
        """
        UTC转换为质心力学时(TDB)
        
        Args:
            utc_dt: UTC时间
            
        Returns:
            TDB时间的儒略日数
        """
        t = self.ts.from_datetime(utc_dt.replace(tzinfo=timezone.utc))
        return t.tdb
    
    def delta_t(self, year: float) -> float:
        """
        获取ΔT值 (TT - UT1)
        
        Args:
            year: 年份（可以是小数）
            
        Returns:
            ΔT值（秒）
            
        Note:
            对于历史数据使用实测值，未来数据使用预测模型
            ΔT = 32.184 + TAI-UTC + (UT1-UTC)
        """
        year_int = int(year)
        if year_int in self.delta_t_table:
            # 线性插值
            if year_int < year:
                next_year = year_int + 1
                if next_year in self.delta_t_table:
                    fraction = year - year_int
                    return (self.delta_t_table[year_int] * (1 - fraction) + 
                           self.delta_t_table[next_year] * fraction)
            return self.delta_t_table[year_int]
        else:
            # 使用长期预测公式（Morrison & Stephenson 2004）
            t = (year - 2000) / 100
            if year < 1600:
                return 102.3 + 102.0 * t + 25.3 * t**2
            else:
                return 62.92 + 0.32217 * t + 0.005589 * t**2
    
    def julian_centuries_j2000(self, jd: float) -> float:
        """计算从J2000.0起的儒略世纪数"""
        return (jd - 2451545.0) / 36525.0


class AstronomicalCalculator:
    """天文计算器，处理岁差、章动、光行差等"""
    
    def __init__(self):
        self.tc = TimeConverter()
        
    def precession_matrix(self, jd: float) -> np.ndarray:
        """
        计算岁差矩阵（IAU 2006/2000A模型）
        
        Args:
            jd: 儒略日数
            
        Returns:
            3x3岁差旋转矩阵
            
        Reference:
            Capitaine et al. (2003), A&A 412, 567-586
        """
        t = self.tc.julian_centuries_j2000(jd)
        
        # 岁差参数（弧秒）
        # ζA = 2.5976176″ + 2306.0809506″T + 0.3019015″T² + ...
        zeta_a = (2.5976176 + 2306.0809506 * t + 0.3019015 * t**2 +
                  0.0179663 * t**3 - 0.0000327 * t**4 - 0.0000002 * t**5)
        
        # θA = 2004.1917476″T - 0.4269353″T² - 0.0418251″T³ + ...
        theta_a = (2004.1917476 * t - 0.4269353 * t**2 - 0.0418251 * t**3 -
                   0.0000601 * t**4 - 0.0000001 * t**5)
        
        # zA = -2.5976176″ + 2306.0803226″T + 1.0947790″T² + ...
        z_a = (-2.5976176 + 2306.0803226 * t + 1.0947790 * t**2 +
               0.0182273 * t**3 + 0.0000470 * t**4 - 0.0000003 * t**5)
        
        # 转换为弧度
        zeta_a = np.radians(zeta_a / 3600)
        theta_a = np.radians(theta_a / 3600)
        z_a = np.radians(z_a / 3600)
        
        # 构造旋转矩阵
        cos_zeta = np.cos(zeta_a)
        sin_zeta = np.sin(zeta_a)
        cos_theta = np.cos(theta_a)
        sin_theta = np.sin(theta_a)
        cos_z = np.cos(z_a)
        sin_z = np.sin(z_a)
        
        # P = R3(-zA) × R2(θA) × R3(-ζA)
        P = np.array([
            [cos_zeta * cos_z * cos_theta - sin_zeta * sin_z,
             -sin_zeta * cos_z * cos_theta - cos_zeta * sin_z,
             -sin_theta * cos_z],
            [cos_zeta * sin_z * cos_theta + sin_zeta * cos_z,
             -sin_zeta * sin_z * cos_theta + cos_zeta * cos_z,
             -sin_theta * sin_z],
            [cos_zeta * sin_theta,
             -sin_zeta * sin_theta,
             cos_theta]
        ])
        
        return P
    
    def nutation_angles(self, jd: float) -> Tuple[float, float]:
        """
        计算章动角度（IAU 2000B模型简化版）
        
        Args:
            jd: 儒略日数
            
        Returns:
            (Δψ, Δε): 黄经章动和黄赤交角章动（弧度）
        """
        t = self.tc.julian_centuries_j2000(jd)
        
        # 平均参数（弧度）
        # 月亮平近点角
        Ml = np.radians(134.96340251 + (1717915923.2178 * t +
                                        31.8792 * t**2 +
                                        0.051635 * t**3 -
                                        0.00024470 * t**4) / 3600.0)
        
        # 太阳平近点角
        Ms = np.radians(357.52910918 + (129596581.0481 * t -
                                        0.5532 * t**2 +
                                        0.000136 * t**3 -
                                        0.00001149 * t**4) / 3600.0)
        
        # 月亮纬度参数
        F = np.radians(93.27209062 + (1739527262.8478 * t -
                                      12.7512 * t**2 -
                                      0.001037 * t**3 +
                                      0.00000417 * t**4) / 3600.0)
        
        # 月亮到升交点的距离
        D = np.radians(297.85019547 + (1602961601.2090 * t -
                                       6.3706 * t**2 +
                                       0.006593 * t**3 -
                                       0.00003169 * t**4) / 3600.0)
        
        # 月亮升交点平黄经
        Om = np.radians(125.04455501 + (-6962890.5431 * t +
                                        7.4722 * t**2 +
                                        0.007702 * t**3 -
                                        0.00005939 * t**4) / 3600.0)
        
        # 主要章动项（最大的几项）
        dpsi = 0.0  # 黄经章动
        deps = 0.0  # 黄赤交角章动
        
        # 系数表（简化版，只包含最大的几项）
        # [D的系数, M的系数, M'的系数, F的系数, Ω的系数, Δψ正弦系数, Δε余弦系数]
        coeffs = [
            [0, 0, 0, 0, 1, -171996.0 - 174.2 * t, 92025.0 + 8.9 * t],
            [-2, 0, 0, 2, 2, -13187.0 - 1.6 * t, 5736.0 - 3.1 * t],
            [0, 0, 0, 2, 2, -2274.0 - 0.2 * t, 977.0 - 0.5 * t],
            [0, 0, 0, 0, 2, 2062.0 + 0.2 * t, -895.0 + 0.5 * t],
            [0, 1, 0, 0, 0, 1426.0 - 3.4 * t, 54.0 - 0.1 * t],
            [0, 0, 1, 0, 0, 712.0 + 0.1 * t, -7.0],
            [-2, 1, 0, 2, 2, -517.0 + 1.2 * t, 224.0 - 0.6 * t],
            [0, 0, 0, 2, 1, -386.0 - 0.4 * t, 200.0],
            [0, 0, 1, 2, 2, -301.0, 129.0 - 0.1 * t],
            [-2, -1, 0, 2, 2, 217.0 - 0.5 * t, -95.0 + 0.3 * t],
        ]
        
        for coeff in coeffs:
            arg = coeff[0] * D + coeff[1] * Ms + coeff[2] * Ml + coeff[3] * F + coeff[4] * Om
            dpsi += coeff[5] * np.sin(arg)
            deps += coeff[6] * np.cos(arg)
        
        # 转换为弧度（系数单位是0.0001弧秒）
        dpsi = dpsi * 0.0001 / 3600.0 * np.pi / 180.0
        deps = deps * 0.0001 / 3600.0 * np.pi / 180.0
        
        return dpsi, deps
    
    def aberration_correction(self, ra: float, dec: float, jd: float) -> Tuple[float, float]:
        """
        计算恒星光行差修正
        
        Args:
            ra: 赤经（弧度）
            dec: 赤纬（弧度）
            jd: 儒略日数
            
        Returns:
            (Δα, Δδ): 赤经和赤纬的光行差修正（弧度）
            
        Note:
            使用罗恩-塞科夫斯基近似公式
            光行差常数 κ = 20.49552″
        """
        t = self.tc.julian_centuries_j2000(jd)
        
        # 太阳平黄经
        L = np.radians(280.46645 + 36000.76983 * t + 0.0003032 * t**2)
        
        # 地球轨道离心率
        e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t**2
        
        # 近日点黄经
        pi = np.radians(102.93735 + 1.71946 * t + 0.00046 * t**2)
        
        # 光行差常数（弧度）
        k = 20.49552 / 3600.0 * np.pi / 180.0
        
        # 计算修正
        dra = -k * (np.cos(ra) * np.cos(L) + np.sin(ra) * np.sin(L)) / np.cos(dec) + \
              e * k * (np.cos(ra) * np.cos(pi) + np.sin(ra) * np.sin(pi)) / np.cos(dec)
        
        ddec = -k * (np.cos(L) * (np.tan(dec) * np.cos(dec) - np.sin(ra) * np.sin(dec)) +
                     np.cos(ra) * np.sin(dec) * np.sin(L)) + \
               e * k * (np.cos(pi) * (np.tan(dec) * np.cos(dec) - np.sin(ra) * np.sin(dec)) +
                        np.cos(ra) * np.sin(dec) * np.sin(pi))
        
        return dra, ddec
    
    def true_solar_time(self, utc_dt: datetime, longitude: float, 
                       equation_of_time: bool = True) -> datetime:
        """
        计算真太阳时
        
        Args:
            utc_dt: UTC时间
            longitude: 经度（度，东经为正）
            equation_of_time: 是否应用时差修正
            
        Returns:
            真太阳时
            
        Example:
            >>> calc = AstronomicalCalculator()
            >>> utc = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            >>> tst = calc.true_solar_time(utc, 116.4074)  # 北京经度
            >>> print(f"真太阳时: {tst}")
        """
        # 平太阳时修正
        mean_solar_offset = longitude * 4  # 每度经度相当于4分钟
        
        # 时差修正
        if equation_of_time:
            # 使用Astropy计算太阳位置
            location = EarthLocation(lon=longitude*u.deg, lat=0*u.deg)
            time = Time(utc_dt)
            sun = get_sun(time)
            
            # 计算时差（太阳赤经与平均赤经之差）
            # E = M - C + E_0，其中M是平近点角，C是中心差，E_0是常数项
            # 简化计算
            day_of_year = utc_dt.timetuple().tm_yday
            B = 2 * np.pi * (day_of_year - 81) / 365
            E = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
            
            total_offset = mean_solar_offset + E
        else:
            total_offset = mean_solar_offset
        
        # 应用修正
        from datetime import timedelta
        true_solar = utc_dt + timedelta(minutes=total_offset)
        
        logger.info("计算真太阳时", 
                   utc_time=utc_dt.isoformat(),
                   longitude=longitude,
                   offset_minutes=total_offset,
                   true_solar_time=true_solar.isoformat())
        
        return true_solar


def delta_t(year: float) -> float:
    """获取ΔT值的便捷函数"""
    tc = TimeConverter()
    return tc.delta_t(year)


def precession_matrix(jd: float) -> np.ndarray:
    """计算岁差矩阵的便捷函数"""
    calc = AstronomicalCalculator()
    return calc.precession_matrix(jd)


def nutation_angles(jd: float) -> Tuple[float, float]:
    """计算章动角度的便捷函数"""
    calc = AstronomicalCalculator()
    return calc.nutation_angles(jd)


def aberration_correction(ra: float, dec: float, jd: float) -> Tuple[float, float]:
    """计算光行差修正的便捷函数"""
    calc = AstronomicalCalculator()
    return calc.aberration_correction(ra, dec, jd)


def true_solar_time(utc_dt: datetime, longitude: float) -> datetime:
    """计算真太阳时的便捷函数"""
    calc = AstronomicalCalculator()
    return calc.true_solar_time(utc_dt, longitude)


if __name__ == "__main__":
    # 简单测试
    tc = TimeConverter()
    calc = AstronomicalCalculator()
    
    # 测试时间转换
    now = datetime.now(timezone.utc)
    tt_jd = tc.utc_to_tt(now)
    tdb_jd = tc.utc_to_tdb(now)
    
    print(f"当前UTC时间: {now}")
    print(f"TT儒略日: {tt_jd:.6f}")
    print(f"TDB儒略日: {tdb_jd:.6f}")
    print(f"ΔT: {tc.delta_t(now.year):.3f}秒")
    
    # 测试岁差矩阵
    P = calc.precession_matrix(tt_jd)
    print(f"\n岁差矩阵:\n{P}")
    
    # 测试章动
    dpsi, deps = calc.nutation_angles(tt_jd)
    print(f"\n章动: Δψ={dpsi*3600*180/np.pi:.3f}″, Δε={deps*3600*180/np.pi:.3f}″")
    
    # 测试真太阳时
    beijing_lon = 116.4074
    tst = calc.true_solar_time(now, beijing_lon)
    print(f"\n北京真太阳时: {tst}")