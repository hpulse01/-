"""
西方占星数据采集模块
实现星历计算、宫位系统（Placidus）、相位分析、过境、月亮交点、凯龙星等

理论基础：
- 行星：日月水金火木土天海冥 + 月亮交点 + 凯龙星
- 宫位：基于真太阳时的ASC/MC计算
- 相位：合冲刑拱等主要相位
- 过境：行运行星与本命盘的互动
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import structlog
from skyfield.api import load, Topos
from skyfield.timelib import Time
from astropy.time import Time as AstropyTime
from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy import units as u
import ephem

from ..utils.time_astro import TimeConverter, AstronomicalCalculator, true_solar_time

logger = structlog.get_logger()


# 行星常量
PLANETS = {
    "Sun": "太阳",
    "Moon": "月亮", 
    "Mercury": "水星",
    "Venus": "金星",
    "Mars": "火星",
    "Jupiter": "木星",
    "Saturn": "土星",
    "Uranus": "天王星",
    "Neptune": "海王星",
    "Pluto": "冥王星"
}

# 特殊点
SPECIAL_POINTS = {
    "NorthNode": "北交点",
    "SouthNode": "南交点",
    "Chiron": "凯龙星",
    "ASC": "上升点",
    "MC": "中天",
    "DSC": "下降点",
    "IC": "天底"
}

# 黄道十二宫
ZODIAC_SIGNS = [
    "白羊座", "金牛座", "双子座", "巨蟹座",
    "狮子座", "处女座", "天秤座", "天蝎座",
    "射手座", "摩羯座", "水瓶座", "双鱼座"
]

# 宫位名称
HOUSES = [
    "第一宫", "第二宫", "第三宫", "第四宫",
    "第五宫", "第六宫", "第七宫", "第八宫",
    "第九宫", "第十宫", "第十一宫", "第十二宫"
]

# 主要相位（度数: [名称, 容许度]）
ASPECTS = {
    0: ["合相", 8],
    60: ["六分相", 6],
    90: ["刑相", 8],
    120: ["拱相", 8],
    180: ["冲相", 8],
    # 次要相位
    30: ["半六分相", 2],
    45: ["半刑相", 2],
    135: ["倍半刑相", 2],
    150: ["梅花相", 2]
}


@dataclass
class CelestialBody:
    """天体信息"""
    name: str
    longitude: float  # 黄经（度）
    latitude: float   # 黄纬（度）
    distance: float   # 距离（AU）
    speed: float      # 速度（度/天）
    
    @property
    def zodiac_sign(self) -> str:
        """所在星座"""
        sign_idx = int(self.longitude / 30)
        return ZODIAC_SIGNS[sign_idx]
    
    @property
    def sign_degree(self) -> float:
        """星座内度数"""
        return self.longitude % 30
    
    @property
    def is_retrograde(self) -> bool:
        """是否逆行"""
        return self.speed < 0
    
    def position_string(self) -> str:
        """位置描述"""
        deg = int(self.sign_degree)
        min = int((self.sign_degree - deg) * 60)
        retro = "R" if self.is_retrograde else ""
        return f"{self.zodiac_sign} {deg:02d}°{min:02d}' {retro}"


@dataclass
class House:
    """宫位信息"""
    number: int  # 宫位号（1-12）
    cusp: float  # 宫头度数
    
    @property
    def name(self) -> str:
        return HOUSES[self.number - 1]
    
    @property
    def zodiac_sign(self) -> str:
        """宫头所在星座"""
        sign_idx = int(self.cusp / 30)
        return ZODIAC_SIGNS[sign_idx]
    
    @property
    def sign_degree(self) -> float:
        """宫头在星座内的度数"""
        return self.cusp % 30
    
    def contains_longitude(self, longitude: float) -> bool:
        """判断某黄经度数是否在此宫内"""
        next_cusp = self.next_cusp if hasattr(self, 'next_cusp') else (self.cusp + 30) % 360
        
        if self.cusp <= next_cusp:
            return self.cusp <= longitude < next_cusp
        else:  # 跨越0度
            return longitude >= self.cusp or longitude < next_cusp


@dataclass
class Aspect:
    """相位信息"""
    body1: str
    body2: str
    angle: float      # 实际角度
    aspect_type: str  # 相位类型
    orb: float        # 容许度
    applying: bool    # 是否入相位
    
    @property
    def is_exact(self) -> bool:
        """是否精确相位"""
        return abs(self.orb) < 1.0
    
    @property
    def strength(self) -> float:
        """相位强度（0-1）"""
        max_orb = ASPECTS[int(self.angle)][1]
        return 1.0 - abs(self.orb) / max_orb


@dataclass
class NatalChart:
    """出生星盘"""
    birth_datetime: datetime
    longitude: float
    latitude: float
    
    # 行星位置
    planets: Dict[str, CelestialBody] = field(default_factory=dict)
    
    # 特殊点
    points: Dict[str, float] = field(default_factory=dict)
    
    # 宫位
    houses: List[House] = field(default_factory=list)
    
    # 相位
    aspects: List[Aspect] = field(default_factory=list)
    
    def get_planet_house(self, planet_name: str) -> Optional[int]:
        """获取行星所在宫位"""
        if planet_name not in self.planets:
            return None
        
        planet_long = self.planets[planet_name].longitude
        for house in self.houses:
            if house.contains_longitude(planet_long):
                return house.number
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        planets_data = {}
        for name, body in self.planets.items():
            planets_data[name] = {
                "位置": body.position_string(),
                "黄经": body.longitude,
                "黄纬": body.latitude,
                "速度": body.speed,
                "逆行": body.is_retrograde,
                "宫位": self.get_planet_house(name)
            }
        
        houses_data = []
        for house in self.houses:
            houses_data.append({
                "宫位": house.name,
                "宫头": f"{house.zodiac_sign} {house.sign_degree:.2f}°"
            })
        
        aspects_data = []
        for aspect in self.aspects:
            aspects_data.append({
                "星体1": aspect.body1,
                "星体2": aspect.body2,
                "相位": aspect.aspect_type,
                "角度": aspect.angle,
                "容许度": aspect.orb,
                "入相位": aspect.applying
            })
        
        return {
            "出生时间": self.birth_datetime.isoformat(),
            "经度": self.longitude,
            "纬度": self.latitude,
            "行星": planets_data,
            "特殊点": self.points,
            "宫位": houses_data,
            "相位": aspects_data
        }


class AstrologyCalculator:
    """西方占星计算器"""
    
    def __init__(self):
        self.tc = TimeConverter()
        self.ac = AstronomicalCalculator()
        # 加载星历
        self.eph = load('de440.bsp')  # JPL星历
        self.ts = load.timescale()
        
    def calculate_natal_chart(self, birth_dt: datetime, 
                            longitude: float, latitude: float) -> NatalChart:
        """
        计算出生星盘
        
        Args:
            birth_dt: 出生时间（需要时区信息）
            longitude: 经度
            latitude: 纬度
            
        Returns:
            出生星盘
        """
        # 转换为真太阳时
        true_solar_dt = true_solar_time(birth_dt, longitude)
        
        # 创建星盘
        chart = NatalChart(
            birth_datetime=birth_dt,
            longitude=longitude,
            latitude=latitude
        )
        
        # 计算行星位置
        self._calculate_planets(chart, true_solar_dt)
        
        # 计算特殊点（月交点、凯龙星）
        self._calculate_special_points(chart, true_solar_dt)
        
        # 计算宫位（使用Placidus系统）
        self._calculate_houses(chart, true_solar_dt, longitude, latitude)
        
        # 计算相位
        self._calculate_aspects(chart)
        
        logger.info("出生星盘计算完成",
                   asc=chart.points.get("ASC", 0),
                   mc=chart.points.get("MC", 0))
        
        return chart
    
    def _calculate_planets(self, chart: NatalChart, dt: datetime):
        """计算行星位置"""
        # Skyfield时间
        t = self.ts.from_datetime(dt)
        
        # 地球位置
        earth = self.eph['earth']
        
        # 计算各行星
        planet_map = {
            'Sun': self.eph['sun'],
            'Moon': self.eph['moon'],
            'Mercury': self.eph['mercury'],
            'Venus': self.eph['venus'],
            'Mars': self.eph['mars'],
            'Jupiter': self.eph['jupiter barycenter'],
            'Saturn': self.eph['saturn barycenter'],
            'Uranus': self.eph['uranus barycenter'],
            'Neptune': self.eph['neptune barycenter'],
            'Pluto': self.eph['pluto barycenter']
        }
        
        for name, planet_obj in planet_map.items():
            # 地心位置
            astrometric = earth.at(t).observe(planet_obj)
            apparent = astrometric.apparent()
            
            # 黄道坐标
            lat, lon, distance = apparent.ecliptic_latlon()
            
            # 计算速度（通过前后时间点）
            dt_before = dt - timedelta(hours=12)
            dt_after = dt + timedelta(hours=12)
            t_before = self.ts.from_datetime(dt_before)
            t_after = self.ts.from_datetime(dt_after)
            
            app_before = earth.at(t_before).observe(planet_obj).apparent()
            app_after = earth.at(t_after).observe(planet_obj).apparent()
            
            _, lon_before, _ = app_before.ecliptic_latlon()
            _, lon_after, _ = app_after.ecliptic_latlon()
            
            # 速度（度/天）
            speed = (lon_after.degrees - lon_before.degrees)
            if speed > 180:
                speed -= 360
            elif speed < -180:
                speed += 360
            
            chart.planets[name] = CelestialBody(
                name=name,
                longitude=lon.degrees,
                latitude=lat.degrees,
                distance=distance.au,
                speed=speed
            )
    
    def _calculate_special_points(self, chart: NatalChart, dt: datetime):
        """计算特殊点（月交点、凯龙星等）"""
        # 月亮交点（使用平均交点）
        # 北交点回归周期约18.6年
        base_date = datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.tzinfo)
        days_since = (dt - base_date).total_seconds() / 86400
        
        # 2000年1月1日北交点在巨蟹座25度左右
        node_cycle = 6798.36  # 天
        node_daily_motion = -360.0 / node_cycle
        
        north_node_long = (115.0 + days_since * node_daily_motion) % 360
        south_node_long = (north_node_long + 180) % 360
        
        chart.planets["NorthNode"] = CelestialBody(
            name="NorthNode",
            longitude=north_node_long,
            latitude=0.0,
            distance=0.0,
            speed=node_daily_motion
        )
        
        chart.planets["SouthNode"] = CelestialBody(
            name="SouthNode",
            longitude=south_node_long,
            latitude=0.0,
            distance=0.0,
            speed=node_daily_motion
        )
        
        # 凯龙星（简化计算）
        # 实际应使用精确星历
        chiron_period = 50.45 * 365.25  # 天
        chiron_daily = 360.0 / chiron_period
        chiron_long = (50.0 + days_since * chiron_daily) % 360
        
        chart.planets["Chiron"] = CelestialBody(
            name="Chiron",
            longitude=chiron_long,
            latitude=0.0,
            distance=13.7,  # 平均距离
            speed=chiron_daily
        )
    
    def _calculate_houses(self, chart: NatalChart, dt: datetime,
                         longitude: float, latitude: float):
        """计算宫位系统（Placidus）"""
        # 计算恒星时
        t = self.ts.from_datetime(dt)
        
        # 计算地方恒星时（LST）
        gst = t.gast  # 格林威治恒星时（小时）
        lst = (gst + longitude / 15.0) % 24  # 地方恒星时
        
        # 计算中天（MC）
        mc_ra = lst * 15.0  # 转换为度
        
        # 计算上升点（ASC）- 使用球面三角
        # tan(A) = -sin(LST) / (cos(LST) * sin(φ) - tan(ε) * cos(φ))
        # 其中φ是地理纬度，ε是黄赤交角
        
        phi = np.radians(latitude)
        epsilon = np.radians(23.4397)  # 黄赤交角
        lst_rad = np.radians(lst * 15.0)
        
        # 计算ASC
        numerator = -np.sin(lst_rad)
        denominator = np.cos(lst_rad) * np.sin(phi) - np.tan(epsilon) * np.cos(phi)
        asc_ra = np.arctan2(numerator, denominator)
        
        # 转换为黄经
        asc_long = np.degrees(asc_ra) % 360
        mc_long = mc_ra % 360
        
        # 保存关键点
        chart.points["ASC"] = asc_long
        chart.points["MC"] = mc_long
        chart.points["DSC"] = (asc_long + 180) % 360
        chart.points["IC"] = (mc_long + 180) % 360
        
        # 计算12宫位（Placidus系统简化版）
        # 实际Placidus计算非常复杂，这里使用等宫制近似
        houses = []
        for i in range(12):
            house = House(
                number=i + 1,
                cusp=(asc_long + i * 30) % 360
            )
            houses.append(house)
        
        # 设置next_cusp属性用于判断行星所在宫位
        for i in range(12):
            houses[i].next_cusp = houses[(i + 1) % 12].cusp
        
        chart.houses = houses
    
    def _calculate_aspects(self, chart: NatalChart):
        """计算相位"""
        aspects = []
        
        # 获取所有需要计算相位的天体
        bodies = list(chart.planets.keys())
        
        # 加入重要点
        for point in ["ASC", "MC"]:
            if point in chart.points:
                bodies.append(point)
        
        # 计算两两之间的相位
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                body1 = bodies[i]
                body2 = bodies[j]
                
                # 获取黄经
                if body1 in chart.planets:
                    lon1 = chart.planets[body1].longitude
                else:
                    lon1 = chart.points.get(body1, 0)
                
                if body2 in chart.planets:
                    lon2 = chart.planets[body2].longitude
                else:
                    lon2 = chart.points.get(body2, 0)
                
                # 计算角度差
                angle_diff = abs(lon1 - lon2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # 检查是否形成相位
                for aspect_angle, (aspect_name, orb_allowed) in ASPECTS.items():
                    orb = angle_diff - aspect_angle
                    if abs(orb) <= orb_allowed:
                        # 判断入相位还是出相位
                        applying = self._is_applying(chart, body1, body2, angle_diff)
                        
                        aspects.append(Aspect(
                            body1=body1,
                            body2=body2,
                            angle=angle_diff,
                            aspect_type=aspect_name,
                            orb=orb,
                            applying=applying
                        ))
                        break
        
        chart.aspects = aspects
    
    def _is_applying(self, chart: NatalChart, body1: str, body2: str, 
                    current_angle: float) -> bool:
        """判断相位是入相位还是出相位"""
        # 获取速度
        speed1 = 0
        speed2 = 0
        
        if body1 in chart.planets:
            speed1 = chart.planets[body1].speed
        if body2 in chart.planets:
            speed2 = chart.planets[body2].speed
        
        # 如果是静止点（ASC/MC等），速度为0
        # 简化判断：速度快的追速度慢的为入相位
        return abs(speed1) > abs(speed2)
    
    def calculate_transits(self, natal_chart: NatalChart, 
                         transit_dt: datetime) -> Dict[str, Any]:
        """
        计算过境（行运）
        
        Args:
            natal_chart: 出生星盘
            transit_dt: 过境时间
            
        Returns:
            过境信息
        """
        # 计算过境时刻的行星位置
        transit_planets = {}
        t = self.ts.from_datetime(transit_dt)
        earth = self.eph['earth']
        
        planet_map = {
            'Sun': self.eph['sun'],
            'Moon': self.eph['moon'],
            'Mercury': self.eph['mercury'],
            'Venus': self.eph['venus'],
            'Mars': self.eph['mars'],
            'Jupiter': self.eph['jupiter barycenter'],
            'Saturn': self.eph['saturn barycenter'],
            'Uranus': self.eph['uranus barycenter'],
            'Neptune': self.eph['neptune barycenter'],
            'Pluto': self.eph['pluto barycenter']
        }
        
        for name, planet_obj in planet_map.items():
            astrometric = earth.at(t).observe(planet_obj)
            apparent = astrometric.apparent()
            lat, lon, distance = apparent.ecliptic_latlon()
            
            transit_planets[name] = {
                "longitude": lon.degrees,
                "latitude": lat.degrees
            }
        
        # 计算过境相位
        transit_aspects = []
        
        for t_name, t_data in transit_planets.items():
            t_lon = t_data["longitude"]
            
            for n_name, n_body in natal_chart.planets.items():
                n_lon = n_body.longitude
                
                # 计算角度差
                angle_diff = abs(t_lon - n_lon)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # 检查相位
                for aspect_angle, (aspect_name, orb_allowed) in ASPECTS.items():
                    # 过境相位使用更严格的容许度
                    transit_orb = orb_allowed * 0.5
                    orb = angle_diff - aspect_angle
                    
                    if abs(orb) <= transit_orb:
                        transit_aspects.append({
                            "过境星": t_name,
                            "本命星": n_name,
                            "相位": aspect_name,
                            "角度": angle_diff,
                            "容许度": orb
                        })
                        break
        
        return {
            "过境时间": transit_dt.isoformat(),
            "过境行星": transit_planets,
            "过境相位": transit_aspects
        }
    
    def calculate_progressions(self, natal_chart: NatalChart,
                             target_dt: datetime) -> Dict[str, Any]:
        """
        计算推进盘（次限法）
        一天代表一年
        
        Args:
            natal_chart: 出生星盘
            target_dt: 目标时间
            
        Returns:
            推进盘信息
        """
        # 计算年龄
        years = (target_dt - natal_chart.birth_datetime).days / 365.25
        
        # 推进的天数
        prog_days = years
        
        # 计算推进后的时间
        prog_dt = natal_chart.birth_datetime + timedelta(days=prog_days)
        
        # 计算推进盘
        prog_chart = self.calculate_natal_chart(
            prog_dt,
            natal_chart.longitude,
            natal_chart.latitude
        )
        
        # 计算推进盘与本命盘的相位
        prog_aspects = []
        
        for p_name, p_body in prog_chart.planets.items():
            p_lon = p_body.longitude
            
            for n_name, n_body in natal_chart.planets.items():
                n_lon = n_body.longitude
                
                angle_diff = abs(p_lon - n_lon)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                for aspect_angle, (aspect_name, orb_allowed) in ASPECTS.items():
                    # 推进相位使用非常严格的容许度
                    prog_orb = min(orb_allowed * 0.25, 1.0)
                    orb = angle_diff - aspect_angle
                    
                    if abs(orb) <= prog_orb:
                        prog_aspects.append({
                            "推进星": p_name,
                            "本命星": n_name,
                            "相位": aspect_name,
                            "角度": angle_diff,
                            "容许度": orb
                        })
                        break
        
        return {
            "目标时间": target_dt.isoformat(),
            "推进年数": years,
            "推进盘": prog_chart.to_dict(),
            "推进相位": prog_aspects
        }


# 便捷函数
def calculate_natal_chart(birth_dt: datetime, longitude: float,
                        latitude: float) -> NatalChart:
    """计算出生星盘的便捷函数"""
    calculator = AstrologyCalculator()
    return calculator.calculate_natal_chart(birth_dt, longitude, latitude)


def calculate_transits(natal_chart: NatalChart,
                     transit_dt: datetime) -> Dict[str, Any]:
    """计算过境的便捷函数"""
    calculator = AstrologyCalculator()
    return calculator.calculate_transits(natal_chart, transit_dt)


def calculate_progressions(natal_chart: NatalChart,
                         target_dt: datetime) -> Dict[str, Any]:
    """计算推进盘的便捷函数"""
    calculator = AstrologyCalculator()
    return calculator.calculate_progressions(natal_chart, target_dt)


if __name__ == "__main__":
    # 测试
    from datetime import timezone
    
    # 测试数据
    birth_dt = datetime(1990, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    longitude = 116.4074  # 北京
    latitude = 39.9042
    
    print("=== 出生星盘 ===")
    chart = calculate_natal_chart(birth_dt, longitude, latitude)
    
    # 显示行星位置
    print("\n行星位置：")
    for name, body in chart.planets.items():
        chinese_name = PLANETS.get(name, SPECIAL_POINTS.get(name, name))
        house = chart.get_planet_house(name)
        print(f"{chinese_name}: {body.position_string()} - 第{house}宫")
    
    # 显示宫位
    print("\n宫位系统：")
    print(f"ASC（上升点）: {chart.points['ASC']:.2f}°")
    print(f"MC（中天）: {chart.points['MC']:.2f}°")
    
    # 显示主要相位
    print("\n主要相位：")
    for aspect in chart.aspects[:10]:  # 显示前10个
        body1_cn = PLANETS.get(aspect.body1, SPECIAL_POINTS.get(aspect.body1, aspect.body1))
        body2_cn = PLANETS.get(aspect.body2, SPECIAL_POINTS.get(aspect.body2, aspect.body2))
        print(f"{body1_cn} {aspect.aspect_type} {body2_cn} "
              f"(容许度: {aspect.orb:.1f}°)")
    
    # 测试过境
    print("\n=== 当前过境 ===")
    now = datetime.now(timezone.utc)
    transits = calculate_transits(chart, now)
    
    print(f"\n重要过境相位：")
    for t_aspect in transits["过境相位"][:5]:
        t_cn = PLANETS.get(t_aspect["过境星"], t_aspect["过境星"])
        n_cn = PLANETS.get(t_aspect["本命星"], t_aspect["本命星"])
        print(f"过境{t_cn} {t_aspect['相位']} 本命{n_cn}")