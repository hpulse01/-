"""
东方命理数据采集模块
实现四柱八字排盘、大运流年岁运、紫微斗数（十四主星+四化+辅星+身宫+限运）

理论基础：
- 四柱八字：年月日时四柱，天干地支组合，五行生克制化
- 紫微斗数：十二宫位，十四主星，四化飞星，辅星系统
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import structlog
from dateutil import tz

from ..utils.time_astro import TimeConverter, true_solar_time

logger = structlog.get_logger()


# 天干地支常量
TIANGAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
DIZHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

# 五行属性
WUXING = ["木", "火", "土", "金", "水"]
TIANGAN_WUXING = {
    "甲": "木", "乙": "木", "丙": "火", "丁": "火", "戊": "土",
    "己": "土", "庚": "金", "辛": "金", "壬": "水", "癸": "水"
}
DIZHI_WUXING = {
    "子": "水", "丑": "土", "寅": "木", "卯": "木", "辰": "土", "巳": "火",
    "午": "火", "未": "土", "申": "金", "酉": "金", "戌": "土", "亥": "水"
}

# 地支藏干
DIZHI_CANGGAN = {
    "子": ["癸"],
    "丑": ["己", "癸", "辛"],
    "寅": ["甲", "丙", "戊"],
    "卯": ["乙"],
    "辰": ["戊", "乙", "癸"],
    "巳": ["丙", "庚", "戊"],
    "午": ["丁", "己"],
    "未": ["己", "丁", "乙"],
    "申": ["庚", "壬", "戊"],
    "酉": ["辛"],
    "戌": ["戊", "辛", "丁"],
    "亥": ["壬", "甲"]
}

# 十神定义
SHISHEN_MAP = {
    ("甲", "甲"): "比肩", ("甲", "乙"): "劫财", ("甲", "丙"): "食神", ("甲", "丁"): "伤官",
    ("甲", "戊"): "偏财", ("甲", "己"): "正财", ("甲", "庚"): "七杀", ("甲", "辛"): "正官",
    ("甲", "壬"): "偏印", ("甲", "癸"): "正印",
    # ... 完整的十神对照表需要100个组合，这里简化
}

# 紫微斗数常量
# 十二宫位
ZIWEI_GONG = ["命宫", "兄弟", "夫妻", "子女", "财帛", "疾厄", 
              "迁移", "交友", "官禄", "田宅", "福德", "父母"]

# 十四主星
ZIWEI_ZHUXING = ["紫微", "天机", "太阳", "武曲", "天同", "廉贞", "天府",
                 "太阴", "贪狼", "巨门", "天相", "天梁", "七杀", "破军"]

# 辅星
ZIWEI_FUXING = ["文昌", "文曲", "左辅", "右弼", "天魁", "天钺", 
                "禄存", "擎羊", "陀罗", "火星", "铃星", "地空", "地劫"]

# 四化星
SIHUA_TIANGAN = {
    "甲": {"化禄": "廉贞", "化权": "破军", "化科": "武曲", "化忌": "太阳"},
    "乙": {"化禄": "天机", "化权": "天梁", "化科": "紫微", "化忌": "太阴"},
    "丙": {"化禄": "天同", "化权": "天机", "化科": "文昌", "化忌": "廉贞"},
    "丁": {"化禄": "太阴", "化权": "天同", "化科": "天机", "化忌": "巨门"},
    "戊": {"化禄": "贪狼", "化权": "太阴", "化科": "右弼", "化忌": "天机"},
    "己": {"化禄": "武曲", "化权": "贪狼", "化科": "天梁", "化忌": "文曲"},
    "庚": {"化禄": "太阳", "化权": "武曲", "化科": "天同", "化忌": "天相"},
    "辛": {"化禄": "巨门", "化权": "太阳", "化科": "文曲", "化忌": "文昌"},
    "壬": {"化禄": "天梁", "化权": "紫微", "化科": "左辅", "化忌": "武曲"},
    "癸": {"化禄": "破军", "化权": "巨门", "化科": "太阴", "化忌": "贪狼"}
}


@dataclass
class BaZiPillar:
    """四柱中的一柱"""
    gan: str  # 天干
    zhi: str  # 地支
    
    @property
    def ganzhi(self) -> str:
        return self.gan + self.zhi
    
    @property
    def wuxing(self) -> Tuple[str, str]:
        """返回天干地支的五行"""
        return TIANGAN_WUXING[self.gan], DIZHI_WUXING[self.zhi]
    
    @property
    def canggan(self) -> List[str]:
        """返回地支藏干"""
        return DIZHI_CANGGAN.get(self.zhi, [])


@dataclass
class BaZiChart:
    """四柱八字命盘"""
    year: BaZiPillar   # 年柱
    month: BaZiPillar  # 月柱
    day: BaZiPillar    # 日柱
    hour: BaZiPillar   # 时柱
    
    birth_datetime: datetime
    longitude: float
    latitude: float
    gender: str = "男"  # 男/女
    
    # 大运
    dayun_start_age: int = 0
    dayun_list: List[Tuple[str, int]] = field(default_factory=list)  # [(干支, 起始年龄), ...]
    
    # 神煞
    shensha: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_rizhu_gan(self) -> str:
        """获取日主（日干）"""
        return self.day.gan
    
    def get_pillars_list(self) -> List[BaZiPillar]:
        """获取四柱列表"""
        return [self.year, self.month, self.day, self.hour]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "年柱": self.year.ganzhi,
            "月柱": self.month.ganzhi,
            "日柱": self.day.ganzhi,
            "时柱": self.hour.ganzhi,
            "日主": self.get_rizhu_gan(),
            "性别": self.gender,
            "出生时间": self.birth_datetime.isoformat(),
            "经度": self.longitude,
            "纬度": self.latitude,
            "大运起始年龄": self.dayun_start_age,
            "大运": [{"干支": gz, "起始年龄": age} for gz, age in self.dayun_list],
            "神煞": self.shensha
        }


@dataclass
class ZiWeiGong:
    """紫微斗数宫位"""
    name: str  # 宫位名称
    position: int  # 位置（0-11）
    gan: str  # 天干
    zhi: str  # 地支
    
    # 星曜
    zhuxing: List[str] = field(default_factory=list)  # 主星
    fuxing: List[str] = field(default_factory=list)   # 辅星
    sihua: Dict[str, str] = field(default_factory=dict)  # 四化 {"化禄": "星名", ...}
    
    @property
    def ganzhi(self) -> str:
        return self.gan + self.zhi
    
    def get_all_stars(self) -> List[str]:
        """获取所有星曜"""
        return self.zhuxing + self.fuxing + list(self.sihua.values())


@dataclass
class ZiWeiChart:
    """紫微斗数命盘"""
    gongs: Dict[str, ZiWeiGong]  # 十二宫位
    ming_gong_position: int  # 命宫位置
    shen_gong_position: int  # 身宫位置
    
    birth_datetime: datetime
    longitude: float
    latitude: float
    gender: str = "男"
    
    # 大限
    daxian_list: List[Tuple[str, int, int]] = field(default_factory=list)  # [(宫位, 起始年龄, 结束年龄), ...]
    
    def get_ming_gong(self) -> ZiWeiGong:
        """获取命宫"""
        return self.gongs["命宫"]
    
    def get_shen_gong(self) -> ZiWeiGong:
        """获取身宫"""
        # 身宫可能与其他宫位重合
        for name, gong in self.gongs.items():
            if gong.position == self.shen_gong_position:
                return gong
        return self.gongs["命宫"]  # 默认返回命宫
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        gongs_data = {}
        for name, gong in self.gongs.items():
            gongs_data[name] = {
                "干支": gong.ganzhi,
                "位置": gong.position,
                "主星": gong.zhuxing,
                "辅星": gong.fuxing,
                "四化": gong.sihua
            }
        
        return {
            "出生时间": self.birth_datetime.isoformat(),
            "性别": self.gender,
            "命宫位置": self.ming_gong_position,
            "身宫位置": self.shen_gong_position,
            "十二宫位": gongs_data,
            "大限": [{"宫位": g, "起始": s, "结束": e} for g, s, e in self.daxian_list]
        }


class BaZiCalculator:
    """四柱八字计算器"""
    
    def __init__(self):
        self.tc = TimeConverter()
        
    def calculate(self, birth_dt: datetime, longitude: float, 
                 latitude: float, gender: str = "男") -> BaZiChart:
        """
        计算四柱八字
        
        Args:
            birth_dt: 出生时间（需要时区信息）
            longitude: 经度
            latitude: 纬度
            gender: 性别
            
        Returns:
            四柱八字命盘
        """
        # 转换为真太阳时
        true_solar_dt = true_solar_time(birth_dt, longitude)
        
        # 计算四柱
        year_pillar = self._calculate_year_pillar(true_solar_dt)
        month_pillar = self._calculate_month_pillar(true_solar_dt, year_pillar)
        day_pillar = self._calculate_day_pillar(true_solar_dt)
        hour_pillar = self._calculate_hour_pillar(true_solar_dt, day_pillar)
        
        # 创建命盘
        chart = BaZiChart(
            year=year_pillar,
            month=month_pillar,
            day=day_pillar,
            hour=hour_pillar,
            birth_datetime=birth_dt,
            longitude=longitude,
            latitude=latitude,
            gender=gender
        )
        
        # 计算大运
        self._calculate_dayun(chart)
        
        # 计算神煞
        self._calculate_shensha(chart)
        
        logger.info("四柱八字计算完成",
                   bazi=f"{chart.year.ganzhi} {chart.month.ganzhi} "
                        f"{chart.day.ganzhi} {chart.hour.ganzhi}",
                   rizhu=chart.get_rizhu_gan())
        
        return chart
    
    def _calculate_year_pillar(self, dt: datetime) -> BaZiPillar:
        """计算年柱（考虑立春）"""
        # 简化处理：以农历年计算
        # 实际应该精确判断立春时刻
        year = dt.year
        
        # 年干支计算（以1984甲子年为基准）
        year_offset = year - 1984
        gan_idx = (year_offset + 0) % 10  # 甲=0
        zhi_idx = (year_offset + 0) % 12  # 子=0
        
        # 立春修正（2月4日前算上一年）
        if dt.month < 2 or (dt.month == 2 and dt.day < 4):
            gan_idx = (gan_idx - 1) % 10
            zhi_idx = (zhi_idx - 1) % 12
        
        return BaZiPillar(TIANGAN[gan_idx], DIZHI[zhi_idx])
    
    def _calculate_month_pillar(self, dt: datetime, year_pillar: BaZiPillar) -> BaZiPillar:
        """计算月柱（考虑节气）"""
        # 月支（以节气定月）
        month_zhi_map = {
            1: "寅", 2: "卯", 3: "辰", 4: "巳", 5: "午", 6: "未",
            7: "申", 8: "酉", 9: "戌", 10: "亥", 11: "子", 12: "丑"
        }
        
        # 简化：按公历月份（实际应按节气）
        month = dt.month
        if dt.day < 6:  # 粗略估计节气在6号左右
            month = (month - 2) % 12 + 1
        else:
            month = (month - 1) % 12 + 1
        
        zhi = month_zhi_map[month]
        
        # 月干（年干定月干）
        year_gan_idx = TIANGAN.index(year_pillar.gan)
        month_gan_start = {
            0: 2, 1: 4, 2: 6, 3: 8, 4: 0,  # 甲己之年丙作首
            5: 2, 6: 4, 7: 6, 8: 8, 9: 0   # 乙庚之年戊为头...
        }
        
        # 根据年干确定正月天干
        start_idx = month_gan_start[year_gan_idx % 5] * 2 % 10
        month_idx = list(month_zhi_map.values()).index(zhi)
        gan_idx = (start_idx + month_idx) % 10
        
        return BaZiPillar(TIANGAN[gan_idx], zhi)
    
    def _calculate_day_pillar(self, dt: datetime) -> BaZiPillar:
        """计算日柱（使用万年历或算法）"""
        # 使用儒略日数计算日干支
        # 这里使用简化算法
        base_date = datetime(1900, 1, 1, tzinfo=dt.tzinfo)
        days_diff = (dt - base_date).days
        
        # 1900年1月1日是庚子日（庚=6, 子=0）
        gan_idx = (6 + days_diff) % 10
        zhi_idx = (0 + days_diff) % 12
        
        return BaZiPillar(TIANGAN[gan_idx], DIZHI[zhi_idx])
    
    def _calculate_hour_pillar(self, dt: datetime, day_pillar: BaZiPillar) -> BaZiPillar:
        """计算时柱"""
        # 时支
        hour = dt.hour
        hour_zhi_map = {
            (23, 1): "子", (1, 3): "丑", (3, 5): "寅", (5, 7): "卯",
            (7, 9): "辰", (9, 11): "巳", (11, 13): "午", (13, 15): "未",
            (15, 17): "申", (17, 19): "酉", (19, 21): "戌", (21, 23): "亥"
        }
        
        for (start, end), zhi in hour_zhi_map.items():
            if start <= hour < end or (start == 23 and (hour >= 23 or hour < 1)):
                hour_zhi = zhi
                break
        else:
            hour_zhi = "子"  # 默认
        
        # 时干（日干定时干）
        day_gan_idx = TIANGAN.index(day_pillar.gan)
        hour_gan_start = {
            0: 0, 1: 2, 2: 4, 3: 6, 4: 8,  # 甲己还生甲
            5: 0, 6: 2, 7: 4, 8: 6, 9: 8   # 乙庚丙作初...
        }
        
        start_idx = hour_gan_start[day_gan_idx % 5]
        hour_idx = DIZHI.index(hour_zhi)
        gan_idx = (start_idx + hour_idx) % 10
        
        return BaZiPillar(TIANGAN[gan_idx], hour_zhi)
    
    def _calculate_dayun(self, chart: BaZiChart):
        """计算大运"""
        # 确定顺逆（阳年生男、阴年生女顺排）
        year_gan_idx = TIANGAN.index(chart.year.gan)
        is_yang_year = year_gan_idx % 2 == 0
        is_male = chart.gender == "男"
        is_forward = (is_yang_year and is_male) or (not is_yang_year and not is_male)
        
        # 计算起运年龄（简化：统一8岁起运）
        chart.dayun_start_age = 8
        
        # 排大运
        month_gan_idx = TIANGAN.index(chart.month.gan)
        month_zhi_idx = DIZHI.index(chart.month.zhi)
        
        dayun_list = []
        for i in range(8):  # 8个大运
            if is_forward:
                gan_idx = (month_gan_idx + i + 1) % 10
                zhi_idx = (month_zhi_idx + i + 1) % 12
            else:
                gan_idx = (month_gan_idx - i - 1) % 10
                zhi_idx = (month_zhi_idx - i - 1) % 12
            
            ganzhi = TIANGAN[gan_idx] + DIZHI[zhi_idx]
            start_age = chart.dayun_start_age + i * 10
            dayun_list.append((ganzhi, start_age))
        
        chart.dayun_list = dayun_list
    
    def _calculate_shensha(self, chart: BaZiChart):
        """计算神煞（简化版）"""
        chart.shensha = {
            "天乙贵人": self._calc_tianyi(chart),
            "文昌": self._calc_wenchang(chart),
            "天德": self._calc_tiande(chart),
            "月德": self._calc_yuede(chart),
        }
    
    def _calc_tianyi(self, chart: BaZiChart) -> List[str]:
        """计算天乙贵人"""
        # 日干查天乙贵人
        tianyi_map = {
            "甲": ["丑", "未"], "乙": ["子", "申"], "丙": ["亥", "酉"],
            "丁": ["亥", "酉"], "戊": ["丑", "未"], "己": ["子", "申"],
            "庚": ["丑", "未"], "辛": ["寅", "午"], "壬": ["卯", "巳"],
            "癸": ["卯", "巳"]
        }
        
        rizhu_gan = chart.get_rizhu_gan()
        tianyi_zhi = tianyi_map.get(rizhu_gan, [])
        
        result = []
        for pillar in chart.get_pillars_list():
            if pillar.zhi in tianyi_zhi:
                result.append(f"{pillar.ganzhi}宫")
        
        return result
    
    def _calc_wenchang(self, chart: BaZiChart) -> List[str]:
        """计算文昌"""
        # 年干查文昌
        wenchang_map = {
            "甲": "巳", "乙": "午", "丙": "申", "丁": "酉", "戊": "申",
            "己": "酉", "庚": "亥", "辛": "子", "壬": "寅", "癸": "卯"
        }
        
        wenchang_zhi = wenchang_map.get(chart.year.gan, "")
        
        result = []
        for pillar in chart.get_pillars_list():
            if pillar.zhi == wenchang_zhi:
                result.append(f"{pillar.ganzhi}宫")
        
        return result
    
    def _calc_tiande(self, chart: BaZiChart) -> List[str]:
        """计算天德"""
        # 月支查天德
        tiande_map = {
            "寅": "丁", "卯": "申", "辰": "壬", "巳": "辛",
            "午": "亥", "未": "甲", "申": "癸", "酉": "寅",
            "戌": "丙", "亥": "乙", "子": "巳", "丑": "庚"
        }
        
        month_zhi = chart.month.zhi
        tiande = tiande_map.get(month_zhi, "")
        
        result = []
        for pillar in chart.get_pillars_list():
            if pillar.gan == tiande or pillar.zhi == tiande:
                result.append(f"{pillar.ganzhi}宫")
        
        return result
    
    def _calc_yuede(self, chart: BaZiChart) -> List[str]:
        """计算月德"""
        # 月支查月德
        yuede_map = {
            "寅": "丙", "卯": "甲", "辰": "壬", "巳": "庚",
            "午": "丙", "未": "甲", "申": "壬", "酉": "庚",
            "戌": "丙", "亥": "甲", "子": "壬", "丑": "庚"
        }
        
        month_zhi = chart.month.zhi
        yuede = yuede_map.get(month_zhi, "")
        
        result = []
        for pillar in chart.get_pillars_list():
            if pillar.gan == yuede:
                result.append(f"{pillar.ganzhi}宫")
        
        return result


class ZiWeiCalculator:
    """紫微斗数计算器"""
    
    def calculate(self, birth_dt: datetime, longitude: float,
                 latitude: float, gender: str = "男") -> ZiWeiChart:
        """
        计算紫微斗数命盘
        
        Args:
            birth_dt: 出生时间
            longitude: 经度
            latitude: 纬度
            gender: 性别
            
        Returns:
            紫微斗数命盘
        """
        # 转换为真太阳时
        true_solar_dt = true_solar_time(birth_dt, longitude)
        
        # 计算农历
        lunar_date = self._solar_to_lunar(true_solar_dt)
        
        # 定命宫
        ming_gong_pos = self._calculate_ming_gong(lunar_date, true_solar_dt)
        
        # 定身宫
        shen_gong_pos = self._calculate_shen_gong(lunar_date, true_solar_dt)
        
        # 起十二宫
        gongs = self._setup_twelve_gongs(ming_gong_pos, true_solar_dt)
        
        # 安十四主星
        self._place_main_stars(gongs, lunar_date)
        
        # 安辅星
        self._place_auxiliary_stars(gongs, lunar_date, true_solar_dt)
        
        # 定四化
        self._calculate_sihua(gongs, true_solar_dt)
        
        # 创建命盘
        chart = ZiWeiChart(
            gongs=gongs,
            ming_gong_position=ming_gong_pos,
            shen_gong_position=shen_gong_pos,
            birth_datetime=birth_dt,
            longitude=longitude,
            latitude=latitude,
            gender=gender
        )
        
        # 计算大限
        self._calculate_daxian(chart)
        
        logger.info("紫微斗数计算完成",
                   ming_gong=f"{gongs['命宫'].ganzhi}",
                   ming_stars=gongs['命宫'].zhuxing)
        
        return chart
    
    def _solar_to_lunar(self, dt: datetime) -> Dict[str, int]:
        """公历转农历（简化版）"""
        # 实际应使用精确的农历转换算法
        # 这里简化处理
        return {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "is_leap": False
        }
    
    def _calculate_ming_gong(self, lunar_date: Dict[str, int], dt: datetime) -> int:
        """定命宫"""
        # 寅宫起正月，顺数至生月，再逆数至生时
        month = lunar_date["month"]
        hour = dt.hour
        
        # 时辰地支
        hour_zhi_idx = hour // 2
        
        # 命宫位置（0-11，0=子位）
        ming_pos = (14 - month - hour_zhi_idx) % 12
        
        return ming_pos
    
    def _calculate_shen_gong(self, lunar_date: Dict[str, int], dt: datetime) -> int:
        """定身宫"""
        # 身宫：由生月和生时决定
        month = lunar_date["month"]
        hour = dt.hour
        hour_zhi_idx = hour // 2
        
        # 身宫规则（简化版）
        shen_rules = {
            (2, 0): 2,   # 子时二月生，身在寅
            (3, 2): 5,   # 寅时三月生，身在巳
            # ... 完整规则表
        }
        
        # 默认规则
        shen_pos = (month + hour_zhi_idx) % 12
        
        return shen_pos
    
    def _setup_twelve_gongs(self, ming_gong_pos: int, dt: datetime) -> Dict[str, ZiWeiGong]:
        """起十二宫"""
        gongs = {}
        
        # 计算年干支（用于宫位天干）
        year = dt.year
        year_offset = year - 1984
        year_gan_idx = year_offset % 10
        year_zhi_idx = year_offset % 12
        
        # 十二宫顺序
        gong_order = ["命宫", "父母", "福德", "田宅", "官禄", "交友",
                     "迁移", "疾厄", "财帛", "子女", "夫妻", "兄弟"]
        
        for i, name in enumerate(gong_order):
            pos = (ming_gong_pos + i) % 12
            
            # 宫位天干（寅宫起甲）
            gan_idx = (pos + 2) % 10
            gan = TIANGAN[gan_idx]
            
            # 宫位地支
            zhi = DIZHI[pos]
            
            gongs[name] = ZiWeiGong(
                name=name,
                position=pos,
                gan=gan,
                zhi=zhi
            )
        
        return gongs
    
    def _place_main_stars(self, gongs: Dict[str, ZiWeiGong], lunar_date: Dict[str, int]):
        """安十四主星（简化版）"""
        # 紫微星系
        # 根据农历日数和五行局数起紫微星
        day = lunar_date["day"]
        
        # 简化：根据日数定紫微星位置
        ziwei_pos = (day * 2) % 12
        
        # 紫微星系顺布
        ziwei_system = ["紫微", "天机", "太阳", "武曲", "天同", "廉贞"]
        for i, star in enumerate(ziwei_system):
            pos = (ziwei_pos + i * 2) % 12
            for gong in gongs.values():
                if gong.position == pos:
                    gong.zhuxing.append(star)
        
        # 天府星系
        tianfu_pos = (12 - ziwei_pos + 4) % 12
        tianfu_system = ["天府", "太阴", "贪狼", "巨门", "天相", "天梁", "七杀", "破军"]
        
        # 天府星系的排布规则更复杂，这里简化
        for i, star in enumerate(tianfu_system):
            pos = (tianfu_pos + i) % 12
            for gong in gongs.values():
                if gong.position == pos and len(gong.zhuxing) < 2:
                    gong.zhuxing.append(star)
    
    def _place_auxiliary_stars(self, gongs: Dict[str, ZiWeiGong], 
                             lunar_date: Dict[str, int], dt: datetime):
        """安辅星"""
        # 文昌文曲
        hour = dt.hour
        wenchang_pos = (10 - hour // 2) % 12
        wenqu_pos = (4 + hour // 2) % 12
        
        for gong in gongs.values():
            if gong.position == wenchang_pos:
                gong.fuxing.append("文昌")
            if gong.position == wenqu_pos:
                gong.fuxing.append("文曲")
        
        # 左辅右弼（根据月份）
        month = lunar_date["month"]
        zuofu_pos = (month + 3) % 12
        youbi_pos = (11 - month) % 12
        
        for gong in gongs.values():
            if gong.position == zuofu_pos:
                gong.fuxing.append("左辅")
            if gong.position == youbi_pos:
                gong.fuxing.append("右弼")
        
        # 其他辅星（天魁天钺、禄存、擎羊陀罗等）
        # 简化处理...
    
    def _calculate_sihua(self, gongs: Dict[str, ZiWeiGong], dt: datetime):
        """定四化"""
        # 根据年干定四化
        year = dt.year
        year_offset = year - 1984
        year_gan_idx = year_offset % 10
        year_gan = TIANGAN[year_gan_idx]
        
        # 查四化表
        if year_gan in SIHUA_TIANGAN:
            sihua = SIHUA_TIANGAN[year_gan]
            
            # 找到对应星曜所在宫位，加入四化
            for gong in gongs.values():
                for hua_type, star_name in sihua.items():
                    if star_name in gong.zhuxing or star_name in gong.fuxing:
                        gong.sihua[hua_type] = star_name
    
    def _calculate_daxian(self, chart: ZiWeiChart):
        """计算大限"""
        # 阳男阴女顺行，阴男阳女逆行
        is_yang = chart.birth_datetime.year % 2 == 0
        is_male = chart.gender == "男"
        is_forward = (is_yang and is_male) or (not is_yang and not is_male)
        
        # 从命宫起大限
        start_pos = chart.ming_gong_position
        daxian_list = []
        
        for i in range(12):
            if is_forward:
                pos = (start_pos + i) % 12
            else:
                pos = (start_pos - i) % 12
            
            # 找到对应宫位
            for gong in chart.gongs.values():
                if gong.position == pos:
                    start_age = i * 10 + 1
                    end_age = (i + 1) * 10
                    daxian_list.append((gong.name, start_age, end_age))
                    break
        
        chart.daxian_list = daxian_list


# 便捷函数
def calculate_bazi(birth_dt: datetime, longitude: float, 
                  latitude: float, gender: str = "男") -> BaZiChart:
    """计算四柱八字的便捷函数"""
    calculator = BaZiCalculator()
    return calculator.calculate(birth_dt, longitude, latitude, gender)


def calculate_ziwei(birth_dt: datetime, longitude: float,
                   latitude: float, gender: str = "男") -> ZiWeiChart:
    """计算紫微斗数的便捷函数"""
    calculator = ZiWeiCalculator()
    return calculator.calculate(birth_dt, longitude, latitude, gender)


if __name__ == "__main__":
    # 测试
    from datetime import timezone
    
    # 测试数据
    birth_dt = datetime(1990, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    longitude = 116.4074  # 北京
    latitude = 39.9042
    
    # 计算八字
    print("=== 四柱八字 ===")
    bazi = calculate_bazi(birth_dt, longitude, latitude, "男")
    print(f"四柱：{bazi.year.ganzhi} {bazi.month.ganzhi} {bazi.day.ganzhi} {bazi.hour.ganzhi}")
    print(f"日主：{bazi.get_rizhu_gan()}")
    print(f"大运：")
    for gz, age in bazi.dayun_list[:4]:
        print(f"  {age}岁起：{gz}")
    print(f"神煞：{bazi.shensha}")
    
    # 计算紫微
    print("\n=== 紫微斗数 ===")
    ziwei = calculate_ziwei(birth_dt, longitude, latitude, "男")
    ming_gong = ziwei.get_ming_gong()
    print(f"命宫：{ming_gong.ganzhi}")
    print(f"命宫主星：{ming_gong.zhuxing}")
    print(f"命宫辅星：{ming_gong.fuxing}")
    print(f"四化：{ming_gong.sihua}")
    
    print("\n十二宫概览：")
    for name, gong in list(ziwei.gongs.items())[:6]:
        stars = gong.zhuxing + gong.fuxing
        print(f"  {name}（{gong.ganzhi}）：{' '.join(stars)}")