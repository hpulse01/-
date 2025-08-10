from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import math

HEAVENLY_STEMS = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
EARTHLY_BRANCHES = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

ZHI_WEI_MAIN_STARS = [
    "紫微","天机","太阳","武曲","天同","廉贞","天府","太阴","贪狼","巨门","天相","天梁","七杀","破军"
]
FOUR_TRANSFORM = ["化禄","化权","化科","化忌"]


@dataclass
class FourPillars:
    year: Tuple[str, str]
    month: Tuple[str, str]
    day: Tuple[str, str]
    hour: Tuple[str, str]


@dataclass
class EasternFeatures:
    pillars: FourPillars
    da_yun: List[Tuple[int, Tuple[str, str]]]
    liu_nian: List[Tuple[int, Tuple[str, str]]]
    zi_wei: Dict[str, int]
    shen_gong: int
    limited_fortunes: List[Dict]


def sexagenary_index(offset: int) -> Tuple[str, str]:
    stem = HEAVENLY_STEMS[offset % 10]
    branch = EARTHLY_BRANCHES[offset % 12]
    return stem, branch


def compute_four_pillars(birth_dt: datetime) -> FourPillars:
    """计算真·四柱（年、月、日、时）。

    使用简化的索引算法：以 1984-02-02（甲子日）为参考，推算干支。
    这不是民俗排盘的全部细节，但算法完整可复现。
    """
    if birth_dt.tzinfo is None:
        birth_dt = birth_dt.replace(tzinfo=timezone.utc)
    base = datetime(1984, 2, 2, tzinfo=timezone.utc)
    days = (birth_dt.date() - base.date()).days

    # Day pillar
    day_sixty = (days % 60 + 60) % 60
    day_pillar = sexagenary_index(day_sixty)

    # Year pillar approximate by solar year start at Li Chun (~ Feb 4), simplified
    year = birth_dt.year
    lichun = datetime(year, 2, 4, tzinfo=timezone.utc)
    if birth_dt < lichun:
        year -= 1
    year_sixty = (year - 1984) % 60
    year_pillar = sexagenary_index(year_sixty)

    # Month pillar: index from year stem, simple mapping
    month_index = ((birth_dt.month + 10) % 12)
    month_sixty = (year_sixty * 12 + month_index) % 60
    month_pillar = sexagenary_index(month_sixty)

    # Hour pillar: each branch = 2 hours from 23:00, compute branch index
    hour_branch_index = ((birth_dt.hour + 1) // 2) % 12
    hour_sixty = (day_sixty * 12 + hour_branch_index) % 60
    hour_pillar = sexagenary_index(hour_sixty)

    return FourPillars(year=year_pillar, month=month_pillar, day=day_pillar, hour=hour_pillar)


def compute_da_yun(birth_dt: datetime, pillars: FourPillars, num_cycles: int = 8) -> List[Tuple[int, Tuple[str, str]]]:
    """大运：每 10 年一大运，方向根据日主阴阳简化确定（此处固定顺行）。"""
    start_age = 6  # 简化起运年龄
    results: List[Tuple[int, Tuple[str, str]]] = []
    base_index = (HEAVENLY_STEMS.index(pillars.month[0]) - HEAVENLY_STEMS.index(pillars.year[0])) % 10
    for i in range(num_cycles):
        age = start_age + 10 * i
        idx = (base_index + i) % 60
        results.append((age, sexagenary_index(idx)))
    return results


def compute_liu_nian(birth_year: int, span: int = 20) -> List[Tuple[int, Tuple[str, str]]]:
    """流年：从出生年起，返回未来 span 年的年柱。"""
    start_sixty = (birth_year - 1984) % 60
    return [(birth_year + i, sexagenary_index((start_sixty + i) % 60)) for i in range(span)]


def compute_zi_wei(birth_dt: datetime) -> Tuple[Dict[str, int], int]:
    """紫微斗数主星定位（十二宫位索引 0..11），身宫推断。

    采用太阳位置简化映射：将一年映射到 12 宫，保持确定性。
    """
    day_of_year = birth_dt.timetuple().tm_yday
    palace = (day_of_year - 1) // 30
    stars = {name: (palace + i) % 12 for i, name in enumerate(ZHI_WEI_MAIN_STARS)}
    shen_gong = (palace + 4) % 12
    return stars, shen_gong


def compute_limited_fortunes(birth_dt: datetime) -> List[Dict]:
    """限运：在未来 5 年内，按季度生成占断条目。"""
    items: List[Dict] = []
    base = datetime(birth_dt.year, ((birth_dt.month - 1) // 3) * 3 + 1, 1, tzinfo=birth_dt.tzinfo)
    for i in range(20):
        start = base + timedelta(days=90 * i)
        level = ["低","中","高"][i % 3]
        items.append({
            "start": start.isoformat(),
            "end": (start + timedelta(days=90)).isoformat(),
            "fortune": level,
            "confidence": round(0.6 + 0.02 * (i % 5), 2),
        })
    return items


def collect_eastern_features(birth_dt: datetime) -> EasternFeatures:
    """东方体系特征集合。

    Returns
    -------
    EasternFeatures
        包含四柱、大运、流年、紫微主星/身宫、限运列表。
    """
    pillars = compute_four_pillars(birth_dt)
    da_yun = compute_da_yun(birth_dt, pillars)
    liu_nian = compute_liu_nian(birth_dt.year, 20)
    zi_wei, shen = compute_zi_wei(birth_dt)
    limited = compute_limited_fortunes(birth_dt)
    return EasternFeatures(pillars=pillars, da_yun=da_yun, liu_nian=liu_nian, zi_wei=zi_wei, shen_gong=shen, limited_fortunes=limited)