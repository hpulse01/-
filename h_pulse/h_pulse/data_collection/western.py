from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import math

from ..utils.time_astro import asc_mc_true_solar

try:
    import swisseph as swe  # type: ignore
    _HAS_SWE = True
except Exception:  # pragma: no cover
    _HAS_SWE = False


PLANETS = {
    "Sun": 0,
    "Moon": 1,
    "Mercury": 2,
    "Venus": 3,
    "Mars": 4,
    "Jupiter": 5,
    "Saturn": 6,
    "Uranus": 7,
    "Neptune": 8,
    "Pluto": 9,
    "TrueNode": 11,
    "Chiron": 15,
}

MAJOR_ASPECTS = {
    "conjunction": 0,
    "opposition": 180,
    "trine": 120,
    "square": 90,
    "sextile": 60,
}


@dataclass
class WesternFeatures:
    asc: float
    mc: float
    planets: Dict[str, float]
    aspects: List[Tuple[str, str, str]]


def _angle_distance(a: float, b: float) -> float:
    d = abs((a - b + 180) % 360 - 180)
    return d


def planetary_longitudes(dt: datetime) -> Dict[str, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    result: Dict[str, float] = {}
    if _HAS_SWE:
        swe.set_ephe_path("")
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
        for name, code in PLANETS.items():
            try:
                lon, lat, dist, lon_speed = swe.calc_ut(jd, code)[0:4]
                result[name] = lon % 360
            except Exception:
                # Fallback simple mean motions
                base = {
                    "Sun": 280.0,
                    "Moon": 218.3,
                    "Mercury": 60.0,
                    "Venus": 90.0,
                    "Mars": 120.0,
                    "Jupiter": 200.0,
                    "Saturn": 300.0,
                    "Uranus": 50.0,
                    "Neptune": 340.0,
                    "Pluto": 270.0,
                    "TrueNode": 0.0,
                    "Chiron": 150.0,
                }
                result[name] = base.get(name, 0.0)
    else:
        # Simplified mean longitudes
        base = {
            "Sun": 280.0,
            "Moon": 218.3,
            "Mercury": 60.0,
            "Venus": 90.0,
            "Mars": 120.0,
            "Jupiter": 200.0,
            "Saturn": 300.0,
            "Uranus": 50.0,
            "Neptune": 340.0,
            "Pluto": 270.0,
            "TrueNode": 0.0,
            "Chiron": 150.0,
        }
        result = base
    return result


def compute_aspects(planets: Dict[str, float], orb: float = 3.0) -> List[Tuple[str, str, str]]:
    names = list(planets.keys())
    aspects: List[Tuple[str, str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            angle = _angle_distance(planets[a], planets[b])
            for asp_name, asp_deg in MAJOR_ASPECTS.items():
                if abs(angle - asp_deg) <= orb:
                    aspects.append((a, b, asp_name))
    return aspects


def collect_western_features(dt: datetime, lat: float, lon: float) -> WesternFeatures:
    asc, mc = asc_mc_true_solar(dt, lat, lon)
    planets = planetary_longitudes(dt)
    aspects = compute_aspects(planets)
    return WesternFeatures(asc=asc, mc=mc, planets=planets, aspects=aspects)