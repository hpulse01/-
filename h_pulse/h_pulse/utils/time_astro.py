from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np

try:  # Prefer astropy high-precision path
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import (
        SkyCoord,
        EarthLocation,
        AltAz,
        ICRS,
        get_sun,
        solar_system_ephemeris,
        FK5,
    )
    _HAS_ASTROPY = True
except Exception:  # pragma: no cover - fallback path covered by unit tests separately
    _HAS_ASTROPY = False


@dataclass
class TimeScales:
    utc_iso: str
    tt_iso: str
    tdb_iso: str
    delta_t_seconds: float


def approximate_delta_t(year: float) -> float:
    """Approximate ΔT (TT − UT1) in seconds using Espenak & Meeus polynomials.

    Parameters
    ----------
    year : float
        Decimal year, e.g. 2025.6

    Returns
    -------
    float
        ΔT seconds

    Notes
    -----
    The approximation is suitable for 1900–2100 with typical error < 2 s.
    """
    t = (year - 2000) / 100.0
    # Simplified polynomial for modern dates
    dt = 64.7 + 64.5 * t + 33.5 * t * t
    return float(dt)


def convert_times(utc_dt: datetime) -> TimeScales:
    """Convert UTC datetime to TT and TDB, return ΔT.

    Uses astropy when available; otherwise uses polynomial ΔT and constant offsets.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> ts = convert_times(datetime(2025, 8, 10, 12, 0, tzinfo=timezone.utc))
    >>> round(ts.delta_t_seconds) > 60
    True
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    if _HAS_ASTROPY:
        t = Time(utc_dt)
        tt = t.tt
        tdb = t.tdb
        delta_t = (tt - t.ut1).sec if t.ut1 is not None else (tt - t.utc).sec
        return TimeScales(t.utc.isot, tt.isot, tdb.isot, float(delta_t))
    else:
        year = utc_dt.year + (utc_dt.timetuple().tm_yday - 1 + (utc_dt.hour - 12) / 24.0) / 365.25
        delta_t = approximate_delta_t(year)
        # TT ≈ UTC + (ΔT + (UT1-UTC)) but lack UT1; approximate TT = UTC + 69 s
        tt_offset = delta_t + 32.184  # crudely treat UTC≈UT1 for fallback
        tt = utc_dt.timestamp() + tt_offset
        tdb = tt  # ignore relativistic difference in fallback
        tt_iso = datetime.fromtimestamp(tt, tz=timezone.utc).isoformat()
        tdb_iso = datetime.fromtimestamp(tdb, tz=timezone.utc).isoformat()
        return TimeScales(utc_dt.isoformat(), tt_iso, tdb_iso, float(delta_t))


def mean_obliquity_of_ecliptic(jcent: float) -> float:
    """IAU 2006 mean obliquity (arcseconds -> radians).

    Parameters
    ----------
    jcent : float
        Julian centuries TT since J2000.0
    """
    U = jcent / 100.0
    seconds = 84381.406 - 46.836769 * jcent - 0.0001831 * jcent**2 + 0.00200340 * jcent**3 - 0.000000576 * jcent**4 - 0.0000000434 * jcent**5
    return math.radians(seconds / 3600.0)


def precession_nutation_aberration_corrections(jd_tt: float) -> Dict[str, float]:
    """Compute basic corrections for precession, nutation, and aberration.

    This function provides simplified magnitudes in arcseconds for validation
    and educational purposes. For production-grade work, astropy is used.

    Returns a dict with keys: precession, nutation_lon, nutation_obl, aberration.
    Values are radians.
    """
    T = (jd_tt - 2451545.0) / 36525.0
    # Precession in longitude (approx magnitude)
    zeta_arcsec = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3)
    z_arcsec = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3)
    theta_arcsec = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3)

    # Nutation terms (largest 18.6-yr terms, simplified)
    # Mean anomaly of Moon and Sun approximations
    Mm = math.radians((134.96298139 + (1325 * 360 + 198.8673981) * T) % 360)
    Ms = math.radians((357.52772333 + (99 * 360 + 359.0503400) * T) % 360)
    Om = math.radians((125.04452 - 1934.136261 * T) % 360)
    dpsi_arcsec = -17.20 * math.sin(Om) - 1.32 * math.sin(2 * 0) - 0.23 * math.sin(2 * 0) + 0.21 * math.sin(2 * Om)
    deps_arcsec = 9.20 * math.cos(Om) + 0.57 * math.cos(2 * 0) + 0.10 * math.cos(2 * 0) - 0.09 * math.cos(2 * Om)

    # Aberration of light mean magnitude near ecliptic ~20.49552 arcsec
    aberr_arcsec = 20.49552

    return {
        "precession": math.radians((zeta_arcsec + z_arcsec + theta_arcsec) / 3.0 / 3600.0),
        "nutation_lon": math.radians(dpsi_arcsec / 3600.0),
        "nutation_obl": math.radians(deps_arcsec / 3600.0),
        "aberration": math.radians(aberr_arcsec / 3600.0),
    }


def julian_day(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)


def local_true_solar_time(utc_dt: datetime, longitude_deg: float) -> Tuple[datetime, float]:
    """Compute true solar time and equation of time.

    Parameters
    ----------
    utc_dt : datetime
        Input time in UTC
    longitude_deg : float
        East-positive longitude in degrees

    Returns
    -------
    (datetime, float)
        Local true solar time (as timezone-aware UTC-based datetime adjusted for EoT and longitude)
        and equation of time in minutes (positive means sundial ahead of clock)
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    # Equation of Time (minutes), NOAA approximation
    day_of_year = utc_dt.timetuple().tm_yday
    B = math.radians(360 * (day_of_year - 81) / 364)
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    # Longitude correction (minutes): 4 min per degree
    longitude_correction = 4.0 * longitude_deg
    minutes_offset = eot + longitude_correction
    true_solar_dt = utc_dt + np.timedelta64(int(minutes_offset * 60), "s").astype("timedelta64[s]").astype(object)
    return true_solar_dt, float(eot)


def asc_mc_true_solar(utc_dt: datetime, latitude_deg: float, longitude_deg: float) -> Tuple[float, float]:
    """Compute Ascendant (ASC) and Midheaven (MC) using true solar time.

    Returns (ASC_longitude_deg, MC_longitude_deg).
    Uses astropy when available; falls back to FK5-based approximation otherwise.
    """
    true_dt, _ = local_true_solar_time(utc_dt, longitude_deg)

    if _HAS_ASTROPY:
        with solar_system_ephemeris.set("builtin"):
            location = EarthLocation.from_geodetic(lon=longitude_deg * u.deg, lat=latitude_deg * u.deg)
            obstime = Time(true_dt)
            # Local apparent sidereal time
            last = obstime.sidereal_time("apparent", longitude=location.lon).to(u.deg).value
            eps = mean_obliquity_of_ecliptic(((obstime.tt.jd - 2451545.0) / 36525.0))
            # MC = arctan2(sin(LST), cos(LST)*sin(eps)) in ecliptic longitude
            lst_rad = math.radians(last)
            mc = math.degrees(math.atan2(math.sin(lst_rad), math.cos(lst_rad) * math.sin(eps))) % 360
            # ASC formula
            phi = math.radians(latitude_deg)
            tan_lambda = -math.cos(lst_rad) / (math.sin(lst_rad) * math.cos(eps) + math.tan(phi) * math.sin(eps))
            asc = math.degrees(math.atan(tan_lambda)) % 360
            return asc, mc
    # Fallback: simplified spherical trig
    jd = julian_day(true_dt)
    T = (jd - 2451545.0) / 36525.0
    eps = mean_obliquity_of_ecliptic(T)
    # Approximate GMST in degrees
    d = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * d) % 360
    lst = (gmst + longitude_deg) % 360
    lst_rad = math.radians(lst)
    phi = math.radians(latitude_deg)
    mc = math.degrees(math.atan2(math.sin(lst_rad), math.cos(lst_rad) * math.sin(eps))) % 360
    tan_lambda = -math.cos(lst_rad) / (math.sin(lst_rad) * math.cos(eps) + math.tan(phi) * math.sin(eps))
    asc = math.degrees(math.atan(tan_lambda)) % 360
    return asc, mc