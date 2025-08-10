from __future__ import annotations

from datetime import datetime, timezone

from h_pulse.utils.time_astro import convert_times, asc_mc_true_solar


def test_convert_times_delta_t_nonzero():
    ts = convert_times(datetime(2025, 8, 10, 12, 0, tzinfo=timezone.utc))
    assert ts.delta_t_seconds > 50
    assert "T" in ts.tt_iso or "T" in ts.tdb_iso or "+" in ts.tt_iso


def test_asc_mc_return_range():
    asc, mc = asc_mc_true_solar(datetime(2025, 8, 10, 12, 0, tzinfo=timezone.utc), 35.68, 139.76)
    assert 0 <= asc < 360
    assert 0 <= mc < 360