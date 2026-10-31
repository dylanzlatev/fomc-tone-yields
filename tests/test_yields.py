"""Unit tests for ftyields.data.yields (pure functions, synthetic inputs, no network)."""
import io

import numpy as np
import pandas as pd
import pytest

from ftyields.data import yields as Y

CFG = {
    "sample": {"start": "2000-01-01", "end": "2025-12-31"},
    "fred": {"series": {"DGS2": "2Y", "DGS5": "5Y", "DGS10": "10Y", "DGS30": "30Y"}},
    "factors": {
        "level": {"tenors": ["2Y", "5Y", "10Y", "30Y"]},
        "slope": {"long": "30Y", "short": "2Y"},
        "curvature": {"belly": "5Y", "short": "2Y", "long": "30Y"},
    },
}


def test_tenor_column():
    assert Y.tenor_column("10Y") == "y_10y"
    assert Y.tenor_column("1M") == "y_1m"
    assert Y.tenor_column("2Y", "gsw_") == "gsw_2y"


def test_parse_fred_csv_missing_values():
    txt = "observation_date,DGS10\n2020-01-01,\n2020-01-02,1.88\n2020-01-03,.\n2020-01-06,1.80\n"
    s = Y.parse_fred_csv(io.StringIO(txt))
    assert s.name == "DGS10"
    assert s.index.name == "date"
    assert s.dtype == float
    assert s.isna().tolist() == [True, False, True, False]
    assert s["2020-01-06"] == pytest.approx(1.80)


def test_parse_fred_csv_old_header_and_sorting():
    txt = "DATE,DGS2\n2020-01-03,1.5\n2020-01-02,1.6\n"
    s = Y.parse_fred_csv(io.StringIO(txt))
    assert list(s.index) == sorted(s.index)
    assert s.iloc[0] == pytest.approx(1.6)


def test_pct_to_bp():
    s = pd.Series([1.88, 0.07, 4.355])
    out = Y.pct_to_bp(s)
    assert out.tolist() == [188.0, 7.0, 435.5]
    # no float noise such as 187.99999999999997
    assert Y.pct_to_bp(pd.Series([1.15])).iloc[0] == 115.0


def test_compute_factors_formulas():
    df = pd.DataFrame({"y_2y": [100.0], "y_5y": [150.0], "y_10y": [200.0], "y_30y": [300.0]})
    f = Y.compute_factors(df, CFG)
    assert f["level"].iloc[0] == pytest.approx((100 + 150 + 200 + 300) / 4)
    assert f["slope"].iloc[0] == pytest.approx(300 - 100)
    assert f["curvature"].iloc[0] == pytest.approx(2 * 150 - 100 - 300)


def test_compute_factors_nan_propagation():
    df = pd.DataFrame({
        "y_2y": [100.0, 100.0, 100.0],
        "y_5y": [150.0, np.nan, 150.0],
        "y_10y": [200.0, 200.0, 200.0],
        "y_30y": [np.nan, 300.0, 300.0],
    })
    f = Y.compute_factors(df, CFG)
    # row 0: 30Y missing -> all three factors NaN (level uses 30Y, slope, curvature too)
    assert f.iloc[0].isna().all()
    # row 1: 5Y missing -> level and curvature NaN, slope fine
    assert np.isnan(f["level"].iloc[1]) and np.isnan(f["curvature"].iloc[1])
    assert f["slope"].iloc[1] == pytest.approx(200.0)
    assert f.iloc[2].notna().all()


def test_compute_factors_10y_missing_only_hits_level():
    df = pd.DataFrame({"y_2y": [100.0], "y_5y": [150.0], "y_10y": [np.nan], "y_30y": [300.0]})
    f = Y.compute_factors(df, CFG)
    assert np.isnan(f["level"].iloc[0])
    assert f["slope"].notna().iloc[0] and f["curvature"].notna().iloc[0]


def test_trading_days_uses_dgs10_and_window():
    idx = pd.to_datetime(["1999-12-31", "2000-01-03", "2000-01-04", "2000-01-05", "2026-01-02"])
    s = pd.Series([6.0, 6.5, np.nan, 6.6, 4.0], index=idx)
    days = Y.trading_days(s, "2000-01-01", "2025-12-31")
    assert list(days) == [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-05")]


def test_build_cmt_no_fill_and_trading_day_rows():
    d = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"])
    series = {
        "DGS10": pd.Series([6.0, 6.1, np.nan, 6.2], index=d),
        "DGS2": pd.Series([5.0, np.nan, 5.1, 5.2], index=d),
        "DGS5": pd.Series([5.5, 5.6, 5.7, 5.8], index=d),
        "DGS30": pd.Series([np.nan, np.nan, 6.5, 6.6], index=d),
    }
    out = Y.build_cmt(series, CFG)
    # 2000-01-05 dropped (DGS10 missing); no forward fill of DGS2 on 2000-01-04
    assert list(out.index) == [d[0], d[1], d[3]]
    assert np.isnan(out.loc[d[1], "y_2y"])
    assert out.loc[d[0], "y_10y"] == 600.0
    assert out["y_30y"].isna().tolist() == [True, True, False]
    assert out[["level", "slope", "curvature"]].iloc[:2].isna().all().all()
    assert out.loc[d[3], "slope"] == pytest.approx(660.0 - 520.0)


GSW_TXT = """Gurkaynak, Sack, and Wright (2006)
Last Updated: 2026-01-01
Source: description line
Notes: another line

Some more description
,,,
Series Descriptions
SVENY02: zero-coupon yield, 2 years
Date,BETA0,SVENY02,SVENY05,SVENY10,SVENY30
2000-01-05,1,6.1,6.2,6.3,NA
2000-01-04,1,6.0,6.1,6.2,6.4
2000-01-03,1,,6.0,6.1,6.3
"""


def test_parse_gsw_csv_finds_header_and_sorts():
    g = Y.parse_gsw_csv(io.StringIO(GSW_TXT))
    assert list(g.index) == sorted(g.index)
    assert g.index[0] == pd.Timestamp("2000-01-03")
    assert np.isnan(g.loc["2000-01-03", "SVENY02"])
    assert np.isnan(g.loc["2000-01-05", "SVENY30"])
    assert g.loc["2000-01-04", "SVENY10"] == pytest.approx(6.2)


def test_parse_gsw_csv_without_header_raises():
    with pytest.raises(ValueError):
        Y.parse_gsw_csv(io.StringIO("a,b\n1,2\n"))


def test_build_gsw_reindex_bp_and_factors():
    g = Y.parse_gsw_csv(io.StringIO(GSW_TXT))
    days = pd.DatetimeIndex(pd.to_datetime(["2000-01-04", "2000-01-05", "2000-01-06"]))
    out = Y.build_gsw(g, days, CFG)
    assert list(out.columns) == ["gsw_2y", "gsw_5y", "gsw_10y", "gsw_30y", "gsw_level",
                                 "gsw_slope", "gsw_curvature"]
    assert out.loc[days[0], "gsw_10y"] == 620.0
    assert out.loc[days[0], "gsw_slope"] == pytest.approx(640.0 - 600.0)
    # 2000-01-05: 30Y NA -> factors NaN; 2000-01-06 absent from file -> all NaN, no fill
    assert out.loc[days[1]].filter(like="gsw_l").isna().all()
    assert out.loc[days[2]].isna().all()
