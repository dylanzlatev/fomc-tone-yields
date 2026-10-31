"""Unit tests for ftyields.data.build (synthetic calendar and yields, no network)."""
import numpy as np
import pandas as pd
import pytest

from ftyields.data import build as B

CFG = {
    "events": {"yield_close_time_et": "15:30"},
    "fred": {"series_breaks": {"y_30y": [pd.Timestamp("2002-02-19").date(),
                                         pd.Timestamp("2006-02-09").date()]}},
    "factors": {
        "level": {"tenors": ["2Y", "5Y", "10Y", "30Y"]},
        "slope": {"long": "30Y", "short": "2Y"},
        "curvature": {"belly": "5Y", "short": "2Y", "long": "30Y"},
    },
}

# Business days 2002-02-11 .. 2002-02-27 minus Presidents' Day (2002-02-18).
DAYS = pd.bdate_range("2002-02-11", "2002-02-27").drop(pd.Timestamp("2002-02-18"))


def make_cmt():
    n = len(DAYS)
    df = pd.DataFrame({"y_2y": np.arange(n) * 1.0, "y_10y": np.arange(n) * 2.0,
                       "y_5y": np.arange(n) * 3.0, "y_30y": np.arange(n) * 10.0}, index=DAYS)
    df.index.name = "date"
    return df


def D(s):
    return pd.Timestamp(s)


# ---------------------------------------------------------------- change / breaks
def test_change_plain():
    df = make_cmt()
    assert B.change(df, "y_2y", D("2002-02-12"), D("2002-02-13"), {}) == 1.0
    assert B.change(df, "y_10y", D("2002-02-12"), D("2002-02-14"), {}) == 4.0


def test_break_inside_window_is_nan():
    df = make_cmt()
    brk = B.build_breaks(CFG)
    # break 2002-02-19 inside (02-15, 02-20]
    assert np.isnan(B.change(df, "y_30y", D("2002-02-15"), D("2002-02-20"), brk))


def test_break_at_prev_day_not_nan():
    df = make_cmt()
    brk = B.build_breaks(CFG)
    # break at start_day: already in the starting level -> clean
    assert B.change(df, "y_30y", D("2002-02-19"), D("2002-02-20"), brk) == 10.0


def test_break_at_event_day_is_nan():
    df = make_cmt()
    brk = B.build_breaks(CFG)
    assert np.isnan(B.change(df, "y_30y", D("2002-02-15"), D("2002-02-19"), brk))


def test_break_only_affects_listed_columns():
    df = make_cmt()
    brk = B.build_breaks(CFG)
    assert B.change(df, "y_10y", D("2002-02-15"), D("2002-02-20"), brk) == 4.0


def test_change_vectorised_and_missing_endpoint():
    df = make_cmt()
    brk = B.build_breaks(CFG)
    s = np.array([D("2002-02-14"), D("2002-02-15"), D("2002-02-16")], dtype="datetime64[ns]")
    e = np.array([D("2002-02-15"), D("2002-02-19"), D("2002-02-19")], dtype="datetime64[ns]")
    out = B.change(df, "y_30y", s, e, brk)
    assert out[0] == 10.0 and np.isnan(out[1]) and np.isnan(out[2])  # 02-16 is a Saturday
    assert np.isnan(B.change(df, "y_2y", pd.NaT, D("2002-02-19"), brk))


def test_factor_break_inheritance():
    brk = B.build_breaks(CFG)
    # all three factors use 30Y
    for f in ("level", "slope", "curvature"):
        assert brk[f] == [D("2002-02-19"), D("2006-02-09")]
    assert "y_2y" not in brk and "gsw_level" not in brk
    df = make_cmt()
    df["level"], df["slope"] = 1.0, 2.0
    assert np.isnan(B.change(df, "slope", D("2002-02-15"), D("2002-02-20"), brk))
    assert B.change(df, "slope", D("2002-02-19"), D("2002-02-20"), brk) == 0.0


def test_factor_inherits_only_from_used_tenors():
    cfg = {**CFG, "fred": {"series_breaks": {"y_5y": [pd.Timestamp("2002-02-19").date()]}}}
    brk = B.build_breaks(cfg)
    assert set(brk) == {"y_5y", "level", "curvature"}      # slope uses 30Y and 2Y only


def test_gsw_factors_inherit_nothing_from_cmt_breaks():
    gsw = B.factor_breaks(CFG, {"y_30y": [D("2002-02-19")]}, prefix="gsw_", out_prefix="gsw_")
    assert gsw == {}


# ---------------------------------------------------------------- event days / moved
def events_frame(rows):
    return pd.DataFrame(rows, columns=["event_id", "event_type", "meeting_date", "scheduled",
                                       "release_datetime_et", "doc_id"])


def test_event_days_moved_and_reasons():
    cmt = make_cmt()
    gsw = pd.DataFrame({"gsw_2y": np.arange(len(DAYS)) * 1.0,
                        "gsw_slope": np.arange(len(DAYS)) * 1.0}, index=DAYS)
    ev = events_frame([
        ("a", "statement", "2002-02-13", True, "2002-02-13 14:00", "S1"),    # normal
        ("b", "statement", "2002-02-14", True, "2002-02-14 16:00", None),    # after close
        ("c", "statement", "2002-02-16", False, "2002-02-16 12:00", "S3"),   # Saturday
        ("d", "minutes", "2002-02-18", True, "2002-02-18 14:00", "M4"),      # holiday
        ("e", "statement", "2002-02-20", False, "2002-02-20 08:20", "S5"),   # pre-open
    ])
    out = B.event_days_table(ev, cmt, gsw, CFG)
    assert out["event_day"].tolist() == [D("2002-02-13"), D("2002-02-15"), D("2002-02-19"),
                                         D("2002-02-19"), D("2002-02-20")]
    assert out["prev_day"].tolist() == [D("2002-02-12"), D("2002-02-14"), D("2002-02-15"),
                                        D("2002-02-15"), D("2002-02-19")]
    assert out["moved"].tolist() == [False, True, True, True, False]
    assert out["moved_reason"].tolist() == ["", "after close", "weekend",
                                            "holiday/non-trading day", ""]
    assert out["has_doc"].tolist() == [True, False, True, True, True]
    assert out["n_events_same_day"].tolist() == [1, 1, 2, 2, 1]
    # d_ columns: plain, break (event day 02-19 for y_30y), GSW
    assert out["d_y_2y"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert out["d_y_30y"].isna().tolist() == [False, False, True, True, False]
    assert out["d_gsw_slope"].tolist() == [1.0] * 5
    assert not any(c in out.columns for c in B.SURPRISE_COLUMNS)


# ---------------------------------------------------------------- surprise join
def test_match_exact_and_fallback_within_7_days():
    ev = pd.Series({"e1": D("2019-10-11"), "e2": D("2019-10-30")})
    matches, fb, refused = B.match_source_dates(ev, [D("2019-10-04"), D("2019-10-30")])
    assert matches == {"e1": D("2019-10-04"), "e2": D("2019-10-30")}
    assert [(f[0], f[3]) for f in fb] == [("e1", 7)]
    assert refused == []


def test_match_refuses_beyond_7_days():
    ev = pd.Series({"e1": D("2019-10-12")})
    matches, fb, refused = B.match_source_dates(ev, [D("2019-10-04")])
    assert matches == {} and fb == [] and refused == []


def test_match_refuses_row_claimed_by_other_event():
    # row 10-04 is the exact match of e0, so e1 (7 days later) must not take it
    ev = pd.Series({"e0": D("2019-10-04"), "e1": D("2019-10-11")})
    matches, fb, refused = B.match_source_dates(ev, [D("2019-10-04")])
    assert matches == {"e0": D("2019-10-04")}
    assert fb == []
    assert len(refused) == 1 and refused[0][0] == "e1" and "claimed by e0" in refused[0][2]


def test_match_contested_fallback_goes_to_nearest_event():
    ev = pd.Series({"e1": D("2019-10-08"), "e2": D("2019-10-12")})
    matches, fb, _ = B.match_source_dates(ev, [D("2019-10-10")])
    # equidistant ... each event is 2 days away: only one may take it (first by id)
    assert len(matches) == 1
    ev = pd.Series({"e1": D("2019-10-07"), "e2": D("2019-10-12")})
    matches, _, _ = B.match_source_dates(ev, [D("2019-10-10")])
    assert matches == {"e2": D("2019-10-10")}


def test_attach_surprises_with_fallback_and_missing_report():
    ev = events_frame([
        ("s1", "statement", "2019-09-18", True, "2019-09-18 14:00", "S1"),
        ("s2", "statement", "2019-10-11", False, "2019-10-11 14:00", "S2"),
        ("s3", "statement", "2019-10-30", True, "2019-10-30 14:00", "S3"),
    ])
    tbl = pd.DataFrame({
        "date": ["2019-09-18", "2019-10-04", "2019-10-30", "2019-12-11"],
        "bs_time_et": ["14:00", None, "14:00", None],
        "bs_unscheduled": [0, None, 0, None],
        "bs_mps": [1.5, np.nan, -2.0, np.nan],
        "bs_mps_orth": [1.0, np.nan, -1.0, np.nan],
        "usmpd_stmt": [0.5, 3.0, np.nan, 1.0],  # s3 missing in usmpd although covered
        "jk_time_et": ["14:00", None, None, "14:00"],
        "jk_pc1": [0.1, np.nan, np.nan, 0.2],
    })
    out, log, fbs, missing = B.attach_surprises(ev, tbl, B.STATEMENT_SOURCES)
    assert out["usmpd_stmt"].tolist()[:2] == [0.5, 3.0]          # s2 via fallback
    assert [(f[0], f[1], f[3]) for f in fbs] == [("usmpd_stmt", "s2", D("2019-10-04"))]
    assert out.loc[1, "surprise_fallback"] == "usmpd_stmt:2019-10-04;"
    assert np.isnan(out.loc[1, "bs_mps"])                         # bs has no row near 10-11
    assert out.loc[2, "bs_mps"] == -2.0
    miss = {(m[0], m[1]) for m in missing}
    assert ("usmpd_stmt", "s3") in miss and ("bs", "s2") in miss
    assert ("jk", "s2") in miss and ("jk", "s3") in miss
    assert any(line.startswith("FALLBACK usmpd_stmt") for line in log)


def test_attach_surprises_minutes():
    ev = events_frame([("m1", "minutes", "2019-09-18", True, "2019-10-09 14:00", "M1")])
    tbl = pd.DataFrame({"date": ["2019-10-09"], "usmpd_min": [-0.7]})
    out, log, _, missing = B.attach_surprises(ev, tbl, B.MINUTES_SOURCES)
    assert out.loc[0, "usmpd_min"] == -0.7 and missing == []


def test_mps_control():
    panel = pd.DataFrame({
        "event_type": ["statement", "statement", "minutes", "minutes"],
        "bs_mps": [1.5, np.nan, np.nan, np.nan]})
    out = B.add_mps_control(panel)
    assert out.loc[0, "mps_control"] == 1.5
    assert np.isnan(out.loc[1, "mps_control"])
    assert out.loc[2, "mps_control"] == 0.0 and out.loc[3, "mps_control"] == 0.0


# ---------------------------------------------------------------- coverage / acceptance
def _panel():
    return pd.DataFrame({
        "event_type": ["statement", "statement", "minutes"],
        "release_datetime_et": ["2001-01-03 13:13", "2001-03-20 14:00", "2001-02-01 14:00"],
        "scheduled": [False, True, True], "has_doc": [True, True, False],
        "moved": [False, True, False], "moved_reason": ["", "after close", ""],
        "bs_mps": [1.0, np.nan, np.nan], "usmpd_stmt": [1.0, 2.0, np.nan],
        "usmpd_min": [np.nan, np.nan, 0.3], "jk_pc1": [np.nan, 1.0, np.nan],
        "d_slope": [1.0, np.nan, 2.0], "d_gsw_slope": [1.0, 2.0, 3.0],
    })


def test_coverage_table_totals():
    cov = B.coverage_table(_panel())
    s2001 = cov[(cov.year == "2001") & (cov.event_type == "statement")].iloc[0]
    assert s2001.n_events == 2 and s2001.n_unscheduled == 1 and s2001.n_moved == 1
    assert s2001.n_moved_after_close == 1 and s2001.n_with_bs_mps == 1
    assert s2001.n_with_usmpd == 2 and s2001.n_with_jk == 1 and s2001.n_with_d_slope == 1
    tot = cov[(cov.year == "Total") & (cov.event_type == "all")].iloc[0]
    assert tot.n_events == 3 and tot.n_with_usmpd == 3 and tot.n_with_doc == 2
    tex = B.coverage_latex(cov)
    assert "\\toprule" in tex and "\\bottomrule" in tex and "2001 & 2 & 1 & 2 & 1" in tex


def test_acceptance_checks_flag_problems():
    ed = pd.DataFrame({
        "event_id": ["a", "b"], "release_datetime_et": ["2002-02-13 14:00", "2002-02-13 14:00"],
        "event_day": [D("2002-02-13"), D("2002-02-13")],
        "prev_day": [D("2002-02-12"), D("2002-02-12")]})
    events = pd.DataFrame({"doc_id": ["S1", None]})
    docs = pd.DataFrame({"doc_id": ["S1"], "doc_type": ["statement"],
                         "meeting_date_ds": ["2002-02-13"], "text": ["x"]})
    cfg = {"sample": {"start": "2000-01-01", "end": "2025-12-31"}}
    assert B.acceptance_checks(ed, docs, events, cfg).empty
    with pytest.raises(AssertionError, match="duplicate"):
        B.acceptance_checks(ed.assign(event_id=["a", "a"]), docs, events, cfg)
    with pytest.raises(AssertionError, match="event_day"):
        B.acceptance_checks(ed.assign(event_day=[D("2002-02-13"), pd.NaT]), docs, events, cfg)
    with pytest.raises(AssertionError, match="shift"):
        B.acceptance_checks(ed.assign(event_day=[D("2002-02-13"), D("2002-02-25")]),
                            docs, events, cfg)
    docs2 = pd.concat([docs, pd.DataFrame({"doc_id": ["M9"], "doc_type": ["minutes"],
                                           "meeting_date_ds": ["2002-03-01"], "text": ["y"]})])
    with pytest.raises(AssertionError, match="unexplained"):
        B.acceptance_checks(ed, docs2, events, cfg)
    docs3 = docs2.assign(text=["x", "x"])        # duplicate text of a matched doc -> explained
    unm = B.acceptance_checks(ed, docs3, events, cfg)
    assert unm["doc_id"].tolist() == ["M9"] and "identical to matched document S1" in unm["reason"][0]
