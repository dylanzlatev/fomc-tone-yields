"""Unit tests for ftyields.data.surprises (pure functions, synthetic inputs, no network)."""
import datetime as dt
import zipfile

import numpy as np
import pandas as pd
import pytest

from ftyields.data import surprises as S


@pytest.mark.parametrize("text, expected", [
    ("2:00pm", "14:00"),
    ("11:30am", "11:30"),
    ("9:45am", "09:45"),
    ("12:30pm", "12:30"),
    ("12:05am", "00:05"),
    ("2:15PM", "14:15"),
    (" 5:00pm ", "17:00"),
])
def test_parse_bs_time(text, expected):
    assert S.parse_bs_time(text) == expected


def test_parse_bs_time_missing_and_time_objects():
    assert np.isnan(S.parse_bs_time(np.nan))
    assert np.isnan(S.parse_bs_time(None))
    assert S.parse_bs_time(dt.time(14, 0)) == "14:00"


def test_parse_bs_time_bad_raises():
    with pytest.raises(ValueError):
        S.parse_bs_time("14:00")


def test_to_bp():
    s = pd.Series([-0.0819, 0.0, 0.1])
    assert S.to_bp(s).tolist() == [-8.19, 0.0, 10.0]


def test_clean_bs_units_and_nan():
    raw = pd.DataFrame({
        "Date": pd.to_datetime(["2023-12-13", "2001-09-17"]),
        "Time": ["2:00pm", "8:20am"],
        "Unscheduled": [0, 1],
        "MPS": [-0.0819, np.nan],
        "MPS_ORTH": [0.0342, np.nan],
    })
    out = S.clean_bs(raw)
    assert out["bs_time_et"].tolist() == ["14:00", "08:20"]
    assert out["bs_unscheduled"].tolist() == [0, 1]
    assert out["bs_mps"].iloc[0] == pytest.approx(-8.19)
    assert out["bs_mps_orth"].iloc[0] == pytest.approx(3.42)
    assert out["bs_mps"].isna().iloc[1] and out["bs_mps_orth"].isna().iloc[1]


def test_clean_usmpd_and_jk():
    u = S.clean_usmpd(pd.DataFrame({"Date": ["2022-06-15"], "STMT": [0.05]}), "STMT",
                      "usmpd_stmt")
    assert u["usmpd_stmt"].iloc[0] == pytest.approx(5.0)
    assert u["date"].iloc[0] == pd.Timestamp("2022-06-15")
    jk = S.clean_jk(pd.DataFrame({"start": ["2005-05-03 14:15"], "pc1": [0.0272]}))
    assert jk["date"].iloc[0] == pd.Timestamp("2005-05-03")
    assert jk["jk_time_et"].iloc[0] == "14:15"
    assert jk["jk_pc1"].iloc[0] == pytest.approx(2.72)


def test_in_window_inclusive():
    df = pd.DataFrame({"date": pd.to_datetime(["1999-12-31", "2000-01-01", "2025-12-31",
                                               "2026-01-01"]), "v": range(4)})
    assert S.in_window(df, "2000-01-01", "2025-12-31")["v"].tolist() == [1, 2]


def test_keep_closest_to_time():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2005-05-03", "2005-05-03", "2005-06-30"]),
        "jk_time_et": ["16:00", "14:15", "14:15"],
        "v": [1, 2, 3],
    })
    kept, dropped = S.keep_closest_to_time(df, "jk_time_et", "14:00")
    assert kept["v"].tolist() == [2, 3]
    assert dropped["v"].tolist() == [1]


def test_join_on_date_outer_and_sorted():
    bs = pd.DataFrame({"date": pd.to_datetime(["2020-03-03", "2020-03-15"]),
                       "bs_mps": [1.0, 2.0]})
    us = pd.DataFrame({"date": pd.to_datetime(["2020-03-15", "2020-04-29"]),
                       "usmpd_stmt": [3.0, 4.0]})
    jk = pd.DataFrame({"date": pd.to_datetime(["2020-03-15"]), "jk_pc1": [5.0]})
    out = S.join_on_date([bs, us, jk])
    assert out["date"].tolist() == list(pd.to_datetime(["2020-03-03", "2020-03-15",
                                                        "2020-04-29"]))
    assert out["bs_mps"].isna().tolist() == [False, False, True]
    assert out["usmpd_stmt"].isna().tolist() == [True, False, False]
    assert out["jk_pc1"].tolist()[1] == 5.0 and out["jk_pc1"].isna().sum() == 2


def test_join_on_date_rejects_duplicates():
    a = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "x": [1, 2]})
    with pytest.raises(ValueError):
        S.join_on_date([a])


def test_disagreements():
    d = pd.to_datetime
    presence = {"bs": pd.Series(d(["2020-01-29", "2020-03-03"])),
                "us": pd.Series(d(["2020-01-29", "2020-03-15"])),
                "jk": pd.Series(d(["2020-01-29", "2020-03-03", "2020-03-15", "2024-01-31"]))}
    out = S.disagreements(presence, lo="2020-01-01", hi="2023-12-31")
    assert out["date"].tolist() == list(d(["2020-03-03", "2020-03-15"]))
    assert out.loc[out["date"] == "2020-03-03", ["bs", "us", "jk"]].iloc[0].tolist() == \
        [True, False, True]
    # without bounds the post-coverage date is reported too
    assert len(S.disagreements(presence)) == 3


def test_safe_extract(tmp_path):
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("mps.csv", "Date,STMT\n")
    dest = tmp_path / "out"
    files = S.safe_extract(zp, dest)
    assert [f.name for f in files] == ["mps.csv"] and (dest / "mps.csv").is_file()

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        S.safe_extract(bad, tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()
