"""Daily Treasury yields: FRED constant-maturity (CMT) and Gurkaynak-Sack-Wright (GSW).

Outputs (all yields in basis points = percent x 100; nothing is filled or interpolated):

* data/raw/fred/{SERIES}.csv                 raw FRED downloads (committed)
* data/processed/yields_cmt_daily.csv        CMT yields + level/slope/curvature (committed)
* data/interim/downloads/feds200628.csv      GSW zero-coupon curve (not committed)
* data/processed/yields_gsw_daily.csv        GSW 2/5/10/30Y + factors (committed)

Trading days are the dates on which FRED publishes DGS10 within the sample window.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.data.download import fetch

FRED_LICENSE = "public domain (FRED / U.S. Treasury)"
GSW_LICENSE = "Federal Reserve Board public data"
GSW_SERIES = {"SVENY02": "gsw_2y", "SVENY05": "gsw_5y", "SVENY10": "gsw_10y",
              "SVENY30": "gsw_30y"}


# --------------------------------------------------------------------------- pure functions
def tenor_column(tenor: str, prefix: str = "y_") -> str:
    """'10Y' -> 'y_10y'."""
    return f"{prefix}{tenor.lower()}"


def parse_fred_csv(src) -> pd.Series:
    """Parse a FRED CSV (path or file-like): first column date, second column value.

    FRED encodes missing observations as '.' or an empty field; both become NaN.
    Returns a float Series (percent) indexed by Timestamp, named after the series.
    """
    df = pd.read_csv(src, dtype=str)
    if df.shape[1] < 2:
        raise ValueError("FRED CSV needs a date column and a value column")
    date_col, val_col = df.columns[0], df.columns[1]
    values = pd.to_numeric(df[val_col].str.strip().replace({".": np.nan, "": np.nan}),
                           errors="coerce")
    out = pd.Series(values.to_numpy(dtype=float), index=pd.to_datetime(df[date_col]),
                    name=val_col)
    out.index.name = "date"
    return out.sort_index()


def pct_to_bp(x):
    """Percent -> basis points (rounded to 6 dp to remove float noise)."""
    return (x * 100.0).round(6)


def compute_factors(df: pd.DataFrame, cfg: dict, prefix: str = "y_",
                    out_prefix: str = "") -> pd.DataFrame:
    """level / slope / curvature (bp) from tenor columns, per cfg['factors'].

    A factor is NaN whenever any of its inputs is NaN (no filling).
    """
    f = cfg["factors"]
    col = lambda t: df[tenor_column(t, prefix)]  # noqa: E731
    level_cols = [col(t) for t in f["level"]["tenors"]]
    level = pd.concat(level_cols, axis=1).mean(axis=1, skipna=False)
    slope = col(f["slope"]["long"]) - col(f["slope"]["short"])
    curv = (2.0 * col(f["curvature"]["belly"]) - col(f["curvature"]["short"])
            - col(f["curvature"]["long"]))
    return pd.DataFrame({f"{out_prefix}level": level, f"{out_prefix}slope": slope,
                         f"{out_prefix}curvature": curv}, index=df.index)


def trading_days(dgs10: pd.Series, start, end) -> pd.DatetimeIndex:
    """Dates with non-missing DGS10 inside [start, end]."""
    s = dgs10.dropna()
    idx = s.index[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return pd.DatetimeIndex(idx, name="date")


def build_cmt(series: dict[str, pd.Series], cfg: dict) -> pd.DataFrame:
    """Assemble the CMT panel (bp) on trading days, with factors."""
    days = trading_days(series["DGS10"], cfg["sample"]["start"], cfg["sample"]["end"])
    cols = {}
    for sid, tenor in cfg["fred"]["series"].items():
        cols[tenor_column(tenor)] = pct_to_bp(series[sid].reindex(days))
    df = pd.DataFrame(cols, index=days)
    return pd.concat([df, compute_factors(df, cfg)], axis=1)


def parse_gsw_csv(src) -> pd.DataFrame:
    """Parse the Fed GSW CSV: skip description lines, find the row starting 'Date'.

    Returns numeric DataFrame (percent) indexed by date. Missing values (NA, empty,
    non-numeric) become NaN.
    """
    if isinstance(src, (str, Path)):
        text = Path(src).read_text(encoding="utf-8", errors="replace")
    else:
        text = src.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hdr = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith("date")),
               None)
    if hdr is None:
        raise ValueError("GSW CSV: no header row starting with 'Date'")
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), dtype=str)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df.apply(lambda c: pd.to_numeric(c, errors="coerce"))


def build_gsw(gsw: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """GSW 2/5/10/30Y in bp on the trading-day index, plus factors."""
    out = pd.DataFrame(index=days)
    for src, name in GSW_SERIES.items():
        out[name] = pct_to_bp(gsw[src].reindex(days))
    return pd.concat([out, compute_factors(out, cfg, prefix="gsw_", out_prefix="gsw_")],
                     axis=1)


# --------------------------------------------------------------------------- pipeline
def _summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"rows: {len(df)}; range: {df.index.min().date()} .. {df.index.max().date()}")
    print("NaN counts:")
    print(df.isna().sum().to_string())
    print("std of daily changes (bp), consecutive trading-day diffs:")
    print(df.diff().std().round(3).to_string())


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    root = repo_root()
    raw = Path(cfg["paths"]["data_raw"]) / "fred"
    series = {}
    for sid in cfg["fred"]["series"]:
        p = fetch(f"fred_{sid}", cfg["sources"]["fred_csv"].format(series=sid),
                  raw / f"{sid}.csv", license=FRED_LICENSE, committed=True, cfg=cfg)
        series[sid] = parse_fred_csv(p)

    cmt = build_cmt(series, cfg)
    out_dir = root / cfg["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmt.reset_index().to_csv(out_dir / "yields_cmt_daily.csv", index=False,
                             float_format="%.8g", lineterminator="\n")
    _summary(cmt, "CMT daily (bp)")

    # known facts (reported, never imposed)
    def _span(mask: pd.Series) -> str:
        d = cmt.index[mask]
        return f"{len(d)} days" + (f" ({d.min().date()} .. {d.max().date()})" if len(d) else "")

    print(f"\ny_30y NaN: {_span(cmt['y_30y'].isna())}")
    sus = cmt.loc["2002-02-19":"2006-02-08", "y_30y"]
    print(f"y_30y in the 2002-02-19..2006-02-08 bond-suspension window: "
          f"{int(sus.notna().sum())} non-NaN of {len(sus)} (expected NaN per task notes)")
    print(f"first non-NaN y_1m: {cmt['y_1m'].first_valid_index().date()}; "
          f"y_1m NaN after that: {int(cmt.loc[cmt['y_1m'].first_valid_index():, 'y_1m'].isna().sum())}")
    print(f"factor NaN: {_span(cmt[['level', 'slope', 'curvature']].isna().any(axis=1))}")
    jump = cmt[["y_20y", "y_30y"]].diff()
    for d in ("2002-02-19", "2006-02-09"):
        print(f"30Y-vs-20Y break check {d}: d(y_20y) = {jump.loc[d, 'y_20y']:+.0f} bp, "
              f"d(y_30y) = {jump.loc[d, 'y_30y']:+.0f} bp")
    for col, a, b in (("y_2y", "2022-03-15", "2022-03-16"),
                      ("y_10y", "2020-03-13", "2020-03-16"),
                      ("y_2y", "2008-12-15", "2008-12-16")):
        print(f"spot {col}: {a} = {cmt.loc[a, col]:.1f}, {b} = {cmt.loc[b, col]:.1f}, "
              f"change = {cmt.loc[b, col] - cmt.loc[a, col]:+.1f} bp")

    # GSW (robustness only)
    gp = fetch("gsw_feds200628", cfg["sources"]["gsw_csv"],
               Path(cfg["downloads"]["dir"]) / "feds200628.csv",
               license=GSW_LICENSE, committed=False, cfg=cfg)
    gsw = build_gsw(parse_gsw_csv(gp), cmt.index, cfg)
    gsw.reset_index().to_csv(out_dir / "yields_gsw_daily.csv", index=False,
                             float_format="%.8g", lineterminator="\n")
    _summary(gsw, "GSW daily (bp)")
    missing = gsw[list(GSW_SERIES.values())].isna().any(axis=1)
    print(f"\ntrading days lacking any GSW 2/5/10/30Y: {int(missing.sum())} of {len(gsw)}")
    if missing.any():
        print("  dates (first 10):", [d.date().isoformat() for d in gsw.index[missing][:10]])
        print("  range:", gsw.index[missing].min().date(), "..", gsw.index[missing].max().date())
    both = pd.concat([cmt["y_10y"], gsw["gsw_10y"]], axis=1).dropna()
    print(f"corr(CMT 10y, GSW 10y) levels: {both.corr().iloc[0, 1]:.5f}; "
          f"mean abs diff {np.abs(both['y_10y'] - both['gsw_10y']).mean():.2f} bp")


if __name__ == "__main__":
    main()
