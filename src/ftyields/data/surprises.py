"""Monetary-policy rate surprises around FOMC announcements, all in basis points.

Sources (units verified in `main`, see the printed report):

* Bauer-Swanson (2023 update) MPS, MPS_ORTH: percentage points  -> x100 = bp
* USMPD (Acosta et al. 2025) STMT / MIN:      percentage points  -> x100 = bp
* Jarocinski-Karadi update pc1:               percent p.a. (pp)  -> x100 = bp

Files without a stated license are downloaded to `downloads.dir` (gitignored); only
the J-K file (CC BY 4.0) is committed, under data/raw/jk/.

Outputs (gitignored, under data/interim/):
    surprises_statements.csv  date, bs_time_et, bs_unscheduled, bs_mps, bs_mps_orth,
                              usmpd_stmt, jk_time_et, jk_pc1
    surprises_minutes.csv     date, usmpd_min
"""
from __future__ import annotations

import zipfile
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.data.download import fetch

NO_LICENSE = "no license stated; SF Fed site policies"
PP_TO_BP = 100.0   # all three sources report percentage points

STATEMENT_COLUMNS = ["date", "bs_time_et", "bs_unscheduled", "bs_mps", "bs_mps_orth",
                     "usmpd_stmt", "jk_time_et", "jk_pc1"]

JK_ATTRIBUTION = """# Attribution

The file `shocks_fed_jk_t.csv` is the updated monetary policy and central bank
information shock series of

Jarocinski, Marek, and Peter Karadi (2020). "Deconstructing Monetary Policy
Surprises - The Role of Information Shocks." American Economic Journal:
Macroeconomics 12(2): 1-43. doi:10.1257/mac.20180090

Source repository: https://github.com/marekjarocinski/jkshocks_update_fed
(file `shocks_fed_jk_t.csv`, fetched from the `main` branch; see
`data/raw/MANIFEST.csv` for the retrieval date and SHA-256).

License: the source data and final dataset are licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. The authors' code is BSD 3-Clause.
The file is unmodified here; unit conversion (percentage points to basis points)
is done downstream in `src/ftyields/data/surprises.py`.
"""


# --------------------------------------------------------------------------- pure functions
def parse_bs_time(value) -> str | float:
    """Bauer-Swanson time text ('2:00pm', '11:30am', '12:05pm') -> 'HH:MM' 24h ET.

    Missing values (NaN/None) return NaN. datetime.time objects are formatted directly.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return np.nan
    if isinstance(value, (dtime, datetime, pd.Timestamp)):
        return f"{value.hour:02d}:{value.minute:02d}"
    text = str(value).strip().lower().replace(" ", "")
    try:
        t = datetime.strptime(text, "%I:%M%p")
    except ValueError as exc:
        raise ValueError(f"unparseable Bauer-Swanson time: {value!r}") from exc
    return f"{t.hour:02d}:{t.minute:02d}"


def to_bp(x):
    """Percentage points -> basis points (rounded to 6 dp to remove float noise)."""
    return (x * PP_TO_BP).round(6)


def in_window(df: pd.DataFrame, start, end, col: str = "date") -> pd.DataFrame:
    """Rows with start <= date <= end (inclusive)."""
    m = (df[col] >= pd.Timestamp(start)) & (df[col] <= pd.Timestamp(end))
    return df.loc[m].reset_index(drop=True)


def clean_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Raw B-S sheet -> date, bs_time_et, bs_unscheduled, bs_mps, bs_mps_orth (bp)."""
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"]).dt.normalize(),
        "bs_time_et": df["Time"].map(parse_bs_time),
        "bs_unscheduled": pd.to_numeric(df["Unscheduled"], errors="coerce").astype("Int64"),
        "bs_mps": to_bp(pd.to_numeric(df["MPS"], errors="coerce")),
        "bs_mps_orth": to_bp(pd.to_numeric(df["MPS_ORTH"], errors="coerce")),
    })
    return out


def clean_usmpd(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """USMPD csv (Date, <col>) -> date, <name> (bp)."""
    return pd.DataFrame({
        "date": pd.to_datetime(df["Date"]).dt.normalize(),
        name: to_bp(pd.to_numeric(df[col], errors="coerce")),
    })


def clean_jk(df: pd.DataFrame) -> pd.DataFrame:
    """J-K csv ('start' = 'YYYY-MM-DD HH:MM', pc1) -> date, jk_time_et, jk_pc1 (bp)."""
    start = pd.to_datetime(df["start"], format="%Y-%m-%d %H:%M")
    return pd.DataFrame({
        "date": start.dt.normalize(),
        "jk_time_et": start.dt.strftime("%H:%M"),
        "jk_pc1": to_bp(pd.to_numeric(df["pc1"], errors="coerce")),
    })


def _minutes_of_day(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def keep_closest_to_time(df: pd.DataFrame, time_col: str, target: str):
    """One row per date: among same-date events keep the one closest to `target` ('HH:MM').

    Returns (kept, dropped). Rows with a missing time are only kept if alone on their date.
    """
    t0 = _minutes_of_day(target)
    dist = df[time_col].map(lambda s: np.inf if pd.isna(s) else abs(_minutes_of_day(s) - t0))
    order = df.assign(_dist=dist, _pos=np.arange(len(df))).sort_values(
        ["date", "_dist", "_pos"])
    keep = ~order["date"].duplicated()
    kept = order[keep].drop(columns=["_dist", "_pos"]).sort_values("date")
    dropped = order[~keep].drop(columns=["_dist", "_pos"]).sort_values("date")
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def join_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join frames on `date`. Each frame must have unique dates."""
    out = None
    for f in frames:
        if f["date"].duplicated().any():
            dups = f.loc[f["date"].duplicated(keep=False), "date"].dt.date.unique()
            raise ValueError(f"duplicate dates in a source frame: {list(dups)[:5]}")
        out = f if out is None else out.merge(f, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def disagreements(presence: dict[str, pd.Series], lo=None, hi=None) -> pd.DataFrame:
    """Dates present in some but not all sources (within [lo, hi] if given).

    `presence` maps source name -> Series of announcement dates. Returns a frame with
    `date` and one boolean column per source.
    """
    dates = sorted(set().union(*[set(s) for s in presence.values()]))
    tab = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    for name, s in presence.items():
        tab[name] = tab["date"].isin(set(s))
    if lo is not None:
        tab = tab[tab["date"] >= pd.Timestamp(lo)]
    if hi is not None:
        tab = tab[tab["date"] <= pd.Timestamp(hi)]
    srcs = list(presence)
    return tab[~tab[srcs].all(axis=1)].reset_index(drop=True)


def safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    """Extract a zip into dest, refusing members that would escape dest."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = (root / info.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe zip member: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
            out.append(target)
    return out


# --------------------------------------------------------------------------- pipeline
def _fmt(v) -> str:
    return "NaN" if pd.isna(v) else f"{v:.3f}"


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    root = repo_root()
    dl = Path(cfg["downloads"]["dir"])
    start, end = cfg["sample"]["start"], cfg["sample"]["end"]
    src = cfg["sources"]

    # ---- Bauer-Swanson
    bs_path = fetch("bauer_swanson_xlsx", src["bauer_swanson_xlsx"],
                    dl / "monetary-policy-surprises-data.xlsx", license=NO_LICENSE,
                    committed=False, cfg=cfg)
    bs_raw = pd.read_excel(bs_path, sheet_name=src["bauer_swanson_sheet"], na_values=["NA"])
    desc = pd.read_excel(bs_path, sheet_name="Description", header=None)
    desc_txt = " ".join(str(v) for v in desc.to_numpy().ravel() if pd.notna(v))
    # Units evidence: description says MPS is scaled so ED4 impact is unity, and ED4 is a
    # futures rate change; the magnitudes below (std of a few bp once x100) confirm pp.
    bs_std_raw = bs_raw["MPS"].std()
    print("=== Bauer-Swanson units ===")
    print("Description sheet mentions: 'scaled so that the impact on ED4 is unity' ->",
          "scaled so that the impact on ED4 is unity" in desc_txt)
    print(f"raw MPS std = {bs_std_raw:.4f}; x100 = {bs_std_raw * 100:.2f} bp; "
          f"raw MPS_ORTH std x100 = {bs_raw['MPS_ORTH'].std() * 100:.2f} bp")
    r = bs_raw[bs_raw["Date"] == "2023-12-13"].iloc[0]
    print(f"2023-12-13 raw MPS = {r['MPS']:.4f} (-> {r['MPS'] * 100:.2f} bp); "
          f"raw ED4 = {r['ED4']:.4f} (futures rate change in pp)")
    print("Conclusion: MPS / MPS_ORTH are in percentage points; x100 = bp.")
    bs = clean_bs(bs_raw)

    # ---- USMPD
    zpath = fetch("usmpd_zip", cfg["sources"]["usmpd_zip"], dl / "monetary-policy-surprises.zip",
                  license=NO_LICENSE, committed=False, cfg=cfg)
    udir = root / dl / "usmpd_mps"
    safe_extract(zpath, udir)
    readme = (udir / "README.md").read_text(encoding="utf-8", errors="replace")
    key = [ln for ln in readme.splitlines() if "units" in ln.lower()]
    print("\n=== USMPD units (README.md) ===")
    for ln in key:
        print("  ", ln.strip())
    print("Conclusion: STMT / MIN are in percentage points (normalised to a 1:1 impact on "
          "the 1-year Treasury yield); x100 = bp.")
    mps = pd.read_csv(udir / "mps.csv", na_values=["NA"])
    mins = pd.read_csv(udir / "mps_minutes.csv", na_values=["NA"])
    us_stmt = clean_usmpd(mps, "STMT", "usmpd_stmt")
    us_min = clean_usmpd(mins, "MIN", "usmpd_min")

    # ---- Jarocinski-Karadi
    jk_path = fetch("jk_shocks_fed_jk_t", src["jk_csv"], "data/raw/jk/shocks_fed_jk_t.csv",
                    license="CC BY 4.0", committed=True, cfg=cfg)
    (jk_path.parent / "ATTRIBUTION.md").write_text(JK_ATTRIBUTION, encoding="utf-8",
                                                   newline="\n")
    jk_readme = fetch("jk_readme", cfg["sources"]["jk_readme"], dl / "jk_README.md", license="CC BY 4.0",
                      committed=False, cfg=cfg)
    jk_lines = [ln for ln in jk_readme.read_text(encoding="utf-8").splitlines()
                if ln.strip().startswith("- pc1")]
    print("\n=== Jarocinski-Karadi units (repo README) ===")
    for ln in jk_lines:
        print("  ", ln.strip())
    jk_raw = pd.read_csv(jk_path)
    print(f"raw pc1 std = {jk_raw['pc1'].std():.4f}; x100 = {jk_raw['pc1'].std() * 100:.2f} bp")
    print("Conclusion: pc1 is scaled to the std of the ED4 surprise, which is in % p.a. "
          "(percentage points); x100 = bp.")
    jk = clean_jk(jk_raw)

    # ---- window + join
    full = {"bauer_swanson": bs, "usmpd_stmt": us_stmt, "jarocinski_karadi": jk,
            "usmpd_minutes": us_min}
    win = {k: in_window(v, start, end) for k, v in full.items()}
    bs_w, us_w, jk_w, min_w = (win[k] for k in
                               ("bauer_swanson", "usmpd_stmt", "jarocinski_karadi",
                                "usmpd_minutes"))
    # Same-day multiple events (J-K 2005-05-03 has 14:15 and 16:00): keep the one closest
    # to the statement release time. Pre-1992 B-S duplicates fall outside the window.
    stmt_time = cfg["events"]["statement_release_time_et"]
    bs_w, bs_drop = keep_closest_to_time(bs_w, "bs_time_et", stmt_time)
    jk_w, jk_drop = keep_closest_to_time(jk_w, "jk_time_et", stmt_time)
    for name, d in (("bs", bs_drop), ("jk", jk_drop)):
        for r in d.itertuples(index=False):
            print(f"NOTE: dropped extra same-day {name} event {r.date.date()} "
                  f"{r[1]} (kept the one closest to {stmt_time})")
    stm = join_on_date([bs_w, us_w, jk_w])[STATEMENT_COLUMNS]
    interim = root / cfg["paths"]["data_interim"]
    interim.mkdir(parents=True, exist_ok=True)
    stm.to_csv(interim / "surprises_statements.csv", index=False, float_format="%.8g",
               lineterminator="\n")
    min_w.sort_values("date").to_csv(interim / "surprises_minutes.csv", index=False,
                                     float_format="%.8g", lineterminator="\n")

    # ---- report
    print(f"\n=== Coverage (sample window {start}..{end}) ===")
    for k in full:
        f, w = full[k], win[k]
        print(f"{k:18s} full file: {len(f):4d} rows {f['date'].min().date()}..{f['date'].max().date()}"
              f" | in window: {len(w):4d} rows {w['date'].min().date()}..{w['date'].max().date()}")
    print(f"joined statements table: {len(stm)} rows")

    pres = {"bs": bs_w["date"], "usmpd": us_w["date"], "jk": jk_w["date"]}
    lo = max(s.min() for s in pres.values())
    hi = min(s.max() for s in pres.values())
    print(f"\n=== Source disagreements on announcement dates, common coverage "
          f"{lo.date()}..{hi.date()} ===")
    dis = disagreements(pres, lo, hi)
    if dis.empty:
        print("none")
    else:
        d2 = dis.copy()
        d2["date"] = d2["date"].dt.date
        print(d2.to_string(index=False))
    outside = disagreements(pres)
    outside = outside[(outside["date"] < lo) | (outside["date"] > hi)]
    print(f"dates outside common coverage (a source simply ends/starts): {len(outside)}; "
          f"before {lo.date()}: {int((outside['date'] < lo).sum())}, "
          f"after {hi.date()}: {int((outside['date'] > hi).sum())}")
    for name, s in pres.items():
        print(f"  {name}: dates after {hi.date()}: {int((s > hi).sum())}")

    print("\nNaN counts in surprises_statements.csv:")
    print(stm.isna().sum().to_string())
    print("\nstd (bp):", stm[["bs_mps", "bs_mps_orth", "usmpd_stmt", "jk_pc1"]].std().round(2)
          .to_dict())
    print("\nPairwise correlations on common dates (n):")
    cols = ["bs_mps", "usmpd_stmt", "jk_pc1"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = stm[[a, b]].dropna()
            # OLS slope = corr * sd_b / sd_a (np.cov/polyfit crash in this env's BLAS)
            slope = both[a].corr(both[b]) * both[b].std() / both[a].std()
            print(f"  corr({a}, {b}) = {both[a].corr(both[b]):.3f}  n={len(both)}  "
                  f"OLS slope of {b} on {a} = {slope:.2f}")
    both = stm[["bs_mps", "bs_mps_orth"]].dropna()
    print(f"  corr(bs_mps, bs_mps_orth) = {both['bs_mps'].corr(both['bs_mps_orth']):.3f}")

    print("\nSpot values (bp):")
    for d in ("2001-01-03", "2008-01-22", "2020-03-15", "2022-06-15"):
        row = stm[stm["date"] == pd.Timestamp(d)]
        if row.empty:
            print(f"  {d}: no row in any source")
            continue
        r = row.iloc[0]
        print(f"  {d}: bs {r['bs_time_et']} unsched={r['bs_unscheduled']} "
              f"mps={_fmt(r['bs_mps'])} orth={_fmt(r['bs_mps_orth'])} | "
              f"usmpd_stmt={_fmt(r['usmpd_stmt'])} | jk {r['jk_time_et']} "
              f"pc1={_fmt(r['jk_pc1'])}")

    print(f"\nMinutes (USMPD MIN): {len(min_w)} rows in window, "
          f"{min_w['date'].min().date()}..{min_w['date'].max().date()}, "
          f"NaN {int(min_w['usmpd_min'].isna().sum())}, "
          f"std {min_w['usmpd_min'].std():.2f} bp")


if __name__ == "__main__":
    main()
