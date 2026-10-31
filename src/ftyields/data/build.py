"""Phase 1 integration: event days, event-day yield changes, surprise join, coverage.

Runs the stage modules (yields, surprises, corpus, fomc_calendar) and then builds

  data/processed/event_days.csv      (committed) one row per release event with its event
                                     day, previous trading day and every yield/factor change
                                     d_<col> in bp. NO surprise values (unclear license).
  data/interim/event_panel.csv       (gitignored) event_days + surprise columns + mps_control
  results/tables/coverage_events.csv (committed) counts by year x event type
  results/tables/coverage_events.tex compact booktabs version for the paper

Conventions
  * trading days = dates in yields_cmt_daily.csv; event day from
    fomc_calendar.map_to_trading_day; prev_day = the trading day just before it.
  * d_<col> = y(event_day) - y(prev_day), in bp. A change is NaN whenever a configured
    series break lies in (prev_day, event_day]; factors inherit the breaks of the tenors
    they are built from (`change`, `factor_breaks`; reused for multi-day horizons later).
  * Surprise rows are joined on the release calendar date. If a source has no row on
    that date, the nearest unclaimed row within +-7 calendar days is used and logged
    (`match_source_dates`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.data import corpus, fomc_calendar, surprises, yields
from ftyields.data.yields import tenor_column

MAX_GAP_DAYS = 7          # +-window for fallback surprise matching
MAX_EVENT_SHIFT_DAYS = 5  # acceptance bound on event_day - release date
MOVED_REASONS = ["weekend", "holiday/non-trading day", "after close"]
REASON_COLS = {"weekend": "n_moved_weekend", "holiday/non-trading day": "n_moved_holiday",
               "after close": "n_moved_after_close"}
# Surprise source groups of surprises_statements.csv: columns and the column whose
# presence marks "this source has a row on that date".
STATEMENT_SOURCES = {
    "bs": (["bs_time_et", "bs_unscheduled", "bs_mps", "bs_mps_orth"], "bs_time_et"),
    "usmpd_stmt": (["usmpd_stmt"], "usmpd_stmt"),
    "jk": (["jk_time_et", "jk_pc1"], "jk_time_et"),
}
MINUTES_SOURCES = {"usmpd_min": (["usmpd_min"], "usmpd_min")}
SURPRISE_COLUMNS = ["bs_time_et", "bs_unscheduled", "bs_mps", "bs_mps_orth", "usmpd_stmt",
                    "jk_time_et", "jk_pc1", "usmpd_min"]


# --------------------------------------------------------------------------- breaks
def factor_breaks(cfg: dict, breaks: dict, prefix: str = "y_", out_prefix: str = "") -> dict:
    """Breaks inherited by the yield-curve factors from the tenor columns they use.

    `breaks` maps tenor column (e.g. 'y_30y') -> list of break dates. Returns
    {factor column: sorted list of Timestamps} for factors with at least one break.
    """
    f = cfg["factors"]
    uses = {
        "level": f["level"]["tenors"],
        "slope": [f["slope"]["long"], f["slope"]["short"]],
        "curvature": [f["curvature"]["belly"], f["curvature"]["short"], f["curvature"]["long"]],
    }
    out = {}
    for name, tenors in uses.items():
        dates = sorted({pd.Timestamp(d) for t in tenors
                        for d in breaks.get(tenor_column(t, prefix), [])})
        if dates:
            out[f"{out_prefix}{name}"] = dates
    return out


def build_breaks(cfg: dict) -> dict:
    """All break dates per column: configured tenor breaks + inherited factor breaks.

    GSW columns carry no configured breaks, so the GSW factors inherit none.
    """
    raw = {c: [pd.Timestamp(d) for d in ds] for c, ds in cfg["fred"].get("series_breaks", {}).items()}
    out = {c: sorted(ds) for c, ds in raw.items()}
    out.update(factor_breaks(cfg, raw, prefix="y_", out_prefix=""))
    out.update(factor_breaks(cfg, raw, prefix="gsw_", out_prefix="gsw_"))
    return out


def change(df: pd.DataFrame, col: str, start_day, end_day, breaks: dict):
    """y(end_day) - y(start_day) for column `col` of a date-indexed frame.

    NaN if either endpoint is missing from the frame / NaN, or if any break date b of
    `col` satisfies start_day < b <= end_day (a break at start_day is already in the
    start level, so it does not contaminate the change; a break at end_day does).
    `start_day`/`end_day` may be scalars or equal-length array-likes; returns a float
    for scalar input, else a numpy array.
    """
    scalar = np.ndim(start_day) == 0 and np.ndim(end_day) == 0
    s = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(start_day))).to_numpy(dtype="datetime64[ns]")
    e = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(end_day))).to_numpy(dtype="datetime64[ns]")
    s, e = np.broadcast_arrays(s, e)
    series = df[col]
    y0 = series.reindex(pd.DatetimeIndex(s)).to_numpy(dtype=float)
    y1 = series.reindex(pd.DatetimeIndex(e)).to_numpy(dtype=float)
    out = y1 - y0
    for b in breaks.get(col, []):
        b64 = np.datetime64(pd.Timestamp(b), "ns")
        out = np.where((s < b64) & (b64 <= e), np.nan, out)
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------- event days
def moved_reason(release_dt: pd.Timestamp, event_day: pd.Timestamp,
                 trading_days: pd.DatetimeIndex) -> str:
    """'' if the event day is the release date, else why it moved."""
    rel_day = release_dt.normalize()
    if event_day == rel_day:
        return ""
    if rel_day.weekday() >= 5:
        return "weekend"
    if rel_day not in trading_days:
        return "holiday/non-trading day"
    return "after close"


def event_days_table(events: pd.DataFrame, cmt: pd.DataFrame, gsw: pd.DataFrame,
                     cfg: dict) -> pd.DataFrame:
    """Event days, previous trading days, moved flags and all d_<col> changes (bp)."""
    days = pd.DatetimeIndex(cmt.index).normalize()
    rel = pd.to_datetime(events["release_datetime_et"])
    eday = fomc_calendar.map_to_trading_day(rel, days, cfg["events"]["yield_close_time_et"])
    pos = days.searchsorted(eday.to_numpy(dtype="datetime64[ns]"))
    prev = pd.Series([days[p - 1] if (not pd.isna(d) and p > 0) else pd.NaT
                      for d, p in zip(eday, pos)], index=events.index)

    out = events[["event_id", "event_type", "meeting_date", "scheduled",
                  "release_datetime_et"]].copy()
    out["event_day"] = eday
    out["prev_day"] = prev
    reasons = [moved_reason(r, d, days) if not pd.isna(d) else "no event day"
               for r, d in zip(rel, eday)]
    out["moved"] = [r != "" for r in reasons]
    out["moved_reason"] = reasons
    out["doc_id"] = events["doc_id"]
    out["has_doc"] = events["doc_id"].notna()
    out["n_events_same_day"] = eday.map(eday.value_counts())

    breaks = build_breaks(cfg)
    for df in (cmt, gsw):
        for col in df.columns:
            out[f"d_{col}"] = change(df, col, out["prev_day"].to_numpy(dtype="datetime64[ns]"),
                                     out["event_day"].to_numpy(dtype="datetime64[ns]"), breaks)
    return out


# --------------------------------------------------------------------------- surprises
def match_source_dates(event_dates: pd.Series, source_dates, max_gap: int = MAX_GAP_DAYS):
    """Match events to source rows by date.

    event_dates: Series indexed by event_id of release calendar dates.
    source_dates: iterable of the source's row dates (unique).
    Exact date matches come first and claim their rows. Each remaining event may take the
    nearest row within +-max_gap calendar days that no other event claims (an exact match
    or a closer fallback); ties for nearest are refused as ambiguous.

    Returns (matches {event_id: source Timestamp}, fallbacks [(event_id, event_date,
    source_date, gap_days)], refused [(event_id, event_date, reason)]).
    """
    src = sorted({pd.Timestamp(d).normalize() for d in source_dates})
    srcset = set(src)
    ev = {eid: pd.Timestamp(d).normalize() for eid, d in event_dates.items()}
    matches, claimed_by = {}, {}
    for eid, d in ev.items():
        if d in srcset:
            matches[eid] = d
            claimed_by.setdefault(d, eid)
    pairs, nearest_tie = [], set()
    for eid, d in ev.items():
        if eid in matches:
            continue
        cands = sorted(((abs((s - d).days), s) for s in src
                        if s not in claimed_by and abs((s - d).days) <= max_gap))
        if len(cands) > 1 and cands[0][0] == cands[1][0]:
            nearest_tie.add(eid)
            continue
        pairs.extend((gap, eid, s) for gap, s in cands)
    fallbacks, used_rows = [], set()
    for gap, eid, s in sorted(pairs, key=lambda x: (x[0], x[1], x[2])):
        if eid in matches or s in used_rows:
            continue
        matches[eid] = s
        used_rows.add(s)
        fallbacks.append((eid, ev[eid], s, gap))
    refused = []
    for eid, d in ev.items():
        if eid in matches:
            continue
        near_claimed = [(s, claimed_by[s]) for s in src
                        if s in claimed_by and 0 < abs((s - d).days) <= max_gap]
        if eid in nearest_tie:
            refused.append((eid, d, "ambiguous: two nearest unclaimed rows equally close"))
        elif near_claimed:
            s, other = near_claimed[0]
            refused.append((eid, d, f"row {s.date()} within {max_gap}d is claimed by {other}"))
    return matches, fallbacks, refused


def attach_surprises(events: pd.DataFrame, table: pd.DataFrame, sources: dict,
                     max_gap: int = MAX_GAP_DAYS):
    """Join surprise columns onto `events` (one event type) by release date.

    table: date + columns of `sources` ({name: (columns, presence column)}).
    Returns (frame indexed like events with all surprise columns + 'surprise_fallback',
    log lines [str], fallbacks [(source, event_id, event_date, source_date)],
    missing [(source, event_id, event_date)]).
    """
    tbl = table.copy()
    tbl["date"] = pd.to_datetime(tbl["date"]).dt.normalize()
    rel_date = pd.to_datetime(events["release_datetime_et"]).dt.normalize()
    rel_date.index = events["event_id"].to_numpy()
    out = pd.DataFrame(index=events.index)
    all_cols = [c for cols, _ in sources.values() for c in cols]
    for c in all_cols:
        out[c] = np.nan if c not in ("bs_time_et", "jk_time_et") else None
    fb_text = pd.Series("", index=events.index)
    log, fallbacks, missing = [], [], []
    id_to_idx = pd.Series(events.index.to_numpy(), index=events["event_id"].to_numpy())
    for name, (cols, presence) in sources.items():
        sub = tbl[tbl[presence].notna()]
        if sub["date"].duplicated().any():
            raise ValueError(f"source {name}: duplicate dates in surprise table")
        matches, fbs, refused = match_source_dates(rel_date, sub["date"], max_gap)
        by_date = sub.set_index("date")
        for eid, sdate in matches.items():
            idx = id_to_idx[eid]
            for c in cols:
                out.at[idx, c] = by_date.at[sdate, c]
        for eid, ed, sd, gap in fbs:
            msg = (f"FALLBACK {name}: {eid} release {ed.date()} <- source row {sd.date()} "
                   f"({(sd - ed).days:+d} days)")
            log.append(msg)
            fallbacks.append((name, eid, ed, sd))
            fb_text.at[id_to_idx[eid]] = (fb_text.at[id_to_idx[eid]] + f"{name}:{sd.date()};")
        for eid, ed, reason in refused:
            log.append(f"REFUSED {name}: {eid} release {ed.date()}: {reason}")
        if len(sub):
            lo, hi = sub["date"].min(), sub["date"].max()
            for eid, ed in rel_date.items():
                if eid not in matches and lo <= ed <= hi:
                    missing.append((name, eid, ed))
                    log.append(f"NO SURPRISE {name}: {eid} release {ed.date()} "
                               f"(source covers {lo.date()}..{hi.date()})")
    out["surprise_fallback"] = fb_text
    return out, log, fallbacks, missing


def add_mps_control(panel: pd.DataFrame) -> pd.DataFrame:
    """mps_control = bs_mps for statements, 0.0 for minutes (no policy action)."""
    panel = panel.copy()
    bs = pd.to_numeric(panel.get("bs_mps", np.nan), errors="coerce")
    panel["mps_control"] = np.where(panel["event_type"] == "minutes", 0.0, bs)
    return panel


# --------------------------------------------------------------------------- coverage
def coverage_counts(g: pd.DataFrame) -> dict:
    """One coverage row (counts) for a group of events."""
    et = g["event_type"]
    usm = np.where(et == "statement", g["usmpd_stmt"].notna(), g["usmpd_min"].notna())
    row = {
        "n_events": len(g),
        "n_scheduled": int(g["scheduled"].sum()),
        "n_unscheduled": int((~g["scheduled"].astype(bool)).sum()),
        "n_with_doc": int(g["has_doc"].sum()),
        "n_moved": int(g["moved"].sum()),
    }
    for reason, col in REASON_COLS.items():
        row[col] = int((g["moved_reason"] == reason).sum())
    row.update({
        "n_with_bs_mps": int(g["bs_mps"].notna().sum()),
        "n_with_usmpd": int(np.sum(usm)),
        "n_with_jk": int(g["jk_pc1"].notna().sum()),
        "n_with_d_slope": int(g["d_slope"].notna().sum()),
        "n_with_d_gsw_slope": int(g["d_gsw_slope"].notna().sum()),
    })
    return row


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows = year x event_type, a 'Total' row per type, and an 'all' grand total."""
    p = panel.copy()
    p["year"] = pd.to_datetime(p["release_datetime_et"]).dt.year
    rows = []
    for et in ("statement", "minutes"):
        sub = p[p["event_type"] == et]
        for y, g in sub.groupby("year"):
            rows.append({"year": str(y), "event_type": et, **coverage_counts(g)})
        rows.append({"year": "Total", "event_type": et, **coverage_counts(sub)})
    rows.append({"year": "Total", "event_type": "all", **coverage_counts(p)})
    return pd.DataFrame(rows)


def coverage_latex(cov: pd.DataFrame) -> str:
    """Compact booktabs table, years down, statements and minutes side by side."""
    stm = cov[cov["event_type"] == "statement"].set_index("year")
    mnt = cov[cov["event_type"] == "minutes"].set_index("year")
    years = [y for y in stm.index if y != "Total"]
    for y in mnt.index:
        if y != "Total" and y not in years:
            years.append(y)
    years = sorted(years) + ["Total"]

    def cell(df, y, col):
        return str(int(df.at[y, col])) if y in df.index else "0"

    lines = [
        r"\begin{tabular}{l rrrrrrr rrrr}",
        r"\toprule",
        r" & \multicolumn{7}{c}{Statements} & \multicolumn{4}{c}{Minutes} \\",
        r"\cmidrule(lr){2-8}\cmidrule(lr){9-12}",
        r"Year & $N$ & Unsch. & Doc & Moved & BS & USMPD & $\Delta$slope"
        r" & $N$ & Doc & Moved & USMPD \\",
        r"\midrule",
    ]
    for y in years:
        if y == "Total":
            lines.append(r"\midrule")
        s = [cell(stm, y, c) for c in ("n_events", "n_unscheduled", "n_with_doc", "n_moved",
                                       "n_with_bs_mps", "n_with_usmpd", "n_with_d_slope")]
        m = [cell(mnt, y, c) for c in ("n_events", "n_with_doc", "n_moved", "n_with_usmpd")]
        lines.append(" & ".join([y] + s + m) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- acceptance
def unmatched_documents(docs: pd.DataFrame, events: pd.DataFrame, start, end) -> pd.DataFrame:
    """In-sample corpus documents whose doc_id is not an event's doc_id, with a reason.

    Reason: text byte-identical to a document that IS matched (the dataset repeats one
    minutes text under several meeting dates), else 'NO REASON' (caller raises).
    """
    in_sample = docs[(docs["meeting_date_ds"] >= str(pd.Timestamp(start).date()))
                     & (docs["meeting_date_ds"] <= str(pd.Timestamp(end).date()))]
    matched_ids = set(events["doc_id"].dropna())
    text_owner = {}
    for r in docs[docs["doc_id"].isin(matched_ids)].itertuples(index=False):
        text_owner.setdefault(r.text, r.doc_id)
    rows = []
    for r in in_sample[~in_sample["doc_id"].isin(matched_ids)].itertuples(index=False):
        owner = text_owner.get(r.text)
        reason = (f"text identical to matched document {owner} (dataset repeats the text under "
                  f"another meeting date; no separate release event)") if owner else "NO REASON"
        rows.append({"doc_id": r.doc_id, "doc_type": r.doc_type,
                     "meeting_date_ds": r.meeting_date_ds, "reason": reason})
    return pd.DataFrame(rows, columns=["doc_id", "doc_type", "meeting_date_ds", "reason"])


def acceptance_checks(ed: pd.DataFrame, docs: pd.DataFrame, events: pd.DataFrame, cfg: dict):
    """Raise AssertionError on any failed acceptance criterion; returns unmatched docs."""
    assert not ed["event_id"].duplicated().any(), "duplicate event_id"
    assert ed["event_day"].notna().all(), \
        f"events without event_day: {ed.loc[ed['event_day'].isna(), 'event_id'].tolist()}"
    assert ed["prev_day"].notna().all(), \
        f"events without prev_day: {ed.loc[ed['prev_day'].isna(), 'event_id'].tolist()}"
    shift = (ed["event_day"] - pd.to_datetime(ed["release_datetime_et"]).dt.normalize()).dt.days
    assert (shift >= 0).all() and (shift <= MAX_EVENT_SHIFT_DAYS).all(), \
        f"event_day shift outside [0, {MAX_EVENT_SHIFT_DAYS}] days: " \
        f"{ed.loc[(shift < 0) | (shift > MAX_EVENT_SHIFT_DAYS), 'event_id'].tolist()}"
    assert (ed["event_day"] > ed["prev_day"]).all(), "prev_day must precede event_day"
    unm = unmatched_documents(docs, events, cfg["sample"]["start"], cfg["sample"]["end"])
    bad = unm[unm["reason"] == "NO REASON"]
    assert bad.empty, f"in-sample documents matching no event and unexplained: " \
                      f"{bad['doc_id'].tolist()}"
    return unm


# --------------------------------------------------------------------------- pipeline
def _fmt(v, nd=1) -> str:
    return "NaN" if pd.isna(v) else f"{v:.{nd}f}"


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    root = repo_root()
    for stage in (yields, surprises, corpus, fomc_calendar):
        print(f"\n##### stage module: {stage.__name__}")
        stage.main(cfg)

    proc = root / cfg["paths"]["data_processed"]
    interim = root / cfg["paths"]["data_interim"]
    tables = root / cfg["paths"]["results_tables"]
    for d in (interim, tables):
        d.mkdir(parents=True, exist_ok=True)

    cmt = pd.read_csv(proc / "yields_cmt_daily.csv", parse_dates=["date"]).set_index("date")
    gsw = pd.read_csv(proc / "yields_gsw_daily.csv", parse_dates=["date"]).set_index("date")
    events = pd.read_csv(proc / "fomc_events.csv", dtype={"doc_id": "string"})
    events["scheduled"] = events["scheduled"].astype(bool)
    docs = pd.read_parquet(root / cfg["paths"]["data_raw"] / "text" / "fomc_documents.parquet")

    print("\n##### integration: event days")
    ed = event_days_table(events, cmt, gsw, cfg)
    unm = acceptance_checks(ed, docs, events, cfg)

    # ---- surprises (statements, minutes) joined on release calendar date
    st_tbl = pd.read_csv(interim / "surprises_statements.csv")
    mi_tbl = pd.read_csv(interim / "surprises_minutes.csv")
    parts, log = [], []
    for et, tbl, sources in (("statement", st_tbl, STATEMENT_SOURCES),
                             ("minutes", mi_tbl, MINUTES_SOURCES)):
        ev = events[events["event_type"] == et]
        part, lg, _, _ = attach_surprises(ev, tbl, sources)
        parts.append(part)
        log += lg
    sur = pd.concat(parts).reindex(events.index)
    for c in SURPRISE_COLUMNS:
        if c not in sur:
            sur[c] = np.nan
    sur["surprise_fallback"] = sur["surprise_fallback"].fillna("")
    panel = pd.concat([ed, sur[SURPRISE_COLUMNS + ["surprise_fallback"]]], axis=1)
    panel = add_mps_control(panel)

    # ---- outputs
    def fmt_dates(df):
        df = df.copy()
        for c in ("event_day", "prev_day"):
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
        return df

    fmt = dict(index=False, float_format="%.8g", lineterminator="\n")
    fmt_dates(ed).to_csv(proc / "event_days.csv", **fmt)
    fmt_dates(panel).to_csv(interim / "event_panel.csv", **fmt)
    cov = coverage_table(panel)
    cov.to_csv(tables / "coverage_events.csv", index=False, lineterminator="\n")
    (tables / "coverage_events.tex").write_text(coverage_latex(cov), encoding="utf-8",
                                                newline="\n")

    # ---- verification summary
    print("\n##### VERIFICATION SUMMARY")
    print(f"events: {len(ed)} ({ed['event_type'].value_counts().to_dict()}); "
          f"unique event_id: {ed['event_id'].is_unique}; "
          f"scheduled {int(ed['scheduled'].sum())}, unscheduled {int((~ed['scheduled']).sum())}; "
          f"with doc {int(ed['has_doc'].sum())}")
    print(f"distinct event days: {ed['event_day'].nunique()}; "
          f"events sharing an event day with another event: {int((ed['n_events_same_day'] > 1).sum())}")
    print(f"corpus documents: {len(docs)}; in-sample unmatched documents: {len(unm)} "
          f"(out-of-sample documents excluded: "
          f"{int((docs['meeting_date_ds'] > str(cfg['sample']['end'])).sum() + (docs['meeting_date_ds'] < str(cfg['sample']['start'])).sum())})")
    if len(unm):
        print(unm.to_string(index=False))

    print("\nsurprise join log:")
    print("\n".join(log) if log else "  (nothing to report)")

    moved = panel[panel["moved"]]
    print(f"\nmoved events ({len(moved)}):")
    print(moved[["event_id", "release_datetime_et", "event_day", "moved_reason"]]
          .assign(event_day=lambda d: pd.to_datetime(d["event_day"]).dt.date)
          .to_string(index=False))

    uns = panel[~panel["scheduled"]]
    print(f"\nunscheduled events ({len(uns)}):")
    print(uns[["event_id", "release_datetime_et", "event_day", "d_y_2y", "d_y_10y", "bs_mps"]]
          .assign(event_day=lambda d: pd.to_datetime(d["event_day"]).dt.date)
          .round(2).to_string(index=False))

    stm = panel[panel["event_type"] == "statement"]
    print("\n10 largest |d_y_2y| statement events:")
    top = stm.reindex(stm["d_y_2y"].abs().sort_values(ascending=False).index).head(10)
    print(top[["event_id", "release_datetime_et", "event_day", "scheduled", "d_y_2y", "d_y_10y",
               "bs_mps"]].assign(event_day=lambda d: pd.to_datetime(d["event_day"]).dt.date)
          .round(2).to_string(index=False))

    print("\nstd (bp) of d_y_2y / d_y_10y:")
    all_diff = cmt[["y_2y", "y_10y"]].diff()
    stm_days = stm.drop_duplicates("event_day")
    min_days = panel[panel["event_type"] == "minutes"].drop_duplicates("event_day")
    for label, a, b, n in (("statement days", stm_days["d_y_2y"], stm_days["d_y_10y"], len(stm_days)),
                           ("minutes days", min_days["d_y_2y"], min_days["d_y_10y"], len(min_days)),
                           ("all trading days", all_diff["y_2y"], all_diff["y_10y"],
                            len(all_diff))):
        print(f"  {label:17s} n={n:5d}  d_y_2y {_fmt(a.std(), 2):>6s}  d_y_10y {_fmt(b.std(), 2):>6s}")

    sched = stm[stm["scheduled"]][["d_y_2y", "bs_mps"]].dropna()
    print(f"\ncorr(d_y_2y, bs_mps) on scheduled statement days: "
          f"{sched['d_y_2y'].corr(sched['bs_mps']):.3f} (n={len(sched)})")
    print(f"\nwrote {(proc / 'event_days.csv').relative_to(root).as_posix()}, "
          f"{(interim / 'event_panel.csv').relative_to(root).as_posix()}, "
          f"results/tables/coverage_events.csv/.tex")


if __name__ == "__main__":
    main()
