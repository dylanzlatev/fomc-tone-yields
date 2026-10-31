"""Authoritative FOMC release calendar, reconciled across independent sources.

The text dataset's dates (corpus.py) are NOT reliable event dates, so the event
calendar is built from four independent sources and every disagreement is logged:

  1. USMPD workbook (SF Fed): statement and minutes release timestamps.
  2. Bauer-Swanson workbook (SF Fed): statement release times (through 2023-12-13).
  3. Federal Reserve website: meeting/call dates, statement release dates (from the
     statement URLs and, for conflicts, the press-release page itself) and minutes
     release dates.
  4. Trillion Dollar Words master file: independent minutes release dates.

Outputs (config paths.data_processed):
  fomc_events.csv            one row per release event inside the sample window
  calendar_reconciliation.csv  every disagreement between sources and its resolution

Resolution rules
  * statements, release date: Fed website > USMPD > Bauer-Swanson > text dataset
  * statements, release time: Fed press release (if it states a time) > USMPD >
    Bauer-Swanson > config default
  * minutes, release date: Fed website > majority of (TDW, USMPD, text dataset);
    an unresolvable tie is labelled UNRESOLVED (never guessed)
  * minutes, release time: USMPD > config default

Also here: `map_to_trading_day`, the event-day rule used downstream.
"""
from __future__ import annotations

import re
import time as _time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ftyields.config import load_config, repo_root
from ftyields.data.download import fetch

FED_ROOT = "https://www.federalreserve.gov"
FED_LICENSE = "U.S. government work (federalreserve.gov), public domain"
SFFED_LICENSE = "no license stated; SF Fed site policies"
TDW_LICENSE = "CC BY-NC 4.0"
TDW_URL = ("https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/"
           "data/master_files/master_mm_final_Oct_2024.xlsx")
SLEEP_S = 1.0           # politeness delay between requests to federalreserve.gov
PROXIMITY_DAYS = 7      # window for fallback matching of misdated documents/rows

SOURCES = ["fed", "usmpd", "bauer_swanson", "tdw", "dataset"]

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_SLUG_RE = re.compile(r"(?<!\d)((?:19|20)\d{2}[01]\d[0-3]\d)(?!\d)")
_FED_DATE_RE = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})")


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #
def _month(tok: str) -> int:
    return _MONTHS[tok.strip().lower()[:3]]


def parse_fed_date(s: str) -> date | None:
    """'Feb 20, 2008' / 'February 20, 2008' / 'April 05, 2011' -> date (else None)."""
    m = _FED_DATE_RE.search(s or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), _month(m.group(1)), int(m.group(2)))
    except (KeyError, ValueError):
        return None


def slug_date(href: str | None) -> date | None:
    """First 8-digit YYYYMMDD in a Fed URL (statement / minutes slugs)."""
    if not href:
        return None
    m = _SLUG_RE.search(href)
    if not m:
        return None
    s = m.group(1)
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None


_HEAD_RE = re.compile(
    r"^\s*(?P<m1>[A-Za-z]+(?:/[A-Za-z]+)?)\s+(?P<d1>\d{1,2})"
    r"(?:\s*-\s*(?:(?P<m2>[A-Za-z]+)\s+)?(?P<d2>\d{1,2}))?"
    r"\s*(?P<tag>\([^)]*\))?\s*(?P<kind>Meeting|Conference Call)?\s*-\s*(?P<year>\d{4})\s*$")


def parse_heading(text: str) -> dict | None:
    """Parse a Fed panel heading such as 'April/May 30-1 Meeting - 2013'.

    Returns dict(date=<last day of the meeting/call>, kind=meeting|call|unscheduled|
    notation_vote|cancelled, scheduled=bool) or None if the heading is not a meeting.
    """
    m = _HEAD_RE.match(re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip())
    if not m:
        return None
    year = int(m.group("year"))
    d1 = int(m.group("d1"))
    d2 = int(m.group("d2")) if m.group("d2") else d1
    m1 = m.group("m1")
    start_month = _month(m1.split("/")[0])
    if "/" in m1:
        end_month = _month(m1.split("/")[1])
    elif m.group("m2"):
        end_month = _month(m.group("m2"))
    elif d2 < d1:
        end_month = start_month % 12 + 1
    else:
        end_month = start_month
    end_year = year + (1 if end_month < start_month else 0)
    tag = (m.group("tag") or "").lower()
    if "cancel" in tag:
        kind = "cancelled"
    elif "notation" in tag:
        kind = "notation_vote"
    elif "unscheduled" in tag:
        kind = "unscheduled"
    elif m.group("kind") == "Conference Call":
        kind = "call"
    else:
        kind = "meeting"
    return {"date": date(end_year, end_month, d2), "kind": kind, "scheduled": kind == "meeting"}


def _statement_href(anchors) -> str | None:
    """First link that is the FOMC statement (text starts with 'Statement', dated slug)."""
    for a in anchors:
        txt = a.get_text(" ", strip=True).lower()
        href = a.get("href", "")
        if txt.startswith("statement") and "implementation" not in txt and slug_date(href):
            return href
    return None


_MIN_REL_RE = re.compile(r"Minutes[^()]{0,60}\(\s*Released\s+([^)]+)\)", re.I | re.S)
_MIN_HOST_RE = re.compile(r"See end of minutes of ([A-Za-z]+ [\d/\-]+) meeting", re.I)


def parse_historical_page(html: str) -> list[dict]:
    """Occasions (meetings, calls, notation votes) from a fomchistorical{year}.htm page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for panel in soup.select("div.panel"):
        head = panel.select_one(".panel-heading")
        if head is None:
            continue
        title = head.get_text(" ", strip=True)
        info = parse_heading(title)
        if info is None:
            continue
        text = panel.get_text(" ", strip=True)
        stmt = _statement_href(panel.find_all("a"))
        min_rel = min_doc = min_href = None
        mrel = _MIN_REL_RE.search(text)
        if mrel:
            min_rel = parse_fed_date(mrel.group(1))
            for p in panel.find_all(["p", "div"]):
                if p.find(["p", "div"]) is None and "Released" in p.get_text() \
                        and re.search(r"Minutes", p.get_text()):
                    for a in p.find_all("a"):
                        if slug_date(a.get("href")):
                            min_href = a.get("href")
                            break
                    break
            min_doc = slug_date(min_href) or info["date"]
        host = _MIN_HOST_RE.search(text)
        out.append({**info, "heading": title, "stmt_href": stmt,
                    "stmt_slug_date": slug_date(stmt),
                    "min_release": min_rel, "min_doc_date": min_doc, "min_href": min_href,
                    "min_host_text": host.group(1) if host else None})
    return out


def parse_calendar_page(html: str) -> list[dict]:
    """Occasions from the current fomccalendars.htm page (2021 onward)."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for panel in soup.select("div.panel"):
        h = panel.select_one(".panel-heading")
        ym = re.match(r"\s*(\d{4}) FOMC Meetings", h.get_text(" ", strip=True)) if h else None
        if not ym:
            continue
        year = int(ym.group(1))
        for row in panel.select("div.fomc-meeting"):
            mo = row.select_one(".fomc-meeting__month")
            dd = row.select_one(".fomc-meeting__date")
            if mo is None or dd is None:
                continue
            dtext = dd.get_text(" ", strip=True).replace("*", "").strip()
            title = f"{mo.get_text(' ', strip=True)} {dtext} - {year}"
            info = parse_heading(title)
            if info is None:
                continue
            stmt = None
            for a in row.find_all("a", href=True):
                if re.search(r"/pressreleases/monetary\d{8}a\.htm$", a["href"]):
                    stmt = a["href"]
                    break
            min_rel = min_doc = min_href = None
            mdiv = row.select_one(".fomc-meeting__minutes")
            if mdiv is not None:
                mrel = re.search(r"\(\s*Released\s+([^)]+)\)", mdiv.get_text(" ", strip=True))
                if mrel:
                    min_rel = parse_fed_date(mrel.group(1))
                    for a in mdiv.find_all("a", href=True):
                        if "fomcminutes" in a["href"] and slug_date(a["href"]):
                            min_href = a["href"]
                            break
                    min_doc = slug_date(min_href) or info["date"]
            out.append({**info, "heading": title, "stmt_href": stmt,
                        "stmt_slug_date": slug_date(stmt),
                        "min_release": min_rel, "min_doc_date": min_doc, "min_href": min_href,
                        "min_host_text": None})
    return out


_PR_DATE_RE = re.compile(
    r"([A-Z][a-z]+ \d{1,2}, \d{4})\s+(?:(?!For immediate release|For release at).){0,400}?"
    r"(?:For immediate release|For release at\s+(\d{1,2}):(\d{2})\s*([ap])\.m\.)", re.S)


def parse_press_release(html: str) -> dict:
    """Release date (and time, when stated) from a Fed press-release page.

    Pages carry a title line like 'June 28, 2007 FOMC statement ... For immediate
    release' or '... For release at 9:15 p.m. EDT'. Returns {'date', 'time'} with None
    for anything not found.
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("#article") or soup.select_one("#content") or soup
    text = body.get_text(" ", strip=True)
    res = {"date": None, "time": None}
    m = re.search(r"Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})", text)
    if m:
        res["date"] = parse_fed_date(m.group(1))
    m2 = _PR_DATE_RE.search(text)
    if m2:
        res["date"] = res["date"] or parse_fed_date(m2.group(1))
        if m2.group(2):
            hh = int(m2.group(2)) % 12 + (12 if m2.group(4).lower() == "p" else 0)
            res["time"] = f"{hh:02d}:{m2.group(3)}"
    return res


# --------------------------------------------------------------------------- #
# workbook loaders
# --------------------------------------------------------------------------- #
def load_usmpd(path: str | Path) -> dict:
    """USMPD workbook -> {'statements': df, 'minutes': df, 'last_update': str}.

    statements: date (release date), time ('HH:MM'), unscheduled (bool), sep (bool)
    minutes:    date (release date), time ('HH:MM')
    """
    path = Path(path)
    st = pd.read_excel(path, sheet_name="Statements", usecols=["Date", "date_time", "SEP", "Unscheduled"])
    mi = pd.read_excel(path, sheet_name="Minutes", usecols=["Date", "date_time"])
    readme = pd.read_excel(path, sheet_name="README", header=None)
    last = ""
    for v in readme.to_numpy().ravel():
        if isinstance(v, str) and v.strip().lower().startswith("last update"):
            last = v.strip()
            break
    return {"statements": usmpd_frame(st, with_flags=True), "minutes": usmpd_frame(mi, with_flags=False),
            "last_update": last}


def usmpd_frame(df: pd.DataFrame, with_flags: bool) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(df["Date"]).dt.normalize(),
                        "time": pd.to_datetime(df["date_time"]).dt.strftime("%H:%M")})
    if with_flags:
        out["unscheduled"] = df["Unscheduled"].astype(int).astype(bool)
        out["sep"] = df["SEP"].astype(int).astype(bool)
    return out.sort_values("date").reset_index(drop=True)


def parse_bs_time(s) -> str | None:
    """'2:00pm' / '11:30am' -> 'HH:MM' (None if missing/unparseable)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*([ap])m\s*$", str(s), re.I)
    if not m:
        return None
    hh = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
    return f"{hh:02d}:{m.group(2)}"


def bs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Bauer-Swanson 'FOMC' sheet (read with na_values=['NA']) -> date, time, unscheduled."""
    out = pd.DataFrame({"date": pd.to_datetime(df["Date"]).dt.normalize(),
                        "time": pd.Series([parse_bs_time(v) for v in df["Time"]], dtype=object),
                        "unscheduled": df["Unscheduled"].fillna(0).astype(int).astype(bool)})
    return out.sort_values(["date", "time"]).reset_index(drop=True)


def load_bs(path: str | Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, na_values=["NA"], usecols=["Date", "Time", "Unscheduled"])
    return bs_frame(df)


def tdw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Trillion Dollar Words master file -> meeting_date (last day), release (date)."""
    def end_date(s: str) -> pd.Timestamp:
        parts = str(s).split("/")           # 'April/May/1/2013' -> month is parts[-3]
        return pd.Timestamp(year=int(parts[-1]), month=_month(parts[-3]), day=int(parts[-2]))
    out = pd.DataFrame({"meeting_date": [end_date(s) for s in df["EndDate"]],
                        "release": pd.to_datetime(df["ReleaseDate"]).dt.normalize(),
                        "url_date": [pd.Timestamp(slug_date(u)) if slug_date(u) else pd.NaT
                                     for u in df["Url"]]})
    return out.sort_values("meeting_date").reset_index(drop=True)


def load_tdw(path: str | Path) -> pd.DataFrame:
    return tdw_frame(pd.read_excel(path))


# --------------------------------------------------------------------------- #
# downloads
# --------------------------------------------------------------------------- #
def _cached_fetch(name: str, url: str, dest: Path, license: str, committed: bool, cfg: dict) -> Path:
    """fetch() plus a 1s politeness sleep when a real network request was made."""
    existed = Path(dest).exists() if Path(dest).is_absolute() else (repo_root() / dest).exists()
    p = fetch(name, url, dest, license=license, committed=committed, cfg=cfg)
    if not existed and "federalreserve.gov" in url:
        _time.sleep(SLEEP_S)
    return p


def download_sources(cfg: dict) -> dict[str, Path]:
    d = cfg["downloads"]["dir"]
    src = cfg["sources"]
    paths = {
        "usmpd": _cached_fetch("usmpd_xlsx", src["usmpd_xlsx"], f"{d}/USMPD.xlsx", SFFED_LICENSE, False, cfg),
        "bs": _cached_fetch("bauer_swanson_xlsx", src["bauer_swanson_xlsx"],
                            f"{d}/monetary-policy-surprises-data.xlsx", SFFED_LICENSE, False, cfg),
        "tdw": _cached_fetch("tdw_master_mm_final_oct_2024", TDW_URL,
                             f"{d}/master_mm_final_Oct_2024.xlsx", TDW_LICENSE, False, cfg),
    }
    return paths


def fetch_fed_occasions(cfg: dict) -> list[dict]:
    """Meetings/calls/notation votes from the Fed calendar page (2021+) and the
    historical pages (<= 2020); every page is cached under downloads/fed_pages."""
    d = cfg["downloads"]["dir"] + "/fed_pages"
    start_year = int(str(cfg["sample"]["start"])[:4])
    occ: list[dict] = []
    p = _cached_fetch("fed_calendar_page", cfg["sources"]["fed_calendar"], f"{d}/fomccalendars.htm",
                      FED_LICENSE, False, cfg)
    for o in parse_calendar_page(Path(p).read_text(encoding="utf-8", errors="replace")):
        if o["date"].year >= 2021:
            occ.append({**o, "source": "fed_calendar"})
    # one year before the sample: the minutes of the Dec 1999 meeting were released 2000-02-03
    for y in range(start_year - 1, 2021):
        url = cfg["sources"]["fed_historical"].format(year=y)
        p = _cached_fetch(f"fed_historical_{y}", url, f"{d}/fomchistorical{y}.htm", FED_LICENSE, False, cfg)
        for o in parse_historical_page(Path(p).read_text(encoding="utf-8", errors="replace")):
            if o["date"].year == y or (o["date"].year == y + 1 and o["date"].month == 1):
                occ.append({**o, "source": f"fed_historical_{y}"})
    occ.sort(key=lambda o: (o["date"], o["heading"]))
    return occ


def fetch_press_release(href: str, cfg: dict) -> dict:
    """Fetch (cached) a Fed press-release page and parse its date/time."""
    d = cfg["downloads"]["dir"] + "/fed_pages"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", href.strip("/").replace(".htm", ""))
    p = _cached_fetch(f"fed_press_{slug}", urljoin(FED_ROOT, href), f"{d}/press_{slug}.htm",
                      FED_LICENSE, False, cfg)
    return parse_press_release(Path(p).read_text(encoding="utf-8", errors="replace"))


# --------------------------------------------------------------------------- #
# helpers on dates / values
# --------------------------------------------------------------------------- #
def _ts(x) -> pd.Timestamp | None:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    return pd.Timestamp(x).normalize()


def _fmt(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    return str(v)


def resolve_value(values: dict, priority: list[str]) -> tuple[object, str, str]:
    """Resolve disagreeing source values.

    values: {source: value or None}. Returns (resolved, resolved_by, reason). Walks
    `priority`; a source that is absent is skipped. The special entry 'majority' in the
    priority list means: most common value among the remaining sources, UNRESOLVED if tied.
    """
    present = {k: v for k, v in values.items() if v is not None and not (not isinstance(v, str) and pd.isna(v))}
    for src in priority:
        if src == "majority":
            rest = {k: v for k, v in present.items() if k not in priority[:priority.index("majority")]}
            if not rest:
                continue
            counts = pd.Series(list(rest.values())).value_counts()
            if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
                val = counts.index[0]
                return val, "majority(" + ",".join(k for k, v in rest.items() if v == val) + ")", \
                    f"{int(counts.iloc[0])} of {len(rest)} remaining sources agree"
            return None, "UNRESOLVED", "tie between remaining sources"
        if src in present:
            return present[src], src, f"priority rule: {' > '.join(priority)}"
    return None, "UNRESOLVED", "no source available"


def is_non_business_day(d: pd.Timestamp) -> str:
    """'weekend' / 'US federal holiday' / '' (federal calendar, not the NYSE calendar)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    d = pd.Timestamp(d).normalize()
    if d.weekday() >= 5:
        return "weekend (" + d.day_name() + ")"
    hol = USFederalHolidayCalendar().holidays(start=d, end=d)
    return "US federal holiday" if len(hol) else ""


# --------------------------------------------------------------------------- #
# the event-day rule (used later with the real FRED trading calendar)
# --------------------------------------------------------------------------- #
def map_to_trading_day(release_datetimes_et: pd.Series, trading_days: pd.DatetimeIndex,
                       close_time_et: str) -> pd.Series:
    """Event day t for each release datetime (naive, Eastern Time).

    t = the release date if it is a trading day AND the release time is strictly before
    `close_time_et`; otherwise the next trading day strictly after the release date.
    Returns normalized Timestamps (NaT where input is NaT or no later trading day exists
    in `trading_days`), indexed like the input.
    """
    rel = pd.to_datetime(release_datetimes_et)
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().unique().sort_values()
    hh, mm = (int(x) for x in close_time_et.split(":")[:2])
    close_minutes = hh * 60 + mm
    out = []
    for ts in rel:
        if pd.isna(ts):
            out.append(pd.NaT)
            continue
        day = ts.normalize()
        minutes = ts.hour * 60 + ts.minute
        pos = days.searchsorted(day, side="left")
        is_trading = pos < len(days) and days[pos] == day
        if is_trading and minutes < close_minutes:
            out.append(day)
            continue
        nxt = days.searchsorted(day, side="right")
        out.append(days[nxt] if nxt < len(days) else pd.NaT)
    return pd.Series(pd.to_datetime(out), index=rel.index, name=getattr(release_datetimes_et, "name", None))



# --------------------------------------------------------------------------- #
# matching and reconciliation
# --------------------------------------------------------------------------- #
def _present(v) -> bool:
    return v is not None and (isinstance(v, str) or not pd.isna(v))


def _doc_lookup(docs: pd.DataFrame) -> dict:
    """{doc_id: namedtuple} with meeting_ts / release_ts as normalized Timestamps."""
    d = docs[["doc_id", "doc_type", "meeting_date_ds", "release_date_ds"]].copy()
    d["meeting_ts"] = pd.to_datetime(d["meeting_date_ds"]).dt.normalize()
    d["release_ts"] = pd.to_datetime(d["release_date_ds"], errors="coerce").dt.normalize()
    return {r.doc_id: r for r in d.itertuples(index=False)}


def match_docs(events: list[dict], docs: pd.DataFrame) -> set:
    """Attach a text-dataset document to each event (sets ev['doc_id'], ev['doc_match']).

    Pass 1: same type and dataset meeting date == event meeting date.
    Pass 2 (events still without a document): the single still-unmatched document of the
    same type whose dataset meeting date is within PROXIMITY_DAYS of the event's, and, where
    both are known, whose dataset release date is within PROXIMITY_DAYS of the event's
    release date. Undated minutes rows (NaN release) are never matched in pass 2: they are
    duplicates of other documents. Ambiguity leaves the event without a document.
    Returns the set of used doc_ids.
    """
    lookup = _doc_lookup(docs)
    used: set[str] = set()
    for ev in events:
        prefix = "S" if ev["event_type"] == "statement" else "M"
        did = f"{prefix}{pd.Timestamp(ev['meeting_date']).strftime('%Y%m%d')}"
        if did in lookup:
            ev["doc_id"], ev["doc_match"] = did, "exact"
            used.add(did)
    for ev in events:
        if ev.get("doc_id"):
            continue
        prefix = "S" if ev["event_type"] == "statement" else "M"
        md = pd.Timestamp(ev["meeting_date"])
        cands = []
        for did, r in lookup.items():
            if not did.startswith(prefix) or did in used:
                continue
            if abs((r.meeting_ts - md).days) > PROXIMITY_DAYS:
                continue
            rel = ev.get("release_date")
            if pd.isna(r.release_ts):
                if prefix == "M":
                    continue
            elif _present(rel) and abs((r.release_ts - rel).days) > PROXIMITY_DAYS:
                continue
            cands.append(did)
        if len(cands) == 1:
            ev["doc_id"], ev["doc_match"] = cands[0], "proximity"
            used.add(cands[0])
    return used


class Recon:
    """Collects reconciliation rows (one per disagreement)."""

    COLS = ["event_id", "event_type", "field", *[f"v_{s}" for s in SOURCES], "resolution",
            "resolved_by", "reason"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, event_id, event_type, field, values: dict, resolution, resolved_by, reason):
        row = {"event_id": event_id, "event_type": event_type, "field": field,
               "resolution": _fmt(resolution), "resolved_by": resolved_by, "reason": reason}
        for s in SOURCES:
            row[f"v_{s}"] = _fmt(values.get(s))
        self.rows.append(row)

    def frame(self, keep_ids: set | None = None) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=self.COLS)
        if keep_ids is not None:
            df = df[df["event_id"].isin(keep_ids)]
        return df.sort_values(["event_id", "field"]).reset_index(drop=True)


def _distinct(values: dict) -> set:
    return {v for v in values.values() if _present(v)}


def _closest(target: pd.Timestamp, candidates: list, window: int = PROXIMITY_DAYS):
    """Candidate nearest to `target` within `window` days (None if none)."""
    best = None
    for c in candidates:
        gap = abs((c - target).days)
        if gap <= window and (best is None or gap < best[0]):
            best = (gap, c)
    return best[1] if best else None


def _transposed(a, b) -> bool:
    """True if two dates in the same month differ by reversed day digits (13 vs 31)."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return a.to_period("M") == b.to_period("M") and a.day != b.day and int(f"{a.day:02d}"[::-1]) == b.day


def _differs(vals: dict, rel) -> str:
    bits = []
    for s, v in vals.items():
        if _present(v) and _present(rel) and _fmt(v) != _fmt(rel):
            hint = " (day digits transposed?)" if _transposed(v, rel) else ""
            bits.append(f"{s}={_fmt(v)}{hint}")
    return ("; overruled: " + ", ".join(bits)) if bits else ""


# ------------------------------- statements -------------------------------- #
def build_statement_events(occ: list[dict], usmpd: pd.DataFrame, bs: pd.DataFrame, recon: Recon,
                           fetch_pr=None, window: tuple | None = None) -> list[dict]:
    """Statement release events from Fed occasions + USMPD + Bauer-Swanson (no dataset yet).

    Matching of a workbook row to a Fed occasion: pass 1 by the statement URL date (the Fed's
    own release date), pass 2 by the meeting/call date. Workbook rows that no occasion claims
    become 'orphan' events (reported). Conflicts among Fed/USMPD/B-S release dates or times
    trigger one fetch of the Fed press-release page (cached) to read its printed date/time.
    """
    u_rows = {r.date: r for r in usmpd.itertuples(index=False)}
    b_rows: dict = {}
    for r in bs.itertuples(index=False):
        b_rows[r.date] = r                      # duplicates on one date: keep the latest time
    u_used: set = set()
    b_used: set = set()
    events: list[dict] = []
    for o in occ:
        if o["kind"] == "cancelled":
            continue
        events.append({"event_type": "statement", "meeting_date": _ts(o["date"]), "occ": o,
                       "slug": _ts(o["stmt_slug_date"]), "u": None, "b": None})

    for pass_no in (1, 2):
        for e in events:
            o = e["occ"]
            if pass_no == 2 and o["stmt_href"] is None and o["kind"] in ("meeting",):
                continue                       # a plain meeting without statement: nothing to claim
            key = e["slug"] if pass_no == 1 else e["meeting_date"]
            if key is None:
                continue
            if e["u"] is None and key in u_rows and key not in u_used:
                e["u"] = u_rows[key]
                u_used.add(key)
            if e["b"] is None and key in b_rows and key not in b_used:
                e["b"] = b_rows[key]
                b_used.add(key)
    events = [e for e in events if e["occ"]["stmt_href"] is not None or e["u"] is not None
              or e["b"] is not None]

    orphan_dates = sorted((set(u_rows) - u_used) | (set(b_rows) - b_used))
    for d in orphan_dates:
        events.append({"event_type": "statement", "meeting_date": d, "occ": None, "slug": None,
                       "u": u_rows.get(d) if d not in u_used else None,
                       "b": b_rows.get(d) if d not in b_used else None, "orphan": True})

    for e in events:
        o = e["occ"]
        eid = f"stmt_{e['meeting_date'].strftime('%Y%m%d')}"
        e["event_id"] = eid
        u, b = e["u"], e["b"]
        vals_date = {"fed": e["slug"], "usmpd": _ts(u.date) if u is not None else None,
                     "bauer_swanson": _ts(b.date) if b is not None else None}
        vals_time = {"usmpd": u.time if u is not None else None,
                     "bauer_swanson": b.time if b is not None else None}
        notes: list[str] = []
        if o is None:
            notes.append("not listed on the Fed meeting pages (workbook row only)")
        elif o["kind"] != "meeting":
            notes.append({"call": "conference call", "unscheduled": "unscheduled meeting",
                          "notation_vote": "notation vote (no meeting)"}[o["kind"]])
        d_conf = len(_distinct(vals_date)) > 1
        t_conf = len({v for v in vals_time.values() if v}) > 1
        no_time = not any(vals_time.values())
        anchor = e["slug"] if e["slug"] is not None else e["meeting_date"]
        in_window = window is None or (window[0] <= anchor <= window[1])      # no fetching for 2026+
        if (d_conf or t_conf or no_time) and o is not None and o["stmt_href"] and fetch_pr is not None                 and in_window:
            try:
                pr = fetch_pr(o["stmt_href"])
            except Exception as exc:        # 404/network: continue, and say so
                pr = {"date": None, "time": None}
                notes.append(f"Fed press-release page unavailable ({type(exc).__name__})")
            if pr["date"] is not None:
                page = _ts(pr["date"])
                if e["slug"] is not None and page != e["slug"]:
                    recon.add(eid, "statement", "fed_url_slug_vs_page_date", {"fed": page}, page, "fed",
                              f"the statement URL carries {_fmt(e['slug'])} but the Fed press-release page "
                              f"itself is dated {_fmt(page)}; the page date is used (URL slug is a typo)")
                    notes.append(f"Fed statement URL slug {_fmt(e['slug'])} is a typo; page dated {_fmt(page)}")
                vals_date["fed"] = page
            if pr["time"]:
                vals_time["fed"] = pr["time"]
                notes.append(f"Fed press release states 'For release at {pr['time']}'")
        if o is not None and o["stmt_href"] is None:
            notes.append("Fed lists no statement page for this event")
        rel, by, why = resolve_value(vals_date, ["fed", "usmpd", "bauer_swanson"])
        e.update({"vals_date": vals_date, "vals_time": vals_time, "notes": notes,
                  "release_date": rel, "scheduled": bool(o["scheduled"]) if o is not None else
                  not ((u is not None and u.unscheduled) or (b is not None and b.unscheduled))})
    return events


def finalize_statements(events: list[dict], docs: pd.DataFrame, cfg: dict, recon: Recon,
                        bs_range: tuple) -> list[dict]:
    """Match documents, then resolve release date/time/scheduled and log disagreements."""
    default_t = str(cfg["events"]["statement_release_time_et"])
    lookup = _doc_lookup(docs)
    match_docs(events, docs)
    for e in events:
        eid, o = e["event_id"], e["occ"]
        ds = lookup.get(e.get("doc_id"))
        vd = dict(e["vals_date"])
        vd["dataset"] = ds.release_ts if ds is not None and not pd.isna(ds.release_ts) else None
        e["vals_date"] = vd
        rel, by, why = resolve_value(vd, ["fed", "usmpd", "bauer_swanson", "dataset"])
        if len(_distinct(vd)) > 1:
            recon.add(eid, "statement", "release_date", vd, rel, by, why + _differs(vd, rel))
        e["release_date"] = rel
        if ds is not None and ds.meeting_ts != e["meeting_date"]:
            recon.add(eid, "statement", "meeting_date",
                      {"fed": e["meeting_date"] if o is not None else None, "dataset": ds.meeting_ts},
                      e["meeting_date"], "fed" if o is not None else "workbook",
                      "dataset document is dated differently from the meeting/call; it was matched by "
                      "proximity (the only unmatched statement within 7 days, with consistent release "
                      "date) and only its text is used")
            e["notes"].append(f"dataset document {ds.doc_id} is mis-dated vs the Fed ({_fmt(ds.meeting_ts)})")
        vt = dict(e["vals_time"])
        tval, tby, twhy = resolve_value(vt, ["fed", "usmpd", "bauer_swanson"])
        if len({v for v in vt.values() if v}) > 1:
            extra = ""
            if tby == "usmpd" and vt.get("bauer_swanson"):
                gap = abs(_minutes(vt["usmpd"]) - _minutes(vt["bauer_swanson"]))
                extra = (f"; Bauer-Swanson differs by {gap} min (USMPD carries the minute-level "
                         "timestamp; B-S times look rounded). Same trading day either way")
            recon.add(eid, "statement", "release_time", vt, tval, tby, twhy + extra)
        if tval is None:
            tval, tby = default_t, "config_default"
        e["time"] = tval
        e["time_source"] = {"fed": "fed_press_release", "usmpd": "usmpd", "bauer_swanson": "bauer_swanson",
                            "config_default": "config_default"}[tby]
        flags = {"fed": bool(o["scheduled"]) if o is not None else None,
                 "usmpd": (not e["u"].unscheduled) if e["u"] is not None else None,
                 "bauer_swanson": (not e["b"].unscheduled) if e["b"] is not None else None}
        if len(_distinct(flags)) > 1:
            recon.add(eid, "statement", "scheduled", flags, e["scheduled"], "fed" if o is not None else "workbook",
                      "Fed heading (meeting vs conference call/unscheduled/notation vote) wins")
        # presence in each source (only where the source covers the date)
        pres = {"fed": "yes" if o is not None else "no",
                "usmpd": "yes" if e["u"] is not None else "no",
                "bauer_swanson": "yes" if e["b"] is not None else "no"}
        rd = _ts(rel)
        if not (bs_range[0] <= rd <= bs_range[1]):
            pres["bauer_swanson"] = None
        if rd is not None and len(_distinct(pres)) > 1:
            weekend = " (Saturday/Sunday release: no same-day market window)" if rd.weekday() >= 5 else ""
            recon.add(eid, "statement", "present_in_source", pres, "event kept", "fed+usmpd",
                      "listed by the Fed website and/or USMPD but missing from another source; "
                      "kept, since the Fed website is authoritative" + weekend)
        e["n_sources_agree"] = sum(1 for s in ("fed", "usmpd", "bauer_swanson", "dataset")
                                   if _present(vd.get(s)) and _ts(vd[s]) == _ts(rel))
        _calendar_notes(e, rel, statement=True)
        e["release_dt"] = _stamp(rel, e["time"])
    return events


def _calendar_notes(e: dict, rel, statement: bool) -> None:
    """Notes on releases that do not fall on the meeting day / on a business day."""
    if not _present(rel):
        return
    rel = _ts(rel)
    if statement and rel != e["meeting_date"]:
        e["notes"].append(f"released {rel.strftime('%Y-%m-%d')}, not on the meeting/call date "
                          f"{e['meeting_date'].strftime('%Y-%m-%d')}")
    why = is_non_business_day(rel)
    if why:
        e["notes"].append(f"release on a non-business day: {why}")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")[:2]
    return int(h) * 60 + int(m)


def _stamp(d, hhmm: str):
    if not _present(d):
        return None
    return pd.Timestamp(f"{pd.Timestamp(d).strftime('%Y-%m-%d')} {hhmm}")


# -------------------------------- minutes ---------------------------------- #
def build_minutes_events(occ: list[dict], tdw: pd.DataFrame, usmpd_min: pd.DataFrame, docs: pd.DataFrame,
                         cfg: dict, recon: Recon) -> list[dict]:
    """Minutes release events keyed by the date of the meeting whose minutes document is released.

    A conference call whose minutes sit inside another meeting's minutes (e.g. the Aug 10 and
    Aug 16, 2007 calls inside the Sep 18, 2007 minutes) does not get its own event.
    """
    default_t = str(cfg["events"]["minutes_release_time_et"])
    lookup = _doc_lookup(docs)
    by_date = {_ts(o["date"]): o for o in occ}
    ev: dict = {}
    for o in occ:
        if o["min_release"] is None or o["kind"] == "cancelled":
            continue
        key = _ts(o["min_doc_date"])
        rel = _ts(o["min_release"])
        e = ev.setdefault(key, {"event_type": "minutes", "meeting_date": key, "fed": rel, "tdw": None,
                                "u": None, "occs": [], "notes": []})
        if e["fed"] != rel:
            e["notes"].append(f"Fed pages disagree on release ({_fmt(e['fed'])} vs {_fmt(rel)})")
        e["occs"].append(o)
    for e in ev.values():
        host = by_date.get(e["meeting_date"])
        e["scheduled"] = bool(host["scheduled"]) if host is not None else False
        calls = sorted(_ts(o["date"]) for o in e["occs"] if _ts(o["date"]) != e["meeting_date"])
        if calls:
            e["notes"].append("document also contains the minutes of call(s) " +
                              ", ".join(c.strftime("%Y-%m-%d") for c in calls))
    events = list(ev.values())
    # TDW (by meeting end date), linked by release date if the meeting date differs
    for r in tdw.itertuples(index=False):
        key = _ts(r.meeting_date)
        hit = ev.get(key)
        if hit is None:
            hit = next((e for e in events if e["fed"] == r.release and e["tdw"] is None), None)
            if hit is not None:
                hit["notes"].append(f"TDW dates this meeting {_fmt(key)}")
        if hit is None:
            hit = {"event_type": "minutes", "meeting_date": key, "fed": None, "tdw": None, "u": None,
                   "occs": [], "notes": ["not on the Fed pages parsed here (TDW only)"],
                   "scheduled": True}
            ev[key] = hit
            events.append(hit)
        hit["tdw"] = r.release
    for e in events:
        e["event_id"] = f"min_{e['meeting_date'].strftime('%Y%m%d')}"
        e["release_date"] = e["fed"] if e["fed"] is not None else e["tdw"]
    match_docs(events, docs)
    for e in events:
        ds = lookup.get(e.get("doc_id"))
        e["ds"] = ds
        e["ds_release"] = ds.release_ts if ds is not None and not pd.isna(ds.release_ts) else None
    # USMPD minutes rows: exact release-date match with any other source, else nearest within 7 days
    u_left = list(usmpd_min.itertuples(index=False))
    for r in list(u_left):
        cands = [e for e in events if e["u"] is None and
                 r.date in {_ts(e["fed"]) if e["fed"] is not None else None,
                            _ts(e["tdw"]) if e["tdw"] is not None else None, e["ds_release"]}]
        if len(cands) >= 1:
            cands[0]["u"] = r
            u_left.remove(r)
    # leftovers: (a) order-based pairing -- a row lying between two matched rows is paired with the
    # single unmatched event lying in the same gap (catches a mistyped USMPD date, e.g. 2001-12-31
    # for 2001-12-13); (b) otherwise the nearest unmatched event within PROXIMITY_DAYS.
    for r in list(u_left):
        anchors = [e["u"].date for e in events if e["u"] is not None]
        lo = max([a for a in anchors if a < r.date], default=None)
        hi = min([a for a in anchors if a > r.date], default=None)
        in_gap = lambda d: (lo is None or d > lo) and (hi is None or d < hi)      # noqa: E731
        cands = [e for e in events if e["u"] is None and e["release_date"] is not None
                 and in_gap(_ts(e["release_date"]))]
        rows_in_gap = [x for x in u_left if in_gap(x.date)]
        if len(cands) == 1 and len(rows_in_gap) == 1:
            cands[0]["u"] = r
            u_left.remove(r)
    for r in list(u_left):
        pool = [e for e in events if e["u"] is None and e["release_date"] is not None]
        near = _closest(r.date, [_ts(e["release_date"]) for e in pool])
        hit = next((e for e in pool if _ts(e["release_date"]) == near), None) if near is not None else None
        if hit is not None:
            hit["u"] = r
            u_left.remove(r)
    for r in u_left:                           # rows no event explains
        events.append({"event_type": "minutes", "meeting_date": r.date, "fed": None, "tdw": None, "u": r,
                       "occs": [], "notes": ["USMPD row only (no Fed/TDW/dataset counterpart)"],
                       "scheduled": True, "event_id": f"min_{r.date.strftime('%Y%m%d')}",
                       "release_date": r.date, "doc_id": None, "ds": None, "ds_release": None})
    return events


def finalize_minutes(events: list[dict], cfg: dict, recon: Recon, usmpd_range: tuple, tdw_range: tuple) -> list[dict]:
    default_t = str(cfg["events"]["minutes_release_time_et"])
    for e in events:
        eid = e["event_id"]
        u = e["u"]
        vd = {"fed": e["fed"], "tdw": e["tdw"], "usmpd": _ts(u.date) if u is not None else None,
              "dataset": e["ds_release"]}
        rel, by, why = resolve_value(vd, ["fed", "majority"])
        if len(_distinct(vd)) > 1:
            if rel is None:
                e["notes"].append("UNRESOLVED minutes release date: " + why)
            recon.add(eid, "minutes", "release_date", vd, rel, by, why + _differs(vd, rel))
        if rel is None and len(_distinct(vd)) == 1:
            rel = next(iter(_distinct(vd)))
        e["release_date"] = rel
        ds = e.get("ds")
        if ds is not None and ds.meeting_ts != e["meeting_date"]:
            recon.add(eid, "minutes", "meeting_date",
                      {"fed": e["meeting_date"] if e["occs"] else None, "dataset": ds.meeting_ts},
                      e["meeting_date"], "fed" if e["occs"] else "tdw",
                      "dataset document is dated differently from the meeting; matched by proximity "
                      "(only unmatched minutes within 7 days, release date consistent); only its text is used")
            e["notes"].append(f"dataset document {ds.doc_id} is mis-dated vs the Fed ({_fmt(ds.meeting_ts)})")
        if u is not None:
            tval, tsrc = u.time, "usmpd"
            if u.time != default_t:
                e["notes"].append(f"USMPD time {u.time} differs from the usual {default_t}")
        else:
            tval, tsrc = default_t, "config_default"
        e["time"], e["time_source"] = tval, tsrc
        pres = {"fed": "yes" if e["fed"] is not None else "no", "tdw": "yes" if e["tdw"] is not None else "no",
                "usmpd": "yes" if u is not None else "no"}
        rd = _ts(rel)
        if rd is not None:
            if not (tdw_range[0] <= rd <= tdw_range[1]):
                pres["tdw"] = None
            if not (usmpd_range[0] <= rd <= usmpd_range[1]):
                pres["usmpd"] = None
            if len(_distinct(pres)) > 1:
                recon.add(eid, "minutes", "present_in_source", pres, "event kept", "fed",
                          "listed by the Fed website but missing from another source covering that date")
        e["n_sources_agree"] = sum(1 for s in ("fed", "tdw", "usmpd", "dataset")
                                   if _present(vd.get(s)) and _ts(vd[s]) == _ts(rel))
        _calendar_notes(e, rel, statement=False)
        e["release_dt"] = _stamp(rel, e["time"])
    return events


# ------------------------------- assembly ---------------------------------- #
EVENT_COLS = ["event_id", "event_type", "meeting_date", "scheduled", "release_datetime_et", "time_source",
              "doc_id", "n_sources_agree", "notes"]


def assemble_events(stmt: list[dict], mins: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for e in stmt + mins:
        rdt = e.get("release_dt")
        if rdt is None or pd.isna(rdt):
            rows.append({**_row(e, None), "_in": True})       # unresolved: keep, loudly
            continue
        if not (start <= pd.Timestamp(rdt).normalize() <= end):
            continue
        rows.append(_row(e, rdt))
    df = pd.DataFrame(rows, columns=EVENT_COLS)
    return df.sort_values(["release_datetime_et", "event_type", "event_id"]).reset_index(drop=True)


def _row(e: dict, rdt) -> dict:
    return {"event_id": e["event_id"], "event_type": e["event_type"],
            "meeting_date": e["meeting_date"].strftime("%Y-%m-%d"), "scheduled": bool(e["scheduled"]),
            "release_datetime_et": rdt.strftime("%Y-%m-%d %H:%M") if rdt is not None else "",
            "time_source": e.get("time_source", ""), "doc_id": e.get("doc_id") or "",
            "n_sources_agree": e.get("n_sources_agree", 0), "notes": "; ".join(e["notes"])}


# --------------------------------- main ------------------------------------ #
def _read_docs_meta(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `python -m ftyields.data.corpus` first")
    return pd.read_parquet(path)


def _p(*a) -> None:
    print(*a, flush=True)


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    root = repo_root()
    start = pd.Timestamp(str(cfg["sample"]["start"]))
    end = pd.Timestamp(str(cfg["sample"]["end"]))
    out_dir = root / cfg["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = download_sources(cfg)
    occ = fetch_fed_occasions(cfg)
    usmpd = load_usmpd(paths["usmpd"])
    bs = load_bs(paths["bs"], cfg["sources"]["bauer_swanson_sheet"])
    tdw = load_tdw(paths["tdw"])
    docs_full = _read_docs_meta(root / cfg["paths"]["data_raw"] / "text" / "fomc_documents.parquet")
    docs = docs_full.drop(columns=["text"])
    _p(f"USMPD: {usmpd['last_update']}; statements {len(usmpd['statements'])}, minutes {len(usmpd['minutes'])}")
    _p(f"Bauer-Swanson rows {len(bs)} ({bs['date'].min().date()}..{bs['date'].max().date()}); "
       f"TDW rows {len(tdw)}; Fed occasions parsed {len(occ)}")

    recon = Recon()
    stm = build_statement_events(occ, usmpd["statements"], bs, recon,
                                 fetch_pr=lambda href: fetch_press_release(href, cfg), window=(start, end))
    stm = finalize_statements(stm, docs, cfg, recon, (bs["date"].min(), bs["date"].max()))
    mins = build_minutes_events(occ, tdw, usmpd["minutes"], docs, cfg, recon)
    mins = finalize_minutes(mins, cfg, recon,
                            (usmpd["minutes"]["date"].min(), usmpd["minutes"]["date"].max()),
                            (tdw["release"].min(), tdw["release"].max()))

    # duplicates in the dataset (same text under different doc_ids)
    text_groups = docs_full.groupby("text")["doc_id"].apply(list)
    dups = {d: [x for x in g if x != d] for g in text_groups for d in g if len(g) > 1}
    for e in stm + mins:
        if e.get("doc_id") in dups:
            e["notes"].append("dataset also holds identical text as " + ",".join(sorted(dups[e["doc_id"]])))

    events = assemble_events(stm, mins, start, end)
    events.to_csv(out_dir / "fomc_events.csv", index=False)
    recon_df = recon.frame(set(events["event_id"]))
    recon_df.to_csv(out_dir / "calendar_reconciliation.csv", index=False)
    report(events, recon_df, occ, stm, mins, docs_full, dups, usmpd, bs, start, end)


def report(events, recon_df, occ, stm, mins, docs_full, dups, usmpd, bs, start, end) -> None:
    _p("\n=== events per year x type ===")
    ev = events.copy()
    ev["year"] = ev["release_datetime_et"].str[:4]
    _p(ev.pivot_table(index="year", columns="event_type", values="event_id", aggfunc="count",
                      fill_value=0, margins=True).to_string())

    _p("\n=== unscheduled statements (scheduled == False) ===")
    un = ev[(ev["event_type"] == "statement") & (~ev["scheduled"])]
    for r in un.itertuples():
        _p(f"{r.event_id}  meeting {r.meeting_date}  release {r.release_datetime_et} ET "
           f"[{r.time_source}]  doc={r.doc_id or '-'}  sources={r.n_sources_agree}  {r.notes}")

    _p("\n=== USMPD vs Bauer-Swanson unscheduled sets, 2000-2023 ===")
    s_u = usmpd["statements"]; s_u = s_u[(s_u["date"] >= start) & (s_u["date"] <= pd.Timestamp("2023-12-31"))]
    s_b = bs[(bs["date"] >= start) & (bs["date"] <= pd.Timestamp("2023-12-31"))]
    _p(f"USMPD unscheduled: {int(s_u['unscheduled'].sum())}; Bauer-Swanson unscheduled: "
       f"{int(s_b['unscheduled'].sum())}")
    ud = set(s_u.loc[s_u["unscheduled"], "date"]); bd = set(s_b.loc[s_b["unscheduled"], "date"])
    _p("in USMPD only (by date):", sorted(d.strftime("%Y-%m-%d") for d in ud - bd))
    _p("in B-S only (by date):", sorted(d.strftime("%Y-%m-%d") for d in bd - ud))
    for e in stm:
        u, b = e["u"], e["b"]
        if (u is not None and u.unscheduled) or (b is not None and b.unscheduled):
            if _ts(e["release_date"]) is not None and not (start <= _ts(e["release_date"]) <= pd.Timestamp("2023-12-31")):
                continue
            ut = f"{u.date.date()} {u.time}" if u is not None else "absent"
            bt = f"{b.date.date()} {b.time}" if b is not None else "absent"
            same = "" if (u is not None and b is not None and u.date == b.date) else "   <-- differs"
            _p(f"  {e['event_id']}: USMPD {ut} | B-S {bt}{same}")
            for r in recon_df[recon_df["event_id"] == e["event_id"]].itertuples():
                _p(f"      [{r.field}] -> {r.resolution} by {r.resolved_by}: {r.reason}")

    _p("\n=== releases on weekends / US federal holidays ===")
    for r in ev.itertuples():
        why = is_non_business_day(pd.Timestamp(r.release_datetime_et[:10]))
        if why:
            _p(f"{r.event_id}  {r.release_datetime_et}  {why}")

    _p("\n=== several minutes documents on one release date ===")
    mm = ev[ev["event_type"] == "minutes"]
    by_rel: dict = {}
    for o in occ:
        if o["min_release"] is not None:
            by_rel.setdefault(_ts(o["min_release"]), []).append(o)
    for d, os_ in sorted(by_rel.items()):
        if len(os_) > 1 and start <= d <= end:
            _p(f"Fed pages: {d.date()} lists minutes for " + ", ".join(
                f"{_ts(o['date']).date()} ({o['kind']}, doc {_ts(o['min_doc_date']).date()})" for o in os_)
               + " -> one document, one event "
               + ",".join(sorted({"min_" + _ts(o["min_doc_date"]).strftime("%Y%m%d") for o in os_})))
    dm = docs_full[docs_full["doc_type"] == "minutes"].dropna(subset=["release_date_ds"])
    for d, g in dm.groupby("release_date_ds"):
        if len(g) > 1 and start <= pd.Timestamp(d) <= end:
            _p(f"dataset: release {d} is shared by {sorted(g['doc_id'])}")
    dup_ev = mm[mm["release_datetime_et"].str[:10].duplicated(keep=False)]
    _p(f"events sharing a release date: {sorted(dup_ev['event_id'])}")

    _p("\n=== minutes of conference calls / non-meeting occasions (no separate minutes event) ===")
    for o in occ:
        if o["kind"] in ("call", "unscheduled", "notation_vote"):
            d = _ts(o["date"])
            if not (start <= d <= end):
                continue
            if o["min_doc_date"] is not None and _ts(o["min_doc_date"]) != d:
                _p(f"{d.date()} {o['kind']}: minutes inside document {_ts(o['min_doc_date']).date()} "
                   f"(released {_ts(o['min_release']).date()})")
            elif o["min_host_text"]:
                _p(f"{d.date()} {o['kind']}: minutes at end of the minutes of {o['min_host_text']}")
            elif o["min_release"] is None:
                _p(f"{d.date()} {o['kind']}: no minutes listed")

    ids = set(ev["doc_id"]) - {""}
    d_in = docs_full[(pd.to_datetime(docs_full["meeting_date_ds"]) >= start) &
                     (pd.to_datetime(docs_full["meeting_date_ds"]) <= end)]
    _p("\n=== dataset documents (meeting date in window) matching no event ===")
    un_docs = d_in[~d_in["doc_id"].isin(ids)]
    for r in un_docs.sort_values("doc_id").itertuples():
        dup = f"identical text to {','.join(sorted(dups[r.doc_id]))}" if r.doc_id in dups else "no duplicate"
        _p(f"{r.doc_id}  meeting {r.meeting_date_ds}  release {r.release_date_ds or 'NaN'}  {dup}")
    _p(f"({len(un_docs)} documents; {len(d_in)} documents in window; {len(docs_full)} total)")

    _p("\n=== events without a dataset document ===")
    for r in ev[ev["doc_id"] == ""].itertuples():
        _p(f"{r.event_id}  {r.release_datetime_et}  scheduled={r.scheduled}")

    _p("\n=== reconciliation summary ===")
    if len(recon_df):
        _p(recon_df.groupby(["event_type", "field", "resolved_by"]).size().to_string())
    unres = recon_df[recon_df["resolved_by"] == "UNRESOLVED"]
    _p(f"UNRESOLVED cases: {len(unres)}")
    for r in unres.itertuples():
        _p(r)
    in_win = lambda e: _present(e.get("release_dt")) and start <= pd.Timestamp(e["release_dt"]).normalize() <= end  # noqa: E731
    orph = [e["event_id"] for e in stm + mins if in_win(e) and (e.get("orphan") or any(
        "only" in n and ("workbook" in n or "USMPD row" in n or "TDW only" in n) for n in e["notes"]))]
    _p(f"events found in a workbook/TDW but not on the Fed pages (orphans): {orph}")
    bad = ev[(ev["release_datetime_et"] == "")]
    _p(f"events with no release datetime: {len(bad)}")


if __name__ == "__main__":
    main()
