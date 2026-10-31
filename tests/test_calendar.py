"""Tests for ftyields.data.calendar (parsers, reconciliation rules, trading-day mapping). No network."""
from datetime import date

import pandas as pd
import pytest

from ftyields.data import calendar as cal

CFG = {"events": {"statement_release_time_et": "14:00", "minutes_release_time_et": "14:00"}}


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,end,kind,sched", [
    ("February 1-2 Meeting - 2000", date(2000, 2, 2), "meeting", True),
    ("July 31-August 1  Meeting - 2012", date(2012, 8, 1), "meeting", True),
    ("April/May 30-1 Meeting - 2013", date(2013, 5, 1), "meeting", True),
    ("Jan/Feb 31-1 Meeting - 2017", date(2017, 2, 1), "meeting", True),
    ("January 3 Conference Call - 2001", date(2001, 1, 3), "call", False),
    ("October 16 (unscheduled) - 2013", date(2013, 10, 16), "unscheduled", False),
    ("March  4 (unscheduled) - 2014", date(2014, 3, 4), "unscheduled", False),
    ("March 15 (unscheduled) Meeting - 2020", date(2020, 3, 15), "unscheduled", False),
    ("March 19 (notation vote) - 2020", date(2020, 3, 19), "notation_vote", False),
    ("March 17-18 (cancelled) Meeting - 2020", date(2020, 3, 18), "cancelled", False),
])
def test_parse_heading(text, end, kind, sched):
    h = cal.parse_heading(text)
    assert h == {"date": end, "kind": kind, "scheduled": sched}


def test_parse_heading_rejects_non_meetings():
    assert cal.parse_heading("Beige Book - 2008") is None
    assert cal.parse_heading("Transcripts and other historical materials") is None


def test_parse_fed_date_variants():
    assert cal.parse_fed_date("Feb 20, 2008") == date(2008, 2, 20)
    assert cal.parse_fed_date("(Released February 19, 2025)") == date(2025, 2, 19)
    assert cal.parse_fed_date("April 05, 2011") == date(2011, 4, 5)
    assert cal.parse_fed_date("no date here") is None


def test_slug_date():
    assert cal.slug_date("/newsevents/pressreleases/monetary20190501a.htm") == date(2019, 5, 1)
    assert cal.slug_date("/boarddocs/press/general/2000/20000202/") == date(2000, 2, 2)
    assert cal.slug_date("/fomc/minutes/20070918.htm#august10") == date(2007, 9, 18)
    assert cal.slug_date("/monetarypolicy/files/FOMC_LongerRunGoals_201201.pdf") is None
    assert cal.slug_date(None) is None


# --------------------------------------------------------------------------- #
# Fed page parsers
# --------------------------------------------------------------------------- #
HIST_HTML = """
<html><body>
<div class="panel panel-default"><div class="panel-heading"><h5>January 21 Conference Call - 2008</h5></div>
 <div class="row divided-row panel-body"><div class="col-xs-12 col-md-12">
  <p><a href="/newsevents/press/monetary/20080122b.htm">Statement</a></p>
  Minutes: <EM>See end of minutes of January 29/30 meeting</EM>
  <p><a href="/monetarypolicy/files/FOMC20080121confcall.pdf">Transcript (628 KB PDF)</a></p>
 </div></div></div>
<div class="panel panel-default"><div class="panel-heading"><h5>January 29-30 Meeting - 2008</h5></div>
 <div class="row divided-row panel-body"><div class="col-xs-12 col-md-6">
  <p><a href="/newsevents/press/monetary/20080130a.htm">Statement</a></p>
  <p>Minutes (Released Feb 20, 2008):<br /><a href="/monetarypolicy/fomcminutes20080130.htm">HTML</a> |
     <a href="/monetarypolicy/files/fomcminutes20080130.pdf">364 KB PDF</a></p>
 </div></div></div>
<div class="panel panel-default"><div class="panel-heading"><h5>August 10 Conference Call - 2007</h5></div>
 <div class="row divided-row panel-body"><div class="col-xs-12 col-md-12">
  <p><a href="/newsevents/press/monetary/20070810a.htm">Statement</a></p>
  <p><a href="/fomc/minutes/20070918.htm#august10">Minutes</a> (Released Oct 9, 2007)</p>
 </div></div></div>
<div class="panel panel-default"><div class="panel-heading"><h5>March 17-18 (cancelled) Meeting - 2020</h5></div>
 <div class="row divided-row panel-body"><div class="col-xs-12"></div></div></div>
<div class="panel panel-default"><div class="panel-heading"><h5>Some other box</h5></div></div>
</body></html>
"""


def test_parse_historical_page():
    occ = {o["date"]: o for o in cal.parse_historical_page(HIST_HTML)}
    call = occ[date(2008, 1, 21)]
    assert call["kind"] == "call" and not call["scheduled"]
    assert call["stmt_slug_date"] == date(2008, 1, 22)          # statement came out the next day
    assert call["min_release"] is None and call["min_host_text"] == "January 29/30"
    meet = occ[date(2008, 1, 30)]
    assert meet["scheduled"] and meet["stmt_slug_date"] == date(2008, 1, 30)
    assert meet["min_release"] == date(2008, 2, 20) and meet["min_doc_date"] == date(2008, 1, 30)
    hosted = occ[date(2007, 8, 10)]
    assert hosted["min_release"] == date(2007, 10, 9)
    assert hosted["min_doc_date"] == date(2007, 9, 18)          # minutes live inside the Sep 18 document
    assert occ[date(2020, 3, 18)]["kind"] == "cancelled" and occ[date(2020, 3, 18)]["stmt_href"] is None
    assert len(occ) == 4                                          # the non-meeting box is ignored


CAL_HTML = """
<html><body>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2019 FOMC Meetings</a></h4></div>
 <div class="row fomc-meeting">
  <div class="fomc-meeting__month col-xs-5"><strong>April/May</strong></div>
  <div class="fomc-meeting__date col-xs-4">30-1</div>
  <div class="col-xs-12"><strong>Statement:</strong><br>
     <a href="/monetarypolicy/files/monetary20190501a1.pdf">PDF</a> |
     <a href="/newsevents/pressreleases/monetary20190501a.htm">HTML</a><br>
     <a href="/newsevents/pressreleases/monetary20190501a1.htm">Implementation Note</a></div>
  <div class="col-xs-12 fomc-meeting__minutes"><strong>Minutes:</strong><br>
     <a href="/monetarypolicy/files/fomcminutes20190501.pdf">PDF</a> |
     <a href="/monetarypolicy/fomcminutes20190501.htm">HTML</a><br> (Released May 22, 2019)</div>
 </div>
 <div class="fomc-meeting--shaded row fomc-meeting">
  <div class="fomc-meeting__month"><strong>August</strong></div>
  <div class="fomc-meeting__date">22 (notation vote)</div>
  <div class="col-xs-12"><a href="/newsevents/pressreleases/monetary20190822a.htm">Statement on Longer-Run Goals</a></div>
  <div class="col-xs-12 fomc-meeting__minutes"></div>
 </div>
 <div class="row fomc-meeting">
  <div class="fomc-meeting__month"><strong>September</strong></div>
  <div class="fomc-meeting__date">17-18*</div>
  <div class="col-xs-12"></div>
  <div class="col-xs-12 fomc-meeting__minutes"></div>
 </div>
</div></body></html>
"""


def test_parse_calendar_page():
    occ = {o["date"]: o for o in cal.parse_calendar_page(CAL_HTML)}
    assert set(occ) == {date(2019, 5, 1), date(2019, 8, 22), date(2019, 9, 18)}
    m = occ[date(2019, 5, 1)]
    assert m["scheduled"] and m["stmt_slug_date"] == date(2019, 5, 1)      # 'a1' implementation note ignored
    assert m["min_release"] == date(2019, 5, 22) and m["min_doc_date"] == date(2019, 5, 1)
    nv = occ[date(2019, 8, 22)]
    assert nv["kind"] == "notation_vote" and not nv["scheduled"] and nv["stmt_slug_date"] == date(2019, 8, 22)
    future = occ[date(2019, 9, 18)]
    assert future["stmt_href"] is None and future["min_release"] is None


def test_parse_press_release_date_and_time():
    imm = "<html><body><div id='article'>June 28, 2007 FOMC statement For immediate release Share The FOMC decided</div></body></html>"
    assert cal.parse_press_release(imm) == {"date": date(2007, 6, 28), "time": None}
    pm = ("<html><body><div id='article'>May 09, 2010 FOMC statement: Federal Reserve, European Central Bank "
          "announce reestablishment of swap facilities For release at 9:15 p.m. EDT Share In response</div></body></html>")
    assert cal.parse_press_release(pm) == {"date": date(2010, 5, 9), "time": "21:15"}
    am = "<html><body><div id='article'>October 11, 2019 Statement Regarding Monetary Policy For release at 11:00 a.m. EDT Share</div></body></html>"
    assert cal.parse_press_release(am) == {"date": date(2019, 10, 11), "time": "11:00"}
    noon = "<html><body><div id='article'>March 1, 2011 Statement For release at 12:30 p.m. EST Share</div></body></html>"
    assert cal.parse_press_release(noon)["time"] == "12:30"
    assert cal.parse_press_release("<html><body>nothing</body></html>") == {"date": None, "time": None}


# --------------------------------------------------------------------------- #
# workbook loaders (xlsx-like frames and tiny real xlsx files)
# --------------------------------------------------------------------------- #
def test_parse_bs_time():
    assert cal.parse_bs_time("2:00pm") == "14:00"
    assert cal.parse_bs_time("11:30am") == "11:30"
    assert cal.parse_bs_time("12:30pm") == "12:30"
    assert cal.parse_bs_time("12:05am") == "00:05"
    assert cal.parse_bs_time(None) is None
    assert cal.parse_bs_time(float("nan")) is None
    assert cal.parse_bs_time("garbage") is None


def test_bs_frame():
    df = pd.DataFrame({"Date": pd.to_datetime(["2008-01-22", "2008-01-30"]),
                       "Time": ["8:30am", "2:15pm"], "Unscheduled": [1, 0]})
    out = cal.bs_frame(df)
    assert list(out["time"]) == ["08:30", "14:15"] and list(out["unscheduled"]) == [True, False]


def test_tdw_frame_handles_cross_month_end_dates():
    df = pd.DataFrame({
        "EndDate": ["April/May/1/2013", "Jan/31/2023", "June/25/2008"],
        "ReleaseDate": pd.to_datetime(["2013-05-22", "2023-02-22", "2008-07-16"]),
        "Url": ["/monetarypolicy/fomcminutes20130501.htm", "/monetarypolicy/fomcminutes20230201.htm",
                "/monetarypolicy/fomc20080625.htm"]})
    out = cal.tdw_frame(df)
    assert list(out["meeting_date"]) == [pd.Timestamp("2008-06-25"), pd.Timestamp("2013-05-01"),
                                         pd.Timestamp("2023-01-31")]
    assert out["release"].iloc[1] == pd.Timestamp("2013-05-22")


def test_load_usmpd_and_bs_from_small_xlsx(tmp_path):
    p = tmp_path / "USMPD.xlsx"
    with pd.ExcelWriter(p) as xw:
        pd.DataFrame([[None, None], ["Last update: 17 September 2026", None]]).to_excel(
            xw, sheet_name="README", header=False, index=False)
        pd.DataFrame({"Date": pd.to_datetime(["2008-01-22", "2008-01-30"]),
                      "date_time": pd.to_datetime(["2008-01-22 08:20", "2008-01-30 14:15"]),
                      "SEP": [0, 1], "Unscheduled": [1, 0], "MP1": [0.1, 0.2]}).to_excel(
            xw, sheet_name="Statements", index=False)
        pd.DataFrame({"Date": pd.to_datetime(["2008-02-20"]),
                      "date_time": pd.to_datetime(["2008-02-20 13:00"]), "MP1": [0]}).to_excel(
            xw, sheet_name="Minutes", index=False)
    u = cal.load_usmpd(p)
    assert u["last_update"] == "Last update: 17 September 2026"
    assert list(u["statements"]["time"]) == ["08:20", "14:15"]
    assert list(u["statements"]["unscheduled"]) == [True, False]
    assert list(u["minutes"]["time"]) == ["13:00"]

    q = tmp_path / "bs.xlsx"
    with pd.ExcelWriter(q) as xw:
        pd.DataFrame({"Date": pd.to_datetime(["2008-01-22", "2023-12-13"]), "Time": ["8:30am", "NA"],
                      "Unscheduled": [1, 0], "FF1": ["NA", 0.1]}).to_excel(xw, sheet_name="FOMC (update 2023)", index=False)
    b = cal.load_bs(q, "FOMC (update 2023)")
    assert b["time"].tolist() == ["08:30", None]


# --------------------------------------------------------------------------- #
# reconciliation rule
# --------------------------------------------------------------------------- #
def test_resolve_value_priority():
    ts = pd.Timestamp
    v = {"fed": ts("2008-01-22"), "usmpd": ts("2008-01-22"), "bauer_swanson": ts("2008-01-22"),
         "dataset": ts("2008-01-21")}
    val, by, _ = cal.resolve_value(v, ["fed", "usmpd", "bauer_swanson", "dataset"])
    assert val == ts("2008-01-22") and by == "fed"
    v["fed"] = None
    val, by, _ = cal.resolve_value(v, ["fed", "usmpd", "bauer_swanson", "dataset"])
    assert by == "usmpd"


def test_resolve_value_majority_and_tie():
    ts = pd.Timestamp
    prio = ["fed", "majority"]
    val, by, _ = cal.resolve_value({"fed": ts("2003-12-11"), "tdw": ts("2003-12-11"), "usmpd": ts("2003-12-10")}, prio)
    assert val == ts("2003-12-11") and by == "fed"                       # Fed wins even against others
    val, by, _ = cal.resolve_value({"fed": None, "tdw": ts("2003-12-11"), "usmpd": ts("2003-12-10"),
                                    "dataset": ts("2003-12-11")}, prio)
    assert val == ts("2003-12-11") and by.startswith("majority")
    val, by, _ = cal.resolve_value({"fed": None, "tdw": ts("2003-12-11"), "usmpd": ts("2003-12-10")}, prio)
    assert val is None and by == "UNRESOLVED"                            # a tie is never guessed


# --------------------------------------------------------------------------- #
# event building on synthetic data
# --------------------------------------------------------------------------- #
def _occ(d, kind="meeting", slug=None, href=None, min_rel=None, min_doc=None, host=None):
    return {"date": d, "kind": kind, "scheduled": kind == "meeting", "heading": str(d), "stmt_href": href,
            "stmt_slug_date": slug, "min_release": min_rel, "min_doc_date": min_doc, "min_href": None,
            "min_host_text": host}


def _docs(rows):
    return pd.DataFrame(rows, columns=["doc_id", "doc_type", "meeting_date_ds", "release_date_ds"])


def _statement_fixture():
    occ = [
        _occ(date(2007, 6, 28), slug=date(2007, 6, 18), href="/newsevents/press/monetary/20070618a.htm"),
        _occ(date(2008, 1, 21), "call", slug=date(2008, 1, 22), href="/newsevents/press/monetary/20080122b.htm"),
        _occ(date(2008, 1, 30), slug=date(2008, 1, 30), href="/newsevents/press/monetary/20080130a.htm"),
        _occ(date(2008, 8, 5), slug=date(2008, 8, 5), href="/newsevents/press/monetary/20080805a.htm"),
        _occ(date(2010, 5, 9), "call", slug=date(2010, 5, 9), href="/newsevents/press/monetary/20100509a.htm"),
        _occ(date(2019, 10, 4), "unscheduled", slug=date(2019, 10, 11),
             href="/newsevents/pressreleases/monetary20191011a.htm"),
    ]
    usmpd = pd.DataFrame({
        "date": pd.to_datetime(["2007-06-28", "2008-01-22", "2008-01-30", "2008-08-05", "2010-05-09", "2019-10-04"]),
        "time": ["14:15", "08:20", "14:15", "14:15", "21:15", "11:00"],
        "unscheduled": [False, True, False, False, True, True], "sep": False})
    bs = pd.DataFrame({
        "date": pd.to_datetime(["2007-06-28", "2008-01-22", "2008-01-30", "2008-08-05", "2019-10-11"]),
        "time": ["14:15", "08:30", "14:15", "14:15", "11:00"],
        "unscheduled": [False, True, False, False, True]})
    docs = _docs([
        ("S20070628", "statement", "2007-06-28", "2007-06-28"),
        ("S20080121", "statement", "2008-01-21", "2008-01-21"),       # dataset says the call day
        ("S20080130", "statement", "2008-01-30", "2008-01-30"),
        ("S20080808", "statement", "2008-08-08", "2008-08-08"),       # mis-dated (really Aug 5)
        ("S20191004", "statement", "2019-10-04", "2019-10-11"),
    ])
    return occ, usmpd, bs, docs


def _run_statements():
    occ, usmpd, bs, docs = _statement_fixture()
    recon = cal.Recon()
    calls = []

    def fake_pr(href):
        calls.append(href)
        if "20070618a" in href:
            return {"date": date(2007, 6, 28), "time": None}
        if "20191011a" in href:
            return {"date": date(2019, 10, 11), "time": "11:00"}
        return {"date": None, "time": None}

    ev = cal.build_statement_events(occ, usmpd, bs, recon, fetch_pr=fake_pr)
    ev = cal.finalize_statements(ev, docs, CFG, recon, (bs["date"].min(), bs["date"].max()))
    return {e["event_id"]: e for e in ev}, recon.frame(), calls


def test_statement_call_released_next_day():
    ev, recon, _ = _run_statements()
    e = ev["stmt_20080121"]
    assert e["release_dt"] == pd.Timestamp("2008-01-22 08:20")      # USMPD minute beats B-S 08:30
    assert e["time_source"] == "usmpd" and not e["scheduled"]
    assert e["doc_id"] == "S20080121" and e["n_sources_agree"] == 3  # dataset's 2008-01-21 overruled
    rows = recon[recon["event_id"] == "stmt_20080121"].set_index("field")
    assert rows.loc["release_date", "resolution"] == "2008-01-22" and rows.loc["release_date", "v_dataset"] == "2008-01-21"
    assert rows.loc["release_time", "resolution"] == "08:20" and rows.loc["release_time", "resolved_by"] == "usmpd"


def test_statement_unscheduled_with_wrong_usmpd_date():
    ev, recon, calls = _run_statements()
    e = ev["stmt_20191004"]
    assert e["release_dt"] == pd.Timestamp("2019-10-11 11:00")      # Fed + B-S + dataset beat USMPD's 10-04
    assert e["doc_id"] == "S20191004" and not e["scheduled"]
    row = recon[(recon["event_id"] == "stmt_20191004") & (recon["field"] == "release_date")].iloc[0]
    assert row["v_usmpd"] == "2019-10-04" and row["resolved_by"] == "fed" and row["resolution"] == "2019-10-11"


def test_statement_url_typo_uses_page_date():
    ev, recon, _ = _run_statements()
    e = ev["stmt_20070628"]
    assert e["release_dt"] == pd.Timestamp("2007-06-28 14:15")
    assert "fed_url_slug_vs_page_date" in set(recon["field"])


def test_statement_no_fetch_without_conflict():
    ev, recon, calls = _run_statements()
    assert not any("20080130a" in c or "20100509a" in c for c in calls)
    assert "stmt_20080130" in ev and ev["stmt_20080130"]["n_sources_agree"] == 4
    assert not recon[recon["event_id"] == "stmt_20080130"].shape[0]


def test_statement_missing_in_bauer_swanson_kept():
    ev, recon, _ = _run_statements()
    e = ev["stmt_20100509"]
    assert e["release_dt"] == pd.Timestamp("2010-05-09 21:15") and not e["scheduled"]
    row = recon[(recon["event_id"] == "stmt_20100509") & (recon["field"] == "present_in_source")].iloc[0]
    assert row["v_bauer_swanson"] == "no" and row["resolution"] == "event kept"


def test_statement_dataset_doc_matched_by_proximity():
    ev, recon, _ = _run_statements()
    e = ev["stmt_20080805"]
    assert e["doc_id"] == "S20080808"
    row = recon[(recon["event_id"] == "stmt_20080805") & (recon["field"] == "meeting_date")].iloc[0]
    assert row["v_fed"] == "2008-08-05" and row["v_dataset"] == "2008-08-08"
    rd = recon[(recon["event_id"] == "stmt_20080805") & (recon["field"] == "release_date")].iloc[0]
    assert rd["resolution"] == "2008-08-05"


def test_statement_default_time_when_no_source():
    occ = [_occ(date(2025, 8, 22), "notation_vote", slug=date(2025, 8, 22),
                href="/newsevents/pressreleases/monetary20250822a.htm")]
    empty_u = pd.DataFrame({"date": pd.to_datetime([]), "time": [], "unscheduled": [], "sep": []})
    empty_b = pd.DataFrame({"date": pd.to_datetime([]), "time": [], "unscheduled": []})
    recon = cal.Recon()
    ev = cal.build_statement_events(occ, empty_u, empty_b, recon, fetch_pr=lambda h: {"date": None, "time": None})
    ev = cal.finalize_statements(ev, _docs([]), CFG, recon, (pd.Timestamp("1988-01-01"), pd.Timestamp("2023-12-13")))
    assert ev[0]["release_dt"] == pd.Timestamp("2025-08-22 14:00") and ev[0]["time_source"] == "config_default"
    assert ev[0].get("doc_id") is None


# ---- minutes ---- #
def _minutes_fixture():
    occ = [
        _occ(date(2001, 10, 2), slug=date(2001, 10, 2), href="x", min_rel=date(2001, 11, 8), min_doc=date(2001, 10, 2)),
        _occ(date(2001, 11, 6), slug=date(2001, 11, 6), href="x", min_rel=date(2001, 12, 13), min_doc=date(2001, 11, 6)),
        _occ(date(2001, 12, 11), slug=date(2001, 12, 11), href="x", min_rel=date(2002, 1, 31), min_doc=date(2001, 12, 11)),
        _occ(date(2007, 8, 10), "call", slug=date(2007, 8, 10), href="x", min_rel=date(2007, 10, 9), min_doc=date(2007, 9, 18)),
        _occ(date(2007, 8, 16), "call", slug=date(2007, 8, 17), href="x", min_rel=date(2007, 10, 9), min_doc=date(2007, 9, 18)),
        _occ(date(2007, 9, 18), slug=date(2007, 9, 18), href="x", min_rel=date(2007, 10, 9), min_doc=date(2007, 9, 18)),
        _occ(date(2008, 1, 9), "call", host="January 29/30"),
    ]
    tdw = pd.DataFrame({"meeting_date": pd.to_datetime(["2001-10-02", "2001-11-06", "2001-12-11", "2007-09-18"]),
                        "release": pd.to_datetime(["2001-11-08", "2001-12-13", "2002-01-31", "2007-10-09"]),
                        "url_date": pd.NaT})
    usmpd_min = pd.DataFrame({"date": pd.to_datetime(["2001-11-08", "2001-12-31", "2002-01-31", "2007-10-09"]),
                              "time": ["14:00", "14:00", "14:00", "14:00"]})
    docs = _docs([
        ("M20011106", "minutes", "2001-11-06", "2001-12-13"),
        ("M20070810", "minutes", "2007-08-10", "2007-10-09"),         # duplicates of the Sep 18 document
        ("M20070816", "minutes", "2007-08-16", "2007-10-09"),
        ("M20070918", "minutes", "2007-09-18", "2007-10-09"),
        ("M20080109", "minutes", "2008-01-09", None),
    ])
    return occ, tdw, usmpd_min, docs


def _run_minutes():
    occ, tdw, usmpd_min, docs = _minutes_fixture()
    recon = cal.Recon()
    ev = cal.build_minutes_events(occ, tdw, usmpd_min, docs, CFG, recon)
    ev = cal.finalize_minutes(ev, CFG, recon, (pd.Timestamp("2001-02-01"), pd.Timestamp("2026-08-19")),
                              (pd.Timestamp("1996-03-29"), pd.Timestamp("2024-10-09")))
    return {e["event_id"]: e for e in ev}, recon.frame()


def test_minutes_one_event_per_document_not_per_call():
    ev, _ = _run_minutes()
    assert set(ev) == {"min_20011002", "min_20011106", "min_20011211", "min_20070918"}
    e = ev["min_20070918"]
    assert e["release_dt"] == pd.Timestamp("2007-10-09 14:00")
    assert any("2007-08-10" in n and "2007-08-16" in n for n in e["notes"])
    assert e["doc_id"] == "M20070918"                                # exact meeting-date match wins


def test_minutes_mistyped_usmpd_date_is_paired_and_overruled():
    ev, recon = _run_minutes()
    e = ev["min_20011106"]
    assert e["release_dt"] == pd.Timestamp("2001-12-13 14:00") and e["u"] is not None
    row = recon[(recon["event_id"] == "min_20011106") & (recon["field"] == "release_date")].iloc[0]
    assert row["v_usmpd"] == "2001-12-31" and row["resolved_by"] == "fed"
    assert "transposed" in row["reason"]


def test_minutes_fed_missing_uses_majority():
    occ, tdw, usmpd_min, docs = _minutes_fixture()
    occ = [o for o in occ if o["date"] != date(2001, 11, 6)]          # Fed page lacks that meeting
    usmpd_min.loc[1, "date"] = pd.Timestamp("2001-12-13")
    recon = cal.Recon()
    ev = cal.build_minutes_events(occ, tdw, usmpd_min, docs, CFG, recon)
    ev = cal.finalize_minutes(ev, CFG, recon, (pd.Timestamp("2001-02-01"), pd.Timestamp("2026-08-19")),
                              (pd.Timestamp("1996-03-29"), pd.Timestamp("2024-10-09")))
    e = {x["event_id"]: x for x in ev}["min_20011106"]
    assert e["release_dt"] == pd.Timestamp("2001-12-13 14:00") and e["fed"] is None
    assert e["u"] is not None and any("TDW only" in n for n in e["notes"])


def test_minutes_unresolved_tie_is_flagged_not_guessed():
    occ = []
    tdw = pd.DataFrame({"meeting_date": pd.to_datetime(["2009-01-28"]), "release": pd.to_datetime(["2009-02-18"]),
                        "url_date": pd.NaT})
    usmpd_min = pd.DataFrame({"date": pd.to_datetime(["2009-02-19"]), "time": ["14:00"]})
    recon = cal.Recon()
    ev = cal.build_minutes_events(occ, tdw, usmpd_min, _docs([]), CFG, recon)
    ev = cal.finalize_minutes(ev, CFG, recon, (pd.Timestamp("2001-02-01"), pd.Timestamp("2026-08-19")),
                              (pd.Timestamp("1996-03-29"), pd.Timestamp("2024-10-09")))
    frame = recon.frame()
    assert (frame["resolved_by"] == "UNRESOLVED").any()
    assert any("UNRESOLVED" in n for n in ev[0]["notes"])


def test_assemble_filters_to_sample_window_and_formats():
    ev, _ = _run_minutes()
    stm, _, _ = _run_statements()
    df = cal.assemble_events(list(stm.values()), list(ev.values()), pd.Timestamp("2007-01-01"), pd.Timestamp("2008-12-31"))
    assert list(df.columns) == cal.EVENT_COLS
    assert df["release_datetime_et"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$").all()
    assert df["release_datetime_et"].str[:4].isin(["2007", "2008"]).all()
    assert "min_20011106" not in set(df["event_id"])
    assert df["event_id"].is_unique


def test_is_non_business_day():
    assert cal.is_non_business_day(pd.Timestamp("2020-03-15")).startswith("weekend")
    assert cal.is_non_business_day(pd.Timestamp("2004-11-11")) == "US federal holiday"
    assert cal.is_non_business_day(pd.Timestamp("2008-01-22")) == ""
    assert cal.is_non_business_day(pd.Timestamp("2008-01-21")) == "US federal holiday"   # MLK day


# --------------------------------------------------------------------------- #
# map_to_trading_day
# --------------------------------------------------------------------------- #
@pytest.fixture
def trading_days():
    days = pd.bdate_range("2020-03-02", "2020-04-30")
    return days.drop(pd.Timestamp("2020-04-10"))           # one weekday holiday (Good Friday)


def _map(trading_days, stamp, close="15:30"):
    return cal.map_to_trading_day(pd.Series(pd.to_datetime([stamp])), trading_days, close).iloc[0]


def test_map_weekday_afternoon_same_day(trading_days):
    assert _map(trading_days, "2020-03-18 14:00") == pd.Timestamp("2020-03-18")


def test_map_sunday_goes_to_monday(trading_days):
    assert _map(trading_days, "2020-03-15 17:00") == pd.Timestamp("2020-03-16")


def test_map_pre_open_same_day(trading_days):
    assert _map(trading_days, "2020-03-23 08:20") == pd.Timestamp("2020-03-23")


def test_map_after_close_next_day(trading_days):
    assert _map(trading_days, "2020-03-17 16:00") == pd.Timestamp("2020-03-18")


def test_map_close_time_is_exclusive(trading_days):
    assert _map(trading_days, "2020-03-17 15:29") == pd.Timestamp("2020-03-17")
    assert _map(trading_days, "2020-03-17 15:30") == pd.Timestamp("2020-03-18")


def test_map_holiday_goes_to_next_trading_day(trading_days):
    assert _map(trading_days, "2020-04-10 14:00") == pd.Timestamp("2020-04-13")


def test_map_friday_after_close_goes_to_monday(trading_days):
    assert _map(trading_days, "2020-03-20 16:00") == pd.Timestamp("2020-03-23")


def test_map_preserves_index_and_handles_nat_and_end_of_calendar(trading_days):
    s = pd.Series(pd.to_datetime(["2020-03-18 14:00", None, "2020-04-30 16:00"]), index=[10, 20, 30], name="rel")
    out = cal.map_to_trading_day(s, trading_days, "15:30")
    assert list(out.index) == [10, 20, 30]
    assert out.iloc[0] == pd.Timestamp("2020-03-18")
    assert pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])        # no later trading day in the calendar


def test_map_works_with_other_close_times(trading_days):
    assert _map(trading_days, "2020-03-18 15:45", close="16:00") == pd.Timestamp("2020-03-18")
    assert _map(trading_days, "2020-03-18 16:00", close="16:00") == pd.Timestamp("2020-03-19")
