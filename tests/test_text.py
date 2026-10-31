"""Tests for ftyields.text (cleaning, splitting, section tags, masking, de-duplication); no network."""
import pandas as pd
import pytest

from ftyields.text import clean as C
from ftyields.text.build import (build_unique, count_table, counts_tex, document_sentences)
from ftyields.text.mask import Masker, make_uid, normalize
from ftyields.text.sections import SectionTagger, classify_heading
from ftyields.text.split import is_artifact, split_sentences


# ---------------------------------------------------------------- splitter

def test_split_basic_and_abbreviations():
    t = "Mr. Greenspan said that the U.S. economy grew. Ms. Yellen disagreed, e.g. on prices. The end came at 2:15 p.m. on Tuesday."
    s = split_sentences(t)
    assert s == [
        "Mr. Greenspan said that the U.S. economy grew.",
        "Ms. Yellen disagreed, e.g. on prices.",
        "The end came at 2:15 p.m. on Tuesday.",
    ]


def test_split_initials_decimals_fractions():
    t = ("Ben S. Bernanke and Roger W. Ferguson, Jr. voted for a 2.5 percent target. "
         "The funds rate was lowered to 1/4 percent. Inflation was 1.9 percent.")
    s = split_sentences(t)
    assert len(s) == 3
    assert s[0].startswith("Ben S. Bernanke and Roger W. Ferguson, Jr.")
    assert s[1] == "The funds rate was lowered to 1/4 percent."


def test_split_us_sentence_end_and_month_abbrev():
    s = split_sentences("Growth slowed in the U.S. The euro area was stable. Data on Sept. 11 were revised, etc. Prices rose.")
    assert s[0] == "Growth slowed in the U.S."
    assert s[1] == "The euro area was stable."
    assert len(s) == 3 and s[2].endswith("Prices rose.") and "Sept. 11" in s[2]


def test_split_ellipsis_parentheses_quotes():
    s = split_sentences('The statement said "policy remains accommodative... for some time." Participants agreed (see Table 1.) The vote was unanimous.')
    assert len(s) == 3
    assert s[0].endswith('for some time."')
    assert "(see Table 1.)" in s[1]


def test_split_merges_short_fragments():
    s = split_sentences("Yes. The Committee decided to keep the target range unchanged at this meeting.")
    assert len(s) == 1 and s[0].startswith("Yes. The Committee")
    s = split_sentences("The Committee decided to keep the target range unchanged at this meeting. Done.")
    assert len(s) == 1


def test_split_enumerator_and_roman_numerals_do_not_split():
    s = split_sentences("A. To buy and sell securities in the open market for the account of the Bank.")
    assert len(s) == 1
    s = split_sentences("All arrangements are subject to annual review; ii. The Manager shall report to the Committee at each meeting.")
    assert len(s) == 1


# ---------------------------------------------------------------- cleaning

MINUTES = """Minutes of the Federal Open Market Committee
January 24-25, 2012
A joint meeting of the Federal Open Market Committee and the Board of Governors was held in the offices of the Board of Governors in Washington, D.C., on Tuesday, January 24, 2012, at 10:00 a.m., and continued on Wednesday, January 25, 2012, at 8:30 a.m.
PRESENT:
Ben Bernanke, Chairman
William C. Dudley, Vice Chairman
Elizabeth Duke
William B. English, Secretary and Economist
Michael S. Gibson, Director, Division of Banking Supervision and Regulation, Board of Governors
Daniel Cooper,2 Economist, Federal Reserve Bank of Boston
Developments in Financial Markets and Open Market Operations
The Manager reported on developments in financial markets. By unanimous vote, the Committee ratified the Desk's domestic transactions over the intermeeting period.
Staff Review of the Economic Situation
The information reviewed at the January 24-25 meeting suggested that economic activity was expanding moderately.1
Committee Policy Action
Members agreed that the stance of policy should stay highly accommodative. Mr. Lacker dissented because he preferred to omit the time period.
Voting for this action: Ben Bernanke, William C. Dudley, and Elizabeth Duke.
Voting against this action: Jeffrey M. Lacker.
The meeting adjourned at 1:15 p.m. on January 25, 2012.
Notation Vote
By notation vote completed on February 9, 2012, the Committee unanimously approved the minutes of the Committee meeting held on December 13, 2011.
Secretary
1. The Federal Open Market Committee is referenced as the "FOMC" in these minutes. Return to text
"""


def _sentences(text, doc_type="minutes"):
    cd = C.clean_document(text, doc_type)
    return cd, document_sentences(cd, doc_type)


def test_minutes_boilerplate_attendance_and_votes_removed():
    cd, sents = _sentences(MINUTES)
    txt = " || ".join(s["text"] for s in sents)
    for bad in ["Minutes of the Federal Open Market Committee", "joint meeting", "PRESENT", "Chairman", "Gibson",
                "Voting for this action", "adjourned", "Notation", "Return to text", "referenced as the"]:
        assert bad not in txt, bad
    assert "The Manager reported on developments in financial markets." in txt
    assert "Mr. Lacker dissented because he preferred to omit the time period." in txt
    assert "Daniel Cooper" in cd.names and "William C. Dudley" in cd.names
    # footnote marker removed from content sentence
    assert "moderately." in txt and "moderately.1" not in txt


def test_dissent_explanations_survive_but_pure_tallies_do_not():
    for s, keep in [
        ("Voting for the FOMC monetary policy action were: Ben S. Bernanke, Chairman; Janet L. Yellen.", False),
        ("Voting against this action: None.", False),
        ("Voting against this action: Thomas M. Hoenig.", False),
        ("Voting against the action was Jeffrey M. Lacker, who preferred to omit the description of the time period.", True),
        ("Voting against was Charles I. Plosser who objected to the guidance.", True),
        ("Votes for this action: Messrs. Greenspan, McDonough, and Parry.", False),
        ("The meeting adjourned at 11:50 a.m. on February 2, 2000.", False),
        ("Chairman Greenspan dissented because he preferred an increase.", True),
    ]:
        assert (C.drop_sentence(s) is not None) is keep, s
    # vote-for + vote-against in one sentence keeps only the dissent part
    out = C.drop_sentence("Voting for the action were: A. B. Smith; C. D. Jones. Voting against was X. Y. Lacker, who preferred no change.")
    assert out is not None


def test_statement_boilerplate_removed_and_wrapped_lines_joined():
    raw = ("The Federal Open Market Committee decided today to keep its target for \n"
           "        the federal funds rate at 1 percent.\n"
           "       The Committee perceives the upside and downside risks to the attainment \n"
           "        of sustainable growth for the next few quarters are roughly equal.\n"
           "       Voting for the FOMC monetary policy actions were: Alan Greenspan, Chairman; \n"
           "        Timothy F. Geithner, Vice Chairman; Ben S. Bernanke; and William Poole.\n"
           "        2004 Monetary policy \n"
           " Home | News and \n"
           "events \n"
           "Accessibility\n"
           "       Last update: May 4, 2004\n")
    cd, sents = _sentences(raw, "statement")
    texts = [s["text"] for s in sents]
    assert texts[0] == "The Federal Open Market Committee decided today to keep its target for the federal funds rate at 1 percent."
    assert len(texts) == 2
    assert all(s["section"] == "statement" for s in sents)
    assert {"Alan Greenspan", "Timothy F. Geithner"} <= cd.names


def test_encoding_repair():
    t = C.fix_text("Todayâ\u0080\u0099s action; the Committeeï¿½s view; up ï¿½ percentage point; "
                   "January 30â\u0080\u009331, 2024")
    assert "Today’s action" in t
    assert "Committee's view" in t
    assert "[FRAC] percentage point" in t
    assert "30–31" in t


# ---------------------------------------------------------------- sections

def test_section_headings_new_format():
    tg = SectionTagger(has_headings=True)
    tg.heading("Staff Review of the Economic Situation")
    assert tg.section == "staff_economic" and not tg.procedural
    for h, sec in [("Staff Review of the Financial Situation", "staff_financial"),
                   ("Staff Economic Outlook", "staff_outlook"),
                   ("Participants' Views on Current Conditions and the Economic Outlook", "participants_views"),
                   ("Committee Policy Actions", "policy_action"),
                   ("Developments in Financial Markets and Open Market Operations", "financial_markets")]:
        tg.heading(h)
        assert tg.section == sec
    tg.heading("Annual Organizational Matters")
    assert tg.section == "other" and tg.procedural
    assert classify_heading("Ben S. Bernanke") == (None, None)


def test_section_heuristics_old_format():
    tg = SectionTagger(has_headings=False)
    assert tg.section == "other" and tg.procedural
    seq = [
        ("The Manager of the System Open Market Account reported on recent developments.", "financial_markets"),
        ("The information reviewed at this meeting suggested that the economy expanded.", "staff_economic"),
        ("At its meeting on March 16, 2004, the Federal Open Market Committee decided to keep its target.", "staff_financial"),
        ("The staff forecast prepared for this meeting suggested that the economy would expand.", "staff_outlook"),
        ("In the Committee's discussion of current and prospective economic developments, members noted.", "participants_views"),
        ("In the Committee's discussion of policy for the intermeeting period, all members favored no change.", "policy_action"),
        ("At the conclusion of the discussion, the Committee voted to authorize.", "policy_action"),
    ]
    for para, sec in seq:
        tg.paragraph(para)
        assert tg.section == sec, para
    assert not tg.procedural


# ---------------------------------------------------------------- masking

@pytest.fixture(scope="module")
def mk():
    return Masker({"Ben S. Bernanke", "Jeffrey M. Lacker", "Esther L. George", "Janet L. Yellen"})


def test_mask_names(mk):
    assert mk("Voting against was Jeffrey M. Lacker, who preferred no change.") == "Voting against was [NAME], who preferred no change."
    assert mk("Chairman Greenspan said that Ms. Yellen and Mr. Hoenig agreed.") == "[NAME] said that [NAME] and [NAME] agreed."
    assert mk("Messrs. Fisher, Kocherlakota, and Plosser dissented.") == "[NAME], [NAME], [NAME] dissented." or \
        mk("Messrs. Fisher, Kocherlakota, and Plosser dissented.") == "[NAME], [NAME], and [NAME] dissented."
    assert mk("Governor Bowman and President Daly spoke.") .startswith("[NAME] and [NAME]")
    assert mk("Esther George dissented.") == "[NAME] dissented."
    # roles without names are not masked
    assert mk("The Chairman and the Vice Chair of the Committee.") == "The Chairman and the Vice Chair of the Committee."
    assert mk("The President of the Federal Reserve Bank of Boston.") == "The President of the Federal Reserve Bank of Boston."


def test_mask_dates_and_years(mk):
    assert mk("At the January 27-28 meeting, the vote was held on May 3.") == "At the [DATE] meeting, the vote was held on [DATE]."
    assert mk("Effective March 16, 2022, the Desk will act.") == "Effective [DATE], the Desk will act."
    assert mk("The Committee met on Tuesday, January 26, 2016, at 12:00 p.m.") == "The Committee met on [DATE], at 12:00 p.m."
    assert mk("in the fourth quarter of 2008") == "in the fourth quarter of [YEAR]"
    assert mk("through the end of 2019, and in mid-2013") == "through the end of [YEAR], and in mid-[YEAR]"
    assert mk("The staff expects growth in 2024 and 2025.") == "The staff expects growth in [YEAR] and [YEAR]."
    assert mk("By the end of June 2012, the Desk will buy securities.") == "By the end of [DATE], the Desk will buy securities."


def test_mask_leaves_economic_numbers(mk):
    for s in ["The target range is 1/4 to 1/2 percent.", "Inflation was 2.5 percent in the fourth quarter.",
              "The Committee will purchase $600 billion of securities over the next year.",
              "The 10-year yield rose 15 basis points over the medium term.",
              "Payrolls rose by 2,000 jobs and the rate was 5-3/4 percent.",
              "The Committee seeks 2 percent inflation in the second half of the year."]:
        assert mk(s) == s


# ---------------------------------------------------------------- uid / dedup / prev_text

def test_uid_stable_and_quote_normalised():
    a = "The Committee seeks “maximum employment” and 2 percent inflation."
    b = 'the  committee seeks "maximum employment"\nand 2 percent inflation.'
    assert normalize(a) == normalize(b)
    assert make_uid(normalize(a)) == make_uid(normalize(b))
    assert make_uid("x") == make_uid("x") and make_uid("x") != make_uid("y")
    assert make_uid("x").startswith("u") and len(make_uid("x")) == 13
    assert make_uid("abc") == "u" + __import__("hashlib").sha1(b"abc").hexdigest()[:12]


def _toy():
    rows = []
    def add(doc, dt, date, idx, sec, txt):
        rows.append(dict(sent_id=f"{doc}_{idx:04d}", doc_id=doc, doc_type=dt, event_id="e", release_date=date,
                         sent_idx=idx, section=sec, text=txt, text_masked=txt, prev_text="", prev_text_masked="",
                         uid=make_uid(normalize(txt)), n_words=len(txt.split()), is_procedural=False))
    add("S1", "statement", "2010-01-01", 0, "statement", "The Committee seeks maximum employment and price stability.")
    add("S2", "statement", "2011-01-01", 0, "statement", "The Committee seeks maximum employment and price stability.")
    add("M1", "minutes", "2011-02-01", 0, "participants_views", "The Committee seeks maximum employment and price stability.")
    add("M1", "minutes", "2011-02-01", 1, "policy_action", "Members agreed to keep the target range unchanged today.")
    return pd.DataFrame(rows)


def test_unique_table_and_counts():
    df = _toy()
    u = build_unique(df).set_index("uid")
    top = u.loc[make_uid(normalize("The Committee seeks maximum employment and price stability."))]
    assert top["n_occurrences"] == 3 and top["n_docs"] == 3 and top["doc_types"] == "both"
    assert top["first_release_date"] == "2010-01-01"
    assert top["sections"] == "participants_views|statement"
    assert len(u) == 2
    tab = count_table(df)
    allrow = tab[(tab.doc_type == "all") & (tab.year == "all")].iloc[0]
    assert allrow["documents"] == 3 and allrow["sentences"] == 4 and allrow["unique_uids"] == 2
    assert "Statements" in counts_tex(tab) and "\\begin{tabular}" in counts_tex(tab)


def test_prev_text_stays_within_document():
    from ftyields.text.build import build_sentences
    docs = pd.DataFrame({
        "doc_id": ["S20100101", "S20110101"], "doc_type": ["statement", "statement"],
        "event_id": ["e1", "e2"], "release_date": ["2010-01-01", "2011-01-01"],
        "text": ["The Committee decided to keep the target range unchanged. Inflation remained subdued over the past year.",
                 "Economic activity expanded at a moderate pace in recent months. Job gains remained solid across sectors."],
    })
    tc = {"min_fragment_words": 4, "min_sentence_words": 3, "uid_hex_len": 12}
    df, _ = build_sentences(docs, tc)
    d1 = df[df.doc_id == "S20110101"].sort_values("sent_idx")
    assert d1.iloc[0]["prev_text"] == "" and d1.iloc[0]["prev_text_masked"] == ""
    assert d1.iloc[1]["prev_text"] == d1.iloc[0]["text"]
    d0 = df[df.doc_id == "S20100101"].sort_values("sent_idx")
    assert d0.iloc[0]["prev_text"] == ""
    assert list(d0["sent_id"]) == ["S20100101_0000", "S20100101_0001"]


# ---------------------------------------------------------------- short sentences vs splitter artifacts

def test_short_complete_sentences_survive_with_neighbours():
    t = ("Job gains have moderated since early last year but remain strong. Inflation remained elevated. "
         "The Committee decided to maintain the target range for the federal funds rate. Employment rose. "
         "Household spending continued to expand at a solid pace in recent months.")
    s = split_sentences(t, min_words=4)
    assert s == [
        "Job gains have moderated since early last year but remain strong.",
        "Inflation remained elevated.",
        "The Committee decided to maintain the target range for the federal funds rate.",
        "Employment rose.",
        "Household spending continued to expand at a solid pace in recent months.",
    ]
    # first / last position too
    assert split_sentences("Employment rose. Spending was strong across most categories of goods and services.") == [
        "Employment rose.", "Spending was strong across most categories of goods and services."]
    assert split_sentences("Spending was strong across most categories of goods and services. Inflation remained elevated.") == [
        "Spending was strong across most categories of goods and services.", "Inflation remained elevated."]


def test_short_sentences_survive_document_pipeline():
    raw = ("Information received since the Committee met suggests that activity is expanding. Inflation remained elevated. "
           "The Committee decided to keep the target range unchanged at this meeting. Employment rose.")
    cd = C.clean_document(raw, "statement")
    texts = [s["text"] for s in document_sentences(cd, "statement", 4, 2)]
    assert "Inflation remained elevated." in texts and "Employment rose." in texts and len(texts) == 4


def test_genuine_artifacts_still_merge():
    assert is_artifact("S.", 4) and is_artifact("1.", 4) and is_artifact("(a)", 4) and is_artifact("Yes", 4)
    assert is_artifact("and so on.", 4)                     # lowercase start
    assert is_artifact("Core inflation remained", 4)       # no terminal punctuation
    assert not is_artifact("Employment rose.", 4) and not is_artifact("Inflation remained elevated.", 4)
    assert not is_artifact("Core inflation remained contained and stable.", 4)
    s = split_sentences("The vote was taken by Messrs. Hoenig. 2. The Committee then adjourned for lunch at noon.")
    assert len(s) <= 2 and all(len(x.split()) >= 2 for x in s)
    assert split_sentences("Yes. The Committee decided to keep the target range unchanged at this meeting.") == [
        "Yes. The Committee decided to keep the target range unchanged at this meeting."]
    assert split_sentences("The Committee decided to keep the target range unchanged at this meeting. Done") == [
        "The Committee decided to keep the target range unchanged at this meeting. Done"]
