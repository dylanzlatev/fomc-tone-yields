"""Section tagging for minutes paragraphs.

Newer minutes (2009+) carry headings; older minutes do not and are tagged with
paragraph-opening heuristics. Output sections: financial_markets, staff_economic,
staff_financial, staff_outlook, participants_views, policy_action, other.
`is_procedural` marks organizational boilerplate (election of officers, authorizations,
directives of the Committee's rules), which is kept but tagged 'other'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ftyields.text.clean import ROLE_WORDS

SECTIONS = ["financial_markets", "staff_economic", "staff_financial", "staff_outlook",
            "participants_views", "policy_action", "other"]

_AP = "['’]"
_H = [
    (re.compile(r"^(?:developments in financial markets|market developments and open market|system liquidity programs)", re.I), "financial_markets"),
    (re.compile(r"^staff review of the economic (?:and financial )?situation", re.I), "staff_economic"),
    (re.compile(r"^staff review of the financial situation", re.I), "staff_financial"),
    (re.compile(r"^staff economic outlook", re.I), "staff_outlook"),
    (re.compile(r"^(?:meeting )?participants" + _AP + r" views", re.I), "participants_views"),
    (re.compile(r"^committee policy actions?", re.I), "policy_action"),
]
_PROC_HEAD = re.compile(r"^(?:annual organizational matters|selection of committee officers?)$", re.I)
_TOPIC = re.compile(
    r"^(?:review of|role of|monetary policy|long-run|longer-run|system open market account|update from|"
    r"discussion|financial stability|balance sheet|policy normalization|framework|a review|"
    r"staff presentation|presentation|briefing|report|special|assessment)", re.I)
_KEEP_UPPER = re.compile(r"^STATEMENT ON LONGER-RUN GOALS", re.I)
_FUNC = {"of", "in", "and", "for", "from", "on"}

# Older (heading-less) minutes: paragraph-opening triggers.
_T_PA = re.compile(
    r"^(?:in (?:the committee" + _AP + r"s |their |its )?(?:discussion|consideration|deliberations?)s? (?:of|on|regarding) "
    r"(?:the )?(?:current |near-term |immediate )?(?:monetary )?polic(?:y|ies)"
    r"|at the conclusion of (?:this|the|their) (?:discussion|meeting)"
    r"|the vote (?:also )?encompassed|votes? (?:for|against)|the committee voted"
    r"|following the discussion,? the committee)", re.I)
_T_FM = re.compile(r"^the manager (?:of the system open market account )?(?:also )?reported\b", re.I)
_T_SE = re.compile(r"^(?:the )?information (?:reviewed|available|provided|received)\b.*\b(?:meeting|committee)", re.I)
_T_SF = re.compile(
    r"^(?:at its (?:[a-z]+ ){0,2}meeting|at the (?:previous|last|committee" + _AP + r"s previous) meeting"
    r"|open market operations during the intermeeting|the committee" + _AP + r"s decision at its"
    r"|(?:investors|market participants) (?:had )?(?:largely |generally )?(?:anticipated|expected)"
    r"|the (?:federal open market )?committee" + _AP + r"s (?:interest rate |policy )?(?:decision|action) (?:at|in) (?:its|the))", re.I)
_T_SO = re.compile(
    r"^(?:in preparation for this meeting,? the staff|the staff(?:" + _AP + r"s)? (?:forecast|projection|economic|outlook|"
    r"revised|lowered|raised|marked|trimmed)|the forecast prepared by the staff|in the forecast prepared"
    r"|the economic forecast prepared by the staff)", re.I)
_T_PV = re.compile(
    r"^(?:in the committee" + _AP + r"s (?:review|discussion|assessment)|in their (?:review|discussion|assessment|comments|consideration)"
    r"|in the course of the committee" + _AP + r"s|committee members (?:noted|commented|discussed)"
    r"|members (?:commented|noted|discussed|agreed that)\b|participants (?:noted|commented|discussed))", re.I)


def classify_heading(h: str) -> tuple[str | None, bool | None]:
    """(section, procedural) implied by a heading, or (None, None) if the heading is not informative."""
    h = re.sub(r"\s+", " ", h.strip())
    for rx, sec in _H:
        if rx.search(h):
            return sec, False
    if _PROC_HEAD.match(h):
        return "other", True
    if _KEEP_UPPER.match(h):
        return "other", False
    base = re.sub(r"\(.*?\)", "", h).strip()
    if base.isupper() and len(base) > 8:
        return "other", True
    if ":" in h or re.search(r"\b(?:Mr|Ms|Mrs|Messrs)\.", h):
        return None, None
    if _TOPIC.match(h):
        return "other", False
    words = h.replace(",", " ").split()
    if (len(words) >= 3 and any(w.lower() in _FUNC for w in words)
            and not any(w in ROLE_WORDS for w in words) and not re.match(r"^[A-Z]\. ", h)
            and sum(w[0].isupper() for w in words if w.lower() not in _FUNC | {"the"}) >= 2):
        return "other", False                       # a special-topic heading
    return None, None


@dataclass
class SectionTagger:
    has_headings: bool
    section: str = "other"
    procedural: bool = True
    combined: bool = False      # heading "Participants' Views and Committee Policy Action" (2009)

    def heading(self, h: str) -> None:
        self.combined = bool(re.search(r"views and committee policy action", h, re.I))
        sec, proc = classify_heading(h)
        if sec is not None:
            self.section, self.procedural = sec, bool(proc)

    def paragraph(self, text: str) -> None:
        """Update the state from the opening of a paragraph (heading-less minutes only)."""
        t = text.strip().lstrip("\"“")
        if self.has_headings:
            if self.combined and self.section == "participants_views" and _T_PA.match(t):
                self.section = "policy_action"
            return
        if self.section == "policy_action":
            return
        if _T_PA.match(t):
            self.section, self.procedural = "policy_action", False
        elif _T_FM.match(t) and self.section == "other":
            self.section, self.procedural = "financial_markets", False
        elif _T_SE.match(t) and self.section in ("other", "financial_markets"):
            self.section, self.procedural = "staff_economic", False
        elif _T_SO.match(t) and self.section in ("staff_economic", "staff_financial", "financial_markets", "other"):
            self.section, self.procedural = "staff_outlook", False
        elif _T_SF.match(t) and self.section == "staff_economic":
            self.section, self.procedural = "staff_financial", False
        elif _T_PV.match(t) and self.section in ("staff_outlook", "staff_financial", "staff_economic", "financial_markets"):
            self.section, self.procedural = "participants_views", False


def doc_has_headings(headings: list[str]) -> bool:
    return any(classify_heading(h)[0] in ("staff_economic", "participants_views", "policy_action",
                                          "financial_markets", "staff_outlook", "staff_financial")
               for h in headings)
