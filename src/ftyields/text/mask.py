"""Masking for blind labelling: person names -> [NAME], dates -> [DATE], years -> [YEAR].

Economic numbers (percentages, fractions, dollar amounts) and generic time phrases
("the fourth quarter", "next year") are left untouched; "the fourth quarter of 2008"
becomes "the fourth quarter of [YEAR]".
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

NAME, DATE, YEAR = "[NAME]", "[DATE]", "[YEAR]"

_CAPNAME = r"[A-Z][A-Za-z'’\-À-ſ]+"
_STOP_AFTER_TITLE = {
    "Federal", "Board", "Bank", "Banks", "Reserve", "Open", "Market", "Committee", "System", "Treasury",
    "United", "States", "Fed", "FOMC", "Of", "The", "And", "Government", "Congress", "Deputy",
    "Assistant", "General", "Economist", "Staff", "Account", "Council", "Joint",
}
_TITLES = (r"(?:Vice\s+Chair(?:man|woman)?|Chair(?:man|woman)?|Governor|President|Secretary|Director|"
           r"Treasury\s+Secretary|Senator|Representative|Prime\s+Minister)")
_INITIALS_NAME = re.compile(
    rf"\b(?!(?:Federal|Reserve|Bank|Board|Regulation|Section|Article|Title|Part|Plan|Phase|Appendix|Annex|Paragraph)\b)"
    rf"{_CAPNAME}(?:\s+[A-Z]\.){{1,2}}\s+(?!(?:The|A|An|In|It|This|That|These|Those|At|On|As|For|Of|To|By|With)\b){_CAPNAME}"
    rf"(?:,?\s+(?:Jr\.|Sr\.|II|III))?")
_HON = re.compile(rf"\b(?:Mr|Ms|Mrs|Dr|Miss)\.\s*(?:{_CAPNAME}\s+)?{_CAPNAME}")
_HON_PL = re.compile(
    rf"\b(?:Messrs|Mses|Mmes)\.\s+{_CAPNAME}"
    rf"(?:(?:,\s+(?:and\s+)?|\s+and\s+)(?!(?:Mr|Ms|Mrs|Dr|Messrs|Mses|Mmes)\b){_CAPNAME})*")
_TITLE_NAME = re.compile(rf"\b{_TITLES}\s+((?:{_CAPNAME})(?:\s+[A-Z]\.)?(?:\s+{_CAPNAME})?)")
_TITLE_TOKEN = re.compile(r"\b(?:Vice\s+Chair(?:man|woman)?|Chair(?:man|woman)?|Governor|President|Secretary|Director)\s+\[NAME\]")

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
          r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?")
_DOW = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?(?!\d)"
_YEARNUM = r"(?:19[9]\d|20[0-2]\d)"
_NOTMONEY = r"(?!\s*(?:percent|%|basis|percentage|billion|million|trillion|thousand))"
_DATE_FULL = re.compile(
    rf"(?:\b{_DOW}(?:\s*[-–—]\s*{_DOW})?,?\s+)?\b{_MONTH}\s+{_DAY}"
    rf"(?:\s*[-–—]\s*(?:{_MONTH}\s+)?{_DAY})?{_NOTMONEY}(?:\s*,\s*{_YEARNUM}|\s+{_YEARNUM}(?!\s*(?:percent|%)))?")
_DATE_MY = re.compile(rf"\b{_MONTH}\s+{_YEARNUM}(?!\d|,\d{{3}}|\.\d|\s*(?:percent|%|basis))")
_DATE_NUM = re.compile(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b")
_YEAR = re.compile(
    rf"(?<![\d,.$/])(?<!\d-)(?<!\d–)(?:{_YEARNUM})(?:\s*[-–—/]\s*(?:{_YEARNUM}|\d{{2}})(?!\d))?"
    rf"(?!\d|,\d{{3}}|\.\d|%|\s*(?:percent|basis|million|billion|trillion|jobs|barrels|points))")
_TITLES_PL = re.compile(rf"\b(?:Presidents|Governors|Directors)\s+{_CAPNAME}(?:(?:,\s+(?:and\s+)?|\s+and\s+){_CAPNAME})*")


class Masker:
    def __init__(self, names: Iterable[str] = ()):
        variants: set[str] = set()
        for n in names:
            toks = n.split()
            if len(toks) < 2:
                continue
            variants.add(n)
            variants.add(f"{toks[0]} {toks[-1]}")
        self._full = None
        if variants:
            alt = "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in sorted(variants, key=len, reverse=True))
            self._full = re.compile(rf"(?<![A-Za-z])(?:{alt})(?![A-Za-z])")
        self.n_names = len(variants)

    def mask_names(self, t: str) -> str:
        if self._full is not None:
            t = self._full.sub(NAME, t)
        t = _INITIALS_NAME.sub(NAME, t)
        t = _HON_PL.sub(self._plural, t)
        t = _HON.sub(NAME, t)
        t = re.sub(r"\b(?:Mr|Ms|Mrs|Dr|Miss)\.\s*\[NAME\]", NAME, t)

        def plural_title(m: re.Match) -> str:
            first = m.group().split()[1]
            return m.group() if first in _STOP_AFTER_TITLE else NAME

        t = _TITLES_PL.sub(plural_title, t)

        def title(m: re.Match) -> str:
            first = m.group(1).split()[0]
            return m.group(0) if first in _STOP_AFTER_TITLE else NAME

        t = _TITLE_NAME.sub(title, t)
        t = _TITLE_TOKEN.sub(NAME, t)
        return t

    @staticmethod
    def _plural(m: re.Match) -> str:
        parts = re.split(r",\s+(?:and\s+)?|\s+and\s+", re.sub(r"^(?:Messrs|Mses|Mmes)\.\s+", "", m.group()))
        return ", ".join([NAME] * len(parts)) if len(parts) > 2 else " and ".join([NAME] * len(parts))

    @staticmethod
    def mask_dates(t: str) -> str:
        t = _DATE_FULL.sub(DATE, t)
        t = _DATE_MY.sub(DATE, t)
        t = _DATE_NUM.sub(DATE, t)
        return t

    @staticmethod
    def mask_years(t: str) -> str:
        return _YEAR.sub(YEAR, t)

    def __call__(self, t: str) -> str:
        t = self.mask_names(t)
        t = self.mask_dates(t)
        t = self.mask_years(t)
        return t


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                         "—": "-", "−": "-"})


def normalize(masked: str) -> str:
    """Lower-cased, whitespace- and quote-normalised form used for de-duplication."""
    t = masked.translate(_QUOTES).lower()
    t = t.replace("--", "-")
    return re.sub(r"\s+", " ", t).strip()


def make_uid(norm: str, n: int = 12) -> str:
    return "u" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:n]
