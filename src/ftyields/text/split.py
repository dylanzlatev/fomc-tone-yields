"""Sentence splitter for FOMC prose (no NLTK/spaCy).

`split_sentences(paragraph)` returns sentences with protection for abbreviations
(Mr., U.S., a.m., Jan., ...), initials (Ben S. Bernanke), decimals/fractions (2.5, 1/4),
ellipses and parentheses. Fragments shorter than `min_words` are merged into their
neighbour (the next sentence, or the previous one for the last fragment).
"""
from __future__ import annotations

import re

# Abbreviations that never end a sentence.
NEVER = {
    "mr", "mrs", "ms", "dr", "messrs", "mses", "mmes", "prof", "sen", "rep", "gov", "gen", "lt",
    "col", "sgt", "capt", "st", "vs", "v", "no", "nos", "fig", "vol", "al", "approx", "ca", "cf",
    "ft", "mt", "viz", "ii", "iii", "iv",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "e.g", "i.e", "ph.d", "a.k.a",
}
# Abbreviations that may end a sentence: boundary only if the next word looks like a sentence start.
SOFT = {"jr", "sr", "inc", "co", "corp", "ltd", "etc", "u.s", "u.k", "d.c", "p.m", "a.m", "u.s.a",
        "n.y", "u.n", "e.u"}

STARTERS = {
    "The", "A", "An", "In", "It", "Its", "This", "That", "These", "Those", "They", "Their", "There",
    "However", "At", "On", "For", "As", "Some", "Many", "Most", "Several", "Participants", "Members",
    "Staff", "Committee", "With", "While", "Although", "If", "Meanwhile", "Overall", "Looking",
    "Both", "Other", "Net", "Total", "Since", "Over", "Following", "He", "She", "We", "Our", "But",
    "And", "Also", "Such", "One", "Two", "Three", "Another", "All", "Each", "Early", "Late",
    "Policymakers", "Manager", "Mr.", "Ms.", "Chairman", "Chair", "Consumer", "Business",
    "Inflation", "Employment", "Real", "Financial", "Monetary", "Conditions", "Growth", "After",
    "Before", "During", "To", "By", "Of", "Under", "Between", "Against", "Votes", "Voting", "Vote",
}

TITLE_STARTERS = {"Governor", "President", "Vice", "Chairman", "Chair", "Messrs.", "Ms.", "Mr."}

_ELLIPSIS =re.compile(r"\.{3,}|(?:\.\s){2,}\.")
_BOUND = re.compile(r"""
    (?P<punct>[.?!])               # candidate terminator
    (?P<close>["'”’)\]]*) # closing quotes / parens
    (?P<ws>\s+)
    (?=(?P<open>["“‘(\[]*)(?P<nxt>[A-Z0-9]))
""", re.X)
_PREV_TOKEN = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]+$")
_NEXT_WORD = re.compile(r"[\"“‘(\[]*([A-Za-z][A-Za-z.'’-]*)")


def _depth_before(text: str, idx: int) -> int:
    seg = text[:idx]
    return seg.count("(") - seg.count(")")


def _is_boundary(text: str, m: re.Match) -> bool:
    punct = m.group("punct")
    start = m.start("punct")
    if punct in "?!":
        # still require paren depth 0
        return _depth_before(text, m.end("close")) <= 0
    before = text[:start]
    # paren depth must return to 0 after the closing characters
    if _depth_before(text, m.end("close")) > 0:
        return False
    # enumerator at very start: "1. The Federal ..." / "A. To buy ..."
    if re.fullmatch(r"\(?\d{1,2}", before.strip()) or re.fullmatch(r"[A-Za-z]", before.strip()):
        return False
    mt = _PREV_TOKEN.search(before)
    if mt is None:
        return True
    tok = mt.group()
    low = tok.lower()
    # single capital letter: an initial (Ben S. Bernanke, Roger W. Ferguson)
    if len(tok) == 1 and (tok.isupper() or (tok == "i" and re.fullmatch(r"(?:.*[;:]\s+)?i", before.strip()))):
        return False
    if low in NEVER:
        return False
    # internal-dot tokens such as "U.S", "p.m" are soft abbreviations
    if low in SOFT:
        nw = _NEXT_WORD.match(text, m.end("ws"))
        starters = STARTERS | TITLE_STARTERS if low in ("jr", "sr") else STARTERS
        return bool(nw and nw.group(1) in starters and not m.group("nxt").isdigit())
    # dotted initials like "J.P." : never a boundary unless followed by a starter word
    if "." in tok and re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", tok):
        nw = _NEXT_WORD.match(text, m.end("ws"))
        return bool(nw and nw.group(1) in STARTERS)
    # a digit-led next token after a period needs a non-abbreviation context: allowed.
    return True


def _n_words(s: str) -> int:
    return len(s.split())


def split_sentences(paragraph: str, min_words: int = 4) -> list[str]:
    """Split one paragraph into sentences (fragments < min_words merged into a neighbour)."""
    text = re.sub(r"\s+", " ", paragraph).strip()
    if not text:
        return []
    text = _ELLIPSIS.sub("…", text)
    cuts = []
    for m in _BOUND.finditer(text):
        if _is_boundary(text, m):
            cuts.append(m.end("close"))
    parts, prev = [], 0
    for c in cuts:
        parts.append(text[prev:c].strip())
        prev = c
    parts.append(text[prev:].strip())
    parts = [p.replace("…", "...") for p in parts if p]
    return _merge_short(parts, min_words)


def _merge_short(parts: list[str], min_words: int) -> list[str]:
    if len(parts) <= 1:
        return parts
    out: list[str] = []
    carry = ""
    for p in parts:
        cur = (carry + " " + p).strip() if carry else p
        if _n_words(cur) < min_words:
            carry = cur
            continue
        out.append(cur)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out
