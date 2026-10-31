"""Cleaning of raw FOMC statement / minutes text into paragraphs and headings.

Pipeline for one document (`clean_document`):
  1. repair encoding damage (UTF-8 read as Latin-1, '?' replacement characters, NBSP, ...)
  2. drop web boilerplate lines, footnote lines and the footnote section
  3. drop the attendance block (PRESENT: / Attendance) and harvest its names (gazetteer)
  4. re-flow hard-wrapped lines into blocks, split blocks into headings and paragraphs
Sentence-level removals (vote tallies, adjournment, ...) live in `drop_sentence`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- characters

_MOJI = re.compile("[Â-ß][\u0080-¿]|[à-ï][\u0080-¿]{2}|[ð-ô][\u0080-¿]{3}")
_C1 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "–", "\x97": "—",
       "\x85": "...", "\x99": ""}
FRAC_TOKEN = "[FRAC]"


def _unmoji(m: re.Match) -> str:
    s = m.group()
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fix_text(t: str) -> str:
    """Repair mojibake and normalise whitespace characters (keeps en/em dashes and curly quotes)."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # U+FFFD stored as UTF-8 bytes read as Latin-1 ("ï¿½"): apostrophes, vis-a-vis, fractions
    t = t.replace("ï¿½", "�")
    t = _MOJI.sub(_unmoji, t)
    for k, v in _C1.items():
        t = t.replace(k, v)
    t = re.sub(r"(?<=[A-Za-z])�(?=s\b)", "'", t)            # Committee<?>s
    t = re.sub(r"(?<=[a-z]s)�(?=[\s,.;])", "'", t)            # countries<?> experiences
    t = re.sub(r"(?<=[Vv]is-)�(?=-vis)", "à", t)          # vis-<?>-vis
    t = re.sub(r"(?<=[a-z])�(?=[a-z])", "'", t)                # generic in-word apostrophe
    t = re.sub(r"(?<=communiqu)�", "é", t)
    t = re.sub(r"(?<=[.,;!?])�(?=\s|$)", '"', t)               # closing quotation mark
    t = re.sub(r"(?:(?<=\s)|^)�(?=[A-Za-z$])", '"', t, flags=re.M)  # opening quotation mark
    t = t.replace("�", FRAC_TOKEN)                            # a lost 1/4, 1/2 or 3/4 glyph
    t = t.replace("­", "")                                    # soft hyphen
    t = t.replace("‑", "-")                                   # non-breaking hyphen
    t = re.sub("[       ​﻿]", " ", t)
    return re.sub(r"[ 	]+", " ", t)


_FOOTNOTE_MARK = re.compile(r"(?<=[a-z\)\"”’])\d{1,2}(?=[\s,;)]|$)|(?<=[a-z]\.)\d{1,2}(?=\s|$)|(?<=\)\.)\d{1,2}(?=\s|$)|(?<=[a-z][,;:])\d{1,2}(?=\s|$)")


def strip_footnote_marks(t: str) -> str:
    """Remove footnote reference digits glued to words/punctuation ('percent.2 ', 'Committee1')."""
    return _FOOTNOTE_MARK.sub("", t)


# --------------------------------------------------------------------------- boilerplate

_BOILER_LINE = re.compile(
    r"""^(?:
        return\ to\ (?:top|text)\b.*|share|home(?:\ \|.*)?|news\ and\ events|accessibility|contact\ us|
        last\ update(?:d)?\s*:.*|release\ date\s*:.*|for\ immediate\ release.*|for\ release\ at.*|
        implementation\ note\ issued.*|\d{4}\ monetary\ policy|footnotes?|skip\ to\ .*|back\ to\ top|
        minutes\ of\ the\ federal\ open\ market\ committee|secretary|notation\ votes?|
        (?:text|pdf|print|html)\ ?(?:version)?|federal\ reserve\ board|board\ of\ governors\ of\ the\ federal\ reserve\ system|
        press\ releases?|monetary\ policy|fomc|recent\ postings|
        (?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*[-–—]?\s*\d{0,2}\s*,?\s*\d{4}
    )$""", re.I | re.X)

_RETURN_TO = re.compile(r"\s*Return to (?:top|text)\s*", re.I)

_PRESENT_START = re.compile(r"^(?:(?:PRESENT|Present)\s*:\s*|PRESENT\s*$|Attendance\s*$)(.*)$")

# Headings recognised as section markers (also used by `sections`).
KNOWN_HEADING = re.compile(
    r"""^(?:
        developments\ in\ financial\ markets.*|staff\ review\ of\ the\ (?:economic|financial)\ situation|
        staff\ economic\ outlook|participants['’]\ views.*|committee\ policy\ actions?|
        annual\ organizational\ matters|selection\ of\ committee\ officers?|notation\ votes?|
        review\ of\ monetary\ policy\ strategy.*|statement\ on\ longer-run\ goals.*|
        market\ developments\ and\ open\ market\ operations|system\ liquidity\ programs.*|
        staff\ review\ of\ the\ economic\ and\ financial\ situation|
        (?:meeting\ )?participants['\u2019]\ views\ and\ committee\ policy\ actions?|conference\ calls?
    )$""", re.I | re.X)

ROLE_WORDS = {
    "Board", "Governors", "Governor", "Federal", "Reserve", "Bank", "Banks", "Open", "Market",
    "Committee", "System", "Account", "Division", "Divisions", "Office", "Offices", "Director",
    "Directors", "Associate", "Assistant", "Deputy", "Senior", "Vice", "President", "Presidents",
    "Chairman", "Chair", "Secretary", "Economist", "Economists", "Counsel", "General", "Adviser",
    "Advisers", "Advisor", "Advisors", "Manager", "Analyst", "Officer", "Officers", "Specialist",
    "Section", "Chief", "Special", "Project", "Principal", "First", "Executive", "Staff",
    "Research", "Statistics", "Monetary", "Affairs", "International", "Finance", "Banking",
    "Supervision", "Regulation", "Financial", "Stability", "Alternate", "Members", "Member",
    "Interim", "Acting", "Technology", "Records", "Management", "Information", "Payment",
    "Payments", "Operations", "Policy", "Research", "Members", "Group", "Administrator", "Systems",
    "Economic", "Planning", "Control", "Markets", "Review", "Consumer", "Community", "Affairs",
    "Reserve", "Treasury", "Services", "Technology", "Analysis", "Strategic", "Communications",
    "Public", "Legal", "Congressional", "Liaison", "Assistants", "Specialists", "Associates",
    "Attendance", "Present", "Policy", "Fellow", "Fellows", "Staff", "Program", "Programs",
    "Reports", "Statistical", "Administration", "Secretariat", "Open", "Data", "Chicago",
    "Dallas", "Boston", "Atlanta", "Richmond", "Cleveland", "Philadelphia", "Minneapolis",
    "Kansas", "City", "Louis", "York", "Francisco", "Angeles", "Charlotte", "New", "San", "St", "St.",
    "Los",
}
NOT_NAME_WORDS = ROLE_WORDS | {
    "Outlook", "Situation", "Matters", "Developments", "Strategy", "Normalization", "Framework",
    "Implementation", "Balance", "Sheet", "Update", "Discussion", "Reinvestment", "Tools",
    "Practices", "Vote", "Votes", "Action", "Actions", "Participants", "Views", "Current",
    "Conditions", "Zealand", "States", "Kingdom", "Korea", "Hong", "Kong", "Saudi", "Arabia",
    "United", "New", "San", "St.", "Mexico", "Canada", "Japan", "England", "Euro", "Amount",
    "Foreign", "Currency", "Swiss", "National", "European", "Central", "Bank", "Australian",
    "Brazilian", "Danish", "Norwegian", "Swedish", "Pounds", "Mexican", "Canadian", "Singapore",
    "Korean", "Japanese", "Authorization", "Directive", "Procedural", "Instructions", "Domestic",
    "Transactions", "Securities", "Lending", "Additional", "General", "Other", "Warehousing",
    "Spot", "Forward", "Holdings", "Reciprocal", "Arrangements", "Standing", "Dollar", "Liquidity",
    "Swaps", "Maximum", "Notation", "Disclosure", "Minutes", "January", "February", "March",
    "April", "May", "June", "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Footnotes", "Return", "Text", "Top",
}
_LOWER_OK = {"and", "of", "the", "to", "respectively", "for", "in", "on", "as", "de", "van", "der",
             "von", "la", "du", "dos", "y", "del", "den", "di", "with", "pro", "tem", "emeritus"}
_NAME_TOKEN = re.compile(r"^(?:[A-Z][A-Za-z'’\-À-ſ]+|[A-Z]\.|[A-Z]\.[A-Z]\.|[a-z]{1,2}[A-Z]\.)$")
_HON = re.compile(r"^(?:Mr|Ms|Mrs|Dr|Messrs|Mses|Mmes)\.$")


_HON_GLUE = re.compile(r"(?<=[a-z\)])(?=(?:Mr|Ms|Mrs|Messrs|Mses|Dr)\.)")


def _clean_tokens(line: str) -> list[str]:
    s = _HON_GLUE.sub(" ", line)
    s = re.sub(r"\s*\b\d{1,2}\s*/(?=\s)", "", s)                      # '2/' footnote marks
    s = re.sub(r"(?<=[A-Za-z.]),?\s?\d{1,2}(?=[,\s]|$)", "", s)
    s = re.sub(r"(?<=[a-z])\.$", "", s.strip())                       # wrapped role fragment 'Governors.'
    return s.replace(",", " , ").replace(";", " ; ").split()


def is_attendance_line(line: str) -> bool:
    """Is this (non-empty) line part of an attendance / staff list?"""
    s = line.strip()
    if not s:
        return True
    if KNOWN_HEADING.match(re.sub(r"\d+$", "", s).strip()):
        return False
    if re.match(r"^[_\s]*\d{1,2}\s*[/.]?\s*(?:Attended|Return)", s) or "Return to text" in s:
        return True
    if re.match(r"^(?:Attended|Participated)\b", s):
        return True
    toks = _clean_tokens(s)
    if len(toks) > 600:
        return False
    # sentence terminal period followed by more text means prose
    if re.search(r"[a-z]{3,}(?<!Messrs)(?<!Mses)(?<!Mmes)\.\s+[A-Z][a-z]+\s+[a-z]+", s):
        return False
    low = [t for t in toks if t[0].islower() and t not in _LOWER_OK and not re.match(r"^[a-z]{1,2}[A-Z]\.", t)]
    if low:
        return False
    has_comma = "," in s
    starts_hon = bool(_HON.match(toks[0])) if toks else False
    caps = [t for t in toks if t not in {",", ";"}]
    role_only = bool(caps) and all((t in ROLE_WORDS) or t in _LOWER_OK or _HON.match(t) for t in caps)
    pure_name = (2 <= len(caps) <= 5 and not has_comma
                 and all(_NAME_TOKEN.match(t) or t in {"Jr.", "Sr.", "II", "III", "IV"} or t in _LOWER_OK for t in caps)
                 and not any(t in NOT_NAME_WORDS for t in caps))
    surname_list = (has_comma and len(caps) >= 2
                    and all(_NAME_TOKEN.match(t) or t == "and" for t in caps) and not any(t in NOT_NAME_WORDS for t in caps))
    if starts_hon or role_only or pure_name or surname_list:
        return True
    if has_comma and any(t in ROLE_WORDS for t in caps):
        return True
    return False


_GLUE = re.compile(
    r"(?<=[a-z\)])(?=(?:The|By|In|A|An|It|At|On|With|After|Following|Unanimous|Advice|Secretary’s|Secretary's)\s+[A-Za-z'’]+)"
    r"|(?<=[a-z\)\.])(?=[A-Z][a-z]+ [a-z]+ [a-z]+\b)")


def _split_glue(line: str) -> tuple[str, str]:
    """Split 'respectivelyBy unanimous vote, ...' into (list part, body part)."""
    for m in _GLUE.finditer(line):
        if m.start() > 8 and line[m.start() - 1] not in " ":
            return line[:m.start()], line[m.start():]
    return line, ""


def extract_names(line: str) -> set[str]:
    """Names ('Ben S. Bernanke') from an attendance / vote / officer line (>= 2 tokens)."""
    names: set[str] = set()
    s = re.sub(r"^(?:PRESENT|Present|Attendance)\s*:?", "", line.strip())
    if re.match(r"^Vot", s):
        s = re.sub(r"^.*?\b(?:were|was|action|actions)\s*:?\s+(?=[A-Z])", "", s)
        s = re.sub(r",?\s+and\s+", ", ", s)
    for piece in s.split(";"):
        last = None
        for seg in piece.split(","):
            seg = re.sub(r"(?<=[A-Za-z.])\d{1,2}$", "", seg.strip())
            seg = re.sub(r"^(?:and|with)\s+", "", seg).strip()
            seg = re.sub(r"\s+(?:Chair(?:man)?|Vice Chair(?:man)?)$", "", seg)
            if not seg:
                continue
            if seg in ("Jr.", "Sr.", "II", "III", "IV") and last:
                continue
            toks = seg.split()
            if (2 <= len(toks) <= 5 and all(_NAME_TOKEN.match(t) or t in _LOWER_OK - {"and", "of", "the", "to", "for", "in", "on", "as", "with", "respectively"}
                                              for t in toks)
                    and not any(t in NOT_NAME_WORDS for t in toks) and not _HON.match(toks[0])):
                names.add(" ".join(toks))
                last = seg
            else:
                break
    return names


# --------------------------------------------------------------------------- line merging

_ABBR_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Messrs|Mses|St|vs|No|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e|Jr|Sr|Inc|Co|Corp)|\b[A-Z]|U\.S|U\.K|D\.C|a\.m|p\.m)\.$")
_SOFT_END = re.compile(r"(?:Jr|Sr|Inc|Co|Corp|U\.S|U\.K|D\.C|a\.m|p\.m|etc)\.$")
_NEVER_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Messrs|Mses|St|vs|No|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e)|\b[A-Z])\.$")
_TERMINAL = re.compile(r"""[.?!:]["'”’)\]]*\d{0,2}\s*$""")
_STOP_LOWER = {"of", "in", "and", "the", "for", "to", "on", "with", "a", "an", "from", "by", "at", "as", "or", "vs"}
_FIRST_WORD = re.compile(r"[\"“(\[]*([A-Za-z][A-Za-z.'’-]*)")


def _starter(nxt: str) -> bool:
    from ftyields.text.split import STARTERS
    m = _FIRST_WORD.match(nxt)
    return bool(m and m.group(1) in STARTERS)


def _ends_sentence(prev: str, nxt: str | None) -> bool:
    if not _TERMINAL.search(prev):
        return False
    core = re.sub(r"[\"'”’)\]\d\s]+$", "", prev)
    if _NEVER_END.search(core):
        return False
    if _SOFT_END.search(core):
        return bool(nxt and _starter(nxt))
    return True


def heading_like(line: str) -> bool:
    """Short title-like line without terminal punctuation (heading, name, list entry)."""
    s = line.strip()
    if not s or len(s) > 160:
        return False
    if re.search(r"[.?!:;,]\s*$", s) and not re.search(r"\.\)$", s):
        return False
    if s.startswith("(") and s.endswith(")"):
        return True
    words = s.split()
    if len(words) > 20:
        return False
    if s.isupper():
        return True
    content = [w for w in words if w.lower() not in _STOP_LOWER]
    if not content:
        return False
    cap = sum(1 for w in content if w[0].isupper() or w[0].isdigit() or w[0] in "(\"“")
    return cap / len(content) >= 0.7


def merge_lines(lines: list[str]) -> list[str]:
    """Re-flow hard-wrapped text: a line break is a paragraph break unless it is a soft wrap."""
    out: list[str] = []
    cur = ""
    for ln in lines:
        s = re.sub(r"\s+", " ", ln).strip()
        if not s:
            continue
        if not cur:
            cur = s
            continue
        if heading_like(cur) and not (s[0].islower()):
            join = False
        elif _ends_sentence(cur, s):
            join = False
        elif _TERMINAL.search(cur) and heading_like(s) and len(s.split()) >= 2:
            join = False
        elif _TERMINAL.search(cur) and not _SOFT_END.search(re.sub(r"[\"')\d\s]+$", "", cur)) \
                and not _NEVER_END.search(re.sub(r"[\"')\d\s]+$", "", cur)):
            join = False
        elif not s[0].isupper():
            join = True
        else:
            join = len(cur) >= 50
        if join:
            cur = cur + " " + s
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- document

@dataclass
class Block:
    kind: str   # 'heading' | 'para'
    text: str


@dataclass
class CleanDoc:
    blocks: list[Block] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    n_attendance_lines: int = 0


def _drop_attendance(lines: list[str], names: set[str]) -> tuple[list[str], int]:
    """Remove the PRESENT:/Attendance block; returns (lines, number removed)."""
    out: list[str] = []
    i, n, removed = 0, len(lines), 0
    while i < n:
        s = lines[i].strip()
        m = _PRESENT_START.match(s)
        if not m:
            out.append(lines[i])
            i += 1
            continue
        rest = m.group(1)
        removed += 1
        if rest:
            keep, body = _split_glue(rest)
            names |= extract_names(keep)
            if body:
                out.append(body)
                i += 1
                continue
        i += 1
        while i < n:
            ln = _HON_GLUE.sub(" ", lines[i])
            if not ln.strip():
                i += 1
                continue
            keep, body = _split_glue(ln)
            if body and is_attendance_line(keep):
                names |= extract_names(keep)
                removed += 1
                out.append(body)
                i += 1
                break
            if is_attendance_line(ln):
                names |= extract_names(ln)
                removed += 1
                i += 1
                continue
            break
        # continue scanning (only the first block per document is expected)
    return out, removed


_GLUED_HEAD = re.compile(
    r"^\s*(Notation Votes?|Staff Economic Outlook|Conference Calls?|Annual Organizational Matters|"
    r"Developments in Financial Markets and the Federal Reserve['\u2019]s Balance Sheet|"
    r"Staff Review of the (?:Economic and Financial|Economic|Financial) Situation|"
    r"(?:Meeting )?Participants['\u2019] Views and Committee Policy Actions?|"
    r"Participants['\u2019] Views on Current Conditions and the Economic Outlook|Committee Policy Actions?|"
    r"Market Developments and Open Market Operations|System Liquidity Programs and Balance Sheet)(?=[A-Z])")


def _split_glued_heading(line: str) -> list[str]:
    """Headings of 2007-09 minutes are glued to the next paragraph ('Staff Economic OutlookIn the ...')."""
    m = _GLUED_HEAD.match(line)
    if not m:
        return [line]
    return [m.group(1), line[m.end():]]


_ROLE_SPLIT = re.compile(r"(?<=[a-z\.])(?=(?:Vice Chair|Chair|Secretary|Deputy|Assistant|General Counsel|Economist|Associate|Manager)\b)")
_PART = {"de", "van", "der", "von", "la", "du", "del", "den", "di"}


def _names_only(line: str) -> set[str] | None:
    """Names if the whole line is a comma/semicolon list of person names, else None."""
    s = re.sub(r"(?<=[A-Za-z.]),?\s?\d{1,2}(?=[,\s]|$)", "", line.strip())
    s = re.sub(r",?\s+and\s+", ", ", s)
    out: set[str] = set()
    segs = [x.strip() for x in re.split(r"[;,]", s) if x.strip()]
    if not segs:
        return None
    for seg in segs:
        if seg in ("Jr.", "Sr.", "II", "III", "IV"):
            continue
        toks = seg.split()
        if not (2 <= len(toks) <= 5) or any(t in NOT_NAME_WORDS for t in toks):
            return None
        if not all(_NAME_TOKEN.match(t) or t in _PART for t in toks) or _HON.match(toks[0]):
            return None
        out.add(" ".join(toks))
    return out or None


def _role_only(line: str) -> bool:
    toks = [t for t in re.sub(r"[,;]", " ", line).split()]
    return bool(toks) and all(t in ROLE_WORDS or t in _LOWER_OK for t in toks) and any(t in ROLE_WORDS for t in toks)


def harvest_officer_names(lines: list[str]) -> set[str]:
    """Names from 'Name / Role' pairs and 'Name, President of the Federal Reserve Bank of X' lines."""
    names: set[str] = set()
    pending: set[str] = set()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        parts = [p for p in _ROLE_SPLIT.split(s) if p.strip()] if len(s) < 80 else [s]
        for part in parts:
            part = part.strip()
            if re.search(r",\s+(?:First Vice )?President of the Federal Reserve Bank", part):
                names |= extract_names(part)
                pending = set()
                continue
            ns = _names_only(part)
            if ns:
                pending |= ns
            elif _role_only(part):
                names |= pending
                pending = set()
            else:
                pending = set()
    return names


def clean_document(text: str, doc_type: str) -> CleanDoc:
    t = fix_text(text)
    t = _RETURN_TO.sub("\n", t)
    raw_lines = t.split("\n")
    # drop the footnote section: from a 'Footnotes' line while lines start with a digit
    lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        s = raw_lines[i].strip()
        if re.search(r"(?:^|[a-z0-9])Footnotes?$", s) and doc_type == "minutes":
            j = i + 1
            while j < len(raw_lines) and (not raw_lines[j].strip() or re.match(r"^\d{1,2}\.?\s", raw_lines[j].strip())):
                j += 1
            i = j
            continue
        lines.append(raw_lines[i])
        i += 1
    # footnote bodies that survive 'Return to text' removal: numbered lines in the tail
    cd = CleanDoc()
    # numbered footnote lines at the end of the document ("1. The Federal Open ..." / "2. Attended ...")
    k = len(lines)
    while k > 0 and (not lines[k - 1].strip() or re.match(r"^\d{1,2}\.\s+\S", lines[k - 1].strip())):
        k -= 1
    tail = [x for x in lines[k:] if x.strip()]
    if tail and all(re.match(r"^\d{1,2}\.\s+\S", x.strip()) for x in tail) and len(tail) < 40:
        lines = lines[:k]
    if doc_type == "minutes":
        lines, removed = _drop_attendance(lines, cd.names)
        cd.n_attendance_lines = removed
    # officer / member lists (name line followed by a role line) -> gazetteer
    if doc_type == "minutes":
        cd.names |= harvest_officer_names(lines)
    lines = [x for ln in lines for x in _split_glued_heading(ln)]
    lines = [ln for ln in lines if not _BOILER_LINE.match(ln.strip())]
    lines = [strip_footnote_marks(ln) for ln in lines]
    blocks = merge_lines(lines)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # vote paragraphs feed the gazetteer (statements and minutes); every sentence of the block
        if re.match(r"^Vot(?:ing|es?)\b", b):
            for sent in re.split(r"(?<=[a-z])\.\s+(?=Vot)", b):
                cd.names |= extract_names(sent)
        if heading_like(b):
            cd.blocks.append(Block("heading", re.sub(r"\s*\d{1,2}$", "", b)))
        elif not _TERMINAL.search(b) and _fragment_like(b):
            cd.blocks.append(Block("heading", b))
        else:
            cd.blocks.append(Block("para", b))
    return cd


def _fragment_like(b: str) -> bool:
    """Unterminated block that is a list/heading rather than prose."""
    words = b.split()
    if len(words) <= 25:
        return True
    cap = sum(1 for w in words if w[0].isupper() or w[0].isdigit())
    return cap / len(words) >= 0.75


# --------------------------------------------------------------------------- sentence-level rules

_WHO = re.compile(r"\b(?:who|because|preferred|dissent\w*|favou?red|opposed|believed|supported|thought|would|wanted|expressed|voted)\b", re.I)
_VOTE_AGAINST = re.compile(r"\bVot(?:ing|es?)\b(?:\s*\([^)]*\))?\s+against\b")
_DROP_PATTERNS = [
    re.compile(r"^(?:A|An) (?:joint )?meeting of the Federal Open Market Committee\b.*\bwas held\b", re.I),
    re.compile(r"^(?:The|This) meeting (?:was )?adjourned\b", re.I),
    re.compile(r"^It (?:also )?was agreed that\b.*\bnext (?:regularly scheduled )?meeting of the Committee would be held\b", re.I),
    re.compile(r"^By notation vote\b.*\b(?:approved|ratified)\b.*\bminutes\b", re.I),
    re.compile(r"^By unanimous vote, the minutes of the meeting of the Federal Open Market Committee held on\b.*\bwere approved", re.I),
    re.compile(r"^(?:For media inquiries|Implementation Note issued|Release Date|For release at|For immediate release)\b", re.I),
    re.compile(r"^The elected members and alternate members were as follows:?", re.I),
    re.compile(r".*,\s+as alternate\.?$", re.I),
    re.compile(r"^(?:Present|Attendance)\s*:", re.I),
    re.compile(r"^Minutes of the Federal Open Market Committee\b", re.I),
    re.compile(r"^(?:Return to (?:top|text)|Last update)\b", re.I),
    re.compile(r"^Secretary$", re.I),
    re.compile(r"^Secretary['’]s notes?:?$", re.I),
    re.compile(r"^\*\s*The (?:HTML|PDF) version\b", re.I),
    re.compile(r"^[a-z][^.]{0,40}\.$"),        # orphan lowercase fragment ('has been corrected.')
    re.compile(r"^.{0,60}\bwas held in the offices of the Board\b", re.I),
    re.compile(r"^[_\s]*\d{1,2}\s*[/.]?\s+Attended\b", re.I),
    re.compile(r"^Attended\b(?!.*\bthe Committee voted)", re.I),
]


def drop_sentence(s: str) -> str | None:
    """Return the sentence (possibly trimmed) or None if it is non-content."""
    s = s.strip()
    if not s:
        return None
    if re.match(r"^Vot(?:ing|es?)\b(?:\s*\([^)]*\))?\s+(?:for|in favor)\b", s):
        m = _VOTE_AGAINST.search(s)
        if m and _WHO.search(s[m.start():]):
            return s[m.start():]
        return None
    if re.match(r"^Vot(?:ing|es?)\b(?:\s*\([^)]*\))?\s+against\b", s) or re.match(r"^Vot(?:e|es|ing)\s+(?:in\s+)?opposition\b", s):
        return s if _WHO.search(s) else None
    for p in _DROP_PATTERNS:
        if p.match(s):
            return None
    core = re.sub(r"\b(?:Mr|Ms|Mrs|Dr|Messrs|Mses|Mmes)\.\s*|\band\b|,", " ", s).split()
    if len(core) >= 3 and re.match(r"^(?:Messrs|Mses|Mmes|Mr|Ms|Mrs)\.", s) and all(w[0].isupper() or w in ("Jr.",) for w in core):
        return None        # roll-call list of honorific + surname
    words = s.rstrip(".").replace(",", " ").split()
    if len(words) <= 14 and words and all(w[0].isupper() or w[0].isdigit() or w in {"and", "of", "the", "for", "to"} for w in words) \
            and not re.search(r"\b(?:is|are|was|were|will|has|have|had)\b", s):
        return None        # title / name list fragment, e.g. 'J. Virgil Mattingly, Jr.'
    return s


_ENUM = re.compile(r"^(?:\(?[A-Za-z0-9]{1,2}\)|[A-Z]\.|\d{1,2}\.|(?:i{1,3}|iv)\.)\s+(?=[A-Z\"“])")
_LEAD_DATE = re.compile(r"^(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:\s*[-–—]\s*\d{1,2})?,?\s+\d{4})\s+(?=A (?:joint )?meeting\b)")


def strip_leading(s: str) -> str:
    s = _LEAD_DATE.sub("", s)
    s = re.sub(r"^Minutes of the Federal Open Market Committee\s+", "", s)
    s = re.sub(r"^(?:Notation Votes?|Conference Calls?)\s+(?=[A-Z])", "", s)
    s = re.sub(r"^Footnote\s+\d{1,2}\.?\s+", "", s)
    s = _ENUM.sub("", s)
    return s.strip()
