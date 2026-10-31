"""Label schema and output validation.

Input batch line:  {"id": str, "prev_text": str, "text": str}
Output line:       {"id": str, "aspects": [{"function", "topic", "direction"}], "confidence": str}

Validation is strict. The only leniency: a surrounding markdown code fence and trailing
whitespace (including CRLF line ends) are stripped, and a note is recorded when a fence
was removed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FUNCTIONS = ("ASSESSMENT", "OUTLOOK", "POLICY", "PROCEDURAL")
TOPICS = ("INFLATION", "LABOR", "ACTIVITY", "FINANCIAL", "GENERAL")
TOPIC_FUNCTIONS = ("ASSESSMENT", "OUTLOOK")   # functions that carry a topic
DIRECTIONS = (-2, -1, 0, 1, 2)
CONFIDENCES = ("low", "med", "high")
MAX_ASPECTS = 2

_ASPECT_KEYS = {"function", "topic", "direction"}
_LABEL_KEYS = {"id", "aspects", "confidence"}
_FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_-]*$")


@dataclass
class Report:
    batch_path: Path | None
    output_path: Path | None
    n_expected: int = 0
    n_valid: int = 0
    errors: dict[str, list[str]] = field(default_factory=dict)   # per id
    global_errors: list[str] = field(default_factory=list)       # not attributable to one id
    notes: list[str] = field(default_factory=list)
    labels: dict[str, dict] = field(default_factory=dict)        # valid, parsed labels

    @property
    def ok(self) -> bool:
        return not self.errors and not self.global_errors

    def add(self, id_: str, msg: str) -> None:
        self.errors.setdefault(id_, []).append(msg)

    def summary(self) -> str:
        head = (f"{'PASS' if self.ok else 'FAIL'}: {self.n_valid}/{self.n_expected} ids valid, "
                f"{len(self.errors)} ids with errors, {len(self.global_errors)} global errors")
        lines = [head]
        lines += [f"  note: {n}" for n in self.notes]
        lines += [f"  global: {e}" for e in self.global_errors]
        for id_, errs in self.errors.items():
            lines += [f"  {id_}: {e}" for e in errs]
        return "\n".join(lines)


def _is_int(x: Any) -> bool:
    return type(x) is int   # rejects bool (subclass of int), float, str


def validate_label(obj: Any, allow_extra_keys: bool = False) -> list[str]:
    """Return the list of schema errors for one parsed output line (empty list = valid)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["line is not a JSON object"]
    extra = set(obj) - _LABEL_KEYS
    missing = _LABEL_KEYS - set(obj)
    if missing:
        errs.append(f"missing keys: {sorted(missing)}")
    if extra and not allow_extra_keys:
        errs.append(f"unexpected keys: {sorted(extra)}")
    if "id" in obj and not isinstance(obj["id"], str):
        errs.append("id is not a string")
    if "confidence" in obj:
        c = obj["confidence"]
        if not (isinstance(c, str) and c in CONFIDENCES):
            errs.append(f"bad confidence {c!r} (allowed {list(CONFIDENCES)})")
    if "aspects" in obj:
        aspects = obj["aspects"]
        if not isinstance(aspects, list):
            errs.append("aspects is not a list")
        else:
            if not 1 <= len(aspects) <= MAX_ASPECTS:
                errs.append(f"{len(aspects)} aspects (need 1-{MAX_ASPECTS})")
            for i, a in enumerate(aspects):
                errs += [f"aspect {i}: {e}" for e in _validate_aspect(a)]
    return errs


def _validate_aspect(a: Any) -> list[str]:
    if not isinstance(a, dict):
        return ["not a JSON object"]
    if set(a) != _ASPECT_KEYS:
        return [f"keys {sorted(a)} != {sorted(_ASPECT_KEYS)}"]
    errs: list[str] = []
    f, t, d = a["function"], a["topic"], a["direction"]
    f_ok = isinstance(f, str) and f in FUNCTIONS
    if not f_ok:
        errs.append(f"bad function {f!r} (allowed {list(FUNCTIONS)})")
    if f_ok and f in TOPIC_FUNCTIONS:
        if not (isinstance(t, str) and t in TOPICS):
            errs.append(f"{f} needs topic in {list(TOPICS)}, got {t!r}")
    elif f_ok:
        if t is not None:
            errs.append(f"{f} must have topic null, got {t!r}")
    elif t is not None and not (isinstance(t, str) and t in TOPICS):
        errs.append(f"bad topic {t!r}")
    if not _is_int(d):
        errs.append(f"direction must be a JSON integer, got {type(d).__name__} {d!r}")
    elif d not in DIRECTIONS:
        errs.append(f"direction {d} outside -2..2")
    elif f == "PROCEDURAL" and d != 0:
        errs.append(f"PROCEDURAL direction must be 0, got {d}")
    return errs


def read_batch(batch_path: str | Path) -> list[dict]:
    """Read a batch JSONL file (our own file; malformed content raises ValueError)."""
    items = []
    with open(batch_path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                raise ValueError(f"{batch_path}: blank line {n}")
            obj = json.loads(line)
            if not isinstance(obj.get("id"), str):
                raise ValueError(f"{batch_path}: line {n} has no string id")
            items.append(obj)
    return items


def split_output_lines(text: str) -> tuple[list[tuple[int, str]], list[str], list[str]]:
    """Strip trailing whitespace and an optional code fence.

    Returns ([(lineno, line)], notes, errors); lineno refers to the original file.
    """
    notes: list[str] = []
    raw = text.rstrip().splitlines()
    if not raw:
        return [], notes, ["output is empty"]
    start, end = 0, len(raw)
    if raw[0].strip().startswith("```"):
        if _FENCE_OPEN.match(raw[0].strip()) and len(raw) >= 2 and raw[-1].strip() == "```":
            start, end = 1, len(raw) - 1
            notes.append("surrounding markdown code fence stripped")
        else:
            return [], notes, ["malformed or unbalanced markdown code fence"]
    return [(i + 1, raw[i].rstrip()) for i in range(start, end)], notes, []


def validate_output(batch_path: str | Path, output_path: str | Path,
                    allow_extra_keys: bool = False) -> Report:
    """Validate one labeller output file against its batch file."""
    rep = Report(Path(batch_path), Path(output_path))
    batch_ids = [it["id"] for it in read_batch(batch_path)]
    batch_set = set(batch_ids)
    rep.n_expected = len(batch_ids)

    try:
        text = Path(output_path).read_bytes().decode("utf-8")
    except FileNotFoundError:
        rep.global_errors.append("output file does not exist")
        return rep
    except UnicodeDecodeError as e:
        rep.global_errors.append(f"output is not valid UTF-8: {e}")
        return rep

    lines, notes, errs = split_output_lines(text)
    rep.notes += notes
    rep.global_errors += errs

    seen: dict[str, list[tuple[int, dict]]] = {}
    for lineno, line in lines:
        if not line.strip():
            rep.global_errors.append(f"line {lineno}: blank line")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            rep.global_errors.append(f"line {lineno}: invalid JSON ({e.msg})")
            continue
        id_ = obj.get("id") if isinstance(obj, dict) else None
        if not isinstance(id_, str):
            rep.global_errors.append(f"line {lineno}: missing or non-string id")
            continue
        seen.setdefault(id_, []).append((lineno, obj))

    for id_ in batch_ids:
        if id_ not in seen:
            rep.add(id_, "missing from output")
    for id_, occ in seen.items():
        if id_ not in batch_set:
            rep.add(id_, "extra id not in batch")
            continue
        if len(occ) > 1:
            rep.add(id_, f"appears {len(occ)} times (lines {[n for n, _ in occ]})")
            continue
        lineno, obj = occ[0]
        label_errs = validate_label(obj, allow_extra_keys=allow_extra_keys)
        for e in label_errs:
            rep.add(id_, f"line {lineno}: {e}")
        if not label_errs:
            rep.labels[id_] = obj
    rep.n_valid = len([i for i in batch_ids if i in rep.labels])
    return rep


def load_valid_output(batch_path: str | Path, output_path: str | Path) -> dict[str, dict]:
    """Labels from an output file, or raise ValueError if it does not validate."""
    rep = validate_output(batch_path, output_path)
    if not rep.ok:
        raise ValueError(f"{output_path} is invalid:\n{rep.summary()}")
    return rep.labels
