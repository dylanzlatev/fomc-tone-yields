"""Human validation sheet: Dylan labels 100 sentences drawn from inside the gold set.

Drawing the 100 from the 350-sentence gold set lets the human labels be compared
directly with the gold majority vote (and with every individual labeler). Sampling is
stratified by doc_types group x policy_like(text) (the heuristic from sample.py), with a
small per-stratum floor so the rare statement/both strata are represented, seed
cfg["seed"] + labelling.human.seed_offset (200). The sheet is then put in random order.

    python -m ftyields.label.human

Outputs (labels/human/): human_sheet.csv (UTF-8 with BOM, label columns blank) and
INSTRUCTIONS.md (written once by hand; not regenerated here).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.label import schema
from ftyields.label.sample import stratified_sample

SHEET_NAME = "human_sheet.csv"
LABEL_COLUMNS = ["function_1", "topic_1", "direction_1", "function_2", "topic_2", "direction_2",
                 "confidence", "notes"]
SHEET_COLUMNS = ["id", "prev_text", "text"] + LABEL_COLUMNS
_ASPECT_COLS = (("function_1", "topic_1", "direction_1"), ("function_2", "topic_2", "direction_2"))
_NULLS = {"", "null", "none", "na", "n/a", "nan", "-", "--", "—", "–"}
_CONF_ALIASES = {"medium": "med", "mid": "med", "m": "med", "l": "low", "h": "high"}


class SheetHasLabelsError(RuntimeError):
    """Refusing to overwrite a sheet that already contains labels or notes."""


# ----------------------------------------------------------------------------------
# drawing and writing


def sheet_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return repo_root() / cfg["paths"]["labels_human"] / SHEET_NAME


def draw_human_items(cfg: dict | None = None, gold_items: pd.DataFrame | None = None) -> pd.DataFrame:
    """100 items from the gold set: columns id, prev_text, text, plus the strata, random order."""
    cfg = cfg or load_config()
    hcfg = cfg["labelling"].get("human", {})
    if gold_items is None:
        gold_items = pd.read_csv(repo_root() / cfg["paths"]["labels_gold"] / "gold_items.csv",
                                 dtype=str, keep_default_na=False)
    seed = int(cfg["seed"]) + int(hcfg.get("seed_offset", 200))
    pool = pd.DataFrame({"uid": gold_items["id"], "prev_text_masked": gold_items["prev_text"],
                         "text_masked": gold_items["text"], "doc_types": gold_items["doc_types"],
                         "sections": gold_items["sections"] if "sections" in gold_items else "",
                         "first_release_date": (gold_items["first_release_date"]
                                                if "first_release_date" in gold_items else "2000-01-01")})
    drawn = stratified_sample(pool, n=int(cfg["labelling"]["human_n"]), seed=seed,
                              strata=("doc_type_group", "policy_like"),
                              floor=int(hcfg.get("floor", 5)))
    order = np.random.default_rng(seed).permutation(len(drawn))
    drawn = drawn.iloc[order].reset_index(drop=True)
    return pd.DataFrame({"id": drawn["uid"], "prev_text": drawn["prev_text_masked"],
                         "text": drawn["text_masked"], "doc_type_group": drawn["doc_type_group"],
                         "policy_like": drawn["policy_like"]})


def blank_sheet(items: pd.DataFrame) -> pd.DataFrame:
    out = items[["id", "prev_text", "text"]].copy()
    for c in LABEL_COLUMNS:
        out[c] = ""
    return out[SHEET_COLUMNS]


def _read_sheet_raw(path: Path) -> pd.DataFrame:
    """Read a sheet as strings; tolerates a BOM, cp1252 (older Excel) and ';' delimiters."""
    last: Exception | None = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc, sep=";")
            return df
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"cannot decode {path}: {last}")


def sheet_has_labels(path: Path) -> bool:
    """True if any fillable cell (labels, confidence, notes) of an existing sheet is non-blank.

    An unreadable or malformed existing file counts as 'has labels' (never overwritten).
    """
    try:
        df = _read_sheet_raw(Path(path))
    except Exception:
        return True
    cols = [c for c in LABEL_COLUMNS if c in df.columns]
    if not cols:
        return True
    return any((df[c].str.strip() != "").any() for c in cols)


def write_sheet(items: pd.DataFrame, path: Path) -> str:
    """Write the blank sheet. Returns 'written' or 'unchanged'.

    Raises SheetHasLabelsError if the existing file has any non-blank label/notes cell.
    Writing identical blank content again is a no-op.
    """
    path = Path(path)
    data = blank_sheet(items).to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    if path.exists():
        if sheet_has_labels(path):
            raise SheetHasLabelsError(
                f"{path} already contains labels or notes; refusing to overwrite it. "
                "Move it aside yourself if you really want a fresh blank sheet.")
        if path.read_bytes() == data:
            return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


# ----------------------------------------------------------------------------------
# reading a filled sheet


def _clean(x) -> str:
    return " ".join(str(x).replace(" ", " ").split())


def _norm_enum(raw: str, allowed: tuple, what: str, errs: list[str]):
    if raw.lower() in _NULLS:
        return None
    u = raw.upper()
    if u not in allowed:
        errs.append(f"{what}: {raw!r} is not one of {list(allowed)}")
        return None
    return u


def _norm_direction(raw: str, what: str, errs: list[str]):
    s = raw.replace("−", "-").replace("–", "-").replace("—", "-").replace(" ", "")
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        errs.append(f"{what}: {raw!r} is not an integer from -2 to 2")
        return None
    if v != int(v) or int(v) not in schema.DIRECTIONS:
        errs.append(f"{what}: {raw!r} is not an integer from -2 to 2")
        return None
    return int(v)


def _parse_row(row: pd.Series) -> tuple[dict | None, list[str], bool]:
    """(label or None, errors, blank). blank = every label cell empty (not yet labelled)."""
    if all(_clean(row.get(c, "")) == "" for c in LABEL_COLUMNS if c != "notes"):
        return None, [], True
    errs: list[str] = []
    aspects = []
    for k, cols in enumerate(_ASPECT_COLS, start=1):
        raw = [_clean(row.get(c, "")) for c in cols]
        if raw[0] == "" and raw[2] == "" and raw[1].lower() in _NULLS:
            if k == 1:
                errs.append("aspect 1 is empty")
            continue
        e: list[str] = []
        f = _norm_enum(raw[0], schema.FUNCTIONS, cols[0], e)
        if raw[0] == "":
            e.append(f"{cols[0]}: missing")
        t = _norm_enum(raw[1], schema.TOPICS, cols[1], e)
        d = _norm_direction(raw[2], cols[2], e)
        if raw[2] == "":
            e.append(f"{cols[2]}: missing")
        errs += e
        if not e:
            aspects.append({"function": f, "topic": t, "direction": d})
    conf_raw = _clean(row.get("confidence", "")).lower()
    conf = _CONF_ALIASES.get(conf_raw, conf_raw)
    if conf == "":
        errs.append("confidence: missing")
    label = {"id": str(row["id"]), "aspects": aspects, "confidence": conf}
    if not errs:
        errs += schema.validate_label(label)
    return (None if errs else label), errs, False


def parse_human_sheet(path: str | Path) -> tuple[dict[str, dict], dict[str, list[str]], int]:
    """(labels, errors, n_blank): valid labels by id, row-level errors by id, blank-row count."""
    df = _read_sheet_raw(Path(path))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in SHEET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    labels: dict[str, dict] = {}
    errors: dict[str, list[str]] = {}
    n_blank = 0
    seen: set[str] = set()
    for _, row in df.iterrows():
        id_ = _clean(row["id"])
        if not id_:
            continue
        row = row.copy()
        row["id"] = id_
        label, errs, blank = _parse_row(row)
        if id_ in seen:
            errors.setdefault(id_, []).append("duplicate id in sheet")
            continue
        seen.add(id_)
        if blank:
            n_blank += 1
        elif errs:
            errors[id_] = errs
        else:
            labels[id_] = label
    return labels, errors, n_blank


def read_human_sheet(path: str | Path, on_error: str = "warn") -> dict[str, dict]:
    """Filled sheet -> {id: label} in the codebook JSON label schema.

    Rows with every label cell blank are skipped (not yet labelled). Rows with problems are
    left out and reported: on_error "warn" (default) emits one warning listing them,
    "raise" raises ValueError, "ignore" stays silent. Use parse_human_sheet for the details.
    """
    if on_error not in ("warn", "raise", "ignore"):
        raise ValueError("on_error must be 'warn', 'raise' or 'ignore'")
    labels, errors, _ = parse_human_sheet(path)
    if errors and on_error != "ignore":
        msg = f"{path}: {len(errors)} row(s) with errors:\n" + "\n".join(
            f"  {i}: {'; '.join(e)}" for i, e in errors.items())
        if on_error == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return labels


# ----------------------------------------------------------------------------------


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    items = draw_human_items(cfg)
    path = sheet_path(cfg)
    try:
        status = write_sheet(items, path)
    except SheetHasLabelsError as e:
        print(f"human: {e}")
        return
    print(f"human: {status} {path} ({len(items)} items)")
    print(items.groupby(["doc_type_group", "policy_like"]).size().to_string())


if __name__ == "__main__":
    main()
