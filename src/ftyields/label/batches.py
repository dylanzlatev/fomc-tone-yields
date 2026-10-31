"""Batch files for labelling subagents, and tracking which batches still need output.

Layout (under labels/batches and labels/out, from config.yaml `paths`):
  labels/batches/<set>/batch_NNNN.jsonl   input, one {"id","prev_text","text"} per line
  labels/batches/<set>/manifest.csv       batch, n_items, first_id, last_id, sha256
  labels/out/<set>/<labeler>/batch_NNNN.jsonl            valid outputs
  labels/out/<set>/<labeler>/rejected/batch_NNNN.<k>.jsonl   invalid outputs, moved aside
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ftyields import config
from ftyields.label.schema import validate_output

MANIFEST = "manifest.csv"
MANIFEST_COLUMNS = ["batch", "n_items", "first_id", "last_id", "sha256"]
_SET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def batches_root(batches_dir: str | Path | None = None) -> Path:
    return Path(batches_dir) if batches_dir is not None else config.path("labels_batches")


def out_root(out_dir: str | Path | None = None) -> Path:
    return Path(out_dir) if out_dir is not None else config.path("labels_out")


def _check_name(name: str, what: str) -> None:
    if not _SET_RE.match(name):
        raise ValueError(f"invalid {what} name {name!r}")


def batch_name(i: int) -> str:
    return f"batch_{i:04d}.jsonl"


def to_items(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to columns id, prev_text, text.

    Accepts either (id, prev_text, text) or the sentences_unique names
    (uid, prev_text_masked, text_masked).
    """
    rename = {"uid": "id", "prev_text_masked": "prev_text", "text_masked": "text"}
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})
    missing = {"id", "text"} - set(out.columns)
    if missing:
        raise ValueError(f"items are missing columns {sorted(missing)}")
    if "prev_text" not in out.columns:
        out = out.assign(prev_text="")
    out = out[["id", "prev_text", "text"]].copy()
    out["id"] = out["id"].astype(str)
    for c in ("prev_text", "text"):
        out[c] = out[c].where(out[c].notna(), "").astype(str)
    return out.reset_index(drop=True)


def _render_batches(items: pd.DataFrame, batch_size: int) -> list[tuple[str, bytes, list[str]]]:
    """[(file name, bytes, ids)] with items sorted by id (deterministic)."""
    items = to_items(items)
    if items["id"].duplicated().any():
        dup = items.loc[items["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"duplicate id in items, e.g. {dup!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    items = items.sort_values("id", kind="mergesort").reset_index(drop=True)
    rendered = []
    for b, start in enumerate(range(0, len(items), batch_size), 1):
        chunk = items.iloc[start:start + batch_size]
        lines = [json.dumps({"id": r.id, "prev_text": r.prev_text, "text": r.text},
                            ensure_ascii=False) for r in chunk.itertuples()]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        rendered.append((batch_name(b), data, list(chunk["id"])))
    return rendered


def _manifest_rows(rendered) -> list[dict]:
    return [{"batch": name, "n_items": len(ids), "first_id": ids[0], "last_id": ids[-1],
             "sha256": hashlib.sha256(data).hexdigest()} for name, data, ids in rendered]


def read_manifest(set_name: str, batches_dir: str | Path | None = None) -> pd.DataFrame:
    path = batches_root(batches_dir) / set_name / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"no manifest for set {set_name!r} at {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_batches(items: pd.DataFrame, set_name: str, batch_size: int,
                 batches_dir: str | Path | None = None) -> list[Path]:
    """Write batch files and manifest for a set; returns the batch paths in order.

    Items are sorted by id, so the result is independent of input order. If the set
    already exists with identical content this is a no-op returning the same paths; if
    the content differs a FileExistsError is raised and nothing is touched.
    """
    _check_name(set_name, "set")
    rendered = _render_batches(items, batch_size)
    if not rendered:
        raise ValueError("no items to batch")
    rows = _manifest_rows(rendered)
    set_dir = batches_root(batches_dir) / set_name
    paths = [set_dir / name for name, _, _ in rendered]

    # a committed README.md (e.g. a licence note) is not batch content
    if set_dir.exists() and any(p.name != "README.md" for p in set_dir.iterdir()):
        mpath = set_dir / MANIFEST
        if not mpath.is_file():
            raise FileExistsError(f"{set_dir} exists and has no manifest; refusing to overwrite")
        old = pd.read_csv(mpath, dtype=str, keep_default_na=False)
        new = pd.DataFrame(rows).astype(str)
        same = (list(old.columns) == MANIFEST_COLUMNS and old.equals(new[MANIFEST_COLUMNS])
                and all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
                        for p, r in zip(paths, rows)))
        if not same:
            raise FileExistsError(
                f"set {set_dir.name!r} already exists with different content; "
                "use a new set name (existing batches are never overwritten)")
        return paths

    set_dir.mkdir(parents=True, exist_ok=True)
    for (name, data, _), p in zip(rendered, paths):
        p.write_bytes(data)
    with open(set_dir / MANIFEST, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return paths


def batch_paths(set_name: str, batches_dir: str | Path | None = None) -> list[Path]:
    """Batch files of an existing set, in manifest order."""
    man = read_manifest(set_name, batches_dir)
    return [batches_root(batches_dir) / set_name / b for b in man["batch"]]


def output_path(set_name: str, labeler: str, batch: str | Path,
                out_dir: str | Path | None = None) -> Path:
    return out_root(out_dir) / set_name / labeler / Path(batch).name


def _reject(out_file: Path, errors_text: str) -> Path:
    rej_dir = out_file.parent / "rejected"
    rej_dir.mkdir(exist_ok=True)
    k = 1
    stem = out_file.stem
    while (rej_dir / f"{stem}.{k}.jsonl").exists():
        k += 1
    target = rej_dir / f"{stem}.{k}.jsonl"
    out_file.replace(target)
    (rej_dir / f"{stem}.{k}.errors.txt").write_text(errors_text + "\n", encoding="utf-8")
    return target


def pending(set_name: str, labeler: str, batches_dir: str | Path | None = None,
            out_dir: str | Path | None = None) -> list[Path]:
    """Batch files that have no valid output for this labeler.

    An existing output that fails validation is moved (never deleted) to
    <labeler>/rejected/batch_NNNN.<k>.jsonl, with the validation errors in a sibling
    .errors.txt, and its batch is reported as pending.
    """
    _check_name(set_name, "set")
    _check_name(labeler, "labeler")
    todo = []
    for bp in batch_paths(set_name, batches_dir):
        op = output_path(set_name, labeler, bp, out_dir)
        if op.is_file():
            rep = validate_output(bp, op)
            if rep.ok:
                continue
            _reject(op, rep.summary())
        todo.append(bp)
    return todo
