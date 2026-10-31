"""Trillion Dollar Words (TDW) hawkish/dovish/neutral sentence set, as an external benchmark.

Source: Hugging Face dataset `cfg["sources"]["tdw_hf_dataset"]` (gtfintechlab/fomc_communication),
files train.csv and test.csv with columns index, sentence, year, label, orig_index;
label 0 = dovish, 1 = hawkish, 2 = neutral. License CC BY-NC 4.0: the download, the interim
parquet files and the labelling batches are all gitignored (only the manifest row and the
batch-folder README are committed).

    python -m ftyields.label.tdw

Outputs:
  data/interim/downloads/tdw/{train,test}.csv        raw files (gitignored; manifest rows in data/raw/MANIFEST.csv)
  data/interim/tdw_{train,test}.parquet               tdw_id, sentence, year, label_raw, direction
  labels/batches/tdw_test/                            batches of the de-duplicated test sentences
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.data import download
from ftyields.label.batches import make_batches

LICENSE = "CC BY-NC 4.0"
FILES = ("train.csv", "test.csv")
LABEL_NAMES = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_TO_DIRECTION = {0: -1, 1: 1, 2: 0}


def label_to_direction(label_raw) -> int:
    """TDW label (0 dovish, 1 hawkish, 2 neutral) -> direction (-1, +1, 0)."""
    try:
        return LABEL_TO_DIRECTION[int(label_raw)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"unknown TDW label {label_raw!r} (expected 0, 1 or 2)") from None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def prepare(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Raw TDW frame -> tdw_id, sentence, year, label_raw, direction (+ index, orig_index).

    The file's `index` column is NOT unique (it restarts within each source year file), so the
    id uses the 0-based row position in the file: tdw_<split>_<row:04d>. `index` and
    `orig_index` are kept for traceability.
    """
    raw = raw.reset_index(drop=True)
    out = pd.DataFrame({
        "tdw_id": [f"tdw_{split}_{i:04d}" for i in range(len(raw))],
        "sentence": raw["sentence"].astype(str).to_numpy(),
        "year": raw["year"].astype(int).to_numpy(),
        "label_raw": raw["label"].astype(int).to_numpy(),
        "index": raw["index"].astype(int).to_numpy(),
        "orig_index": raw["orig_index"].astype(int).to_numpy()})
    out["direction"] = out["label_raw"].map(label_to_direction).astype(int)
    return out[["tdw_id", "sentence", "year", "label_raw", "direction", "index", "orig_index"]]


def drop_train_duplicates(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(test without sentences also in train, the dropped test rows).

    Match is on the sentence text after whitespace normalisation.
    """
    in_train = set(train["sentence"].map(_norm))
    dup = test["sentence"].map(_norm).isin(in_train)
    return test[~dup].reset_index(drop=True), test[dup].reset_index(drop=True)


def download_files(cfg: dict) -> dict[str, Path]:
    """hf_hub_download train/test into data/interim/downloads/tdw/ and record manifest rows."""
    from huggingface_hub import hf_hub_download

    repo = cfg["sources"]["tdw_hf_dataset"]
    root = repo_root()
    dest_dir = root / cfg["downloads"]["dir"] / "tdw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    mpath = download._manifest_path(cfg)
    rows = download._read_manifest(mpath)
    paths = {}
    for fname in FILES:
        local = Path(hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                                     local_dir=str(dest_dir)))
        paths[fname] = local
        name = f"tdw_hf_{Path(fname).stem}"
        if name not in rows or rows[name].get("sha256") != download.sha256(local):
            rows[name] = {
                "name": name,
                "url": f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}",
                "local_path": local.resolve().relative_to(root).as_posix(),
                "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": download.sha256(local),
                "bytes": local.stat().st_size,
                "license": LICENSE,
                "committed": "no",
            }
    download._write_manifest(mpath, rows)
    return paths


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    tcfg = cfg["labelling"]["tdw"]
    root = repo_root()
    paths = download_files(cfg)
    train = prepare(pd.read_csv(paths["train.csv"]), "train")
    test_all = prepare(pd.read_csv(paths["test.csv"]), "test")
    test, dups = drop_train_duplicates(train, test_all)

    interim = root / cfg["paths"]["data_interim"]
    interim.mkdir(parents=True, exist_ok=True)
    train.to_parquet(interim / "tdw_train.parquet", index=False)
    test.to_parquet(interim / "tdw_test.parquet", index=False)

    items = pd.DataFrame({"id": test["tdw_id"], "prev_text": "", "text": test["sentence"]})
    batch_paths = make_batches(items, tcfg["set"], int(tcfg["batch_size"]))

    def dist(df):
        return {LABEL_NAMES[k]: int(v) for k, v in df["label_raw"].value_counts().sort_index().items()}

    summary = {"train": len(train), "test_raw": len(test_all), "test_duplicates_dropped": len(dups),
               "test": len(test), "test_within_duplicates": int(test["sentence"].map(_norm).duplicated().sum()),
               "train_labels": dist(train), "test_labels": dist(test),
               "n_batches": len(batch_paths)}
    for k, v in summary.items():
        print(f"tdw: {k}: {v}")
    return summary


if __name__ == "__main__":
    main()
