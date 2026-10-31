"""FOMC statements and minutes corpus (HF dataset vtasca/fomc-statements-minutes).

Writes data/raw/text/fomc_documents.parquet (committed; FOMC texts are U.S. government
works). The dataset's own dates are kept unchanged (`meeting_date_ds`, `release_date_ds`);
they are NOT used as event dates (see calendar.py). All years are kept; filtering to the
sample window happens later.

Columns: doc_id (S|M + meeting date YYYYMMDD), doc_type (statement|minutes),
meeting_date_ds, release_date_ds, text, n_chars.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.data import download as dl

HF_PAGE = "https://huggingface.co/datasets/{name}"
HF_CARD_RAW = "https://huggingface.co/datasets/{name}/raw/main/README.md"
CARD_LICENSE = ("HF dataset card YAML 'license: cc' (CC variant unspecified); underlying FOMC texts "
                "are U.S. government works (public domain)")
TYPE_MAP = {"Statement": ("S", "statement"), "Minute": ("M", "minutes")}


def load_hf(name: str) -> pd.DataFrame:
    """Load the HF dataset's `train` split as a DataFrame.

    Tries the local HF cache first (HF_HUB_OFFLINE=1) and falls back to the network.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    try:
        ds = load_dataset(name)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:                                    # constants are read at import time: reset them too
            import datasets.config as dcfg
            import huggingface_hub.constants as hcfg
            dcfg.HF_HUB_OFFLINE = False
            hcfg.HF_HUB_OFFLINE = False
        except Exception:
            pass
        ds = load_dataset(name)
    return ds["train"].to_pandas()


def build_documents(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw HF frame (Date, Release Date, Type, Text) -> documents frame with unique doc_id."""
    bad = set(raw["Type"].unique()) - set(TYPE_MAP)
    if bad:
        raise ValueError(f"unexpected Type values: {sorted(bad)}")
    prefix = raw["Type"].map(lambda t: TYPE_MAP[t][0])
    out = pd.DataFrame({
        "doc_id": prefix + raw["Date"].str.replace("-", "", regex=False),
        "doc_type": raw["Type"].map(lambda t: TYPE_MAP[t][1]),
        "meeting_date_ds": raw["Date"],
        "release_date_ds": raw["Release Date"].where(raw["Release Date"].notna(), None),
        "text": raw["Text"],
    })
    out["n_chars"] = out["text"].str.len().astype("int64")
    dup = out["doc_id"][out["doc_id"].duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"doc_id is not unique: {sorted(dup.unique())}")
    return out.sort_values(["meeting_date_ds", "doc_type"]).reset_index(drop=True)


def record_manifest(cfg: dict, name: str, url: str, local: Path, license: str, committed: bool) -> None:
    """Upsert a manifest row for a file that did not come through download.fetch()."""
    mpath = dl._manifest_path(cfg)
    rows = dl._read_manifest(mpath)
    digest = dl.sha256(local)
    old = rows.get(name, {})
    keep_time = old.get("sha256") == digest and old.get("retrieved_utc")
    rows[name] = {
        "name": name, "url": url, "local_path": local.relative_to(repo_root()).as_posix(),
        "retrieved_utc": keep_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": digest, "bytes": local.stat().st_size, "license": license,
        "committed": "yes" if committed else "no",
    }
    dl._write_manifest(mpath, rows)


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    name = cfg["text"]["hf_dataset"]
    root = repo_root()

    card = dl.fetch(f"hf_card_{name.split('/')[-1].replace('-', '_')}", HF_CARD_RAW.format(name=name),
                    f"{cfg['downloads']['dir']}/hf_card_{name.split('/')[-1].replace('-', '_')}.md",
                    license=CARD_LICENSE, committed=False, cfg=cfg)
    record_manifest(cfg, f"hf_card_{name.split('/')[-1].replace('-', '_')}", HF_CARD_RAW.format(name=name),
                    Path(card), CARD_LICENSE, False)

    docs = build_documents(load_hf(name))
    out = root / cfg["paths"]["data_raw"] / "text" / "fomc_documents.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out, index=False)
    record_manifest(cfg, "hf_fomc_statements_minutes", HF_PAGE.format(name=name), out, CARD_LICENSE, True)

    print(f"wrote {out.relative_to(root).as_posix()}: {len(docs)} documents "
          f"({docs['meeting_date_ds'].min()}..{docs['meeting_date_ds'].max()})")
    print("doc_id unique:", docs["doc_id"].is_unique,
          "| undated release_date_ds:", int(docs["release_date_ds"].isna().sum()))
    year = docs["meeting_date_ds"].str[:4].rename("year")
    print(pd.crosstab(year, docs["doc_type"], margins=True).to_string())
    return docs


if __name__ == "__main__":
    main()
