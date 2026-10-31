"""Training set for the distilled classifier: ~10,000 unique sentences labelled by an LLM.

Pool = every unique sentence except pilot-round uids, gold uids and procedural-only uids
(those are PROCEDURAL by construction and never labelled). From the pool:

  1. ALL sentences that occur in statements (doc_types in {statement, both}) are taken;
  2. the rest up to ``labelling.train_n`` is a stratified draw (doc type x section x era,
     floor per stratum, seed = seed + labelling.train.seed_offset) from the minutes-only uids.

Writes labels/train/train_items.csv (id, prev_text, text, doc_types, sections,
first_release_date) and the batch set ``train`` (labelling.batch_size items per batch).

    python -m ftyields.label.train_set
"""
from __future__ import annotations

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.label.batches import make_batches
from ftyields.label.pilot import procedural_only_uids
from ftyields.label.sample import add_strata_columns, stratified_sample

STATEMENT_TYPES = ("statement", "both")


def excluded_uids(cfg: dict, sentences: pd.DataFrame) -> set[str]:
    """Pilot, gold and procedural-only uids."""
    root = repo_root()
    exclude = procedural_only_uids(sentences)
    pilot_dir = root / cfg["paths"]["labels_pilot"]
    for k in range(1, len(cfg["labelling"]["pilot"]["codebook_versions"]) + 1):
        exclude |= set(pd.read_csv(pilot_dir / f"pilot_r{k}_items.csv")["id"])
    exclude |= set(pd.read_csv(root / cfg["paths"]["labels_gold"] / "gold_items.csv")["id"])
    return exclude


def draw_train(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = repo_root()
    tcfg = cfg["labelling"]["train"]
    n = int(cfg["labelling"]["train_n"])
    proc = root / cfg["paths"]["data_processed"]
    sent = pd.read_parquet(proc / "sentences.parquet", columns=["uid", "is_procedural"])
    uniq = pd.read_parquet(proc / "sentences_unique.parquet")
    exclude = excluded_uids(cfg, sent)

    pool = uniq[~uniq["uid"].isin(exclude)]
    stmt = pool[pool["doc_types"].isin(STATEMENT_TYPES)]
    minutes_only = pool[~pool["doc_types"].isin(STATEMENT_TYPES)]
    if len(stmt) > n:
        raise ValueError(f"{len(stmt)} statement sentences exceed train_n={n}")
    k = n - len(stmt)
    drawn = stratified_sample(minutes_only, n=k, seed=int(cfg["seed"]) + int(tcfg["seed_offset"]),
                              floor=int(tcfg["floor"]))
    sample = pd.concat([stmt, drawn.reindex(columns=stmt.columns)]).sort_values("uid", kind="mergesort")
    assert sample["uid"].is_unique and not (set(sample["uid"]) & exclude)

    items = pd.DataFrame({"id": sample["uid"], "prev_text": sample["prev_text_masked"],
                          "text": sample["text_masked"]})
    meta = sample[["uid", "doc_types", "sections", "first_release_date"]].rename(columns={"uid": "id"})
    out = items.merge(meta, on="id")
    train_dir = root / cfg["paths"]["labels_train"]
    train_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(train_dir / "train_items.csv", index=False, lineterminator="\n")
    paths = make_batches(items, "train", batch_size=int(cfg["labelling"]["batch_size"]))
    print(f"[train_set] pool {len(pool)} (excluded {len(exclude)} uids incl. pilot/gold/procedural); "
          f"statement sentences taken: {len(stmt)}; minutes-only drawn: {len(drawn)}")
    print_composition(sample)
    print(f"[train_set] {len(out)} items in {len(paths)} batches of {cfg['labelling']['batch_size']}")
    return out


def print_composition(sample: pd.DataFrame) -> None:
    s = add_strata_columns(sample)
    print("\nBy doc type:\n" + s["doc_types"].value_counts().to_string())
    print("\nBy section group:\n" + s["section_group"].value_counts().to_string())
    print("\nBy era:\n" + s["era"].value_counts().sort_index().to_string())
    print("\nDoc type x era:\n" + pd.crosstab(s["doc_types"], s["era"]).to_string())


def main(cfg: dict | None = None) -> None:
    draw_train(cfg)


if __name__ == "__main__":
    main()
