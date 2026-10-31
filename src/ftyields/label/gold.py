"""Gold validation set: 350 stratified unique sentences, POLICY-like over-sampled.

Excludes every pilot-round uid and the flagged procedural-only uids. Labelled by three
blind labelers; the majority vote is the gold label set.

    python -m ftyields.label.gold
"""
from __future__ import annotations

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.label.batches import make_batches
from ftyields.label.pilot import pilot_weights, procedural_only_uids
from ftyields.label.sample import stratified_sample


def draw_gold(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = repo_root()
    gcfg = cfg["labelling"]["gold"]
    sent = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences.parquet")
    uniq = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences_unique.parquet")
    exclude = procedural_only_uids(sent)
    pilot_dir = root / cfg["paths"]["labels_pilot"]
    for k in range(1, len(cfg["labelling"]["pilot"]["codebook_versions"]) + 1):
        exclude |= set(pd.read_csv(pilot_dir / f"pilot_r{k}_items.csv")["id"])
    sample = stratified_sample(uniq, n=cfg["labelling"]["gold_n"], seed=cfg["seed"] + 100,
                               weights=lambda d: pilot_weights(d, gcfg), exclude=exclude)
    items = pd.DataFrame({"id": sample["uid"], "prev_text": sample["prev_text_masked"],
                          "text": sample["text_masked"]})
    gold_dir = root / cfg["paths"]["labels_gold"]
    gold_dir.mkdir(parents=True, exist_ok=True)
    meta = sample[["uid", "doc_types", "sections", "first_release_date"]].rename(columns={"uid": "id"})
    items.merge(meta, on="id").to_csv(gold_dir / "gold_items.csv", index=False, lineterminator="\n")
    make_batches(items, "gold", batch_size=gcfg["batch_size"])
    return items


if __name__ == "__main__":
    print(f"gold: {len(draw_gold())} items")
