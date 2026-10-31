"""Codebook pilot rounds: draw a fresh stratified sample per round and write its batch.

Each round excludes every uid used in earlier rounds and every uid that only occurs in
the flagged annual organisational/authorisation blocks (`is_procedural`), which are
PROCEDURAL by construction and never labelled. Statement sentences and policy-like
sentences are up-weighted so the pilot exercises the POLICY rules.

    python -m ftyields.label.pilot --round 1
"""
from __future__ import annotations

import argparse

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.label.batches import make_batches
from ftyields.label.sample import policy_like, stratified_sample


def procedural_only_uids(sentences: pd.DataFrame) -> set[str]:
    """uids whose every occurrence is in a flagged procedural block."""
    flag = sentences.groupby("uid")["is_procedural"].all()
    return set(flag[flag].index)


def pilot_weights(df: pd.DataFrame, pcfg: dict) -> pd.Series:
    w = pd.Series(1.0, index=df.index)
    w[df["doc_types"].isin(["statement", "both"])] *= pcfg["statement_weight"]
    w[df["text_masked"].map(policy_like)] *= pcfg["policy_weight"]
    return w


def draw_round(k: int, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = repo_root()
    pcfg = cfg["labelling"]["pilot"]
    sent = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences.parquet")
    uniq = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences_unique.parquet")
    exclude = procedural_only_uids(sent)
    pilot_dir = root / cfg["paths"]["labels_pilot"]
    for j in range(1, k):
        prev = pd.read_csv(pilot_dir / f"pilot_r{j}_items.csv")
        exclude |= set(prev["id"])
    sample = stratified_sample(uniq, n=pcfg["n_per_round"], seed=cfg["seed"] + k,
                               weights=lambda d: pilot_weights(d, pcfg), exclude=exclude)
    items = pd.DataFrame({"id": sample["uid"], "prev_text": sample["prev_text_masked"],
                          "text": sample["text_masked"]})
    pilot_dir.mkdir(parents=True, exist_ok=True)
    meta = sample[["uid", "doc_types", "sections", "first_release_date"]].rename(columns={"uid": "id"})
    items.merge(meta, on="id").to_csv(pilot_dir / f"pilot_r{k}_items.csv", index=False,
                                      lineterminator="\n")
    make_batches(items, f"pilot_r{k}", batch_size=pcfg["n_per_round"])
    return items


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args(argv)
    items = draw_round(args.round)
    print(f"pilot_r{args.round}: {len(items)} items")


if __name__ == "__main__":
    main()
