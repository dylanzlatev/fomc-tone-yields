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


ROUND_COLUMNS = {
    "function_fleiss_kappa": "function_kappa", "function_all_agree_pct": "function_all_agree_pct",
    "topic_fleiss_kappa": "topic_kappa", "direction_all_exact_pct": "direction_all_exact_pct",
    "direction_alpha_interval": "direction_alpha", "full_match_all_pct": "full_match_pct",
}


def summarize_rounds(cfg: dict | None = None) -> pd.DataFrame:
    """Agreement by pilot round -> results/tables/pilot_agreement.{csv,tex}."""
    cfg = cfg or load_config()
    root = repo_root()
    pilot_dir = root / cfg["paths"]["labels_pilot"]
    codebook_by_round = cfg["labelling"]["pilot"]["codebook_versions"]
    rows = []
    for k, version in enumerate(codebook_by_round, start=1):
        overall = pd.read_csv(pilot_dir / f"r{k}_agreement" / "overall.csv").iloc[0]
        row = {"round": k, "codebook": version, "n_items": int(overall["n_items"])}
        row.update({new: float(overall[old]) for old, new in ROUND_COLUMNS.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    out = root / cfg["paths"]["results_tables"]
    table.to_csv(out / "pilot_agreement.csv", index=False, lineterminator="\n")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Round & Codebook & $n$ & Function $\kappa$ & Topic $\kappa$ & "
             r"Direction exact (\%) & Direction $\alpha$ & Full match (\%) \\", r"\midrule"]
    for r in table.itertuples():
        lines.append(f"{r.round} & {r.codebook} & {r.n_items} & {r.function_kappa:.2f} & "
                     f"{r.topic_kappa:.2f} & {r.direction_all_exact_pct:.0f} & "
                     f"{r.direction_alpha:.2f} & {r.full_match_pct:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "pilot_agreement.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, help="draw this pilot round")
    ap.add_argument("--summarize", action="store_true", help="write the cross-round table")
    args = ap.parse_args(argv)
    if args.round:
        items = draw_round(args.round)
        print(f"pilot_r{args.round}: {len(items)} items")
    if args.summarize:
        print(summarize_rounds().to_string(index=False))


if __name__ == "__main__":
    main()
