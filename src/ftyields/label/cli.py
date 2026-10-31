"""CLI: python -m ftyields.label <cmd>   (thin wrapper over the library functions)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ftyields import config
from ftyields.label import agreement, batches, sample, schema


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=True, dtype=False)
    raise SystemExit(f"unsupported file type: {p}")


def _cfg_default(key: str, fallback):
    return config.load_config().get("labelling", {}).get(key, fallback)


def cmd_sample(a) -> int:
    df = pd.read_parquet(a.sentences or config.path("data_processed") / "sentences_unique.parquet")
    exclude: set[str] = set()
    for s in a.exclude_set or []:
        for bp in batches.batch_paths(s):
            exclude |= {it["id"] for it in schema.read_batch(bp)}
    seed = a.seed if a.seed is not None else int(config.load_config()["seed"])
    if a.kind == "pilot":
        out = sample.pilot_sample(df, n=a.n or _cfg_default("pilot_n", 100), seed=seed, exclude=exclude)
    else:
        out = sample.gold_sample(df, n=a.n or _cfg_default("gold_n", 350), seed=seed, exclude=exclude,
                                 policy_oversample=a.policy_oversample)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} sampled sentences to {a.out}")
    return 0


def cmd_make_batches(a) -> int:
    items = batches.to_items(_read_table(a.items))
    size = a.batch_size or _cfg_default("batch_size", 300)
    paths = batches.make_batches(items, a.set, size)
    print(f"set {a.set}: {len(items)} items, {len(paths)} batches in {paths[0].parent}")
    return 0


def cmd_pending(a) -> int:
    for p in batches.pending(a.set, a.labeler):
        print(p)
    return 0


def cmd_validate(a) -> int:
    if a.batch and a.output:
        reports = [schema.validate_output(a.batch, a.output)]
    else:
        reports = []
        for bp in batches.batch_paths(a.set):
            op = batches.output_path(a.set, a.labeler, bp)
            if op.is_file():
                reports.append(schema.validate_output(bp, op))
            else:
                print(f"{op.name}: no output")
    for r in reports:
        print(f"{r.output_path}\n{r.summary()}")
    return 0 if reports and all(r.ok for r in reports) else 1


def cmd_agree(a) -> int:
    labels = agreement.load_set_labels(a.set, a.labelers, on_missing="skip" if a.allow_partial else "error")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = agreement.agreement_tables(labels)
    t_fa = agreement.agreement_tables(labels, restrict_function_agree=True)
    for name, t in (("", t_all), ("_function_agree", t_fa)):
        t["pairwise"].to_csv(out / f"pairwise{name}.csv", index=False)
        t["overall"].to_csv(out / f"overall{name}.csv", index=False)
    dis = agreement.disagreements(labels, agreement.load_set_texts(a.set))
    dis.to_csv(out / "disagreements.csv", index=False, encoding="utf-8")
    (out / "summary.md").write_text(agreement.render_markdown(a.set, t_all, t_fa, len(dis)), encoding="utf-8")
    print(f"wrote tables, disagreements.csv ({len(dis)} items) and summary.md to {out}")
    return 0


def cmd_majority(a) -> int:
    labels = agreement.load_set_labels(a.set, a.labelers, on_missing="skip" if a.allow_partial else "error")
    df = agreement.majority_vote(labels)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in agreement.majority_records(df):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(df)} majority labels to {a.out} "
          f"({int(df['function'].isna().sum())} without function majority)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ftyields.label")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw a pilot/gold sample from sentences_unique.parquet")
    s.add_argument("kind", choices=["pilot", "gold"])
    s.add_argument("--out", required=True, help="output parquet")
    s.add_argument("--sentences", help="default data/processed/sentences_unique.parquet")
    s.add_argument("--n", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--policy-oversample", type=float, default=2.0)
    s.add_argument("--exclude-set", action="append", help="batch set whose ids to exclude (repeatable)")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("make-batches", help="write batch files for a set")
    s.add_argument("--items", required=True, help="parquet/csv/jsonl with id,prev_text,text "
                                                  "(or uid,prev_text_masked,text_masked)")
    s.add_argument("--set", required=True)
    s.add_argument("--batch-size", type=int, help="default labelling.batch_size")
    s.set_defaults(func=cmd_make_batches)

    s = sub.add_parser("pending", help="list batches without a valid output (moves invalid ones aside)")
    s.add_argument("--set", required=True)
    s.add_argument("--labeler", required=True)
    s.set_defaults(func=cmd_pending)

    s = sub.add_parser("validate", help="validate outputs (exit 1 on any failure)")
    s.add_argument("--set")
    s.add_argument("--labeler")
    s.add_argument("--batch")
    s.add_argument("--output")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("agree", help="agreement tables + markdown summary")
    s.add_argument("--set", required=True)
    s.add_argument("--labelers", nargs="+", required=True)
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--allow-partial", action="store_true")
    s.set_defaults(func=cmd_agree)

    s = sub.add_parser("majority", help="majority-vote JSONL (>= 3 labelers)")
    s.add_argument("--set", required=True)
    s.add_argument("--labelers", nargs="+", required=True)
    s.add_argument("--out", required=True, help="output JSONL path")
    s.add_argument("--allow-partial", action="store_true")
    s.set_defaults(func=cmd_majority)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "validate" and not ((a.batch and a.output) or (a.set and a.labeler)):
        raise SystemExit("validate needs --set and --labeler, or --batch and --output")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
