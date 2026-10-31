"""Validation tables: gold-set agreement, labeler-vs-reference, human-vs-models.

Re-runnable at any time (`python -m ftyields.label.validation`); labelers whose gold
outputs are missing or invalid are skipped with a note (rows with status "pending"), so the
tables fill themselves in as outputs appear.

Everything reuses agreement.py: the primary aspect (largest |direction|, ties -> first
listed), pair_metrics (function / topic / direction / full-match definitions) and
majority_vote. Percentages are 0-100; CSVs hold unrounded values, TeX tables round
(percent 1 decimal, kappa/alpha 2 decimals).

Outputs (results/tables/):
  gold_agreement.{csv,tex}        inter-labeler agreement of the gold labelers + majority-vote counts
  labeler_vs_reference.{csv,tex}  each candidate labeler vs (i) the gold majority and (ii) a
                                  leave-out consensus: the items on which the gold labelers other
                                  than the candidate all agree on the primary label
  human_vs_models.{csv,tex}       the filled human sheet vs the gold majority and vs each labeler
and labels/gold/gold_majority.jsonl (majority-vote labels with an `agreement` field).

For a candidate that is itself a gold labeler, the "gold_majority" reference contains its own
vote (flagged in_gold); the leave-out consensus is the unbiased comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from ftyields.config import load_config, repo_root
from ftyields.label import agreement, human
from ftyields.label.batches import batch_paths, output_path
from ftyields.label.schema import validate_output

METRICS = ["n_items", "function_pct", "function_kappa", "topic_n_items", "topic_pct", "topic_kappa",
           "direction_exact_pct", "direction_within1_pct", "direction_sign_pct", "direction_qwk",
           "full_match_pct"]
NAN = float("nan")


# ----------------------------------------------------------------------------------
# loading (read-only: never moves or rewrites outputs)


def discover_labelers(set_name: str, out_dir: Path) -> list[str]:
    d = Path(out_dir) / set_name
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def load_complete(set_name: str, labelers: list[str], out_dir=None, batches_dir=None
                  ) -> tuple[dict[str, dict[str, dict]], dict[str, str]]:
    """({labeler: {id: label}} for labelers with every batch valid, {labeler: reason skipped})."""
    bps = batch_paths(set_name, batches_dir)
    labels: dict[str, dict[str, dict]] = {}
    skipped: dict[str, str] = {}
    for lab in labelers:
        got: dict[str, dict] = {}
        n_ok = 0
        for bp in bps:
            op = output_path(set_name, lab, bp, out_dir)
            if not op.is_file():
                continue
            rep = validate_output(bp, op)
            if rep.ok:
                got.update(rep.labels)
                n_ok += 1
        if n_ok == len(bps):
            labels[lab] = got
        else:
            skipped[lab] = f"incomplete: {n_ok}/{len(bps)} batches valid"
    return labels, skipped


# ----------------------------------------------------------------------------------
# comparison helpers


def _frame_from_majority(maj: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote result -> primary-label frame (function, topic, direction) by id, without
    the no-majority items."""
    m = maj[maj["function"].notna()].copy()
    out = pd.DataFrame({"function": m["function"].to_numpy(),
                        "topic": m["topic"].to_numpy(),
                        "direction": m["direction"].astype(int).to_numpy()},
                       index=pd.Index(m["id"].to_numpy(), name="id"))
    return out


def _frame(labels: dict[str, dict], ids) -> pd.DataFrame:
    return agreement.primary_frames({"x": labels}, ids)["x"]


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """agreement.pair_metrics of two primary-label frames on their common ids (selected columns)."""
    ids = sorted(set(a.index) & set(b.index))
    if not ids:
        return {m: (0 if m.endswith("n_items") else NAN) for m in METRICS}
    pm = agreement.pair_metrics(a.loc[ids], b.loc[ids])
    return {m: pm[m] for m in METRICS}


def leave_out_reference(labels: dict[str, dict[str, dict]], gold_labelers: list[str],
                        candidate: str) -> pd.DataFrame:
    """Items on which every gold labeler other than `candidate` gives the same primary
    (function, topic, direction); frame indexed by id with that label. Needs >= 2 others."""
    others = [g for g in gold_labelers if g != candidate]
    if len(others) < 2:
        raise ValueError("leave-out consensus needs at least 2 other gold labelers")
    sub = {g: labels[g] for g in others}
    rows, ids = [], []
    for i in agreement.common_ids(sub):
        prims = {agreement.primary(sub[g][i]) for g in others}
        if len(prims) == 1:
            ids.append(i)
            rows.append(next(iter(prims)))
    return pd.DataFrame(rows, index=pd.Index(ids, name="id"), columns=["function", "topic", "direction"])


# ----------------------------------------------------------------------------------
# table 1: gold agreement


def gold_agreement_table(labels: dict[str, dict[str, dict]] | None, gold_labelers: list[str],
                         skipped: dict[str, str] | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(table, majority-vote frame or None). With incomplete labelers: a single pending row."""
    missing = [g for g in gold_labelers if not labels or g not in labels]
    if missing:
        note = "; ".join(f"{g}: {(skipped or {}).get(g, 'no outputs')}" for g in missing)
        return pd.DataFrame([{"status": "pending", "note": note, "scope": "overall",
                              "labelers": "|".join(gold_labelers)}]), None
    sub = {g: labels[g] for g in gold_labelers}
    tabs = agreement.agreement_tables(sub)
    rows = [{"status": "ok", "note": "", "scope": "overall", "labelers": "|".join(sorted(sub)),
             **tabs["overall"].iloc[0].to_dict()}]
    for r in tabs["pairwise"].to_dict("records"):
        rows.append({"status": "ok", "note": "", "scope": "pair",
                     "labelers": f"{r.pop('labeler_a')} vs {r.pop('labeler_b')}", **r})
    maj = agreement.majority_vote(sub, min_labelers=min(3, len(sub)))
    flags = maj["flags"]
    rows.append({"status": "ok", "note": "", "scope": "majority_vote", "labelers": "|".join(sorted(sub)),
                 "n_items": len(maj),
                 "n_no_majority": int(flags.map(lambda f: "no_majority" in f).sum()),
                 "n_plurality_only": int(flags.map(lambda f: "plurality_only" in f).sum()),
                 "n_topic_tie": int(flags.map(lambda f: "topic_tie" in f).sum()),
                 "n_direction_spread": int(flags.map(lambda f: "direction_spread" in f).sum())})
    return pd.DataFrame(rows), maj


# ----------------------------------------------------------------------------------
# table 2: labeler vs reference


def labeler_vs_reference_table(labels: dict[str, dict[str, dict]], skipped: dict[str, str],
                               gold_labelers: list[str], maj: pd.DataFrame | None,
                               candidates: list[str]) -> pd.DataFrame:
    rows = []
    gold_complete = all(g in labels for g in gold_labelers)
    maj_frame = _frame_from_majority(maj) if maj is not None else None
    for x in candidates:
        base = {"labeler": x, "in_gold": x in gold_labelers}
        if x not in labels:
            for ref in ("gold_majority", "leave_out_consensus"):
                rows.append({**base, "reference": ref, "status": "pending",
                             "note": skipped.get(x, "no outputs"), **{m: NAN for m in METRICS}})
            continue
        cand = _frame(labels[x], sorted(labels[x]))
        # (i) gold majority
        if maj_frame is None:
            rows.append({**base, "reference": "gold_majority", "status": "pending",
                         "note": "gold labelers incomplete", **{m: NAN for m in METRICS}})
        else:
            rows.append({**base, "reference": "gold_majority", "status": "ok",
                         "note": f"{len(maj) - len(maj_frame)} no-majority items excluded",
                         "n_reference_pool": len(maj_frame), **compare(cand, maj_frame)})
        # (ii) leave-out consensus
        others = [g for g in gold_labelers if g != x]
        if len(others) < 2 or not all(g in labels for g in others):
            rows.append({**base, "reference": "leave_out_consensus", "status": "pending",
                         "note": "other gold labelers incomplete", **{m: NAN for m in METRICS}})
        else:
            ref = leave_out_reference(labels, gold_labelers, x)
            pool = len(agreement.common_ids({g: labels[g] for g in others}))
            rows.append({**base, "reference": "leave_out_consensus", "status": "ok",
                         "note": f"others agree on {len(ref)}/{pool} items",
                         "n_reference_pool": pool, **compare(cand, ref)})
    cols = ["labeler", "in_gold", "reference", "status", "note", "n_reference_pool"] + METRICS
    return pd.DataFrame(rows).reindex(columns=cols)


# ----------------------------------------------------------------------------------
# table 3: human vs models


def human_vs_models_table(human_labels: dict[str, dict], n_blank: int, n_invalid: int,
                          labels: dict[str, dict[str, dict]], skipped: dict[str, str],
                          maj: pd.DataFrame | None, candidates: list[str],
                          sheet_note: str = "") -> pd.DataFrame:
    info = {"n_human_labelled": len(human_labels), "n_human_blank": n_blank,
            "n_human_invalid": n_invalid}
    refs: list[tuple[str, pd.DataFrame | None, str]] = []
    refs.append(("gold_majority", _frame_from_majority(maj) if maj is not None else None,
                 "gold labelers incomplete"))
    for x in candidates:
        refs.append((x, _frame(labels[x], sorted(labels[x])) if x in labels else None,
                     skipped.get(x, "no outputs")))
    h = _frame(human_labels, sorted(human_labels)) if human_labels else None
    rows = []
    for name, frame, why in refs:
        row = {"reference": name, **info}
        if h is None:
            rows.append({**row, "status": "pending", "note": sheet_note or "no human labels yet",
                         **{m: NAN for m in METRICS}})
        elif frame is None:
            rows.append({**row, "status": "pending", "note": why, **{m: NAN for m in METRICS}})
        else:
            rows.append({**row, "status": "ok", "note": sheet_note, **compare(h, frame)})
    cols = ["reference", "status", "note", "n_human_labelled", "n_human_blank", "n_human_invalid"] + METRICS
    return pd.DataFrame(rows).reindex(columns=cols)


def read_human(cfg: dict) -> tuple[dict[str, dict], int, int, str]:
    """(valid labels, n blank rows, n invalid rows, note) from the human sheet, tolerating its absence."""
    path = human.sheet_path(cfg)
    if not path.is_file():
        return {}, 0, 0, f"no sheet at {path.name}"
    labels, errors, n_blank = human.parse_human_sheet(path)
    note = ""
    if errors:
        note = f"{len(errors)} sheet rows with errors ignored: " + "; ".join(
            f"{i}: {' / '.join(e)}" for i, e in list(errors.items())[:5])
    return labels, n_blank, len(errors), note


# ----------------------------------------------------------------------------------
# output


def _f(x, nd: int) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _esc(s) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _tex(path: Path, header: list[str], rows: list[list[str]], n_text: int = 1) -> None:
    align = "l" * n_text + "r" * (len(header) - n_text)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def _metric_cells(r) -> list[str]:
    return [_f(r["function_pct"], 1), _f(r["function_kappa"], 2), _f(r["topic_pct"], 1),
            _f(r["direction_exact_pct"], 1), _f(r["direction_within1_pct"], 1),
            _f(r["direction_sign_pct"], 1), _f(r["direction_qwk"], 2), _f(r["full_match_pct"], 1)]


_METRIC_HEADER = ["Function acc.", r"Function $\kappa$", "Topic acc.", "Dir. exact", "Dir. within 1",
                  "Dir. sign", r"Dir. $\kappa_w$", "Full match"]


def write_gold_tex(tab: pd.DataFrame, path: Path) -> None:
    header = ["Labelers", "$n$", r"Function $\kappa$", r"Topic $\kappa$", "Dir. exact (\\%)",
              r"Dir. $\kappa_w$ / $\alpha$", "Full match (\\%)"]
    rows = []
    for r in tab.to_dict("records"):
        if r["status"] != "ok":
            rows.append([_esc(r["labelers"]) + " (pending)"] + ["--"] * 6)
        elif r["scope"] == "majority_vote":
            rows.append([f"Majority vote ({int(r['n_no_majority'])} no majority)",
                         str(int(r["n_items"]))] + ["--"] * 5)
        elif r["scope"] == "overall":
            rows.append(["All (Fleiss / Krippendorff)", str(int(r["n_items"])),
                         _f(r["function_fleiss_kappa"], 2), _f(r["topic_fleiss_kappa"], 2),
                         _f(r["direction_all_exact_pct"], 1), _f(r["direction_alpha_interval"], 2),
                         _f(r["full_match_all_pct"], 1)])
        else:
            rows.append([_esc(r["labelers"]), str(int(r["n_items"])), _f(r["function_kappa"], 2),
                         _f(r["topic_kappa"], 2), _f(r["direction_exact_pct"], 1),
                         _f(r["direction_qwk"], 2), _f(r["full_match_pct"], 1)])
    _tex(path, header, rows)


def write_reference_tex(tab: pd.DataFrame, path: Path, lead: list[tuple[str, str]]) -> None:
    """lead: [(header, column)] for the leading text columns."""
    header = [h for h, _ in lead] + ["$n$"] + _METRIC_HEADER
    rows = []
    for r in tab.to_dict("records"):
        n = str(int(r["n_items"])) if r["status"] == "ok" else "pending"
        rows.append([_esc(r[c]) for _, c in lead] + [n] + _metric_cells(r))
    _tex(path, header, rows, n_text=len(lead))


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def build(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = repo_root()
    out = root / cfg["paths"]["results_tables"]
    out.mkdir(parents=True, exist_ok=True)
    out_dir = root / cfg["paths"]["labels_out"]
    gold_labelers = list(cfg["labelling"]["gold"]["labelers"])
    candidates = sorted(set(discover_labelers("gold", out_dir)) | set(gold_labelers))

    labels, skipped = load_complete("gold", candidates, out_dir=out_dir)
    for lab, why in skipped.items():
        print(f"validation: skipping gold labeler {lab!r} ({why})")

    # 1. gold agreement + majority vote
    gold_tab, maj = gold_agreement_table(labels, gold_labelers, skipped)
    _csv(gold_tab, out / "gold_agreement.csv")
    write_gold_tex(gold_tab, out / "gold_agreement.tex")
    if maj is not None:
        path = root / cfg["paths"]["labels_gold"] / "gold_majority.jsonl"
        recs = agreement.majority_records(maj)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=_json_default)
                                  for r in recs) + "\n", encoding="utf-8")
        n_nm = sum("no_majority" in r["agreement"]["flags"] for r in recs)
        print(f"validation: gold majority written ({len(recs)} items, {n_nm} with no majority)")
    else:
        print("validation: gold agreement pending (not all gold labelers complete); "
              "gold_majority.jsonl not written")

    # 2. labeler vs reference
    ref_tab = labeler_vs_reference_table(labels, skipped, gold_labelers, maj, candidates)
    _csv(ref_tab, out / "labeler_vs_reference.csv")
    write_reference_tex(ref_tab, out / "labeler_vs_reference.tex",
                        [("Labeler", "labeler"), ("Reference", "reference")])

    # 3. human vs models
    h_labels, n_blank, n_invalid, note = read_human(cfg)
    hum_tab = human_vs_models_table(h_labels, n_blank, n_invalid, labels, skipped, maj, candidates, note)
    _csv(hum_tab, out / "human_vs_models.csv")
    write_reference_tex(hum_tab, out / "human_vs_models.tex", [("Reference", "reference")])
    print(f"validation: human sheet: {len(h_labels)} labelled, {n_blank} blank, {n_invalid} invalid rows")
    return {"gold": gold_tab, "labeler_vs_reference": ref_tab, "human_vs_models": hum_tab}


def main(cfg: dict | None = None) -> None:
    build(cfg)


if __name__ == "__main__":
    main()
