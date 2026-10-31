"""Inter-labeller agreement and majority vote.

Labels are dicts {"id", "aspects": [...], "confidence"}. Agreement is computed on each
item's PRIMARY aspect: the aspect with the largest |direction| (ties -> first listed).
All metrics (pairwise ones too) use the items labelled by every labeller, so tables are
comparable; the number of items is reported.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ftyields.label.batches import batch_paths, output_path, read_manifest
from ftyields.label.schema import (DIRECTIONS, TOPIC_FUNCTIONS, TOPICS, read_batch,
                                   validate_output)

Labels = Mapping[str, Mapping[str, dict]]   # labeler -> {id: label}

# ----------------------------------------------------------------------------------
# loading


def load_set_labels(set_name: str, labelers: Sequence[str], batches_dir=None, out_dir=None,
                    on_missing: str = "error") -> dict[str, dict[str, dict]]:
    """labeler -> {id: label} from valid outputs of an existing set.

    on_missing: "error" raises if a labeller's batch output is absent or invalid;
    "skip" silently omits those batches.
    """
    if on_missing not in ("error", "skip"):
        raise ValueError("on_missing must be 'error' or 'skip'")
    result: dict[str, dict[str, dict]] = {}
    for lab in labelers:
        d: dict[str, dict] = {}
        for bp in batch_paths(set_name, batches_dir):
            op = output_path(set_name, lab, bp, out_dir)
            rep = validate_output(bp, op)
            if rep.ok:
                d.update(rep.labels)
            elif on_missing == "error":
                raise ValueError(f"labeler {lab!r}, {bp.name}: no valid output\n{rep.summary()}")
        result[lab] = d
    return result


def load_set_texts(set_name: str, batches_dir=None) -> dict[str, tuple[str, str]]:
    """id -> (prev_text, text) for every item of a set."""
    texts = {}
    for bp in batch_paths(set_name, batches_dir):
        for it in read_batch(bp):
            texts[it["id"]] = (it.get("prev_text", ""), it["text"])
    return texts


# ----------------------------------------------------------------------------------
# primary aspect


def primary(label: Mapping) -> tuple[str, str | None, int]:
    """(function, topic, direction) of the aspect with the largest |direction|; ties -> first."""
    aspects = label["aspects"]
    i = min(range(len(aspects)), key=lambda k: (-abs(aspects[k]["direction"]), k))
    a = aspects[i]
    return a["function"], a["topic"], a["direction"]


def primary_frames(labels: Labels, ids: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """labeler -> DataFrame(function, topic, direction) indexed by id, on the common ids."""
    if ids is None:
        ids = common_ids(labels)
    ids = list(ids)
    return {lab: pd.DataFrame([primary(labels[lab][i]) for i in ids], index=pd.Index(ids, name="id"),
                              columns=["function", "topic", "direction"])
            for lab in labels}


def common_ids(labels: Labels) -> list[str]:
    sets = [set(d) for d in labels.values()]
    return sorted(set.intersection(*sets)) if sets else []


# ----------------------------------------------------------------------------------
# agreement statistics


def cohen_kappa(a: Sequence, b: Sequence, categories: Sequence | None = None,
                weights: str | None = None) -> float:
    """Cohen's kappa; weights None (unweighted) or 'quadratic' (numeric categories).

    Returns nan when chance agreement is total (no variation to explain).
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if not a:
        return float("nan")
    cats = list(categories) if categories is not None else sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights is None:
        w = 1.0 - np.eye(k)
    elif weights == "quadratic":
        v = np.array(cats, dtype=float)
        w = (v[:, None] - v[None, :]) ** 2
    else:
        raise ValueError("weights must be None or 'quadratic'")
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / denom)


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from an items x categories matrix of rating counts (equal raters/item)."""
    counts = np.asarray(counts, dtype=float)
    n_items = counts.shape[0]
    n_raters = counts[0].sum()
    if not np.allclose(counts.sum(1), n_raters):
        raise ValueError("every item needs the same number of ratings")
    if n_raters < 2:
        raise ValueError("need at least 2 raters")
    p_j = counts.sum(0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    if p_e == 1:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def fleiss_kappa_labels(ratings: Sequence[Sequence], categories: Sequence | None = None) -> float:
    """Fleiss' kappa from a list (items) of lists (one rating per labeller)."""
    cats = list(categories) if categories is not None else sorted({r for row in ratings for r in row}, key=str)
    idx = {c: i for i, c in enumerate(cats)}
    counts = np.zeros((len(ratings), len(cats)))
    for i, row in enumerate(ratings):
        for r in row:
            counts[i, idx[r]] += 1
    return fleiss_kappa(counts)


def krippendorff_alpha_interval(data: np.ndarray) -> float:
    """Krippendorff's alpha, interval level. data: units x raters, nan = missing.

    alpha = 1 - D_o / D_e with squared-difference metric; units with < 2 values are
    dropped. Returns nan if there is no variation at all (D_e = 0).
    """
    data = np.asarray(data, dtype=float)
    s_o = 0.0
    pooled: list[float] = []
    for row in data:
        v = row[~np.isnan(row)]
        m = len(v)
        if m < 2:
            continue
        pooled.extend(v)
        diff2 = (v[:, None] - v[None, :]) ** 2
        s_o += diff2.sum() / (m - 1)
    n = len(pooled)
    if n < 2:
        return float("nan")
    p = np.array(pooled)
    s_e = ((p[:, None] - p[None, :]) ** 2).sum()
    if s_e == 0:
        return float("nan")
    return float(1.0 - (n - 1) * s_o / s_e)


def _pct(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(100.0 * mask.mean()) if mask.size else float("nan")


def _direction_metrics(da: np.ndarray, db: np.ndarray) -> dict:
    return {
        "direction_exact_pct": _pct(da == db),
        "direction_within1_pct": _pct(np.abs(da - db) <= 1),
        "direction_sign_pct": _pct(np.sign(da) == np.sign(db)),
        "direction_qwk": cohen_kappa(da, db, categories=DIRECTIONS, weights="quadratic"),
    }


def pair_metrics(pa: pd.DataFrame, pb: pd.DataFrame) -> dict:
    """Agreement between two labellers' primary-aspect frames (same index)."""
    fa, fb = pa["function"].to_numpy(), pb["function"].to_numpy()
    out: dict = {"n_items": len(pa),
                 "function_pct": _pct(fa == fb),
                 "function_kappa": cohen_kappa(fa, fb, categories=sorted(set(fa) | set(fb)))}
    both_topic = np.isin(fa, TOPIC_FUNCTIONS) & np.isin(fb, TOPIC_FUNCTIONS)
    ta, tb = pa["topic"].to_numpy()[both_topic], pb["topic"].to_numpy()[both_topic]
    out["topic_n_items"] = int(both_topic.sum())
    out["topic_pct"] = _pct(ta == tb)
    out["topic_kappa"] = cohen_kappa(ta, tb, categories=list(TOPICS)) if len(ta) else float("nan")
    da, db = pa["direction"].to_numpy(dtype=int), pb["direction"].to_numpy(dtype=int)
    out.update(_direction_metrics(da, db))
    subst = (fa != "PROCEDURAL") & (fb != "PROCEDURAL")
    sm = _direction_metrics(da[subst], db[subst]) if subst.any() else {}
    out["direction_subst_n_items"] = int(subst.sum())
    out["direction_subst_exact_pct"] = sm.get("direction_exact_pct", float("nan"))
    out["direction_subst_qwk"] = sm.get("direction_qwk", float("nan"))
    same_topic = (pa["topic"].fillna("").to_numpy() == pb["topic"].fillna("").to_numpy())
    out["full_match_pct"] = _pct((fa == fb) & same_topic & (da == db))
    return out


def overall_metrics(frames: Mapping[str, pd.DataFrame]) -> dict:
    """All-labeller agreement: Fleiss kappa (function, topic), Krippendorff alpha (direction)."""
    labs = list(frames)
    n = len(next(iter(frames.values())))
    f = np.column_stack([frames[l]["function"].to_numpy() for l in labs])
    t = np.column_stack([frames[l]["topic"].fillna("").to_numpy() for l in labs])
    d = np.column_stack([frames[l]["direction"].to_numpy(dtype=float) for l in labs])
    f_agree = (f == f[:, [0]]).all(1)
    out: dict = {"n_items": n, "n_labelers": len(labs)}
    if n:
        out["function_fleiss_kappa"] = fleiss_kappa_labels(f.tolist(), categories=sorted(set(f.ravel())))
        out["function_all_agree_pct"] = _pct(f_agree)
        topic_rows = np.isin(f, TOPIC_FUNCTIONS).all(1)
        out["topic_n_items"] = int(topic_rows.sum())
        out["topic_all_agree_pct"] = _pct((t[topic_rows] == t[topic_rows][:, [0]]).all(1)) if topic_rows.any() else float("nan")
        out["topic_fleiss_kappa"] = (fleiss_kappa_labels(t[topic_rows].tolist(), categories=list(TOPICS))
                                     if topic_rows.any() else float("nan"))
        out["direction_all_exact_pct"] = _pct((d == d[:, [0]]).all(1))
        out["direction_alpha_interval"] = krippendorff_alpha_interval(d)
        subst = (f != "PROCEDURAL").all(1)
        out["direction_subst_alpha_interval"] = (krippendorff_alpha_interval(d[subst]) if subst.any()
                                                 else float("nan"))
        out["full_match_all_pct"] = _pct(f_agree & (t == t[:, [0]]).all(1) & (d == d[:, [0]]).all(1))
    return out


def agreement_tables(labels: Labels, restrict_function_agree: bool = False) -> dict[str, pd.DataFrame]:
    """{"pairwise": per-pair table, "overall": one-row table} on the labelled-by-all items.

    restrict_function_agree: keep only items where all labellers' primary function agrees
    (so topic/direction agreement is conditional on function agreement).
    """
    if len(labels) < 2:
        raise ValueError("need at least 2 labelers")
    ids = common_ids(labels)
    frames = primary_frames(labels, ids)
    if restrict_function_agree:
        funcs = pd.concat([f["function"] for f in frames.values()], axis=1)
        keep = funcs.nunique(axis=1) == 1
        frames = {l: f[keep.to_numpy()] for l, f in frames.items()}
    rows = []
    for a, b in itertools.combinations(sorted(frames), 2):
        rows.append({"labeler_a": a, "labeler_b": b, **pair_metrics(frames[a], frames[b])})
    pairwise = pd.DataFrame(rows)
    overall = pd.DataFrame([{**overall_metrics(frames),
                             "n_items_labelled_by_any": len(set().union(*[set(d) for d in labels.values()]))}])
    return {"pairwise": pairwise, "overall": overall}


# ----------------------------------------------------------------------------------
# disagreements


def _fmt(p: tuple) -> str:
    f, t, d = p
    return f"{f}/{t or '-'}/{d:+d}"


def disagreements(labels: Labels, texts: Mapping[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Items (labelled by all) where the primary labels differ, for human review.

    Columns: id, prev_text, text, differs_on (function/topic/direction), one column per
    labeller (function/topic/direction of its primary aspect) and one with all aspects.
    """
    ids = common_ids(labels)
    labs = sorted(labels)
    rows = []
    for i in ids:
        prims = {l: primary(labels[l][i]) for l in labs}
        if len(set(prims.values())) == 1:
            continue
        diff = [name for k, name in enumerate(("function", "topic", "direction"))
                if len({p[k] for p in prims.values()}) > 1]
        prev, text = (texts or {}).get(i, ("", ""))
        row = {"id": i, "prev_text": prev, "text": text, "differs_on": ",".join(diff)}
        for l in labs:
            row[l] = _fmt(prims[l])
            row[f"{l}__all_aspects"] = " | ".join(_fmt((a["function"], a["topic"], a["direction"]))
                                                  for a in labels[l][i]["aspects"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["id", "prev_text", "text", "differs_on"]
                        + [c for l in labs for c in (l, f"{l}__all_aspects")])


# ----------------------------------------------------------------------------------
# majority vote


def _round_half_away(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x)) if x else 0


def _plurality(values: Sequence, order: Sequence | None = None):
    """(winner, count, tied) by count; ties broken by `order` (else None winner if tied)."""
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    leaders = [v for v, c in counts.items() if c == top]
    if len(leaders) == 1:
        return leaders[0], top, False
    if order is not None:
        leaders.sort(key=lambda v: order.index(v))
        return leaders[0], top, True
    return None, top, True


def majority_vote(labels: Labels, min_labelers: int = 3) -> pd.DataFrame:
    """Majority vote of primary aspects over items labelled by all labellers.

    function: strict plurality; a tie for first place gives function None and flag
      "no_majority". A plurality below 50% is flagged "plurality_only".
    topic: plurality among labellers who chose the winning function (only for
      ASSESSMENT/OUTLOOK; ties resolved by canonical topic order and flagged "topic_tie").
    direction: median of the direction of those same labellers, rounded half away from
      zero (flag "direction_spread" if they do not all agree).
    confidence: "high" if all labellers are in the majority function, "med" if the share is
      >= 2/3, else "low"; the numeric share is in function_share.
    Output columns: id, function, topic, direction, confidence, function_share, n_labelers,
      flags (list), agreement (dict).
    """
    if len(labels) < min_labelers:
        raise ValueError(f"majority vote needs >= {min_labelers} labelers, got {len(labels)}")
    ids = common_ids(labels)
    labs = sorted(labels)
    rows = []
    for i in ids:
        prims = {l: primary(labels[l][i]) for l in labs}
        n = len(labs)
        flags: list[str] = []
        func, count, tied = _plurality([p[0] for p in prims.values()])
        votes = {l: _fmt(p) for l, p in prims.items()}
        if func is None:
            rows.append({"id": i, "function": None, "topic": None, "direction": None,
                         "confidence": "low", "function_share": count / n, "n_labelers": n,
                         "flags": ["no_majority"],
                         "agreement": {"votes": votes, "function_share": count / n,
                                       "flags": ["no_majority"]}})
            continue
        if count * 2 <= n:
            flags.append("plurality_only")
        group = [p for p in prims.values() if p[0] == func]
        topic = None
        topic_share = None
        if func in TOPIC_FUNCTIONS:
            topic, tcount, ttie = _plurality([p[1] for p in group], order=list(TOPICS))
            topic_share = tcount / len(group)
            if ttie:
                flags.append("topic_tie")
        dirs = [p[2] for p in group]
        direction = _round_half_away(float(np.median(dirs)))
        if len(set(dirs)) > 1:
            flags.append("direction_spread")
        share = count / n
        conf = "high" if count == n else ("med" if share >= 2 / 3 else "low")
        rows.append({"id": i, "function": func, "topic": topic, "direction": direction,
                     "confidence": conf, "function_share": share, "n_labelers": n,
                     "flags": flags,
                     "agreement": {"votes": votes, "function_share": share,
                                   "topic_share": topic_share, "directions": dirs, "flags": flags}})
    return pd.DataFrame(rows, columns=["id", "function", "topic", "direction", "confidence",
                                       "function_share", "n_labelers", "flags", "agreement"])


def _none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def majority_records(df: pd.DataFrame) -> list[dict]:
    """Majority-vote DataFrame -> label dicts (same schema + extra "agreement" field).

    Items with no function majority carry one aspect with function/topic/direction null
    and agreement.flags containing "no_majority"; they do not validate against the schema
    on purpose, so they cannot be mistaken for labels.
    """
    recs = []
    for r in df.itertuples():
        direction = None if r.direction is None or pd.isna(r.direction) else int(r.direction)
        recs.append({"id": r.id,
                     "aspects": [{"function": _none(r.function), "topic": _none(r.topic),
                                  "direction": direction}],
                     "confidence": r.confidence,
                     "agreement": r.agreement})
    return recs


# ----------------------------------------------------------------------------------
# markdown summary


def _md_table(df: pd.DataFrame) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def render_markdown(set_name: str, tables: Mapping[str, pd.DataFrame],
                    tables_function_agree: Mapping[str, pd.DataFrame], n_disagreements: int) -> str:
    parts = [f"# Agreement summary: {set_name}", "",
             "Metrics use each item's primary aspect (largest |direction|, ties -> first listed), "
             "on items labelled by all labellers. Percentages are 0-100.", "",
             "## Overall", _md_table(tables["overall"]).replace("nan", "NA"), "",
             "## Pairwise", _md_table(tables["pairwise"]).replace("nan", "NA"), "",
             "## Restricted to items where all labellers agree on function", "",
             "Overall", "", _md_table(tables_function_agree["overall"]).replace("nan", "NA"), "",
             "Pairwise", "", _md_table(tables_function_agree["pairwise"]).replace("nan", "NA"), "",
             f"Items with any disagreement on the primary label: {n_disagreements}", ""]
    return "\n".join(parts)
