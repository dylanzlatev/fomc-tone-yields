"""Distil LLM sentence labels (FUNCTION / TOPIC / DIRECTION) into a local classifier.

Targets ("slots")
-----------------
A label is a set of aspects, each (function, topic, direction). There are 11 slots:
ASSESSMENT x {INFLATION, LABOR, ACTIVITY, FINANCIAL, GENERAL}, OUTLOOK x {same 5}, POLICY.
For a sentence and slot: ``present`` in {0, 1} and, if present, ``direction`` in -2..2.
PROCEDURAL = no slot present. If a label has two aspects in the same slot (forbidden by
the codebook) the one with the larger |direction| is kept (ties: first listed) and the
case is counted and reported.

Model
-----
Shared encoder (``distill.encoder``) + per-slot heads: a presence logit (sigmoid, BCE) and
5 direction logits (CE, masked to present slots). loss = mean BCE + lambda * masked CE.
Input = prev_text </s></s> text (the masked text the labelers saw), truncated to
``distill.max_length`` tokens (the previous sentence is shortened first; counts reported).

Splits are by YEAR of first_release_date (``distill.test_years`` / ``val_years``), because
near-duplicate sentences recur across meetings and a random split leaks. A seeded random
80/10/10 split is reported for comparison, clearly flagged as optimistic.

Inference is fp32 with batch size 1 (no padding), so a sentence's output cannot depend on
batch composition; this is verified on ``distill.batch_check_n`` sentences.

Usage (project interpreter, PYTHONPATH=src):

    python -m ftyields.model.distill                  # train + eval + infer, real labels
    python -m ftyields.model.distill --stages train
    python -m ftyields.model.distill --stages eval infer
    python -m ftyields.model.distill --smoke          # 650 pilot+gold labels, outputs in tmp/
    python -m ftyields.model.distill --memory-probe   # a few FOMC-RoBERTa-large steps, peak GPU memory
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ftyields import config as cfgmod
from ftyields.label import agreement
from ftyields.label.batches import batch_paths, output_path
from ftyields.label.schema import (DIRECTIONS, TOPICS, MAX_ASPECTS, validate_label,
                                   validate_output)

# ----------------------------------------------------------------------------- slots

SLOTS: tuple[tuple[str, str | None], ...] = tuple(
    [(f, t) for f in ("ASSESSMENT", "OUTLOOK") for t in TOPICS] + [("POLICY", None)])
SLOT_NAMES = tuple(f if t is None else f"{f}_{t}" for f, t in SLOTS)
N_SLOTS = len(SLOTS)
N_DIR = len(DIRECTIONS)
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}
DIR_VALUES = np.array(DIRECTIONS, dtype=np.float64)   # class k <-> direction DIR_VALUES[k]


def _aspect_slot(aspect: Mapping) -> int | None:
    """Slot index of one aspect; None for PROCEDURAL."""
    f, t = aspect["function"], aspect.get("topic")
    if f == "PROCEDURAL":
        return None
    return SLOT_INDEX[(f, t if f != "POLICY" else None)]


def label_to_slots(label: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
    """Label dict -> (present (11,) int, direction (11,) int, info).

    direction is 0 where the slot is absent (use `present` as the mask). info counts
    ``duplicate_slot`` (two aspects in one slot; larger |direction| kept, ties first) and
    ``procedural_with_other`` (a PROCEDURAL aspect next to a substantive one; it is dropped).
    """
    present = np.zeros(N_SLOTS, dtype=np.int64)
    direction = np.zeros(N_SLOTS, dtype=np.int64)
    info = {"duplicate_slot": 0, "procedural_with_other": 0}
    aspects = label["aspects"]
    n_proc = sum(a["function"] == "PROCEDURAL" for a in aspects)
    if n_proc and n_proc < len(aspects):
        info["procedural_with_other"] = 1
    for a in aspects:
        i = _aspect_slot(a)
        if i is None:
            continue
        d = int(a["direction"])
        if present[i]:
            info["duplicate_slot"] += 1
            if abs(d) > abs(direction[i]):
                direction[i] = d
        else:
            present[i], direction[i] = 1, d
    return present, direction, info


def slots_to_label(present: np.ndarray, direction: np.ndarray, id_: str = "",
                   confidence: str = "high") -> dict:
    """Inverse of label_to_slots (slot order); no present slot -> one PROCEDURAL aspect."""
    aspects = [_slot_aspect(i, int(direction[i])) for i in range(N_SLOTS) if present[i]]
    if not aspects:
        aspects = [{"function": "PROCEDURAL", "topic": None, "direction": 0}]
    return {"id": id_, "aspects": aspects, "confidence": confidence}


def _slot_aspect(i: int, d: int) -> dict:
    f, t = SLOTS[i]
    return {"function": f, "topic": t, "direction": d}


def labels_to_arrays(labels: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack label_to_slots over labels -> (n, 11) present, (n, 11) direction, summed info."""
    pres, dirs = [], []
    tot = {"duplicate_slot": 0, "procedural_with_other": 0}
    for lab in labels:
        p, d, info = label_to_slots(lab)
        pres.append(p)
        dirs.append(d)
        for k in tot:
            tot[k] += info[k]
    if not pres:
        return np.zeros((0, N_SLOTS), np.int64), np.zeros((0, N_SLOTS), np.int64), tot
    return np.stack(pres), np.stack(dirs), tot


# ----------------------------------------------------------------------------- derived outputs


def expected_direction(p_dir: np.ndarray) -> np.ndarray:
    """E[d] = sum_d d * p(d) over the last axis (5 classes in order -2..2)."""
    return np.asarray(p_dir, dtype=np.float64) @ DIR_VALUES


def derive_label(p_present: np.ndarray, p_dir: np.ndarray, threshold: float = 0.5,
                 max_aspects: int = MAX_ASPECTS, id_: str = "", confidence: str = "high") -> dict:
    """Predicted label in the codebook schema from one sentence's probabilities.

    Slots with p_present >= threshold, at most `max_aspects` of them (highest presence
    probability first, ties -> lower slot index), direction = argmax of the 5 direction
    probabilities. No slot above the threshold -> PROCEDURAL.
    """
    p = np.asarray(p_present, dtype=np.float64)
    order = [i for i in np.argsort(-p, kind="stable") if p[i] >= threshold][:max_aspects]
    aspects = [_slot_aspect(int(i), int(DIRECTIONS[int(np.argmax(p_dir[i]))])) for i in order]
    if not aspects:
        aspects = [{"function": "PROCEDURAL", "topic": None, "direction": 0}]
    return {"id": id_, "aspects": aspects, "confidence": confidence}


def derive_labels(p_present: np.ndarray, p_dir: np.ndarray, ids: Sequence[str],
                  threshold: float = 0.5) -> list[dict]:
    return [derive_label(p_present[i], p_dir[i], threshold, id_=ids[i]) for i in range(len(ids))]


# ----------------------------------------------------------------------------- splits


def year_split(dates: Sequence, test_years: Sequence[int], val_years: Sequence[int]) -> np.ndarray:
    """'train' / 'val' / 'test' per first_release_date, by calendar year."""
    test_years, val_years = set(map(int, test_years)), set(map(int, val_years))
    if test_years & val_years:
        raise ValueError(f"test_years and val_years overlap: {sorted(test_years & val_years)}")
    years = pd.to_datetime(pd.Series(list(dates))).dt.year.to_numpy()
    out = np.full(len(years), "train", dtype=object)
    out[np.isin(years, list(val_years))] = "val"
    out[np.isin(years, list(test_years))] = "test"
    return out


def random_split(n: int, fractions: Mapping[str, float], seed: int) -> np.ndarray:
    """Seeded random 'train' / 'val' / 'test' assignment with the given fractions."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("random_split fractions must sum to 1")
    perm = np.random.default_rng(int(seed)).permutation(n)
    n_val, n_test = int(round(n * fractions["val"])), int(round(n * fractions["test"]))
    out = np.full(n, "train", dtype=object)
    out[perm[:n_val]] = "val"
    out[perm[n_val:n_val + n_test]] = "test"
    return out


# ----------------------------------------------------------------------------- tokenisation


def encode_pair(prev_ids: list[int], text_ids: list[int], cls_id: int, sep_id: int,
                max_length: int) -> tuple[list[int], int, bool, bool]:
    """<s> prev </s></s> text </s> (no prev: <s> text </s>), truncated to max_length.

    The previous sentence is trimmed first (keeping its END, closest to the target); the
    target is trimmed (keeping its start) only if it does not fit alone. Returns
    (ids, full_length, prev_truncated, text_truncated).
    """
    full = len(text_ids) + 2 + ((len(prev_ids) + 2) if prev_ids else 0)
    if full <= max_length:
        ids = ([cls_id] + prev_ids + [sep_id, sep_id] + text_ids + [sep_id]) if prev_ids \
            else ([cls_id] + text_ids + [sep_id])
        return ids, full, False, False
    text_trunc = len(text_ids) > max_length - 2
    if text_trunc:
        text_ids = text_ids[:max_length - 2]
    room_prev = max_length - 4 - len(text_ids)
    if prev_ids and room_prev >= 1:
        ids = [cls_id] + prev_ids[-room_prev:] + [sep_id, sep_id] + text_ids + [sep_id]
    else:
        ids = [cls_id] + text_ids + [sep_id]
    return ids, full, True, text_trunc


def encode_items(tok, prevs: Sequence[str], texts: Sequence[str], max_length: int, chunk: int = 4096
                 ) -> dict[str, Any]:
    """Token ids for every (prev, text) pair plus truncation bookkeeping."""
    ids, full, prev_tr, text_tr = [], [], [], []
    for s in range(0, len(texts), chunk):
        p = tok([x or "" for x in prevs[s:s + chunk]], add_special_tokens=False)["input_ids"]
        t = tok([x or "" for x in texts[s:s + chunk]], add_special_tokens=False)["input_ids"]
        for pi, ti in zip(p, t):
            e, n_full, ptr, ttr = encode_pair(pi, ti, tok.cls_token_id, tok.sep_token_id, max_length)
            ids.append(e)
            full.append(n_full)
            prev_tr.append(ptr)
            text_tr.append(ttr)
    return {"ids": ids, "n_full": np.array(full), "truncated": np.array(prev_tr),
            "text_truncated": np.array(text_tr)}


def collate(seqs: Sequence[Sequence[int]], pad_id: int, pad_to: int | None = None):
    """Right-padded input_ids / attention_mask tensors."""
    import torch
    width = max(len(s) for s in seqs) if pad_to is None else pad_to
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


# ----------------------------------------------------------------------------- loss


def distill_loss(pres_logits, dir_logits, present, direction, lam: float = 1.0):
    """mean BCE(presence) + lam * masked CE(direction); returns (loss, bce, ce).

    pres_logits (B, S); dir_logits (B, S, 5); present (B, S) in {0,1}; direction (B, S)
    in -2..2 (ignored where present = 0). The CE is averaged over present slots only
    (0 if the batch has none).
    """
    import torch
    import torch.nn.functional as F
    pres_logits, dir_logits = pres_logits.float(), dir_logits.float()
    present_f = present.float()
    bce = F.binary_cross_entropy_with_logits(pres_logits, present_f)
    target = (direction + 2).clamp(0, N_DIR - 1).long()
    ce_all = F.cross_entropy(dir_logits.reshape(-1, N_DIR), target.reshape(-1), reduction="none")
    mask = present_f.reshape(-1)
    ce = (ce_all * mask).sum() / mask.sum().clamp(min=1.0)
    return bce + lam * ce, bce, ce


# ----------------------------------------------------------------------------- model


def _from_pretrained(factory, name: str, **kw):
    """Local HF cache first; download from the Hub only if it is not cached."""
    try:
        return factory.from_pretrained(name, local_files_only=True, **kw)
    except (OSError, ValueError, AttributeError):   # not cached (some versions raise AttributeError)
        return factory.from_pretrained(name, **kw)


def make_distiller(encoder: str, pretrained: bool = True, dropout: float = 0.1):
    """Shared RoBERTa encoder + presence head (S) + direction head (S x 5)."""
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class Distiller(nn.Module):
        def __init__(self):
            super().__init__()
            if pretrained:
                self.encoder = _from_pretrained(AutoModel, encoder, add_pooling_layer=False)
            else:
                self.encoder = AutoModel.from_config(_from_pretrained(AutoConfig, encoder),
                                                     add_pooling_layer=False)
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(dropout)
            self.presence = nn.Linear(h, N_SLOTS)
            self.direction = nn.Linear(h, N_SLOTS * N_DIR)

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            h = self.drop(h)
            return self.presence(h), self.direction(h).view(-1, N_SLOTS, N_DIR)

    return Distiller()


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    import torch
    cfgmod.set_seeds(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, seqs: Sequence[Sequence[int]], pad_id: int, device: str, batch_size: int = 1,
            amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(p_present (n, 11), p_dir (n, 11, 5)) in input order.

    batch_size 1 means no padding at all, so each row is independent of the others. For
    larger batches sentences are length-sorted and padded. amp=True is fp16 autocast
    (validation during training only); the default is plain fp32.
    """
    import torch
    n = len(seqs)
    p_pres = np.zeros((n, N_SLOTS), dtype=np.float32)
    p_dir = np.zeros((n, N_SLOTS, N_DIR), dtype=np.float32)
    order = np.argsort([-len(s) for s in seqs], kind="stable") if batch_size > 1 else np.arange(n)
    model.eval()
    use_amp = bool(amp and device.startswith("cuda"))
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            b = order[s:s + batch_size]
            ids, mask = collate([seqs[i] for i in b], pad_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pl, dl = model(ids.to(device), mask.to(device))
            p_pres[b] = torch.sigmoid(pl.float()).cpu().numpy()
            p_dir[b] = torch.softmax(dl.float(), dim=-1).cpu().numpy()
    return p_pres, p_dir


# ----------------------------------------------------------------------------- metrics


def _f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def slot_presence_table(true_present: np.ndarray, pred_present: np.ndarray) -> pd.DataFrame:
    """Per-slot support / precision / recall / F1 of presence."""
    rows = []
    for i, name in enumerate(SLOT_NAMES):
        t, p = true_present[:, i].astype(bool), pred_present[:, i].astype(bool)
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        rows.append({"slot": name, "support": int(t.sum()), "n_pred": int(p.sum()),
                     "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "recall": tp / (tp + fn) if tp + fn else float("nan"),
                     "f1": _f1(tp, fp, fn)})
    return pd.DataFrame(rows)


def presence_f1(true_present: np.ndarray, pred_present: np.ndarray) -> tuple[float, float]:
    """(macro-F1 over slots with any positive truth or prediction, micro-F1)."""
    tab = slot_presence_table(true_present, pred_present)
    macro = float(np.nanmean(tab["f1"])) if tab["f1"].notna().any() else float("nan")
    t, p = true_present.astype(bool), pred_present.astype(bool)
    micro = _f1(int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()))
    return macro, micro


def primary_only(label: Mapping) -> dict:
    """Label reduced to its primary aspect (largest |direction|, ties first)."""
    f, t, d = agreement.primary(label)
    return {**label, "aspects": [{"function": f, "topic": t, "direction": d}]}


METRIC_COLUMNS = ["n_items", "function_pct", "function_kappa", "topic_n_items", "topic_pct",
                  "topic_kappa", "direction_exact_pct", "direction_within1_pct",
                  "direction_sign_pct", "direction_qwk", "full_match_pct", "presence_macro_f1",
                  "presence_micro_f1", "slot_direction_exact_pct"]


def evaluate_labels(true: Mapping[str, Mapping], pred: Mapping[str, Mapping],
                    pred_primary_only: bool = False) -> dict:
    """Agreement of predicted vs reference labels on their common ids.

    Primary-aspect metrics come from agreement.pair_metrics (function accuracy and kappa,
    topic accuracy over ASSESSMENT/OUTLOOK pairs, direction exact / within-1 / sign / QWK,
    full match); slot-level presence macro/micro F1 and direction accuracy on slots present
    in both use all aspects. pred_primary_only reduces the predictions to their primary
    aspect first (for single-aspect references such as the gold majority).
    """
    ids = sorted(set(true) & set(pred))
    if not ids:
        return {c: float("nan") for c in METRIC_COLUMNS} | {"n_items": 0}
    pred = {i: (primary_only(pred[i]) if pred_primary_only else pred[i]) for i in ids}
    pa = pd.DataFrame([agreement.primary(true[i]) for i in ids], columns=["function", "topic", "direction"])
    pb = pd.DataFrame([agreement.primary(pred[i]) for i in ids], columns=["function", "topic", "direction"])
    m = agreement.pair_metrics(pa, pb)
    tp_, td_, _ = labels_to_arrays([true[i] for i in ids])
    pp_, pd_, _ = labels_to_arrays([pred[i] for i in ids])
    m["presence_macro_f1"], m["presence_micro_f1"] = presence_f1(tp_, pp_)
    both = (tp_ == 1) & (pp_ == 1)
    m["slot_direction_exact_pct"] = float(100.0 * (td_[both] == pd_[both]).mean()) if both.any() else float("nan")
    return {c: m.get(c, float("nan")) for c in METRIC_COLUMNS}


def heldout_tex(table: pd.DataFrame) -> str:
    """Compact LaTeX table (percent 1 decimal, kappa 2 decimals)."""
    def f1(x):
        return "--" if pd.isna(x) else f"{x:.1f}"

    def f2(x):
        return "--" if pd.isna(x) else f"{x:.2f}"

    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Evaluation & $n$ & Func.\ acc.\ (\%) & Func.\ $\kappa$ & Topic acc.\ (\%) & "
             r"Dir.\ exact (\%) & Dir.\ $\pm1$ (\%) & Dir.\ sign (\%) & Dir.\ QWK & "
             r"Full match (\%) & Presence macro-F1 \\", r"\midrule"]
    for r in table.itertuples():
        name = str(r.evaluation).replace("_", r"\_")
        lines.append(f"{name} & {int(r.n_items)} & {f1(r.function_pct)} & {f2(r.function_kappa)} & "
                     f"{f1(r.topic_pct)} & {f1(r.direction_exact_pct)} & {f1(r.direction_within1_pct)} & "
                     f"{f1(r.direction_sign_pct)} & {f2(r.direction_qwk)} & {f1(r.full_match_pct)} & "
                     f"{f2(r.presence_macro_f1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- training data


def load_train_labels(cfg: dict, labeler: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Valid labels of the ``train`` set -> DataFrame(id, prev_text, text, first_release_date, label).

    Every batch output is checked with schema.validate_output; invalid or missing files
    are skipped with a warning (their items are simply not used).
    """
    root = cfgmod.repo_root()
    labeler = labeler or cfg["labelling"]["train"]["labeler"]
    items = pd.read_csv(root / cfg["paths"]["labels_train"] / "train_items.csv",
                        dtype=str, keep_default_na=False)
    labels: dict[str, dict] = {}
    n_batches = n_bad = 0
    for bp in batch_paths("train"):
        n_batches += 1
        op = output_path("train", labeler, bp)
        rep = validate_output(bp, op)
        if rep.ok:
            labels.update(rep.labels)
        else:
            n_bad += 1
            print(f"[distill] WARNING: skipping {op} ({rep.summary().splitlines()[0]})")
    if verbose:
        print(f"[distill] train labels from {labeler!r}: {len(labels)} labelled items "
              f"({n_batches - n_bad}/{n_batches} batch files valid)")
    items = items[items["id"].isin(labels)].copy()
    items["label"] = items["id"].map(labels)
    return items.reset_index(drop=True)


def load_smoke_labels(cfg: dict, labeler: str = "sonnet") -> pd.DataFrame:
    """The labels that exist before the real run: pilot rounds (sonnet) + gold majority."""
    root = cfgmod.repo_root()
    pieces = []
    for k in range(1, len(cfg["labelling"]["pilot"]["codebook_versions"]) + 1):
        items = pd.read_csv(root / cfg["paths"]["labels_pilot"] / f"pilot_r{k}_items.csv",
                            dtype=str, keep_default_na=False)
        labels: dict[str, dict] = {}
        for bp in batch_paths(f"pilot_r{k}"):
            rep = validate_output(bp, output_path(f"pilot_r{k}", labeler, bp))
            if not rep.ok:
                print(f"[distill] WARNING: pilot_r{k} {labeler} output invalid, skipped")
                continue
            labels.update(rep.labels)
        items = items[items["id"].isin(labels)].copy()
        items["label"] = items["id"].map(labels)
        pieces.append(items)
    gold_items = pd.read_csv(root / cfg["paths"]["labels_gold"] / "gold_items.csv",
                             dtype=str, keep_default_na=False)
    gold = load_gold_majority(cfg)
    g = gold_items[gold_items["id"].isin(gold)].copy()
    g["label"] = g["id"].map(gold)
    pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    return out[["id", "prev_text", "text", "first_release_date", "label"]]


def load_gold_majority(cfg: dict) -> dict[str, dict]:
    """id -> gold label for items with a valid function majority (others are skipped)."""
    path = cfgmod.repo_root() / cfg["paths"]["labels_gold"] / "gold_majority.jsonl"
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            lab = {k: rec[k] for k in ("id", "aspects", "confidence")}
            if not validate_label(lab):
                out[rec["id"]] = lab
    return out


def label_statistics(labels: Sequence[Mapping]) -> dict:
    """Counts reported before training (slot prevalence, aspects, rule violations)."""
    present, direction, info = labels_to_arrays(labels)
    n = len(labels)
    n_asp = pd.Series([len(l["aspects"]) for l in labels]).value_counts().sort_index().to_dict()
    stats = {"n": n, "n_procedural": int((present.sum(1) == 0).sum()),
             "aspects_per_label": {int(k): int(v) for k, v in n_asp.items()},
             "duplicate_slot_cases": info["duplicate_slot"],
             "procedural_with_other_cases": info["procedural_with_other"],
             "slot_prevalence": {s: int(present[:, i].sum()) for i, s in enumerate(SLOT_NAMES)},
             "direction_counts": {int(d): int(((direction == d) & (present == 1)).sum()) for d in DIRECTIONS}}
    return stats


# ----------------------------------------------------------------------------- training


def _param_groups(model, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def validation_score(true_present, true_dir, p_pres, p_dir, threshold: float) -> dict:
    """Early-stopping score = mean(presence macro-F1, direction accuracy on truly present slots)."""
    macro, _ = presence_f1(true_present, (p_pres >= threshold).astype(int))
    mask = true_present == 1
    pred_d = np.argmax(p_dir, axis=-1) - 2
    acc = float((pred_d[mask] == true_dir[mask]).mean()) if mask.any() else float("nan")
    score = float(np.nanmean([macro, acc]))
    return {"val_presence_macro_f1": macro, "val_direction_acc": acc, "val_score": score}


def train_model(cfg: dict, enc: dict, present: np.ndarray, direction: np.ndarray, split: np.ndarray,
                ckpt: Path, device: str, epochs: int | None = None, grad_accum: int | None = None,
                tag: str = "") -> dict:
    """Fine-tune and save the best (by validation score) weights to `ckpt`; returns a log dict."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    dc = cfg["distill"]
    epochs = int(epochs or dc["epochs"])
    accum = int(grad_accum or dc["grad_accum"])
    bs = int(dc["batch_size"])
    seed = int(cfg["seed"])
    seed_everything(seed)
    pad_id = enc["pad_id"]
    seqs = enc["ids"]
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    if len(tr) == 0:
        raise ValueError("empty training split")
    model = make_distiller(dc["encoder"]).to(device)
    if dc.get("freeze_embeddings", False):
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
    if dc.get("grad_checkpointing", False):
        model.encoder.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(_param_groups(model, float(dc["weight_decay"])), lr=float(dc["lr"]))
    steps_per_epoch = math.ceil(math.ceil(len(tr) / bs) / accum)
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(float(dc["warmup_frac"]) * total), total)
    use_amp = bool(dc["amp"] and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(seed)
    pres_t = torch.tensor(present, dtype=torch.long)
    dir_t = torch.tensor(direction, dtype=torch.long)
    lam = float(dc["lambda_dir"])
    log: list[dict] = []
    best, best_epoch, bad = -math.inf, 0, 0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    print(f"[distill{tag}] train {len(tr)}, val {len(va)}, epochs {epochs}, batch {bs} x accum {accum}, "
          f"steps {total}, amp {use_amp}")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        batches = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        sums = np.zeros(3)
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        for k, b in enumerate(batches, 1):
            ids, mask = collate([seqs[i] for i in b], pad_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pl, dl = model(ids.to(device), mask.to(device))
            loss, bce, ce = distill_loss(pl, dl, pres_t[b].to(device), dir_t[b].to(device), lam)
            scaler.scale(loss / accum).backward()
            sums += np.array([loss.item(), bce.item(), ce.item()]) * len(b)
            n_seen += len(b)
            if k % accum == 0 or k == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        rec = {"epoch": epoch, "train_loss": sums[0] / n_seen, "train_bce": sums[1] / n_seen,
               "train_ce": sums[2] / n_seen, "elapsed_s": time.time() - t0}
        if len(va):
            p_pres, p_dir = predict(model, [seqs[i] for i in va], pad_id, device, batch_size=32, amp=use_amp)
            rec.update(validation_score(present[va], direction[va], p_pres, p_dir, float(dc["threshold"])))
            score = rec["val_score"]
        else:
            score = -rec["train_loss"]
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        rec["improved"] = bool(improved)
        log.append(rec)
        print(f"[distill{tag}] epoch {epoch}: loss {rec['train_loss']:.4f} (bce {rec['train_bce']:.4f}, "
              f"ce {rec['train_ce']:.4f})"
              + (f"  val presence F1 {rec['val_presence_macro_f1']:.3f}, dir acc {rec['val_direction_acc']:.3f}"
                 if len(va) else "") + f"  [{rec['elapsed_s']:.0f}s]{' *' if improved else ''}")
        if bad >= int(dc["patience"]):
            print(f"[distill{tag}] early stop (no improvement for {bad} epochs); best epoch {best_epoch}")
            break
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"epochs_run": len(log), "best_epoch": best_epoch, "best_val_score": best, "log": log,
            "runtime_s": time.time() - t0, "peak_gpu_gib": peak}


def load_trained(cfg: dict, ckpt: Path, encoder: str, device: str):
    import torch
    model = make_distiller(encoder, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


# ----------------------------------------------------------------------------- pipeline


class Run:
    """Paths and settings for one invocation (real or smoke)."""

    def __init__(self, cfg: dict, smoke: bool):
        self.cfg, self.smoke = cfg, smoke
        root = cfgmod.repo_root()
        dc = cfg["distill"]
        if smoke:
            base = root / "tmp" / "distill_smoke"
            self.model_dir, self.tables, self.out = base / "models", base / "tables", base / "distilled_labels.parquet"
            self.fallback = self.out
            self.encoder_name = dc["smoke_encoder"]
        else:
            self.model_dir, self.tables = root / dc["model_dir"], root / dc["tables"]
            self.out, self.fallback = root / dc["out"], root / dc["fallback_out"]
            self.encoder_name = dc["encoder"]
        for d in (self.model_dir, self.tables, self.out.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.device = pick_device()
        self.tok = None

    def resolve_encoder(self) -> None:
        """Smoke: roberta-base if obtainable, else FOMC-RoBERTa. Sets cfg['distill']['encoder']."""
        from transformers import AutoConfig, AutoTokenizer
        name = self.encoder_name
        try:
            self.tok = _from_pretrained(AutoTokenizer, name)
            _from_pretrained(AutoConfig, name)
        except Exception as e:  # noqa: BLE001
            if not self.smoke:
                raise
            print(f"[distill] WARNING: {name} unavailable ({type(e).__name__}); using {self.cfg['distill']['encoder']}")
            name = self.cfg["distill"]["encoder"]
            self.tok = _from_pretrained(AutoTokenizer, name)
        self.encoder_name = name
        self.cfg["distill"]["encoder"] = name
        print(f"[distill] encoder {name}, device {self.device}")

    def ckpt(self, which: str) -> Path:
        return self.model_dir / f"{which}.pt"


def _training_frame(run: Run) -> pd.DataFrame:
    df = load_smoke_labels(run.cfg) if run.smoke else load_train_labels(run.cfg)
    if df.empty:
        raise SystemExit("no training labels found (labels/out/train/<labeler>/ is empty or invalid)")
    return df


def _splits(run: Run, df: pd.DataFrame) -> dict[str, np.ndarray]:
    dc = run.cfg["distill"]
    rnd = random_split(len(df), dc["random_split"], int(run.cfg["seed"]) + 301)
    if run.smoke:
        return {"random": rnd}
    return {"year": year_split(df["first_release_date"], dc["test_years"], dc["val_years"]),
            "random": rnd}


def _describe_split(name: str, df: pd.DataFrame, split: np.ndarray) -> None:
    counts = pd.Series(split).value_counts().reindex(["train", "val", "test"], fill_value=0)
    extra = ""
    if name == "year":
        yrs = pd.to_datetime(df["first_release_date"]).dt.year
        extra = "; test years " + str(sorted(set(yrs[split == "test"]))) + ", val years " + str(sorted(set(yrs[split == "val"])))
    print(f"[distill] {name} split: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + extra)


def stage_train(run: Run) -> None:
    cfg = run.cfg
    dc = cfg["distill"]
    run.resolve_encoder()
    df = _training_frame(run)
    labels = df["label"].tolist()
    stats = label_statistics(labels)
    print("[distill] label statistics: " + json.dumps(stats))
    present, direction, _ = labels_to_arrays(labels)
    enc = encode_items(run.tok, df["prev_text"].tolist(), df["text"].tolist(), int(dc["max_length"]))
    enc["pad_id"] = run.tok.pad_token_id
    n_tr, n_tx = int(enc["truncated"].sum()), int(enc["text_truncated"].sum())
    print(f"[distill] truncation at max_length={dc['max_length']}: {n_tr}/{len(df)} inputs truncated "
          f"({n_tx} of them cut into the target sentence); max length {int(enc['n_full'].max())}")
    runs = {}
    for name, split in _splits(run, df).items():
        _describe_split(name, df, split)
        pd.DataFrame({"id": df["id"], "split": split}).to_csv(run.model_dir / f"split_{name}.csv", index=False)
        epochs = int(dc["smoke_epochs"]) if run.smoke else None
        accum = 1 if run.smoke else None
        runs[name] = train_model(cfg, enc, present, direction, split, run.ckpt(name), run.device,
                                 epochs=epochs, grad_accum=accum, tag=f":{name}")
        print(f"[distill] {name}: best epoch {runs[name]['best_epoch']}, val score "
              f"{runs[name]['best_val_score']:.3f}, {runs[name]['runtime_s']:.0f}s, "
              f"peak GPU {runs[name]['peak_gpu_gib']:.2f} GiB")
    meta = {"encoder": run.encoder_name, "stats": stats, "truncated": n_tr, "text_truncated": n_tx,
            "runs": runs}
    (run.model_dir / "train_log.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")


def _encode(run: Run, prevs, texts) -> dict:
    if run.tok is None:
        run.resolve_encoder()
    enc = encode_items(run.tok, list(prevs), list(texts), int(run.cfg["distill"]["max_length"]))
    enc["pad_id"] = run.tok.pad_token_id
    return enc


def _predict_labels(run: Run, model, df: pd.DataFrame, enc_cache: dict | None = None):
    dc = run.cfg["distill"]
    enc = enc_cache or _encode(run, df["prev_text"], df["text"])
    p_pres, p_dir = predict(model, enc["ids"], enc["pad_id"], run.device, batch_size=int(dc["infer_batch_size"]))
    labels = derive_labels(p_pres, p_dir, df["id"].tolist(), float(dc["threshold"]))
    return {l["id"]: l for l in labels}, p_pres, p_dir


def stage_eval(run: Run) -> pd.DataFrame:
    cfg = run.cfg
    run.resolve_encoder()
    df = _training_frame(run)
    truth = dict(zip(df["id"], df["label"]))
    rows = []
    slot_tabs = []
    for name, split in _splits(run, df).items():
        if not run.ckpt(name).is_file():
            print(f"[distill] no checkpoint for {name!r} split; skipped")
            continue
        test = df[split == "test"].reset_index(drop=True)
        model = load_trained(cfg, run.ckpt(name), run.encoder_name, run.device)
        pred, _, _ = _predict_labels(run, model, test)
        del model
        m = evaluate_labels(truth, pred)
        label = ("year_heldout_test" if name == "year" else
                 "random_split_test (optimistic: near-duplicates leak)")
        if run.smoke:
            label = "smoke_random_split_test"
        rows.append({"evaluation": label, "split": name, "encoder": run.encoder_name,
                     "n_train": int((split == "train").sum()), "n_val": int((split == "val").sum()), **m})
        tp_, _, _ = labels_to_arrays([truth[i] for i in test["id"]])
        pp_, _, _ = labels_to_arrays([pred[i] for i in test["id"]])
        st = slot_presence_table(tp_, pp_)
        st.insert(0, "split", name)
        slot_tabs.append(st)
    table = pd.DataFrame(rows)
    if rows:
        table.to_csv(run.tables / "distill_heldout.csv", index=False, lineterminator="\n")
        (run.tables / "distill_heldout.tex").write_text(heldout_tex(table), encoding="utf-8")
        pd.concat(slot_tabs).to_csv(run.tables / "distill_slots_heldout.csv", index=False, lineterminator="\n")
        print("\n" + table.drop(columns=["encoder"]).round(3).to_string(index=False))
    stage_eval_gold(run)
    return table


def stage_eval_gold(run: Run) -> pd.DataFrame | None:
    """Distilled model (year-split, i.e. production) vs the gold majority. Gold uids are never in training."""
    cfg = run.cfg
    which = "random" if run.smoke else "year"
    if not run.ckpt(which).is_file():
        return None
    root = cfgmod.repo_root()
    gold = load_gold_majority(cfg)
    items = pd.read_csv(root / cfg["paths"]["labels_gold"] / "gold_items.csv", dtype=str, keep_default_na=False)
    items = items[items["id"].isin(gold)].reset_index(drop=True)
    model = load_trained(cfg, run.ckpt(which), run.encoder_name, run.device)
    pred, _, _ = _predict_labels(run, model, items)
    del model
    m = evaluate_labels(gold, pred, pred_primary_only=True)
    note = ("gold items are IN the smoke training set: not an external check" if run.smoke
            else "gold items excluded from training: external check")
    row = {"evaluation": "distilled_vs_gold_majority" + (" (smoke, in-training)" if run.smoke else ""),
           "encoder": run.encoder_name, "note": note, "presence_f1_basis": "primary aspect only", **m}
    table = pd.DataFrame([row])
    table.to_csv(run.tables / "distill_vs_gold.csv", index=False, lineterminator="\n")
    (run.tables / "distill_vs_gold.tex").write_text(heldout_tex(table), encoding="utf-8")
    print("\n" + table.drop(columns=["encoder", "note"]).round(3).to_string(index=False))
    return table


def result_frame(uids: Sequence[str], p_pres: np.ndarray, p_dir: np.ndarray, threshold: float) -> pd.DataFrame:
    """distilled_labels table: uid, aspects (JSON), p_<slot>, e_<slot>, pdir_<slot>_<d>."""
    labels = derive_labels(p_pres, p_dir, list(uids), threshold)
    exp = expected_direction(p_dir)
    cols: dict[str, Any] = {"uid": list(uids),
                            "aspects": [json.dumps(l["aspects"], ensure_ascii=False) for l in labels]}
    for i, s in enumerate(SLOT_NAMES):
        cols[f"p_{s}"] = p_pres[:, i].astype(np.float32)
    for i, s in enumerate(SLOT_NAMES):
        cols[f"e_{s}"] = exp[:, i].astype(np.float32)
    for i, s in enumerate(SLOT_NAMES):
        for k, d in enumerate(DIRECTIONS):
            cols[f"pdir_{s}_{d:+d}"] = p_dir[:, i, k].astype(np.float32)
    return pd.DataFrame(cols)


def stage_infer(run: Run, n_limit: int | None = None) -> pd.DataFrame:
    """Predict the whole corpus (unique masked sentences) in fp32, batch size 1; write the parquet."""
    cfg = run.cfg
    dc = cfg["distill"]
    root = cfgmod.repo_root()
    run.resolve_encoder()
    which = "random" if run.smoke else "year"
    uniq = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences_unique.parquet")
    if run.smoke:
        n_limit = int(dc["smoke_infer_n"])
    if n_limit:
        uniq = uniq.sample(n=min(n_limit, len(uniq)), random_state=int(cfg["seed"])).sort_values("uid")
    uniq = uniq.reset_index(drop=True)
    seed_everything(int(cfg["seed"]))
    model = load_trained(cfg, run.ckpt(which), run.encoder_name, run.device)
    enc = _encode(run, uniq["prev_text_masked"].fillna(""), uniq["text_masked"].fillna(""))
    print(f"[distill] inference on {len(uniq)} unique sentences, fp32, batch size {dc['infer_batch_size']}; "
          f"truncated {int(enc['truncated'].sum())} ({int(enc['text_truncated'].sum())} into the target)")
    t0 = time.time()
    p_pres, p_dir = predict(model, enc["ids"], enc["pad_id"], run.device, batch_size=int(dc["infer_batch_size"]))
    runtime = time.time() - t0
    print(f"[distill] inference {runtime:.0f}s ({len(uniq) / runtime:.1f} sentences/s)")

    # determinism / batch independence on a seeded subset, in shuffled order
    rng = np.random.default_rng(int(cfg["seed"]) + 2)
    sub = rng.permutation(len(uniq))[:int(dc["batch_check_n"])]
    seqs = [enc["ids"][i] for i in sub]
    bs = int(dc["infer_batch_size"])
    p2, d2 = predict(model, seqs, enc["pad_id"], run.device, batch_size=bs)
    diff_rerun = float(max(np.abs(p2 - p_pres[sub]).max(), np.abs(d2 - p_dir[sub]).max()))
    p3, d3 = predict(model, seqs, enc["pad_id"], run.device, batch_size=16)
    diff_b16 = float(max(np.abs(p3 - p_pres[sub]).max(), np.abs(d3 - p_dir[sub]).max()))
    tol = float(dc["batch_check_tol"])
    print(f"[distill] check on {len(sub)} sentences: rerun (shuffled order, batch size {bs}) max |diff| = "
          f"{diff_rerun:.2e} (tol {tol:.0e}); for information, batch size 16 differs by {diff_b16:.2e}")
    if diff_rerun > tol and bs > 1:
        raise RuntimeError(f"outputs depend on batch composition: {diff_rerun} > {tol}")
    if diff_rerun > tol:
        raise RuntimeError(f"non-deterministic inference: {diff_rerun} > {tol}")
    del model

    out = result_frame(uniq["uid"], p_pres, p_dir, float(dc["threshold"]))
    out.to_parquet(run.out, index=False)
    mb = run.out.stat().st_size / 1e6
    if not run.smoke and mb >= float(dc["max_commit_mb"]):
        run.fallback.parent.mkdir(parents=True, exist_ok=True)
        run.out.replace(run.fallback)
        print(f"[distill] output is {mb:.1f} MB (>= {dc['max_commit_mb']}): moved to {run.fallback}; do not commit")
    else:
        print(f"[distill] wrote {run.out} ({mb:.1f} MB)")
    _summarise_output(out)
    return out


def _summarise_output(out: pd.DataFrame) -> None:
    funcs = out["aspects"].map(lambda s: json.loads(s)[0]["function"])
    n_asp = out["aspects"].map(lambda s: len(json.loads(s)) if json.loads(s)[0]["function"] != "PROCEDURAL" else 0)
    print("[distill] predicted primary function shares:\n"
          + funcs.value_counts(normalize=True).round(3).to_string())
    print("[distill] predicted aspects per sentence: " + str(n_asp.value_counts().sort_index().to_dict()))


# ----------------------------------------------------------------------------- memory probe


def memory_probe(cfg: dict, n_opt_steps: int = 2) -> dict:
    """A few optimiser steps of the configured encoder with worst-case (max_length) batches."""
    import torch
    run = Run(cfg, smoke=True)
    run.encoder_name = cfg["distill"]["encoder"]
    run.resolve_encoder()
    dc = cfg["distill"]
    if not run.device.startswith("cuda"):
        raise SystemExit("memory probe needs a GPU")
    seed_everything(int(cfg["seed"]))
    df = load_smoke_labels(cfg)
    enc = encode_items(run.tok, df["prev_text"].tolist(), df["text"].tolist(), int(dc["max_length"]))
    present, direction, _ = labels_to_arrays(df["label"].tolist())
    bs, accum = int(dc["batch_size"]), int(dc["grad_accum"])
    model = make_distiller(run.encoder_name).to(run.device)
    if dc.get("freeze_embeddings", False):
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
    if dc.get("grad_checkpointing", False):
        model.encoder.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(_param_groups(model, float(dc["weight_decay"])), lr=float(dc["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(dc["amp"]))
    torch.cuda.reset_peak_memory_stats()
    model.train()
    pres_t = torch.tensor(present)
    dir_t = torch.tensor(direction)
    L = int(dc["max_length"])
    t0 = time.time()
    k = 0
    for step in range(n_opt_steps):
        for _ in range(accum):
            b = np.arange(k, k + bs) % len(df)
            k += bs
            # worst case: every sequence padded to the full max_length (real tokens, cycled to fill)
            seqs = [(enc["ids"][i] * (L // len(enc["ids"][i]) + 1))[:L] for i in b]
            ids, mask = collate(seqs, run.tok.pad_token_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(dc["amp"])):
                pl, dl = model(ids.to(run.device), mask.to(run.device))
            loss, bce, ce = distill_loss(pl, dl, pres_t[b].to(run.device), dir_t[b].to(run.device),
                                         float(dc["lambda_dir"]))
            scaler.scale(loss / accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        print(f"[probe] opt step {step + 1}: loss {loss.item():.4f}")
    torch.cuda.synchronize()
    res = {"encoder": run.encoder_name, "batch_size": bs, "grad_accum": accum, "seq_len": L,
           "opt_steps": n_opt_steps, "seconds": time.time() - t0,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    print("[probe] " + json.dumps(res))
    return res


# ----------------------------------------------------------------------------- entry points

ALL_STAGES = ("train", "eval", "infer")


def run_pipeline(cfg: dict, stages: Sequence[str] = ALL_STAGES, smoke: bool = False) -> None:
    run = Run(cfg, smoke)
    t0 = time.time()
    if "train" in stages:
        stage_train(run)
    if "eval" in stages:
        stage_eval(run)
    if "infer" in stages:
        stage_infer(run)
    print(f"[distill] done in {time.time() - t0:.0f}s")


def main(cfg: dict | None = None) -> None:
    """Stage entry point (run_all convention): train, evaluate and infer with the real labels."""
    run_pipeline(cfg or cfgmod.load_config(), ALL_STAGES, smoke=False)


def cli(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="650 pilot+gold labels, random split, outputs in tmp/")
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    ap.add_argument("--memory-probe", action="store_true", help="a few steps of the configured encoder; peak GPU memory")
    ap.add_argument("--encoder", help="override distill.encoder (e.g. roberta-base)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override distill.<key> (YAML value), e.g. --set epochs=3 batch_size=4")
    args = ap.parse_args(argv)
    import yaml
    cfg = cfgmod.load_config()
    if args.encoder:
        cfg["distill"]["encoder"] = args.encoder
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg["distill"][k] = yaml.safe_load(v)
    if args.memory_probe:
        memory_probe(cfg)
        return
    run_pipeline(cfg, args.stages, smoke=args.smoke)


if __name__ == "__main__":
    cli()
