"""Phase 5 baseline: sentence-level hawkish/dovish scores from FOMC-RoBERTa.

Stage ``score`` of run_all.py. Reads data/processed/sentences.parquet, scores the raw
(unmasked) ``text`` of every sentence occurrence with ``gtfintechlab/FOMC-RoBERTa``
(identical texts are scored once and mapped back) and writes one row per sent_id to
data/processed/roberta_sentence_scores.parquet:

    sent_id, uid, p_dovish, p_hawkish, p_neutral, score, label, n_tokens, truncated

score = P(hawkish) - P(dovish), in [-1, 1]. The label order of the classification head
(LABEL_0/1/2) lives in config.yaml (``scoring.label_map``), not here.
"""
from __future__ import annotations

import os
import time
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models come from the local HF cache

import numpy as np
import pandas as pd

from ftyields import config as cfgmod

CLASSES = ("dovish", "hawkish", "neutral")
SCORE_COLUMNS = ["sent_id", "uid", "p_dovish", "p_hawkish", "p_neutral", "score",
                 "label", "n_tokens", "truncated"]


# --------------------------------------------------------------------------- pure helpers

def class_indices(id2label: dict, label_map: dict) -> dict[str, int]:
    """Map class name -> logit column, from the model's id2label and the config label_map.

    ``id2label`` is the model config's {index: raw label} (e.g. {0: 'LABEL_0', ...});
    ``label_map`` is the config's {raw label: class name}.
    """
    out: dict[str, int] = {}
    for idx, raw in id2label.items():
        if raw not in label_map:
            raise KeyError(f"Model label {raw!r} is missing from scoring.label_map")
        name = label_map[raw]
        if name in out:
            raise ValueError(f"Class {name!r} mapped twice in scoring.label_map")
        out[name] = int(idx)
    if set(out) != set(CLASSES):
        raise ValueError(f"label_map must cover exactly {CLASSES}, got {sorted(out)}")
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def probs_to_scores(probs: np.ndarray, idx: dict[str, int]) -> pd.DataFrame:
    """(n, 3) class probabilities in model-column order -> named probabilities, score, label."""
    probs = np.asarray(probs, dtype=np.float64)
    df = pd.DataFrame({
        "p_dovish": probs[:, idx["dovish"]],
        "p_hawkish": probs[:, idx["hawkish"]],
        "p_neutral": probs[:, idx["neutral"]],
    })
    df["score"] = df["p_hawkish"] - df["p_dovish"]
    names = np.array(CLASSES)
    df["label"] = names[df[["p_dovish", "p_hawkish", "p_neutral"]].to_numpy().argmax(axis=1)]
    return df


def logits_to_scores(logits: np.ndarray, idx: dict[str, int]) -> pd.DataFrame:
    return probs_to_scores(softmax(logits), idx)


def dedup_texts(texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Unique texts (first-occurrence order) and, per input row, the index of its unique text."""
    first: dict[str, int] = {}
    inverse = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        inverse[i] = first.setdefault(t, len(first))
    return list(first), inverse


def truncated_flag(n_tokens_full: np.ndarray, max_length: int) -> np.ndarray:
    """True where the tokenised length (incl. special tokens) exceeds max_length."""
    return np.asarray(n_tokens_full) > max_length


def length_batches(n_tokens: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Index batches over the texts sorted by token length (longest first, stable)."""
    order = np.argsort(-np.asarray(n_tokens), kind="stable")
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


# --------------------------------------------------------------------------- model side

def load_model(model_name: str, device: str):
    """(tokenizer, eval-mode model on device). Offline: reads the local HF cache only."""
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(device)
    return tok, model


def token_lengths(tok, texts: Sequence[str], chunk: int = 4096) -> np.ndarray:
    """Untruncated token count (with special tokens) of each text."""
    out: list[int] = []
    for i in range(0, len(texts), chunk):
        enc = tok(list(texts[i:i + chunk]), add_special_tokens=True, truncation=False,
                  padding=False)["input_ids"]
        out.extend(len(x) for x in enc)
    return np.asarray(out, dtype=np.int64)


def predict_probs(tok, model, texts: Sequence[str], n_tokens: np.ndarray, *, max_length: int,
                  batch_size: int, fp16: bool, device: str) -> np.ndarray:
    """Softmax probabilities (n, 3), model column order, for ``texts`` (input order kept)."""
    import torch

    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float64)
    use_amp = bool(fp16 and device.startswith("cuda"))
    with torch.inference_mode():
        for b in length_batches(np.minimum(n_tokens, max_length), batch_size):
            enc = tok([texts[i] for i in b], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            probs[b] = softmax(logits.float().cpu().numpy())
    return probs


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- stage

def _output_path(cfg: dict[str, Any]):
    sc = cfg["scoring"]
    root = cfgmod.repo_root()
    return root / sc["scores_out"], root / sc["scores_fallback_out"]


def _summary(df: pd.DataFrame, sents: pd.DataFrame, cfg: dict[str, Any], runtime: float,
             n_unique: int, n_trunc_occ: int, n_trunc_uniq: int, max_len: int) -> None:
    sc = cfg["scoring"]
    thr = float(sc["strong_abs_score"])
    k = int(sc["summary_examples"])
    d = df.merge(sents[["sent_id", "doc_type", "text"]], on="sent_id")
    print(f"\n=== FOMC-RoBERTa scoring summary ===")
    print(f"sentences: {len(df)} occurrences, {n_unique} unique texts")
    print(f"runtime (inference on unique texts): {runtime:.1f}s, "
          f"{n_unique / runtime:.1f} unique sentences/s")
    print(f"truncated at max_length={max_len}: {n_trunc_occ} occurrences, "
          f"{n_trunc_uniq} unique texts")
    g = d.groupby("doc_type")["score"]
    tab = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(),
                        f"share|score|>{thr}": d.assign(s=d["score"].abs() > thr)
                        .groupby("doc_type")["s"].mean()})
    print("\nScore by doc_type:\n" + tab.round(3).to_string())
    print("\nLabel shares (all / by doc_type):")
    shares = pd.crosstab(d["doc_type"], d["label"], normalize="index")
    shares.loc["all"] = d["label"].value_counts(normalize=True)
    print(shares.round(3).to_string())
    uniq = d.drop_duplicates("text")
    pd.set_option("display.max_colwidth", 140)
    print(f"\nMost hawkish ({k} unique texts):")
    for _, r in uniq.nlargest(k, "score").iterrows():
        print(f"  {r.score:+.4f}  {r.text[:200]}")
    print(f"\nMost dovish ({k}):")
    for _, r in uniq.nsmallest(k, "score").iterrows():
        print(f"  {r.score:+.4f}  {r.text[:200]}")
    print(f"\nMost neutral ({k}; highest P(neutral)):")
    for _, r in uniq.nlargest(k, "p_neutral").iterrows():
        print(f"  P(neu)={r.p_neutral:.4f} score={r.score:+.4f}  {r.text[:200]}")


def main(cfg: dict[str, Any]) -> None:
    import torch

    sc = cfg["scoring"]
    root = cfgmod.repo_root()
    cfgmod.set_seeds(int(cfg["seed"]))
    max_len, bs = int(sc["max_length"]), int(sc["batch_size"])
    col = sc["score_text_col"]
    device = pick_device()

    sents = pd.read_parquet(root / cfg["paths"]["data_processed"] / "sentences.parquet")
    texts_all = sents[col].tolist()
    uniq, inverse = dedup_texts(texts_all)
    print(f"[score] {len(sents)} sentences, {len(uniq)} unique raw texts, device={device}")

    tok, model = load_model(sc["roberta_model"], device)
    idx = class_indices(model.config.id2label, sc["label_map"])
    print(f"[score] label columns: {idx}")

    t0 = time.time()
    n_full = token_lengths(tok, uniq)
    print(f"[score] tokenised lengths in {time.time() - t0:.1f}s; max={n_full.max()}")
    trunc_u = truncated_flag(n_full, max_len)
    n_used = np.minimum(n_full, max_len)

    # --- fp16 vs fp32 check on a seeded random subset
    fp16 = bool(sc["fp16"]) and device.startswith("cuda")
    if fp16:
        rng = np.random.default_rng(int(cfg["seed"]))
        sub = rng.choice(len(uniq), size=min(int(sc["fp16_check_n"]), len(uniq)), replace=False)
        sub_t = [uniq[i] for i in sub]
        kw = dict(max_length=max_len, batch_size=bs, device=device)
        p16 = predict_probs(tok, model, sub_t, n_used[sub], fp16=True, **kw)
        p32 = predict_probs(tok, model, sub_t, n_used[sub], fp16=False, **kw)
        s16 = probs_to_scores(p16, idx)["score"].to_numpy()
        s32 = probs_to_scores(p32, idx)["score"].to_numpy()
        max_diff = float(np.abs(s16 - s32).max())
        print(f"[score] fp16 vs fp32 on {len(sub)} sentences: max |score diff| = {max_diff:.6f}")
        if max_diff >= float(sc["fp16_max_abs_diff"]):
            print("[score] diff too large -> using fp32")
            fp16 = False
    print(f"[score] precision: {'fp16 autocast' if fp16 else 'fp32'}")

    # --- full inference on unique texts
    t0 = time.time()
    probs_u = predict_probs(tok, model, uniq, n_full, max_length=max_len, batch_size=bs,
                            fp16=fp16, device=device)
    runtime = time.time() - t0

    # --- determinism: rescore a batch-aligned slice of the sorted order (identical batches)
    n_chk = int(sc["determinism_check_n"])
    order = np.concatenate(length_batches(n_used, bs))
    chk = order[: max(bs, (n_chk // bs) * bs)]
    pc = predict_probs(tok, model, [uniq[i] for i in chk], n_used[chk], max_length=max_len,
                       batch_size=bs, fp16=fp16, device=device)
    det_diff = float(np.abs(pc - probs_u[chk]).max())
    print(f"[score] rerun on {len(chk)} sentences (same batches): max |prob diff| = {det_diff:.2e}")
    if det_diff > float(sc["determinism_tol"]):
        raise RuntimeError(f"Non-deterministic scoring: max diff {det_diff} > {sc['determinism_tol']}")

    # --- batch-composition sensitivity (informational): random subset, different batches
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    rs = rng.choice(len(uniq), size=min(n_chk, len(uniq)), replace=False)
    pr = predict_probs(tok, model, [uniq[i] for i in rs], n_used[rs], max_length=max_len,
                       batch_size=bs, fp16=fp16, device=device)
    print(f"[score] batch-composition sensitivity on {len(rs)} random sentences: "
          f"max |prob diff| = {np.abs(pr - probs_u[rs]).max():.2e}")

    # --- map back to one row per sent_id
    scored_u = probs_to_scores(probs_u, idx)
    scored_u["n_tokens"] = n_full
    scored_u["truncated"] = trunc_u
    out = scored_u.iloc[inverse].reset_index(drop=True)
    out.insert(0, "uid", sents["uid"].to_numpy())
    out.insert(0, "sent_id", sents["sent_id"].to_numpy())
    out = out[SCORE_COLUMNS]
    assert len(out) == len(sents) and out["sent_id"].is_unique

    path, fallback = _output_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1e6
    if size_mb >= float(sc["scores_max_commit_mb"]):
        fallback.parent.mkdir(parents=True, exist_ok=True)
        path.replace(fallback)
        path = fallback
        print(f"[score] output is {size_mb:.1f} MB (>= limit): kept at {path}, do not commit")
    else:
        print(f"[score] wrote {path} ({size_mb:.1f} MB)")

    n_trunc_occ = int(out["truncated"].sum())
    _summary(out, sents, cfg, runtime, len(uniq),
             n_trunc_occ, int(trunc_u.sum()), max_len)
