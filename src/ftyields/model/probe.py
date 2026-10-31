"""Sentence probe: FOMC-RoBERTa score vs FinBERT top label on a few hand-picked sentences.

Writes results/tables/sentence_probe.csv and .tex (booktabs). Sentences and model names come
from config.yaml (``scoring.probe_sentences``, ``scoring.roberta_model``, ``scoring.finbert_model``).
Both models run in fp32 here (tiny input; no reason to lose precision).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ftyields import config as cfgmod
from ftyields.model import score as sc_mod


def latex_escape(s: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(repl.get(c, c) for c in s)


def _probs(tok, model, sentences, device, max_length) -> np.ndarray:
    n = sc_mod.token_lengths(tok, sentences)
    return sc_mod.predict_probs(tok, model, sentences, n, max_length=max_length,
                                batch_size=len(sentences), fp16=False, device=device)


def build_table(cfg: dict[str, Any]) -> pd.DataFrame:
    sc = cfg["scoring"]
    sentences = list(sc["probe_sentences"])
    device = sc_mod.pick_device()
    max_len = int(sc["max_length"])

    tok, model = sc_mod.load_model(sc["roberta_model"], device)
    idx = sc_mod.class_indices(model.config.id2label, sc["label_map"])
    rob = sc_mod.probs_to_scores(_probs(tok, model, sentences, device, max_len), idx)
    del model

    ftok, fmodel = sc_mod.load_model(sc["finbert_model"], device)
    id2label = {int(k): v for k, v in fmodel.config.id2label.items()}
    print(f"[probe] FinBERT id2label (from model config): {id2label}")
    fp = _probs(ftok, fmodel, sentences, device, max_len)
    top = fp.argmax(axis=1)

    return pd.DataFrame({
        "sentence": sentences,
        "roberta_score": rob["score"].round(4),
        "roberta_p_dovish": rob["p_dovish"].round(4),
        "roberta_p_hawkish": rob["p_hawkish"].round(4),
        "roberta_p_neutral": rob["p_neutral"].round(4),
        "roberta_label": rob["label"],
        "finbert_label": [id2label[i] for i in top],
        "finbert_prob": fp[np.arange(len(top)), top].round(4),
    })


def to_latex(df: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrl}", r"\toprule",
             r"Sentence & RoBERTa $P(\mathrm{hawk})-P(\mathrm{dov})$ & FinBERT top label \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{latex_escape(r.sentence)} & {round(r.roberta_score, 2) + 0.0:+.2f} & "
                     f"{latex_escape(r.finbert_label)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main(cfg: dict[str, Any]) -> None:
    df = build_table(cfg)
    stem = cfgmod.repo_root() / cfg["scoring"]["probe_out"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(to_latex(df), encoding="utf-8", newline="\n")
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(df[["sentence", "roberta_score", "roberta_label", "finbert_label", "finbert_prob"]]
          .to_string(index=False))
    print(f"[probe] wrote {stem.with_suffix('.csv')} and .tex")


if __name__ == "__main__":
    main(cfgmod.load_config())
