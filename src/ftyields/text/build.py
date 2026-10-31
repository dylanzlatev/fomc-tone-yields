"""`text` stage: raw FOMC documents -> clean, masked, de-duplicated sentence tables.

Inputs : data/raw/text/fomc_documents.parquet, data/processed/fomc_events.csv
Outputs: data/processed/sentences.parquet, data/processed/sentences_unique.parquet,
         results/tables/sentence_counts.{csv,tex}, tmp/phase2_samples.md (gitignored)

Only documents linked from an event (events.doc_id) are used; release dates come from the
event (authoritative), never from the dataset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ftyields.config import load_config, path as cfg_path, repo_root
from ftyields.text import clean as C
from ftyields.text.mask import Masker, make_uid, normalize
from ftyields.text.sections import SectionTagger, doc_has_headings
from ftyields.text.split import split_sentences

SENT_COLS = ["sent_id", "doc_id", "doc_type", "event_id", "release_date", "sent_idx", "section", "text",
             "text_masked", "prev_text", "prev_text_masked", "uid", "n_words", "is_procedural"]


def _tcfg(cfg: dict[str, Any]) -> dict[str, Any]:
    d = {"min_fragment_words": 4, "min_sentence_words": 3, "uid_hex_len": 12, "sample_per_type": 30,
         "extreme_n": 15, "top_uids": 20, "masked_examples": 20}
    d.update({k: v for k, v in (cfg.get("text") or {}).items() if k in d})
    return d


def load_inputs(cfg: dict[str, Any]) -> pd.DataFrame:
    """Documents linked from an event, with event_id and authoritative release datetime."""
    docs = pd.read_parquet(cfg_path("data_raw", cfg) / "text" / "fomc_documents.parquet")
    ev = pd.read_csv(cfg_path("data_processed", cfg) / "fomc_events.csv", dtype={"doc_id": str})
    ev = ev[ev["doc_id"].notna() & (ev["doc_id"].str.len() > 0)].copy()
    if ev["doc_id"].duplicated().any():
        raise ValueError("doc_id linked from more than one event")
    m = ev.merge(docs[["doc_id", "doc_type", "text"]], on="doc_id", how="left", validate="one_to_one")
    if m["text"].isna().any():
        raise ValueError(f"events reference missing documents: {m.loc[m['text'].isna(), 'doc_id'].tolist()[:5]}")
    bad = m[m["doc_type"] != m["event_type"]]
    if len(bad):
        raise ValueError(f"doc_type != event_type for {bad['doc_id'].tolist()[:5]}")
    m["release_dt"] = pd.to_datetime(m["release_datetime_et"])
    m["release_date"] = m["release_dt"].dt.strftime("%Y-%m-%d")
    return m.sort_values(["release_dt", "doc_id"]).reset_index(drop=True)


def document_sentences(cd: C.CleanDoc, doc_type: str, min_fragment_words: int = 4,
                       min_sentence_words: int = 3) -> list[dict[str, Any]]:
    """Sentences (text, section, is_procedural) of one cleaned document, in order."""
    out: list[dict[str, Any]] = []
    tagger = SectionTagger(has_headings=doc_has_headings([b.text for b in cd.blocks if b.kind == "heading"]))
    for b in cd.blocks:
        if b.kind == "heading":
            if doc_type == "minutes":
                tagger.heading(b.text)
            continue
        para = C.strip_leading(b.text)
        if not para:
            continue
        if doc_type == "minutes":
            tagger.paragraph(para)
        for s in split_sentences(para, min_fragment_words):
            s = C.drop_sentence(C.strip_leading(s))
            if s is None or len(s.split()) < min_sentence_words:
                continue
            if doc_type == "statement":
                out.append({"text": s, "section": "statement", "is_procedural": False})
            else:
                out.append({"text": s, "section": tagger.section, "is_procedural": tagger.procedural})
    return out


def build_sentences(docs: pd.DataFrame, tc: dict[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    cleaned = {r.doc_id: C.clean_document(r.text, r.doc_type) for r in docs.itertuples()}
    names: set[str] = set()
    for cd in cleaned.values():
        names |= cd.names
    masker = Masker(names)
    rows: list[dict[str, Any]] = []
    for r in docs.itertuples():
        sents = document_sentences(cleaned[r.doc_id], r.doc_type, tc["min_fragment_words"],
                                   tc["min_sentence_words"])
        prev, prev_m = "", ""
        for i, s in enumerate(sents):
            masked = masker(s["text"])
            rows.append({
                "sent_id": f"{r.doc_id}_{i:04d}", "doc_id": r.doc_id, "doc_type": r.doc_type,
                "event_id": r.event_id, "release_date": r.release_date, "sent_idx": i,
                "section": s["section"], "text": s["text"], "text_masked": masked,
                "prev_text": prev, "prev_text_masked": prev_m,
                "n_words": len(s["text"].split()), "is_procedural": s["is_procedural"],
            })
            prev, prev_m = s["text"], masked
    df = pd.DataFrame(rows)
    df["uid"] = [make_uid(normalize(t), tc["uid_hex_len"]) for t in df["text_masked"]]
    return df[SENT_COLS], names


def build_unique(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["release_date", "doc_id", "sent_idx"], kind="mergesort")
    first = d.groupby("uid", sort=False).first()
    g = d.groupby("uid", sort=False)
    def _types(x):
        s = set(x)
        return "both" if len(s) > 1 else next(iter(s))
    out = pd.DataFrame({
        "text_masked": first["text_masked"],
        "prev_text_masked": first["prev_text_masked"],
        "n_occurrences": g.size(),
        "n_docs": g["doc_id"].nunique(),
        "first_release_date": g["release_date"].min(),
        "doc_types": g["doc_type"].agg(_types),
        "sections": g["section"].agg(lambda x: "|".join(sorted(set(x)))),
    })
    return out.reset_index()[["uid", "text_masked", "prev_text_masked", "n_occurrences", "n_docs",
                              "first_release_date", "doc_types", "sections"]]


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(year=df["release_date"].str[:4])
    def agg(g: pd.DataFrame, dt: str, yr: str) -> dict[str, Any]:
        return {"doc_type": dt, "year": yr, "documents": g["doc_id"].nunique(), "sentences": len(g),
                "unique_uids": g["uid"].nunique(), "words_median": float(g["n_words"].median()),
                "words_p95": float(np.percentile(g["n_words"], 95))}
    rows = []
    for dt in ["statement", "minutes"]:
        sub = d[d["doc_type"] == dt]
        for yr, g in sub.groupby("year"):
            rows.append(agg(g, dt, yr))
        rows.append(agg(sub, dt, "all"))
    rows.append(agg(d, "all", "all"))
    return pd.DataFrame(rows)


def counts_tex(tab: pd.DataFrame) -> str:
    sel = tab[tab["year"] == "all"]
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Document type & Documents & Sentences & Unique & Words (median) & Words (p95) \\", r"\hline"]
    for r in sel.itertuples():
        lab = {"statement": "Statements", "minutes": "Minutes", "all": "All documents"}[r.doc_type]
        lines.append(f"{lab} & {r.documents:,} & {r.sentences:,} & {r.unique_uids:,} & "
                     f"{r.words_median:.0f} & {r.words_p95:.0f} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def section_distribution(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["doc_type"] == "minutes"].copy()
    yr = m["release_date"].str[:4].astype(int)
    m["era"] = pd.cut(yr, [1999, 2003, 2008, 2013, 2019, 2025], labels=["2000-03", "2004-08", "2009-13", "2014-19", "2020-25"])
    return pd.crosstab(m["era"], m["section"], normalize="index").round(3)


def _fmt(s: str, n: int = 700) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ..."


def write_samples(df: pd.DataFrame, uniq: pd.DataFrame, seed: int, tc: dict[str, Any], out: Path,
                  names: set[str]) -> None:
    L = ["# Phase 2 text samples", "", f"seed = {seed}; sentences = {len(df):,}; unique = {len(uniq):,}; "
         f"gazetteer names = {len(names)}", ""]
    for dt in ["statement", "minutes"]:
        sub = df[df["doc_type"] == dt]
        smp = sub.sample(min(tc["sample_per_type"], len(sub)), random_state=seed).sort_values("sent_id")
        L += [f"## {tc['sample_per_type']} random {dt} sentences", ""]
        for r in smp.itertuples():
            L += [f"- `{r.sent_id}` [{r.section}{' / procedural' if r.is_procedural else ''}]",
                  f"  - text: {_fmt(r.text)}", f"  - masked: {_fmt(r.text_masked)}"]
        L.append("")
    k = tc["extreme_n"]
    L += [f"## {k} longest sentences (words)", ""]
    for r in df.nlargest(k, "n_words").itertuples():
        L.append(f"- `{r.sent_id}` ({r.n_words} w) {_fmt(r.text, 500)}")
    L += ["", f"## {k} shortest sentences", ""]
    for r in df.nsmallest(k, "n_words").itertuples():
        L.append(f"- `{r.sent_id}` ({r.n_words} w) {_fmt(r.text, 300)}")
    L += ["", f"## {tc['top_uids']} most frequent uids", ""]
    for r in uniq.nlargest(tc["top_uids"], "n_occurrences").itertuples():
        L.append(f"- `{r.uid}` n={r.n_occurrences} docs={r.n_docs} ({r.doc_types}): {_fmt(r.text_masked, 300)}")
    mk = df[df["text_masked"].str.contains(r"\[(?:NAME|DATE|YEAR)\]")]
    L += ["", f"## {tc['masked_examples']} random masked sentences (contain a mask token)", ""]
    for r in mk.sample(min(tc["masked_examples"], len(mk)), random_state=seed).itertuples():
        L += [f"- `{r.sent_id}`", f"  - text: {_fmt(r.text, 400)}", f"  - masked: {_fmt(r.text_masked, 400)}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_config()
    tc = _tcfg(cfg)
    docs = load_inputs(cfg)
    df, names = build_sentences(docs, tc)
    uniq = build_unique(df)
    out_dir = cfg_path("data_processed", cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "sentences.parquet", index=False, compression="zstd")
    uniq.to_parquet(out_dir / "sentences_unique.parquet", index=False, compression="zstd")
    tab = count_table(df)
    tdir = cfg_path("results_tables", cfg)
    tdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tdir / "sentence_counts.csv", index=False)
    (tdir / "sentence_counts.tex").write_text(counts_tex(tab), encoding="utf-8")
    write_samples(df, uniq, int(cfg["seed"]), tc, repo_root() / "tmp" / "phase2_samples.md", names)
    print(f"documents: {docs['doc_type'].value_counts().to_dict()}")
    print(f"sentences: {len(df):,} total, {len(uniq):,} unique")
    print(tab[tab["year"] == "all"].to_string(index=False))
    print(section_distribution(df).to_string())
    print(f"gazetteer names: {len(names)}")


if __name__ == "__main__":
    main()
