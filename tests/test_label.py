"""Tests for ftyields.label (synthetic data only, no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from ftyields.label import agreement, batches, cli, sample, schema


# ---------------------------------------------------------------- helpers

def L(id_, *aspects, conf="high"):
    return {"id": id_, "aspects": [{"function": f, "topic": t, "direction": d} for f, t, d in aspects],
            "confidence": conf}


def write_lines(path, recs, fence=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r) if not isinstance(r, str) else r for r in recs) + "\n"
    if fence:
        body = "```json\n" + body + "```\n"
    path.write_text(body, encoding="utf-8")


def items(n=5):
    return pd.DataFrame({"id": [f"s{i}" for i in range(n)],
                         "prev_text": [f"prev {i} é" for i in range(n)],
                         "text": [f"text {i}" for i in range(n)]})


@pytest.fixture
def batch(tmp_path):
    p = tmp_path / "batch_0001.jsonl"
    write_lines(p, [{"id": f"a{i}", "prev_text": "", "text": f"t{i}"} for i in range(3)])
    return p


def good(i):
    return L(i, ("ASSESSMENT", "LABOR", 1))


def check(batch, tmp_path, recs, fence=False):
    out = tmp_path / "out.jsonl"
    write_lines(out, recs, fence=fence)
    return schema.validate_output(batch, out)


# ---------------------------------------------------------------- validator

def test_valid_output_passes(batch, tmp_path):
    r = check(batch, tmp_path, [good("a0"), good("a1"),
                                L("a2", ("POLICY", None, -1), ("PROCEDURAL", None, 0))])
    assert r.ok and r.n_valid == 3 and not r.notes


def test_missing_and_extra_id(batch, tmp_path):
    r = check(batch, tmp_path, [good("a0"), good("a1"), good("zz")])
    assert not r.ok
    assert "missing" in r.errors["a2"][0]
    assert "extra" in r.errors["zz"][0]


def test_duplicate_id(batch, tmp_path):
    r = check(batch, tmp_path, [good("a0"), good("a0"), good("a1"), good("a2")])
    assert not r.ok and "appears 2 times" in r.errors["a0"][0]


@pytest.mark.parametrize("label,frag", [
    (L("a0", ("BOGUS", None, 0)), "bad function"),
    (L("a0", ("POLICY", "LABOR", 1)), "must have topic null"),
    (L("a0", ("ASSESSMENT", None, 1)), "needs topic"),
    (L("a0", ("ASSESSMENT", "WEATHER", 1)), "needs topic"),
    (L("a0", ("ASSESSMENT", "LABOR", "1")), "JSON integer"),
    (L("a0", ("ASSESSMENT", "LABOR", True)), "JSON integer"),
    (L("a0", ("ASSESSMENT", "LABOR", 1.0)), "JSON integer"),
    (L("a0", ("ASSESSMENT", "LABOR", 3)), "outside"),
    (L("a0", ("ASSESSMENT", "LABOR", 1), ("OUTLOOK", "LABOR", 1), ("POLICY", None, 1)), "3 aspects"),
    (L("a0"), "0 aspects"),
    (L("a0", ("PROCEDURAL", None, 1)), "PROCEDURAL direction must be 0"),
    (L("a0", ("ASSESSMENT", "LABOR", 1), conf="certain"), "bad confidence"),
])
def test_validator_catches(batch, tmp_path, label, frag):
    r = check(batch, tmp_path, [label, good("a1"), good("a2")])
    assert not r.ok
    assert any(frag in e for e in r.errors["a0"]), r.errors


def test_extra_and_missing_keys(batch, tmp_path):
    bad = good("a0")
    bad["note"] = "x"
    del_conf = good("a1")
    del del_conf["confidence"]
    r = check(batch, tmp_path, [bad, del_conf, good("a2")])
    assert "unexpected keys" in r.errors["a0"][0]
    assert "missing keys" in r.errors["a1"][0]
    assert schema.validate_label(bad, allow_extra_keys=True) == []


def test_invalid_json_and_blank_lines(batch, tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text(json.dumps(good("a0")) + "\n\n{not json}\n" + json.dumps(good("a1")) + "\n"
                   + json.dumps(good("a2")) + "\n", encoding="utf-8")
    r = schema.validate_output(batch, out)
    assert not r.ok
    assert any("blank line" in e for e in r.global_errors)
    assert any("invalid JSON" in e for e in r.global_errors)


def test_fenced_output_accepted_with_note(batch, tmp_path):
    r = check(batch, tmp_path, [good("a0"), good("a1"), good("a2")], fence=True)
    assert r.ok and any("fence" in n for n in r.notes)


def test_trailing_whitespace_and_crlf_accepted(batch, tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_bytes(("\r\n".join(json.dumps(good(f"a{i}")) + "  " for i in range(3)) + "\r\n\r\n  \n").encode())
    assert schema.validate_output(batch, out).ok


def test_unbalanced_fence_and_empty_and_missing(batch, tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text("```json\n" + json.dumps(good("a0")) + "\n", encoding="utf-8")
    assert not schema.validate_output(batch, out).ok
    out.write_text("", encoding="utf-8")
    assert not schema.validate_output(batch, out).ok
    assert not schema.validate_output(batch, tmp_path / "nope.jsonl").ok


# ---------------------------------------------------------------- batching

def test_batching_deterministic_and_order_independent(tmp_path):
    df = items(7)
    p1 = batches.make_batches(df, "setA", 3, batches_dir=tmp_path / "b1")
    p2 = batches.make_batches(df.sample(frac=1, random_state=1), "setA", 3, batches_dir=tmp_path / "b2")
    assert [p.name for p in p1] == ["batch_0001.jsonl", "batch_0002.jsonl", "batch_0003.jsonl"]
    for a, b in zip(p1, p2):
        assert a.read_bytes() == b.read_bytes()
    man = pd.read_csv(tmp_path / "b1" / "setA" / "manifest.csv")
    assert list(man.columns) == ["batch", "n_items", "first_id", "last_id", "sha256"]
    assert list(man["n_items"]) == [3, 3, 1]
    assert man.loc[0, "first_id"] == "s0" and man.loc[2, "last_id"] == "s6"
    first = p1[0].read_text(encoding="utf-8").splitlines()[0]
    assert "é" in first and json.loads(first)["id"] == "s0"


def test_batching_idempotent_and_refuses_overwrite(tmp_path):
    d = tmp_path / "b"
    df = items(4)
    p = batches.make_batches(df, "s", 2, batches_dir=d)
    before = [x.read_bytes() for x in p]
    assert batches.make_batches(df, "s", 2, batches_dir=d) == p   # identical: no-op
    changed = df.copy()
    changed.loc[0, "text"] = "different"
    with pytest.raises(FileExistsError):
        batches.make_batches(changed, "s", 2, batches_dir=d)
    with pytest.raises(FileExistsError):
        batches.make_batches(df, "s", 3, batches_dir=d)
    assert [x.read_bytes() for x in p] == before


def test_batching_input_validation(tmp_path):
    df = items(3)
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError):
        batches.make_batches(dup, "s", 2, batches_dir=tmp_path)
    with pytest.raises(ValueError):
        batches.make_batches(df, "../evil", 2, batches_dir=tmp_path)
    renamed = df.rename(columns={"id": "uid", "text": "text_masked", "prev_text": "prev_text_masked"})
    assert len(batches.make_batches(renamed, "ok", 5, batches_dir=tmp_path)) == 1


def test_pending_and_rejected(tmp_path):
    bd, od = tmp_path / "b", tmp_path / "o"
    paths = batches.make_batches(items(4), "s", 2, batches_dir=bd)
    assert batches.pending("s", "lab", bd, od) == paths
    # valid output for batch 1, invalid (missing id) for batch 2
    ids1 = [json.loads(l)["id"] for l in paths[0].read_text(encoding="utf-8").splitlines()]
    ids2 = [json.loads(l)["id"] for l in paths[1].read_text(encoding="utf-8").splitlines()]
    write_lines(od / "s" / "lab" / "batch_0001.jsonl", [good(i) for i in ids1])
    bad_file = od / "s" / "lab" / "batch_0002.jsonl"
    write_lines(bad_file, [good(ids2[0])])
    assert batches.pending("s", "lab", bd, od) == [paths[1]]
    assert not bad_file.exists()
    rej = od / "s" / "lab" / "rejected" / "batch_0002.1.jsonl"
    assert rej.is_file()
    # a second invalid output gets k=2; nothing is overwritten
    write_lines(bad_file, [good(ids2[1])])
    assert batches.pending("s", "lab", bd, od) == [paths[1]]
    assert (od / "s" / "lab" / "rejected" / "batch_0002.2.jsonl").is_file() and rej.is_file()
    # now a valid one
    write_lines(bad_file, [good(i) for i in ids2])
    assert batches.pending("s", "lab", bd, od) == []
    assert bad_file.is_file()


# ---------------------------------------------------------------- sampling

def synth_sentences(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2000-01-01") + pd.to_timedelta(rng.integers(0, 26 * 365, n), unit="D")
    texts = [("The Committee decided to lower the target range." if rng.random() < 0.15
              else f"Output grew moderately {i}.") for i in range(n)]
    sections = rng.choice(["policy_action", "staff_economic", "participants_views",
                           "staff_economic;participants_views"], n, p=[.1, .4, .4, .1])
    dt = rng.choice(["statement", "minutes", "both"], n, p=[.2, .7, .1])
    return pd.DataFrame({"uid": [f"u{i:05d}" for i in range(n)], "text_masked": texts,
                         "prev_text_masked": "", "n_occurrences": 1, "n_docs": 1,
                         "first_release_date": dates, "doc_types": dt, "sections": sections})


def test_policy_like():
    assert sample.policy_like("The Committee decided to raise rates")
    assert sample.policy_like("a more ACCOMMODATIVE stance")
    assert not sample.policy_like("Unemployment fell in March.")
    assert not sample.policy_like(None)


def test_eras_and_sections():
    assert sample.era_of("2007-12-31") == "2000-2007"
    assert sample.era_of("2008-01-01") == "2008-2015"
    assert sample.era_of(pd.Timestamp("2025-06-01")) == "2016-2025"
    assert sample.section_group("a;b") == "multiple" and sample.section_group("a") == "a"
    assert sample.section_group(None) == "unknown"


def test_sampler_deterministic_and_order_independent():
    df = synth_sentences()
    a = sample.stratified_sample(df, 80, seed=1)
    b = sample.stratified_sample(df.sample(frac=1, random_state=3), 80, seed=1)
    c = sample.stratified_sample(df, 80, seed=2)
    assert len(a) == 80 and a["uid"].is_unique
    assert list(a["uid"]) == list(b["uid"])
    assert list(a["uid"]) != list(c["uid"])


def test_sampler_exclusion():
    df = synth_sentences()
    pilot = sample.pilot_sample(df, n=50, seed=5)
    gold = sample.gold_sample(df, n=100, seed=5, exclude=pilot["uid"])
    assert len(gold) == 100 and not set(gold["uid"]) & set(pilot["uid"])
    with pytest.raises(ValueError):
        sample.stratified_sample(df.head(10), 20, seed=1)


def test_sampler_strata_floor():
    df = synth_sentences()
    # a tiny stratum: 3 sentences in a unique section/era combination
    tiny = df.index[:3]
    df.loc[tiny, ["sections", "doc_types"]] = ["rare_section", "both"]
    df.loc[tiny, "first_release_date"] = pd.Timestamp("2010-05-05")
    s = sample.stratified_sample(df, 120, seed=4, floor=2)
    full = sample.add_strata_columns(df)
    sam = sample.add_strata_columns(s)
    counts = sam.groupby(["doc_type_group", "section_group", "era"]).size()
    sizes = full.groupby(["doc_type_group", "section_group", "era"]).size()
    assert len(counts) == len(sizes)                     # every non-empty stratum represented
    assert (counts >= np.minimum(2, sizes.loc[counts.index])).all()
    assert counts.loc[("both", "rare_section", "2008-2015")] == 2
    with pytest.raises(ValueError):
        sample.stratified_sample(df, 5, seed=4, floor=2)  # floors alone exceed n


def test_policy_oversampling_raises_share():
    df = synth_sentences(2000)
    base = sample.stratified_sample(df, 200, seed=7, floor=0)["text_masked"].map(sample.policy_like).mean()
    over = sample.stratified_sample(df, 200, seed=7, floor=0, policy_oversample=4.0)["text_masked"] \
        .map(sample.policy_like).mean()
    assert over > base + 0.1


def test_user_weights_column():
    df = synth_sentences(300)
    df["w"] = 0.0
    df.loc[df.index[:100], "w"] = 1.0
    s = sample.stratified_sample(df, 50, seed=1, weights="w", floor=0)
    assert set(s["uid"]) <= set(df["uid"].iloc[:100])


# ---------------------------------------------------------------- statistics

def test_cohen_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    a = rng.choice(["A", "B", "C"], 200)
    b = np.where(rng.random(200) < 0.6, a, rng.choice(["A", "B", "C"], 200))
    assert agreement.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    x = rng.integers(-2, 3, 300)
    y = np.clip(x + rng.integers(-1, 2, 300), -2, 2)
    assert agreement.cohen_kappa(x, y, categories=range(-2, 3), weights="quadratic") == pytest.approx(
        cohen_kappa_score(x, y, weights="quadratic", labels=list(range(-2, 3))))
    assert np.isnan(agreement.cohen_kappa(["A"] * 5, ["A"] * 5))


def test_fleiss_kappa_textbook():
    # Wikipedia worked example: 10 subjects, 14 raters, 5 categories -> kappa = 0.210
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                       [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                       [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert agreement.fleiss_kappa(counts) == pytest.approx(0.210, abs=5e-4)
    # equals Cohen's kappa-like perfect agreement
    assert agreement.fleiss_kappa(np.array([[3, 0], [0, 3], [3, 0]])) == pytest.approx(1.0)
    # label-list interface gives the same answer
    rows = [[1, 1], [1, 2], [2, 2], [1, 1]]
    c = np.array([[2, 0], [1, 1], [0, 2], [2, 0]])
    assert agreement.fleiss_kappa_labels(rows) == pytest.approx(agreement.fleiss_kappa(c))


def test_krippendorff_alpha_textbook():
    # Krippendorff (2011) "Computing Krippendorff's alpha-reliability": 4 coders, 12 units;
    # interval-level alpha = 0.849
    nan = np.nan
    data = np.array([
        [1, 1, nan, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
        [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [nan, 5, 5, 5], [nan, nan, 1, 1], [nan, 3, nan, nan]])
    assert agreement.krippendorff_alpha_interval(data) == pytest.approx(0.849, abs=5e-4)
    # perfect agreement and a sanity check of disagreement
    perfect = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    assert agreement.krippendorff_alpha_interval(perfect) == pytest.approx(1.0)
    assert agreement.krippendorff_alpha_interval(np.array([[1, 2], [2, 1], [1, 2], [2, 1]], float)) < 0


# ---------------------------------------------------------------- agreement / majority

def test_primary_aspect():
    assert agreement.primary(L("x", ("ASSESSMENT", "LABOR", 1), ("POLICY", None, -2))) == ("POLICY", None, -2)
    # tie -> first listed
    assert agreement.primary(L("x", ("ASSESSMENT", "LABOR", 1), ("POLICY", None, -1))) == ("ASSESSMENT", "LABOR", 1)
    assert agreement.primary(L("x", ("OUTLOOK", "GENERAL", 0), ("POLICY", None, 0)))[0] == "OUTLOOK"


def three_labelers():
    return {
        "a": {"1": L("1", ("ASSESSMENT", "LABOR", 1)), "2": L("2", ("POLICY", None, -1)),
              "3": L("3", ("PROCEDURAL", None, 0)), "4": L("4", ("OUTLOOK", "INFLATION", 2))},
        "b": {"1": L("1", ("ASSESSMENT", "LABOR", 2)), "2": L("2", ("POLICY", None, -1)),
              "3": L("3", ("ASSESSMENT", "GENERAL", 0)), "4": L("4", ("OUTLOOK", "ACTIVITY", 1))},
        "c": {"1": L("1", ("ASSESSMENT", "LABOR", 1)), "2": L("2", ("OUTLOOK", "GENERAL", -1)),
              "3": L("3", ("POLICY", None, 0)), "4": L("4", ("OUTLOOK", "INFLATION", 2))},
    }


def test_agreement_tables():
    t = agreement.agreement_tables(three_labelers())
    pw = t["pairwise"].set_index(["labeler_a", "labeler_b"])
    ab = pw.loc[("a", "b")]
    assert ab["n_items"] == 4
    assert ab["function_pct"] == 75.0                  # items 1,2,4 agree
    assert ab["topic_n_items"] == 2                    # items 1 and 4 (3: PROCEDURAL vs ASSESSMENT excluded)
    assert ab["topic_pct"] == 50.0
    assert ab["direction_exact_pct"] == 50.0           # items 2 and 3 equal; 1 (1v2) and 4 (2v1) differ
    assert ab["direction_within1_pct"] == 100.0
    assert ab["full_match_pct"] == 25.0                # only item 2 matches on function+topic+direction
    o = t["overall"].iloc[0]
    assert o["n_items"] == 4 and o["n_labelers"] == 3
    assert o["function_all_agree_pct"] == 50.0         # items 1 and 4
    fa = agreement.agreement_tables(three_labelers(), restrict_function_agree=True)
    assert fa["overall"].iloc[0]["n_items"] == 2
    assert fa["overall"].iloc[0]["function_all_agree_pct"] == 100.0


def test_disagreements_table():
    d = agreement.disagreements(three_labelers(), {"1": ("p", "text one")})
    assert set(d["id"]) == {"1", "2", "3", "4"}
    r1 = d.set_index("id").loc["1"]
    assert r1["differs_on"] == "direction" and r1["text"] == "text one"
    assert r1["a"] == "ASSESSMENT/LABOR/+1" and r1["b"] == "ASSESSMENT/LABOR/+2"


def test_majority_vote_basic_and_ties():
    mv = agreement.majority_vote(three_labelers()).set_index("id")
    assert mv.loc["1", "function"] == "ASSESSMENT" and mv.loc["1", "topic"] == "LABOR"
    assert mv.loc["1", "direction"] == 1                        # median of 1, 2, 1
    assert mv.loc["1", "confidence"] == "high"
    assert mv.loc["2", "function"] == "POLICY" and pd.isna(mv.loc["2", "topic"])
    assert mv.loc["2", "direction"] == -1 and mv.loc["2", "confidence"] == "med"
    # 3-way split on function: no majority -> null + flag
    assert pd.isna(mv.loc["3", "function"]) and "no_majority" in mv.loc["3", "flags"]
    # topic tie among labelers in the majority function (OUTLOOK x3: INFLATION, ACTIVITY, INFLATION)
    assert mv.loc["4", "topic"] == "INFLATION"
    labs = three_labelers()
    labs["c"]["4"] = L("4", ("OUTLOOK", "LABOR", 2))
    mv = agreement.majority_vote(labs).set_index("id")
    assert "topic_tie" in mv.loc["4", "flags"]
    assert mv.loc["4", "topic"] == "INFLATION"                  # canonical order tie-break


def test_majority_direction_median_rounding():
    def lab(d):
        return {"1": L("1", ("ASSESSMENT", "LABOR", d))}
    # four labelers: median of (1, 1, 2, 2) = 1.5 -> 2 ; (-1, -1, -2, -2) -> -2
    labels = {f"l{i}": lab(d) for i, d in enumerate([1, 1, 2, 2])}
    assert agreement.majority_vote(labels).loc[0, "direction"] == 2
    labels = {f"l{i}": lab(d) for i, d in enumerate([-1, -1, -2, -2])}
    assert agreement.majority_vote(labels).loc[0, "direction"] == -2
    labels = {f"l{i}": lab(d) for i, d in enumerate([0, 0, 1, 1])}
    assert agreement.majority_vote(labels).loc[0, "direction"] == 1
    assert agreement._round_half_away(-0.5) == -1 and agreement._round_half_away(0.0) == 0


def test_majority_even_split_and_min_labelers():
    labels = {f"l{i}": {"1": L("1", (f, None if f == "POLICY" else "LABOR", 0 if f != "POLICY" else 1))}
              for i, f in enumerate(["ASSESSMENT", "ASSESSMENT", "POLICY", "POLICY"])}
    row = agreement.majority_vote(labels).iloc[0]
    assert pd.isna(row["function"]) and "no_majority" in row["flags"]
    with pytest.raises(ValueError):
        agreement.majority_vote({"a": {}, "b": {}})


def test_majority_records_schema():
    recs = agreement.majority_records(agreement.majority_vote(three_labelers()))
    by_id = {r["id"]: r for r in recs}
    assert schema.validate_label(by_id["1"], allow_extra_keys=True) == []
    assert "agreement" in by_id["1"]
    assert by_id["3"]["aspects"][0]["function"] is None
    assert json.dumps(recs)   # serialisable


# ---------------------------------------------------------------- CLI end to end

def test_cli_end_to_end(tmp_path, monkeypatch):
    bd, od = tmp_path / "batches", tmp_path / "out"
    monkeypatch.setattr(batches, "batches_root", lambda d=None: bd if d is None else Path(d))
    monkeypatch.setattr(batches, "out_root", lambda d=None: od if d is None else Path(d))
    ipath = tmp_path / "items.csv"
    items(6).to_csv(ipath, index=False)
    assert cli.main(["make-batches", "--items", str(ipath), "--set", "t", "--batch-size", "3"]) == 0
    for lab in "abc":
        for bp in batches.batch_paths("t"):
            ids = [json.loads(l)["id"] for l in bp.read_text(encoding="utf-8").splitlines()]
            write_lines(od / "t" / lab / bp.name, [good(i) for i in ids])
    assert cli.main(["validate", "--set", "t", "--labeler", "a"]) == 0
    assert cli.main(["pending", "--set", "t", "--labeler", "a"]) == 0
    assert cli.main(["agree", "--set", "t", "--labelers", "a", "b", "c", "--out", str(tmp_path / "agree")]) == 0
    assert (tmp_path / "agree" / "summary.md").is_file() and (tmp_path / "agree" / "pairwise.csv").is_file()
    assert cli.main(["majority", "--set", "t", "--labelers", "a", "b", "c", "--out", str(tmp_path / "mv.jsonl")]) == 0
    lines = (tmp_path / "mv.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["aspects"][0]["function"] == "ASSESSMENT"
