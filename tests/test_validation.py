"""Tests for the validation tooling: human sheet, TDW, validation tables (no network)."""
import math

import numpy as np
import pandas as pd
import pytest

from ftyields.config import load_config
from ftyields.label import agreement, batches, human, tdw, validation


def L(id_, *aspects, conf="high"):
    return {"id": id_, "aspects": [{"function": f, "topic": t, "direction": d} for f, t, d in aspects],
            "confidence": conf}


def items(n=6):
    return pd.DataFrame({"id": [f"s{i}" for i in range(n)], "prev_text": [f"prev {i}" for i in range(n)],
                         "text": [f"text, {i} é" for i in range(n)]})


def fill(path, edits):
    """Edit cells of a written sheet: {(id, column): value}."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for (i, col), v in edits.items():
        df.loc[df["id"] == i, col] = v
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ human sheet


def test_sheet_written_with_bom_and_blank_columns(tmp_path):
    p = tmp_path / "sheet.csv"
    assert human.write_sheet(items(), p) == "written"
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(df.columns) == ["id", "prev_text", "text", "function_1", "topic_1", "direction_1",
                                "function_2", "topic_2", "direction_2", "confidence", "notes"]
    assert (df[human.LABEL_COLUMNS] == "").all().all()
    assert df.loc[0, "text"] == "text, 0 é"


def test_blank_sheet_rewrite_is_noop_and_filled_sheet_refused(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    assert human.write_sheet(items(), p) == "unchanged"
    # a different blank draw may replace a blank sheet
    assert human.write_sheet(items(7), p) == "written"
    fill(p, {("s1", "function_1"): "POLICY"})
    before = p.read_bytes()
    with pytest.raises(human.SheetHasLabelsError, match="refusing to overwrite"):
        human.write_sheet(items(7), p)
    assert p.read_bytes() == before


def test_notes_only_also_protects_sheet(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    fill(p, {("s0", "notes"): "check this one"})
    with pytest.raises(human.SheetHasLabelsError):
        human.write_sheet(items(), p)


def test_read_blank_sheet_is_empty(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    labels, errors, n_blank = human.parse_human_sheet(p)
    assert labels == {} and errors == {} and n_blank == 6
    assert human.read_human_sheet(p) == {}


def test_read_filled_rows(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    fill(p, {
        ("s0", "function_1"): " assessment ", ("s0", "topic_1"): "Inflation", ("s0", "direction_1"): "+1",
        ("s0", "confidence"): "HIGH",
        ("s1", "function_1"): "POLICY", ("s1", "direction_1"): "-2", ("s1", "confidence"): "med",
        ("s2", "function_1"): "outlook", ("s2", "topic_1"): "labor", ("s2", "direction_1"): "1",
        ("s2", "function_2"): "OUTLOOK", ("s2", "topic_2"): "FINANCIAL", ("s2", "direction_2"): "−1",
        ("s2", "confidence"): "low", ("s2", "notes"): "two points",
        ("s3", "function_1"): "PROCEDURAL", ("s3", "topic_1"): "null", ("s3", "direction_1"): "0.0",
        ("s3", "confidence"): "high",
    })
    labels = human.read_human_sheet(p, on_error="raise")
    assert set(labels) == {"s0", "s1", "s2", "s3"}
    assert labels["s0"] == L("s0", ("ASSESSMENT", "INFLATION", 1))
    assert labels["s1"] == L("s1", ("POLICY", None, -2), conf="med")
    assert labels["s2"] == L("s2", ("OUTLOOK", "LABOR", 1), ("OUTLOOK", "FINANCIAL", -1), conf="low")
    assert labels["s3"] == L("s3", ("PROCEDURAL", None, 0))
    assert all(isinstance(a["direction"], int) for lab in labels.values() for a in lab["aspects"])
    assert agreement.primary(labels["s2"]) == ("OUTLOOK", "LABOR", 1)   # tie -> first listed


def test_read_reports_row_errors(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    fill(p, {
        ("s0", "function_1"): "ASSESSMENT", ("s0", "topic_1"): "INFLATION", ("s0", "direction_1"): "1",
        ("s0", "confidence"): "high",                                           # valid
        ("s1", "function_1"): "FORECAST", ("s1", "direction_1"): "1", ("s1", "confidence"): "high",
        ("s2", "function_1"): "ASSESSMENT", ("s2", "topic_1"): "LABOR", ("s2", "direction_1"): "3",
        ("s2", "confidence"): "high",
        ("s3", "function_1"): "POLICY", ("s3", "topic_1"): "INFLATION", ("s3", "direction_1"): "1",
        ("s3", "confidence"): "high",                                           # POLICY needs null topic
        ("s4", "function_1"): "OUTLOOK", ("s4", "topic_1"): "GENERAL", ("s4", "direction_1"): "0",
        ("s4", "function_2"): "POLICY",                                         # half-filled aspect 2, no confidence
    })
    labels, errors, n_blank = human.parse_human_sheet(p)
    assert set(labels) == {"s0"} and set(errors) == {"s1", "s2", "s3", "s4"} and n_blank == 1
    assert any("function_1" in e for e in errors["s1"])
    assert any("direction_1" in e for e in errors["s2"])
    assert any("POLICY must have topic null" in e for e in errors["s3"])
    assert any("direction_2" in e for e in errors["s4"]) and any("confidence" in e for e in errors["s4"])
    with pytest.warns(UserWarning, match="4 row"):
        assert set(human.read_human_sheet(p)) == {"s0"}
    with pytest.raises(ValueError, match="s2"):
        human.read_human_sheet(p, on_error="raise")


def test_read_accepts_semicolon_cp1252_sheet(tmp_path):
    p = tmp_path / "sheet.csv"
    p.write_bytes((
        "id;prev_text;text;function_1;topic_1;direction_1;function_2;topic_2;direction_2;confidence;notes\r\n"
        "s0;;café;ASSESSMENT;LABOR;-1;;;;high;\r\n").encode("cp1252"))
    assert human.read_human_sheet(p, on_error="raise")["s0"] == L("s0", ("ASSESSMENT", "LABOR", -1))


def test_draw_human_items_from_gold_stratified():
    cfg = load_config()
    rng = np.random.default_rng(0)
    n = 350
    texts = [("The Committee decided to keep the target range." if i % 3 == 0 else f"Output grew {i}.")
             for i in range(n)]
    gold = pd.DataFrame({"id": [f"u{i:04d}" for i in range(n)], "prev_text": "",
                         "text": texts,
                         "doc_types": rng.choice(["minutes", "statement", "both"], n, p=[.85, .06, .09]),
                         "sections": "policy_action", "first_release_date": "2010-01-01"})
    a = human.draw_human_items(cfg, gold)
    b = human.draw_human_items(cfg, gold)
    assert len(a) == cfg["labelling"]["human_n"] == 100
    assert a["id"].is_unique and set(a["id"]) <= set(gold["id"])
    assert a.equals(b)
    assert list(a["id"]) != sorted(a["id"])           # randomised order
    assert set(a["doc_type_group"]) == {"minutes", "statement", "both"}
    assert a["policy_like"].any() and (~a["policy_like"]).any()


# ------------------------------------------------------------------ TDW


def test_tdw_label_to_direction():
    assert tdw.label_to_direction(0) == -1      # dovish
    assert tdw.label_to_direction(1) == 1       # hawkish
    assert tdw.label_to_direction(2) == 0       # neutral
    assert tdw.label_to_direction("1") == 1
    with pytest.raises(ValueError):
        tdw.label_to_direction(3)


def test_tdw_prepare_ids_unique_despite_repeated_index():
    raw = pd.DataFrame({"index": [0, 0, 1], "sentence": ["a", "b", "c"], "year": [2007, 2011, 2011],
                        "label": [0, 1, 2], "orig_index": [0, 0, 1]})
    out = tdw.prepare(raw, "test")
    assert out["tdw_id"].is_unique and out["tdw_id"].iloc[0] == "tdw_test_0000"
    assert list(out["direction"]) == [-1, 1, 0]
    assert list(out["label_raw"]) == [0, 1, 2]


def test_tdw_drop_train_duplicates():
    train = pd.DataFrame({"sentence": ["Inflation rose.", "Jobs fell.  ", "Other."]})
    test = pd.DataFrame({"tdw_id": ["t0", "t1", "t2", "t3"],
                         "sentence": ["Inflation rose.", "Jobs   fell.", "New one.", "Another."]})
    kept, dropped = tdw.drop_train_duplicates(train, test)
    assert list(kept["tdw_id"]) == ["t2", "t3"]
    assert list(dropped["tdw_id"]) == ["t0", "t1"]


def test_make_batches_ignores_committed_readme(tmp_path):
    (tmp_path / "tdw_test").mkdir()
    (tmp_path / "tdw_test" / "README.md").write_text("licence note")
    paths = batches.make_batches(items(5), "tdw_test", 2, batches_dir=tmp_path)
    assert len(paths) == 3 and (tmp_path / "tdw_test" / "README.md").read_text() == "licence note"
    assert batches.make_batches(items(5), "tdw_test", 2, batches_dir=tmp_path) == paths   # no-op


# ------------------------------------------------------------------ validation tables


def synthetic_labels():
    """4 labelers over 8 items; a, b, c are 'gold', d is the candidate."""
    A = ("ASSESSMENT", "INFLATION", 1)
    P = ("POLICY", None, -2)
    O = ("OUTLOOK", "LABOR", 0)
    base = {f"i{k}": (A if k % 3 == 0 else P if k % 3 == 1 else O) for k in range(8)}
    a = dict(base)
    b = dict(base); b["i0"] = ("ASSESSMENT", "INFLATION", 2)          # disagrees with a, c on i0
    c = dict(base); c["i1"] = ("POLICY", None, -1)                    # disagrees with a, b on i1
    d = dict(base); d["i2"] = ("ASSESSMENT", "LABOR", 0)              # candidate error on i2
    mk = lambda m: {i: L(i, t) for i, t in m.items()}
    return {"a": mk(a), "b": mk(b), "c": mk(c), "d": mk(d)}


def test_leave_out_consensus_four_labelers():
    labels = synthetic_labels()
    # candidate d: others = a, b, c must all agree -> not i0 (b differs) and not i1 (c differs)
    ref = validation.leave_out_reference(labels, ["a", "b", "c"], "d")
    assert list(ref.index) == [f"i{k}" for k in range(2, 8)]
    assert tuple(ref.loc["i3"]) == ("ASSESSMENT", "INFLATION", 1)
    # candidate a (a gold member): others = b, c agree on everything except i0 and i1
    ref_a = validation.leave_out_reference(labels, ["a", "b", "c"], "a")
    assert list(ref_a.index) == [f"i{k}" for k in range(2, 8)]
    # gold of 4: for d with others a, b, c
    ref4 = validation.leave_out_reference(labels, ["a", "b", "c", "d"], "d")
    assert list(ref4.index) == list(ref.index)
    with pytest.raises(ValueError):
        validation.leave_out_reference(labels, ["a", "d"], "d")


def test_labeler_vs_reference_table_uses_agreement_metrics():
    labels = synthetic_labels()
    gold = ["a", "b", "c"]
    maj = agreement.majority_vote({g: labels[g] for g in gold})
    tab = validation.labeler_vs_reference_table(labels, {}, gold, maj, ["a", "b", "c", "d"])
    row = tab[(tab["labeler"] == "d") & (tab["reference"] == "leave_out_consensus")].iloc[0]
    assert row["status"] == "ok" and row["n_items"] == 6 and not row["in_gold"]
    # d errs on i2 only (function ASSESSMENT vs OUTLOOK): 5/6 on function and full match
    assert row["function_pct"] == pytest.approx(100 * 5 / 6)
    assert row["full_match_pct"] == pytest.approx(100 * 5 / 6)
    maj_row = tab[(tab["labeler"] == "d") & (tab["reference"] == "gold_majority")].iloc[0]
    assert maj_row["n_items"] == 8                      # majority resolves every item
    assert maj_row["full_match_pct"] == pytest.approx(100 * 7 / 8)
    # same numbers as agreement.pair_metrics
    ref = validation.leave_out_reference(labels, gold, "d")
    pm = agreement.pair_metrics(agreement.primary_frames({"d": labels["d"]}, ref.index)["d"], ref)
    assert row["direction_qwk"] == pytest.approx(pm["direction_qwk"], nan_ok=True)
    assert tab["in_gold"].sum() == 6   # three gold labelers x two references


def test_incomplete_labelers_are_pending_not_errors():
    labels = synthetic_labels()
    present = {k: labels[k] for k in ("a", "b")}
    skipped = {"c": "incomplete: 0/2 batches valid"}
    gtab, maj = validation.gold_agreement_table(present, ["a", "b", "c"], skipped)
    assert maj is None and list(gtab["status"]) == ["pending"] and "c" in gtab.loc[0, "note"]
    tab = validation.labeler_vs_reference_table(present, skipped, ["a", "b", "c"], None, ["a", "b", "c"])
    assert set(tab.loc[tab["labeler"] == "c", "status"]) == {"pending"}
    assert set(tab.loc[tab["labeler"] == "a", "status"]) == {"pending"}   # no majority yet
    assert tab["function_pct"].isna().all()


def test_gold_agreement_table_counts_and_majority():
    labels = synthetic_labels()
    tab, maj = validation.gold_agreement_table({k: labels[k] for k in "abc"}, ["a", "b", "c"])
    assert set(tab["scope"]) == {"overall", "pair", "majority_vote"}
    m = tab[tab["scope"] == "majority_vote"].iloc[0]
    assert m["n_items"] == 8 and m["n_no_majority"] == 0
    assert (tab["status"] == "ok").all()
    assert len(maj) == 8 and maj.set_index("id").loc["i0", "direction"] == 1   # median of 1, 2, 1


def test_human_vs_models_pending_when_sheet_blank(tmp_path):
    p = tmp_path / "sheet.csv"
    human.write_sheet(items(), p)
    h, errors, n_blank = human.parse_human_sheet(p)
    labels = synthetic_labels()
    tab = validation.human_vs_models_table(h, n_blank, len(errors), labels, {}, None, ["a", "b"])
    assert list(tab["status"]) == ["pending"] * 3        # gold_majority + 2 labelers
    assert tab["function_pct"].isna().all() and tab["direction_exact_pct"].isna().all()
    assert (tab["n_human_labelled"] == 0).all() and (tab["n_human_blank"] == 6).all()


def test_human_vs_models_fills_in_when_labelled():
    labels = synthetic_labels()
    gold = ["a", "b", "c"]
    maj = agreement.majority_vote({g: labels[g] for g in gold})
    human_labels = {i: labels["a"][i] for i in ("i0", "i1", "i2", "i3")}
    human_labels["i2"] = L("i2", ("ASSESSMENT", "LABOR", 0))
    tab = validation.human_vs_models_table(human_labels, 4, 0, labels, {}, maj, ["a", "d"])
    assert list(tab["status"]) == ["ok"] * 3 and (tab["n_items"] == 4).all()
    vs_a = tab[tab["reference"] == "a"].iloc[0]
    assert vs_a["function_pct"] == pytest.approx(75.0) and vs_a["full_match_pct"] == pytest.approx(75.0)
    assert not math.isnan(vs_a["direction_exact_pct"])
