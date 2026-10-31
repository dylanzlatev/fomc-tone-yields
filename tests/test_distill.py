"""CPU tests for the distiller's pure parts (slots, loss, splits, derived labels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ftyields.label.schema import validate_label
from ftyields.model import distill as D


def lab(*aspects, id_="x"):
    return {"id": id_, "confidence": "high",
            "aspects": [{"function": f, "topic": t, "direction": d} for f, t, d in aspects]}


def test_slot_layout():
    assert D.N_SLOTS == 11
    assert D.SLOT_NAMES[0] == "ASSESSMENT_INFLATION" and D.SLOT_NAMES[5] == "OUTLOOK_INFLATION"
    assert D.SLOT_NAMES[-1] == "POLICY"
    assert len(set(D.SLOT_NAMES)) == 11


def test_single_aspect_roundtrip():
    l = lab(("OUTLOOK", "LABOR", -1))
    present, direction, info = D.label_to_slots(l)
    assert present.sum() == 1 and present[D.SLOT_INDEX[("OUTLOOK", "LABOR")]] == 1
    assert direction[D.SLOT_INDEX[("OUTLOOK", "LABOR")]] == -1
    assert info == {"duplicate_slot": 0, "procedural_with_other": 0}
    assert D.slots_to_label(present, direction)["aspects"] == l["aspects"]


def test_two_aspect_roundtrip_and_policy():
    l = lab(("POLICY", None, 2), ("ASSESSMENT", "INFLATION", 1))
    present, direction, _ = D.label_to_slots(l)
    assert present.sum() == 2
    back = D.slots_to_label(present, direction)
    # slot order, not listing order
    assert back["aspects"] == [{"function": "ASSESSMENT", "topic": "INFLATION", "direction": 1},
                               {"function": "POLICY", "topic": None, "direction": 2}]
    assert validate_label({**back, "id": "x"}) == []


def test_procedural_has_no_slot():
    present, direction, info = D.label_to_slots(lab(("PROCEDURAL", None, 0)))
    assert present.sum() == 0 and direction.sum() == 0 and info["procedural_with_other"] == 0
    back = D.slots_to_label(present, direction)
    assert back["aspects"] == [{"function": "PROCEDURAL", "topic": None, "direction": 0}]
    assert validate_label(back) == []


def test_procedural_next_to_other_is_dropped_and_counted():
    present, _, info = D.label_to_slots(lab(("PROCEDURAL", None, 0), ("POLICY", None, 1)))
    assert present.sum() == 1 and info["procedural_with_other"] == 1


def test_duplicate_slot_keeps_larger_abs_direction():
    l = lab(("ASSESSMENT", "LABOR", 1), ("ASSESSMENT", "LABOR", -2))
    present, direction, info = D.label_to_slots(l)
    i = D.SLOT_INDEX[("ASSESSMENT", "LABOR")]
    assert present.sum() == 1 and direction[i] == -2 and info["duplicate_slot"] == 1
    # tie in |direction|: first listed wins
    _, direction, _ = D.label_to_slots(lab(("POLICY", None, 1), ("POLICY", None, -1)))
    assert direction[D.SLOT_INDEX[("POLICY", None)]] == 1


def test_labels_to_arrays_counts():
    labels = [lab(("POLICY", None, 1), ("POLICY", None, 2)), lab(("PROCEDURAL", None, 0))]
    present, direction, info = D.labels_to_arrays(labels)
    assert present.shape == (2, 11) and info["duplicate_slot"] == 1
    assert present[1].sum() == 0
    stats = D.label_statistics(labels)
    assert stats["n_procedural"] == 1 and stats["duplicate_slot_cases"] == 1


def test_masked_loss_ignores_absent_slots():
    torch.manual_seed(0)
    B = 3
    pres = torch.randn(B, D.N_SLOTS)
    dirl = torch.randn(B, D.N_SLOTS, 5)
    present = torch.zeros(B, D.N_SLOTS, dtype=torch.long)
    present[0, 2] = 1
    present[1, 7] = 1
    direction = torch.zeros(B, D.N_SLOTS, dtype=torch.long)
    direction[0, 2], direction[1, 7] = 2, -1
    loss, bce, ce = D.distill_loss(pres, dirl, present, direction, lam=1.0)
    exp_ce = (torch.nn.functional.cross_entropy(dirl[0, 2][None], torch.tensor([4]))
              + torch.nn.functional.cross_entropy(dirl[1, 7][None], torch.tensor([1]))) / 2
    assert torch.allclose(ce, exp_ce, atol=1e-6)
    exp_bce = torch.nn.functional.binary_cross_entropy_with_logits(pres, present.float())
    assert torch.allclose(bce, exp_bce, atol=1e-6)
    assert torch.allclose(loss, exp_bce + exp_ce, atol=1e-6)
    # direction logits of absent slots (and garbage direction targets there) have no effect
    dirl2 = dirl.clone()
    dirl2[2] += 100.0
    dirl2[0, 3] += 50.0
    direction2 = direction.clone()
    direction2[2] = 2
    _, _, ce2 = D.distill_loss(pres, dirl2, present, direction2, lam=1.0)
    assert torch.allclose(ce, ce2, atol=1e-6)
    # lambda scales only the CE term; no present slots -> CE term is 0, no NaN
    l5, b5, c5 = D.distill_loss(pres, dirl, present, direction, lam=0.5)
    assert torch.allclose(l5, b5 + 0.5 * c5)
    none = torch.zeros_like(present)
    l0, b0, c0 = D.distill_loss(pres, dirl, none, direction)
    assert c0.item() == 0.0 and torch.isfinite(l0)


def test_loss_gradient_zero_for_absent_direction_logits():
    pres = torch.zeros(1, D.N_SLOTS, requires_grad=True)
    dirl = torch.zeros(1, D.N_SLOTS, 5, requires_grad=True)
    present = torch.zeros(1, D.N_SLOTS, dtype=torch.long)
    present[0, 4] = 1
    direction = torch.zeros(1, D.N_SLOTS, dtype=torch.long)
    direction[0, 4] = 1
    D.distill_loss(pres, dirl, present, direction)[0].backward()
    g = dirl.grad[0].abs().sum(-1)
    assert g[4] > 0 and g.sum() == pytest.approx(float(g[4]))


def test_year_split_assignment():
    dates = ["2003-05-06", "2006-12-12", "2017-03-15", "2009-01-28", "2010-06-01", "2023-11-01"]
    out = D.year_split(dates, test_years=[2003, 2009, 2014, 2019, 2023], val_years=[2006, 2017])
    assert list(out) == ["test", "val", "val", "test", "train", "test"]
    with pytest.raises(ValueError):
        D.year_split(dates, [2003], [2003])


def test_random_split_seeded_and_sized():
    f = {"train": 0.8, "val": 0.1, "test": 0.1}
    a, b = D.random_split(1000, f, 5), D.random_split(1000, f, 5)
    assert (a == b).all()
    assert (a == "val").sum() == 100 and (a == "test").sum() == 100 and (a == "train").sum() == 800
    assert not (D.random_split(1000, f, 6) == a).all()
    with pytest.raises(ValueError):
        D.random_split(10, {"train": 0.5, "val": 0.1, "test": 0.1}, 1)


def _probs(pres: dict, dirs: dict | None = None):
    p = np.full(D.N_SLOTS, 0.01)
    d = np.full((D.N_SLOTS, 5), 0.2)
    for i, v in pres.items():
        p[i] = v
    for i, k in (dirs or {}).items():
        d[i] = 0.05
        d[i, k] = 0.8
    return p, d


def test_derive_label_threshold_direction_and_order():
    p, d = _probs({1: 0.9, 7: 0.6, 3: 0.4}, {1: 4, 7: 0})
    out = D.derive_label(p, d, threshold=0.5)
    assert out["aspects"] == [{"function": "ASSESSMENT", "topic": "LABOR", "direction": 2},
                              {"function": "OUTLOOK", "topic": "ACTIVITY", "direction": -2}]
    assert validate_label(out) == []
    # lower threshold admits the third slot, but at most 2 aspects are kept (highest probs)
    out3 = D.derive_label(*_probs({1: 0.9, 7: 0.6, 3: 0.55}), threshold=0.5)
    assert [a["topic"] for a in out3["aspects"]] == ["LABOR", "ACTIVITY"]
    # a probability exactly at the threshold counts as present
    assert D.derive_label(*_probs({10: 0.5}), threshold=0.5)["aspects"][0]["function"] == "POLICY"


def test_derive_label_procedural_fallback():
    out = D.derive_label(*_probs({2: 0.49}), threshold=0.5)
    assert out["aspects"] == [{"function": "PROCEDURAL", "topic": None, "direction": 0}]
    assert validate_label(out) == []


def test_derive_labels_batch_ids():
    p = np.stack([_probs({10: 0.9}, {10: 3})[0], _probs({})[0]])
    d = np.stack([_probs({10: 0.9}, {10: 3})[1], _probs({})[1]])
    labs = D.derive_labels(p, d, ["a", "b"], 0.5)
    assert [l["id"] for l in labs] == ["a", "b"]
    assert labs[0]["aspects"][0] == {"function": "POLICY", "topic": None, "direction": 1}
    assert labs[1]["aspects"][0]["function"] == "PROCEDURAL"


def test_expected_direction():
    p = np.zeros((2, D.N_SLOTS, 5))
    p[0, :, 4] = 1.0                                        # all mass on +2
    p[1, :, :] = np.array([0.1, 0.2, 0.4, 0.2, 0.1])         # symmetric
    p[1, 0] = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
    p[1, 1] = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    e = D.expected_direction(p)
    assert e.shape == (2, D.N_SLOTS)
    assert np.allclose(e[0], 2.0)
    assert e[1, 0] == pytest.approx(0.0) and e[1, 1] == pytest.approx(1.5)
    assert e[1, 5] == pytest.approx(0.0)


def test_encode_pair_truncation_keeps_target():
    cls, sep = 0, 2
    prev, text = list(range(10, 30)), list(range(100, 110))
    ids, full, trunc, ttr = D.encode_pair(prev, text, cls, sep, 64)
    assert ids == [cls] + prev + [sep, sep] + text + [sep] and full == 34 and not trunc
    ids, full, trunc, ttr = D.encode_pair(prev, text, cls, sep, 20)
    assert len(ids) == 20 and trunc and not ttr and ids[-11:] == text + [sep]
    assert ids[:1] == [cls] and prev[-6:] == ids[1:7]            # keeps the END of prev
    ids, _, trunc, ttr = D.encode_pair(prev, text, cls, sep, 8)  # target alone is too long
    assert ids == [cls] + text[:6] + [sep] and trunc and ttr
    ids, full, _, _ = D.encode_pair([], text, cls, sep, 64)
    assert ids == [cls] + text + [sep] and full == 12


def test_predict_batch_independent_with_tiny_model():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, 8, padding_idx=1)
            self.p = torch.nn.Linear(8, D.N_SLOTS)
            self.d = torch.nn.Linear(8, D.N_SLOTS * 5)

        def forward(self, ids, mask):
            h = (self.emb(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            return self.p(h), self.d(h).view(-1, D.N_SLOTS, 5)

    torch.manual_seed(0)
    m = Tiny()
    seqs = [[0, 5, 6, 2], [0, 7, 2], [0, 8, 9, 10, 11, 2]]
    p1, d1 = D.predict(m, seqs, 1, "cpu", batch_size=1)
    p3, d3 = D.predict(m, seqs, 1, "cpu", batch_size=3)
    assert np.allclose(p1, p3, atol=1e-6) and np.allclose(d1, d3, atol=1e-6)
    assert np.allclose(d1.sum(-1), 1.0, atol=1e-5)


def test_evaluate_labels_perfect_and_degraded():
    truth = {"a": lab(("ASSESSMENT", "LABOR", 1)), "b": lab(("POLICY", None, -1)),
             "c": lab(("PROCEDURAL", None, 0)), "d": lab(("OUTLOOK", "INFLATION", 2), ("POLICY", None, 1))}
    m = D.evaluate_labels(truth, truth)
    assert m["n_items"] == 4 and m["function_pct"] == 100 and m["full_match_pct"] == 100
    assert m["presence_macro_f1"] == pytest.approx(1.0) and m["direction_exact_pct"] == 100
    pred = dict(truth)
    pred["a"] = lab(("ASSESSMENT", "LABOR", 0))
    pred["c"] = lab(("POLICY", None, 0))
    m2 = D.evaluate_labels(truth, pred)
    assert m2["function_pct"] == 75 and m2["direction_exact_pct"] == 75
    assert m2["full_match_pct"] == 50 and m2["presence_macro_f1"] < 1.0
    # primary-only reduction of the prediction for single-aspect references
    mg = D.evaluate_labels({"d": lab(("OUTLOOK", "INFLATION", 2))}, {"d": truth["d"]}, pred_primary_only=True)
    assert mg["full_match_pct"] == 100 and mg["presence_macro_f1"] == pytest.approx(1.0)


def test_result_frame_schema():
    rng = np.random.default_rng(0)
    p = rng.random((4, D.N_SLOTS)).astype(np.float32)
    d = rng.random((4, D.N_SLOTS, 5)).astype(np.float32)
    d /= d.sum(-1, keepdims=True)
    df = D.result_frame(["u1", "u2", "u3", "u4"], p, d, 0.5)
    assert df.shape == (4, 2 + 11 + 11 + 55)
    assert df["uid"].tolist() == ["u1", "u2", "u3", "u4"]
    assert "p_POLICY" in df and "e_OUTLOOK_GENERAL" in df and "pdir_ASSESSMENT_LABOR_-2" in df
    assert np.allclose(df["e_POLICY"], D.expected_direction(d)[:, 10], atol=1e-5)
    assert all(isinstance(s, str) for s in df["aspects"])


def test_config_keys_present():
    from ftyields.config import load_config
    c = load_config()
    dc = c["distill"]
    assert set(dc["test_years"]).isdisjoint(dc["val_years"])
    assert 0 < dc["threshold"] < 1 and dc["infer_batch_size"] >= 1
    assert c["labelling"]["train"]["labeler"] == "sonnet"
