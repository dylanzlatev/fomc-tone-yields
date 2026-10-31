import numpy as np
import pytest

from ftyields.config import load_config
from ftyields.model import score as S

CFG = load_config()
ID2LABEL = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}


def test_config_label_map_and_idx():
    idx = S.class_indices(ID2LABEL, CFG["scoring"]["label_map"])
    assert idx == {"dovish": 0, "hawkish": 1, "neutral": 2}


def test_label_map_rejects_bad_maps():
    with pytest.raises(KeyError):
        S.class_indices(ID2LABEL, {"LABEL_0": "dovish", "LABEL_1": "hawkish"})
    with pytest.raises(ValueError):
        S.class_indices(ID2LABEL, {"LABEL_0": "dovish", "LABEL_1": "dovish", "LABEL_2": "neutral"})


def test_logits_to_probabilities_and_score_use_config_mapping():
    idx = S.class_indices(ID2LABEL, CFG["scoring"]["label_map"])
    logits = np.array([[0.0, 10.0, 0.0],    # hawkish
                       [10.0, 0.0, 0.0],    # dovish
                       [0.0, 0.0, 10.0],    # neutral
                       [1.0, 1.0, 1.0]])
    df = S.logits_to_scores(logits, idx)
    assert np.allclose(df[["p_dovish", "p_hawkish", "p_neutral"]].sum(axis=1), 1.0)
    assert df["label"].tolist()[:3] == ["hawkish", "dovish", "neutral"]
    assert df["score"].iloc[0] > 0.99 and df["score"].iloc[1] < -0.99
    assert abs(df["score"].iloc[2]) < 1e-3 and abs(df["score"].iloc[3]) < 1e-12
    assert np.allclose(df["score"], df["p_hawkish"] - df["p_dovish"])


def test_alternate_label_order_changes_score_sign():
    swapped = {"LABEL_0": "hawkish", "LABEL_1": "dovish", "LABEL_2": "neutral"}
    idx = S.class_indices(ID2LABEL, swapped)
    df = S.logits_to_scores(np.array([[0.0, 10.0, 0.0]]), idx)
    assert df["score"].iloc[0] < -0.99


def test_dedup_preserves_count_and_order():
    texts = ["b", "a", "b", "c", "a", "b"]
    uniq, inv = S.dedup_texts(texts)
    assert uniq == ["b", "a", "c"]
    assert len(inv) == len(texts)
    assert [uniq[i] for i in inv] == texts


def test_dedup_empty():
    uniq, inv = S.dedup_texts([])
    assert uniq == [] and len(inv) == 0


def test_truncation_flag_boundary():
    flags = S.truncated_flag(np.array([10, 256, 257, 1000]), 256)
    assert flags.tolist() == [False, False, True, True]


def test_length_batches_cover_all_longest_first():
    n = np.array([5, 50, 7, 30, 9])
    batches = S.length_batches(n, 2)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(5))
    assert flat.tolist() == [1, 3, 4, 2, 0]
    assert [len(b) for b in batches] == [2, 2, 1]


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="CUDA not available")
def test_gpu_smoke_roberta_directions():
    sc = CFG["scoring"]
    tok, model = S.load_model(sc["roberta_model"], "cuda")
    idx = S.class_indices(model.config.id2label, sc["label_map"])
    texts = ["Inflation remains elevated.", "The unemployment rate has risen and job gains have slowed."]
    n = S.token_lengths(tok, texts)
    p = S.predict_probs(tok, model, texts, n, max_length=sc["max_length"], batch_size=2,
                        fp16=False, device="cuda")
    df = S.probs_to_scores(p, idx)
    assert df["score"].iloc[0] > 0.5 and df["score"].iloc[1] < -0.5
