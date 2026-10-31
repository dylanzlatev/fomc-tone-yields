"""Tests for ftyields.data.corpus (no network)."""
import pandas as pd
import pytest

from ftyields.data import corpus


def _raw():
    return pd.DataFrame({
        "Date": ["2008-01-30", "2008-01-30", "2008-01-09", "2000-02-02"],
        "Release Date": ["2008-02-20", "2008-01-30", None, "2000-02-02"],
        "Type": ["Minute", "Statement", "Minute", "Statement"],
        "Text": ["minutes text", "stmt", "dup of other minutes", "old statement"],
    })


def test_build_documents_columns_and_ids():
    docs = corpus.build_documents(_raw())
    assert list(docs.columns) == ["doc_id", "doc_type", "meeting_date_ds", "release_date_ds",
                                  "text", "n_chars"]
    assert set(docs["doc_id"]) == {"M20080130", "S20080130", "M20080109", "S20000202"}
    assert docs["doc_id"].is_unique
    row = docs.set_index("doc_id").loc["M20080130"]
    assert row["doc_type"] == "minutes" and row["n_chars"] == len("minutes text")
    assert docs.set_index("doc_id").loc["S20080130", "doc_type"] == "statement"


def test_dates_kept_unchanged_and_missing_release_is_null():
    docs = corpus.build_documents(_raw()).set_index("doc_id")
    assert docs.loc["M20080130", "meeting_date_ds"] == "2008-01-30"
    assert docs.loc["M20080130", "release_date_ds"] == "2008-02-20"
    assert pd.isna(docs.loc["M20080109", "release_date_ds"])


def test_sorted_by_meeting_date():
    docs = corpus.build_documents(_raw())
    assert list(docs["meeting_date_ds"]) == sorted(docs["meeting_date_ds"])


def test_duplicate_doc_id_raises():
    raw = pd.concat([_raw(), _raw().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        corpus.build_documents(raw)


def test_unknown_type_raises():
    raw = _raw()
    raw.loc[0, "Type"] = "Transcript"
    with pytest.raises(ValueError, match="unexpected Type"):
        corpus.build_documents(raw)
