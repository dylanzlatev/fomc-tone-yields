"""Seeded stratified samplers over data/processed/sentences_unique.parquet.

Expected columns: uid, text_masked, prev_text_masked, n_occurrences, n_docs,
first_release_date, doc_types ("statement"|"minutes"|"both"), sections (string such as
"policy_action" or "staff_economic;participants_views").

Strata (defaults): doc_type_group x section_group x era.
  doc_type_group  doc_types as given
  section_group   the section if exactly one is listed, "multiple" if several, else "unknown"
  era             by year of first_release_date: 2000-2007, 2008-2015, 2016-2025

Allocation: every non-empty stratum first gets min(floor, size) items; the remainder is
split in proportion to each stratum's weight mass (sum of item weights, so over-sampled
policy-like sentences raise their strata's share), by largest remainder, capped at
stratum size. Within a stratum items are drawn without replacement with probability
proportional to item weight (Efraimidis-Spirakis keys). Everything is a deterministic
function of (df contents, n, seed, exclude), independent of row order.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ftyields import config

ERAS = (("2000-2007", 2007), ("2008-2015", 2015), ("2016-2025", 9999))
DEFAULT_STRATA = ("doc_type_group", "section_group", "era")

# Transparent keyword heuristic, used ONLY to over-sample policy-relevant sentences.
POLICY_PATTERNS = (
    r"the committee decided",
    r"target range",
    r"federal funds rate",
    r"\bpolicy\b",
    r"accommodat",
    r"purchases",
    r"balance sheet",
    r"appropriate",
    r"dissent",
    r"preferred",
)
_POLICY_RE = re.compile("|".join(POLICY_PATTERNS), re.IGNORECASE)


def policy_like(text: object) -> bool:
    """True if the sentence contains any policy keyword (case-insensitive)."""
    return isinstance(text, str) and _POLICY_RE.search(text) is not None


def era_of(date: object) -> str:
    year = pd.Timestamp(date).year
    for label, last in ERAS:
        if year <= last:
            return label
    return ERAS[-1][0]


def section_group(sections: object) -> str:
    if not isinstance(sections, str) or not sections.strip():
        return "unknown"
    parts = [s for s in sections.split(";") if s.strip()]
    if not parts:
        return "unknown"
    return parts[0].strip() if len(parts) == 1 else "multiple"


def add_strata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with doc_type_group, section_group, era, policy_like columns added."""
    out = df.copy()
    out["doc_type_group"] = out["doc_types"].astype(str)
    out["section_group"] = out["sections"].map(section_group)
    out["era"] = out["first_release_date"].map(era_of)
    out["policy_like"] = out["text_masked"].map(policy_like)
    return out


def _largest_remainder(total: int, shares: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Integer allocation of `total` proportional to shares, with per-slot caps."""
    alloc = np.zeros(len(shares), dtype=int)
    remaining = total
    while remaining > 0:
        open_ = (alloc < caps) & (shares > 0)
        if not open_.any():
            open_ = alloc < caps          # zero-mass leftovers: spread by capacity
            shares = np.where(open_, 1.0, 0.0)
        if not open_.any():
            raise ValueError("cannot allocate: requested sample larger than the pool")
        ideal = remaining * shares * open_ / (shares * open_).sum()
        add = np.minimum(np.floor(ideal).astype(int), caps - alloc)
        alloc += add
        remaining -= int(add.sum())
        if remaining == 0:
            break
        frac = np.where(open_, ideal - np.floor(ideal), -1.0)
        # one unit to the largest remainders (stable tie-break on position)
        for idx in np.argsort(-frac, kind="stable")[:remaining]:
            if open_[idx] and alloc[idx] < caps[idx]:
                alloc[idx] += 1
                remaining -= 1
    return alloc


def stratified_sample(df: pd.DataFrame, n: int, seed: int | None = None,
                      strata: Sequence[str] = DEFAULT_STRATA,
                      weights: str | pd.Series | Callable[[pd.DataFrame], pd.Series] | None = None,
                      policy_oversample: float = 1.0, floor: int = 1,
                      exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """Seeded stratified sample of n rows (without replacement), sorted by uid.

    strata: columns defining strata; derived columns doc_type_group, section_group and
        era are added automatically when named.
    weights: optional per-row weights (column name, Series aligned to df.index, or
        callable df -> Series); multiplied with the policy_like over-sampling factor.
    policy_oversample: factor (>= 1 over-samples) applied to policy_like sentences; 1.0 = off.
    floor: minimum per non-empty stratum (capped at stratum size).
    exclude: uids already used elsewhere; removed from the pool before sampling.
    """
    if seed is None:
        seed = int(config.load_config()["seed"])
    if n < 1:
        raise ValueError("n must be >= 1")
    if policy_oversample <= 0:
        raise ValueError("policy_oversample must be > 0")
    if df["uid"].duplicated().any():
        raise ValueError("uid must be unique in df")

    w_user = None
    if weights is not None:
        if isinstance(weights, str):
            w_user = df[weights]
        elif callable(weights):
            w_user = weights(df)
        else:
            w_user = weights
        w_user = pd.Series(w_user, index=df.index).astype(float)
        if (w_user < 0).any() or w_user.isna().any():
            raise ValueError("weights must be non-negative and non-null")

    pool = add_strata_columns(df)
    pool["_w"] = (w_user if w_user is not None else 1.0)
    pool["_w"] = pool["_w"] * np.where(pool["policy_like"], policy_oversample, 1.0)
    pool["stratum"] = pool[list(strata)].astype(str).agg("|".join, axis=1)

    if exclude is not None:
        pool = pool[~pool["uid"].isin(set(exclude))]
    pool = pool[pool["_w"] > 0].sort_values("uid", kind="mergesort").reset_index(drop=True)
    if n > len(pool):
        raise ValueError(f"requested n={n} but only {len(pool)} eligible sentences")

    groups = sorted(pool["stratum"].unique())
    size = np.array([(pool["stratum"] == g).sum() for g in groups])
    mass = np.array([pool.loc[pool["stratum"] == g, "_w"].sum() for g in groups])
    base = np.minimum(floor, size)
    if base.sum() > n:
        raise ValueError(f"floor={floor} over {len(groups)} strata needs {base.sum()} > n={n}")
    extra = _largest_remainder(n - int(base.sum()), mass, size - base)
    alloc = base + extra

    rng = np.random.default_rng(int(seed))
    u = rng.random(len(pool))
    # Efraimidis-Spirakis: key = log(u) / w, larger is better.
    pool["_key"] = np.log(u) / pool["_w"].to_numpy()
    picked = []
    for g, k in zip(groups, alloc):
        if k:
            sub = pool[pool["stratum"] == g].sort_values(["_key", "uid"], ascending=[False, True])
            picked.append(sub.head(int(k)))
    out = pd.concat(picked).sort_values("uid", kind="mergesort")
    return out.drop(columns=["_w", "_key"]).reset_index(drop=True)


def pilot_sample(df: pd.DataFrame, n: int = 100, seed: int | None = None, **kw) -> pd.DataFrame:
    """Pilot set: stratified, no policy over-sampling by default."""
    kw.setdefault("policy_oversample", 1.0)
    return stratified_sample(df, n=n, seed=seed, **kw)


def gold_sample(df: pd.DataFrame, n: int = 350, seed: int | None = None,
                exclude: Iterable[str] | None = None, policy_oversample: float = 2.0,
                **kw) -> pd.DataFrame:
    """Gold set: stratified with policy-like sentences over-sampled; excludes e.g. the pilot."""
    return stratified_sample(df, n=n, seed=seed, exclude=exclude,
                             policy_oversample=policy_oversample, **kw)
