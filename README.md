# FOMC Tone & the Yield Curve

**Question.** Does what the FOMC *says* move Treasury yields beyond what it *does*
with rates, and which kind of Fed language (policy guidance, economic assessment,
outlook) moves which part of the curve? We run an event study on statement and
minutes release days: release-day changes in yields, slope and curvature (in bp)
are regressed on sentence-level tone indices, controlling for the monetary-policy
rate surprise, with local projections for the dynamics.

Author: Dylan Zlatev

## Setup

The project runs in the `ftyields` conda env. Always use its interpreter:
`C:/Users/dylan/anaconda3/envs/ftyields/python.exe`.

```bash
# if conda is not on PATH, use the full path
C:/Users/dylan/anaconda3/Scripts/conda.exe env create -f environment.yml

pre-commit install
```

## Reproduce

```bash
C:/Users/dylan/anaconda3/envs/ftyields/python.exe run_all.py            # all stages
C:/Users/dylan/anaconda3/envs/ftyields/python.exe run_all.py --dry-run  # show the plan
C:/Users/dylan/anaconda3/envs/ftyields/python.exe run_all.py --only data text
C:/Users/dylan/anaconda3/envs/ftyields/python.exe run_all.py --from score
```

Stages, in order: `data`, `text`, `label`, `score`, `indices`, `analysis`,
`forecast`, `figures`, `paper`. Stages that are not implemented yet are
reported and skipped. All parameters live in `config.yaml`.

Tests: `python.exe -m pytest`.

## Repo layout

```
config.yaml        all parameters (single source of truth)
run_all.py         rebuild entry point
src/ftyields/      package: data, text, label, model, analysis, figures
codebook/          labelling codebook
labels/            pilot, gold, human, train, batches, out
data/              raw, interim (gitignored), processed
results/           tables, figures, macros
paper/             LaTeX source and compiled PDF
tests/             pytest suite
```

## Conventions

- Yield changes and factors (level, slope, curvature) are in basis points (bp).
- Time is measured in trading days.
- Event alignment: a release on trading day t maps to the change from the close
  of t-1 to the close of t (statements are released at 2:00pm ET, before the
  close). Releases on non-trading days map to the next trading day.
- Seeds: a single seed in `config.yaml`, applied via `ftyields.config.set_seeds`.
- Train-only fitting: any scaler, model or threshold is fit on the training
  split only, never on validation/test data.

## Data & licensing

FOMC statements/minutes (Hugging Face `vtasca/fomc-statements-minutes`) and FRED
yields are public. Third-party labels and monetary-policy surprise series are
fetched by download scripts and are not committed to this repo.

TODO: document each source, its license, and the download scripts.

## Status

Phase 0 — setup.

TODO: results summary, paper link, citation.
