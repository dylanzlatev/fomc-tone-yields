# PROGRESS — FOMC Tone & the Yield Curve

Running log kept by the orchestrator session. Read this at the start of every session
(see CLAUDE.md). Newest entries at the top of each section.

## Current state
- **Phase:** 0 done (PR from `phase-0-setup`); starting phase 1 (data).
- **Next:** phase 1 — FRED yields, FOMC event calendar, surprise series, text corpus.
- **Interrupted batch jobs:** none.

## Phase status

| # | Phase | Status | Verified | PR |
|---|---|---|---|---|
| 0 | Setup | done | yes | (this PR) |
| 1 | Data | not started | | |
| 2 | Text | not started | | |
| 3 | Codebook | not started | | |
| 4 | Validation sets | not started | | |
| 5 | Scoring | not started | | |
| 6 | Measurement results | not started | | |
| 7 | Indices | not started | | |
| 8 | Main analysis | not started | | |
| 9 | Forecasting | not started | | |
| 10 | Figures and tables | not started | | |
| 11 | Paper | not started | | |

## Log

### 2026-10-30 — Session 1
- Read CLAUDE.md and HANDOFF.md from `handoff/`. `ANTHROPIC_API_KEY` not set.
- Bootstrap (agreed with Dylan in-session): the GitHub repo was empty, so the single root
  commit on `main` is a `.gitignore` that ignores `handoff/` (commit `e13fb91`). Every
  later change goes branch → PR → merge.
- Copied `handoff/.claude/` (guard hook + deny rules), `CLAUDE.md` and `HANDOFF.md`
  verbatim to the repo root (Dylan approved). Verified the guard is live: it blocked
  `git branch -d …` and a command that mentioned `.claude/hooks`.
- Guard quirk: `mkdir -p` in the same command as `conda env create` trips the
  "modifying a conda env by path" rule (`-p` read as `--prefix`). Workaround: separate
  commands. Not a real problem; noted for subagent briefs.
- **Phase 0 (setup).** Scaffold written by a Sonnet subagent (layout, `config.yaml`,
  `run_all.py`, `src/ftyields/config.py`, README, `.gitignore`, `.pre-commit-config.yaml`,
  tests); reviewed by me, small fixes (TBD phase numbers, README question wording).
  Env `ftyields` built from `environment.yml`. Two fixes were needed (see Decisions):
  transformers capped `<5`, and torch moved 2.5.1 → 2.6.0+cu124.
  **Verified myself:**
  - `ftyields/python.exe`: numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, sklearn 1.9.1,
    statsmodels 0.15.0, torch 2.6.0+cu124, transformers 4.57.6; `pip check` clean.
  - `torch.cuda.is_available() == True` (RTX 2070 SUPER); `from transformers import
    pipeline` works; `gtfintechlab/FOMC-RoBERTa` loads with `HF_HUB_OFFLINE=1` and runs
    on GPU. Label map: **LABEL_0 = dovish, LABEL_1 = hawkish, LABEL_2 = neutral**
    ("Inflation remains elevated." → P(hawk) .999; "The unemployment rate has risen…" →
    P(dov) .986), consistent with the handoff probe and the TDW encoding.
  - `pytest`: 8 passed. `run_all.py --help` and `--dry-run` work (all stages pending).
  - `pre-commit install`; a 60 MB dummy file was refused by `check-added-large-files`,
    and a force-added `dummy_ckpt.pt` was refused by `forbid-artifacts`.
  - `environment.lock.yml` exported (`--no-builds`, prefix removed, channels trimmed to
    conda-forge after confirming every conda package is from conda-forge; PyTorch
    `--extra-index-url` re-added to the pip section so the lock is rebuildable).
- **Research for later phases** (Sonnet with web, notes in `tmp/research_sources.md`,
  to be checked and committed with phase 1/3/4):
  - Bauer–Swanson MPS / MPS_ORTH: SF Fed xlsx, sheet `FOMC (update 2023)`, covers
    1988-02-04 … **2023-12-13 only**; includes unscheduled meetings (flagged); no stated
    license → commit a download script, not the data.
  - Jarociński–Karadi US update (GitHub, CC BY 4.0) and SF Fed USMPD (through 2026) are
    candidate fallbacks / extensions.
  - TDW labels: HF `gtfintechlab/fomc_communication` (train 1,984 / test 496;
    0 = dovish, 1 = hawkish, 2 = neutral; year only), CC BY-NC 4.0 → download script.
  - Shah et al. annotation guide = paper Appendix C.2 / Table 10.

## For Dylan
- (nothing yet)

## Requests for Dylan
- **Branch protection:** `main` now exists on GitHub. Apply `handoff/protect_main.json`
  yourself (agents are banned from changing protection), e.g.
  `gh api -X PUT repos/dylanzlatev/fomc-tone-yields/branches/main/protection --input handoff/protect_main.json`.
  Until then, protection rests on the local guard hook alone.
- **Guard false positives (optional fix, agents can't edit the guard):**
  (1) the `--prefix` conda rule matches `mkdir -p` in the same command;
  (2) the `commit -n` rule (`commit.*[[:space:]]-[a-z]*n…`) matches any shell text where
  "commit" is followed later by a ` -n…` flag, e.g. a README heredoc containing
  "pre-commit install" and a later `conda … -n ftyields`. Workarounds in use: split
  commands; write files with the Write tool instead of heredocs.

## Proposed deviations
- (none)

## Decisions log
- 2026-10-30: seed `20261030` everywhere (config.yaml).
- 2026-10-30: **torch 2.6.0+cu124 instead of the handoff's 2.5.1+cu121.** Unpinned
  transformers resolved to 5.x, which needs torch ≥ 2.6; capping it at 4.57 still fails
  because transformers ≥ 4.50 refuses `torch.load` of `.bin` weights on torch < 2.6
  (CVE-2025-32434), and the cached FOMC-RoBERTa only has `pytorch_model.bin`. 2.6.0 is
  the smallest step that works; the GPU (sm_75) and driver 595 are fine with it.
  transformers stays on 4.x (`<5`) to avoid the v5 API change.
- 2026-10-30: `run_all.py` stops at the first failing implemented stage (later stages
  depend on earlier ones).
