# HANDOFF — FOMC Tone & the Yield Curve (v2 rebuild)

You are the **orchestrator** for this project: an Opus session at high effort, working
autonomously for Dylan Zlatev (sole author). You make the day-to-day decisions, delegate
the work to subagents, and check everything they produce before it counts. This file is
your briefing. `CLAUDE.md` holds the standing rules. `PROGRESS.md` is your running log;
create it on the first session and keep it current.

Written 2026-10-30 by the Claude session that reviewed the old project with Dylan. Every
decision under "Locked decisions" was agreed with him.

**You run fully autonomously and never wait for Dylan.** Where the plan involves his
input (codebook review, his 100 labels, the results story), you leave a note in
`PROGRESS.md` under "For Dylan" and keep going. If the evidence argues against a locked
decision, don't silently change the main design: run the alternative as an **additional**
analysis on its own branch, keep the locked design as the main line, and record the
argument under "Proposed deviations" in `PROGRESS.md`.

---

## 1. Background

### 1.1 The old project (read-only reference)
`C:\Users\dylan\Documents\GT\CS\4641\FedSentiment-NLP` (GitHub: dylanzlatev/FedSentiment-NLP).
This was a Georgia Tech CS 4641 course project, submitted 2026-04. The paper is
`figures/Quantifying_Fed_Rhetoric__NLP_Driven_Yield_Curve_Modeling.pdf`. What it did:
- FRED daily 2/5/10/30Y yields and macro data; PCA on standardized yield levels gives
  Level, Slope and Curvature factors.
- The HF dataset `vtasca/fomc-statements-minutes`: 220 statements and 209 minutes,
  2000–2025. **Minutes are dated by release date, which is correct.**
- FinBERT and FOMC-RoBERTa applied to ~450-token chunks; the document score is the mean
  of the top labels.
- Forecasts of 30-trading-day factor changes with 7 models (random walk, VAR, ElasticNet,
  XGBoost, LightGBM, ensemble, LSTM), yearly walk-forward 2011–2025.

**The result:** the random walk beat every model on every factor (mean RMSE Level 0.347,
Slope 0.129, Curvature 0.048). Sentiment helped only the overfit tree models. For
ElasticNet, the best ML model, removing sentiment *improved* RMSE.

**Flaws you must not repeat:**
1. Granger tests on daily non-stationary factor levels, against a sentiment step function
   with 1–4 day lags, reporting the minimum p-value over lags. Invalid.
2. PCA fit on the full sample and then used in the walk-forward, which leaks future data.
   The leak-free per-window function existed but was never called.
3. Targets were `shift(-30)` with no gap between training and test, so training labels
   overlapped the test year.
4. FOMC-RoBERTa (trained on single sentences) was given 450-token chunks and only the top
   label was kept. Statements are 1–2 chunks, so tone could only be {−1, −.5, 0, .5, 1}.
5. The FinBERT sign map in the code disagreed with the cached scores.
6. Model features came from the minutes; the statements, which markets react to, were
   mostly unused.
7. About 6.5k daily rows held only ~200 distinct sentiment values, and the 30-day targets
   overlapped, so the effective sample size was far smaller than the row count.
8. Macro data was dated by the month it describes, not its publication date
   (look-ahead).
9. No significance tests on any forecast comparison. Units were standardized PCA scores
   that nobody could interpret.

You may read old code and data for reference. **Do not port old code wholesale; rebuild
cleanly.** Reusing the raw text dataset and the cached HF models is fine.

### 1.2 Evidence gathered during the review
**A quick event-study check with the old crude scores** (release-day yield change in bp,
HC1 standard errors):
- Statement days, n = 219: slope (30Y−2Y) on RoBERTa tone level gives b = −1.23bp,
  t = −2.19, R² = .022.
- Minutes days, n = 206: 10Y on the change in tone gives b = +3.48bp, t = 2.14, R² = .021.
- Every other combination was insignificant.

The signs make economic sense but the signal is weak. That points to the *measurement*
as the bottleneck.

**Sentence probe** (FOMC-RoBERTa P(hawkish)−P(dovish) vs FinBERT top label):

| Sentence | RoBERTa | FinBERT |
|---|---|---|
| Employment is strong. | +0.99 | positive |
| The labor market remains strong and the unemployment rate has declined. | +0.99 | **negative** |
| The unemployment rate has risen and job gains have slowed. | −0.97 | negative |
| Inflation has eased substantially. | **−0.15** | positive |
| Inflation remains elevated. | +1.00 | positive |
| The Committee decided to raise the target range for the federal funds rate. | +1.00 | neutral |
| The Committee will be patient as it determines future adjustments. | **0.00** | neutral |
| Strains in financial markets have increased. | **+0.65** (should be dovish) | negative |

**FinBERT measures whether news is good or bad, which is a different construct from
hawkish/dovish.** It is dropped from the pipeline (see section 2). RoBERTa handles plain
macro descriptions well and is weak on forward guidance and financial stress. The paper
should include this table, or an improved version, as a short caution note.

---

## 2. Locked decisions

1. **Research question.** Does what the FOMC *says* move Treasury yields beyond what it
   *does* with rates? Which kind of Fed language moves which part of the curve?
2. **Main design: an event study.** Release-day changes in yields, slope and curvature
   (in bp) on statement days and minutes-release days, regressed on tone indices,
   controlling for the monetary-policy rate surprise (Bauer–Swanson series; fall back to
   another published surprise series only if it is unavailable, and note it). Dynamics
   via **local projections** (Jordà) at h = 0..60 trading days, with HAC confidence
   bands. These **replace** Granger/VAR/VECM.
3. **Forecasting is secondary.** Meeting-to-meeting horizon. Factors defined directly in
   bp (level = mean yield; slope = 30Y−2Y; curvature = 2·5Y−2Y−30Y), with no PCA in the
   main analysis. A gap between training and test data, and **Clark–West** tests for the
   nested sentiment-vs-no-sentiment comparison. Report several horizons. Drop the LSTM
   unless it earns its place.
4. **Tone measurement is per sentence, following a written codebook** (section 3). Scores
   are **absolute** per sentence; anything "relative to the previous release" is computed
   afterwards by differencing indices, never inside the labels.
5. **Separate indices:** policy-guidance tone, economic-assessment tone and outlook tone,
   with optional per-topic breakdowns. Reason: assessment sentences mostly repeat data
   that is already public and priced in, while guidance is new information about the
   rate path. Averaging them dilutes the signal. Which index moves which yield *is* the
   finding. Mention the Fed information effect (Nakamura & Steinsson 2018).
6. **FinBERT is dropped.** It appears only as the caution note.
7. **Scorers:** FOMC-RoBERTa (published baseline, sentence-level, using P(hawk)−P(dov)
   as the score) and a Claude-labelled codebook measure, distilled into a local model
   (section 5).
8. **Validation:**
   - (a) Trillion Dollar Words human labels as an external check on *direction*.
   - (b) **Dylan hand-labels 100 sentences** for function, topic and direction; this is
     the human anchor.
   - (c) 3 blind Claude labelers produce a majority-vote label set. Report it next to the
     human results, and acknowledge that same-family models make correlated mistakes.
9. **No API billing.** All LLM labelling runs as Claude Code subagents on Dylan's Max
   subscription. Never call the Anthropic API or ask for a key. If the session-start hook
   warns that `ANTHROPIC_API_KEY` is set, pause labelling and leave a note (see section 7).
10. **Paper:** arXiv-style LaTeX, sole author **Dylan Zlatev**. Every number comes from
    generated tables or macros, never hand-typed.
11. **New repo** (this one). The old repo stays untouched.

---

## 3. Codebook specification (starting point for v0)

Write `codebook/codebook.md`. It must be **self-contained and blind**: no mention of
yields, markets, the research question or hypotheses. It describes only how to label a
sentence. Version it (v0, v1, …) with a changelog.

**Unit:** one sentence, shown with the sentence before it as context only.

**Level 1 — FUNCTION** (exactly one):
- `ASSESSMENT`: the current or recent state of the economy, based on data already
  observed. Usually past or present tense. *"Job gains have been solid."*
- `OUTLOOK`: the Committee's or participants' projections, expectations or risk
  assessments. *"Inflation is expected to rise to 2 percent over the medium term."*
- `POLICY`: decisions, forward guidance, balance-sheet actions, dissents, views on what
  policy should be. *"Further increases may be appropriate."*
- `PROCEDURAL`: mandate boilerplate, attendance, administrative or voting mechanics with
  no stance, staff logistics. Excluded from every index.

**Level 2 — TOPIC** (only for ASSESSMENT and OUTLOOK):
- `INFLATION`: prices, inflation expectations, energy and commodity prices.
- `LABOR`: employment, unemployment, payrolls, participation, wages.
- `ACTIVITY`: GDP, spending, investment, housing, production, trade.
- `FINANCIAL`: markets, credit, banking stress, the dollar, global financial spillovers.

**DIRECTION:** an integer from −2 to +2, with anchor sentences for every value. Hawkish
is positive and means the sentence implies a higher expected path for the policy rate.
- Higher inflation, a tighter labor market or stronger activity → hawkish.
- Weakness, falling inflation or rising unemployment → dovish.
- **Financial stress → dovish.**

Base the topic rules on the Shah, Paturi & Chava (2023) annotation guide. Read it and
cite it.

**Edge rules:**
- **Mixed sentences:** at most 2 aspects, each with its own function, topic and
  direction.
- **Hedges** ("may", "some participants"): keep the sign, lower the strength by one, with
  a minimum of 1.
- **Dissents:** POLICY, with the direction the dissenter wanted.
- **No outside knowledge:** dates and names are stripped from what the labeler sees, and
  it must not reason about what happened later.

**Output schema** (JSONL, one line per sentence):
`{"id": str, "aspects": [{"function": ..., "topic": ...|null, "direction": int}], "confidence": "low|med|high"}`

---

## 4. Roadmap and acceptance criteria

Work the phases in order. A phase is done only when its acceptance check passes **and**
you have verified it yourself (section 7). Log each phase in `PROGRESS.md`.

| # | Phase | Deliverable | Acceptance check |
|---|---|---|---|
| 0 | Setup | Repo layout (section 6). **Conda env `ftyields` built from a committed `environment.yml`** (see "Project environment" in section 6), plus a committed `environment.lock.yml`. One rebuild script (`run_all.py` or Makefile), README skeleton with setup instructions, `.gitignore`, large-file pre-commit hook | `C:/Users/dylan/anaconda3/envs/ftyields/python.exe` imports torch with `torch.cuda.is_available() == True`, imports transformers (including `from transformers import pipeline`), statsmodels, pandas and sklearn with no numpy conflict, and loads `gtfintechlab/FOMC-RoBERTa` from the HF cache offline; `run_all --help` works; the hook refuses a test commit of a dummy large file |
| 1 | Data | FRED yields (1M or 3M, 1Y, 2Y, 5Y, 7Y, 10Y, 20Y, 30Y), FOMC event calendar (statement and minutes release dates, **unscheduled meetings**, e.g. 2020-03-03 and 2020-03-15 (a Sunday)), rate-surprise series, text corpus | Every statement and minutes document maps to a trading day; unscheduled meetings and weekend releases handled explicitly; coverage table written |
| 2 | Text | Sentence table: `doc_id, doc_type, release_date, sent_idx, text, prev_text`; minutes cleaned (headers, attendance, vote tallies removed); **exact-duplicate sentences de-duplicated with a mapping back** (statements are heavily boilerplate, which saves a lot of labelling) | 30 random sentences per document type look clean (paste them into PROGRESS.md); counts reported, total and unique |
| 3 | Codebook | v0, then a pilot: 3 blind labelers on 100 stratified sentences; revise; repeat until pairwise agreement plateaus (expect 2–3 rounds) | Agreement per round logged; final codebook frozen and tagged; add a "For Dylan: codebook ready to review" note to PROGRESS.md and **continue without waiting**. If his later feedback changes a definition, relabel then |
| 4 | Validation sets | (a) a 350-sentence stratified set (over-sample POLICY) labelled by 3 blind agents with majority vote; (b) a **100-sentence sheet for Dylan** (CSV with blank columns plus a one-page instruction drawn from the codebook), drawn without overlap or deliberately overlapping, your choice; document it; (c) TDW human labels loaded | Write the sheet to `labels/human/`, note it under "For Dylan", and **continue without waiting**. All human-vs-model tables are generated by a script that reruns once the sheet is filled; until then, report the majority-vote and TDW validation and mark the human column "pending" |
| 5 | Scoring | Sentence-level RoBERTa on all sentences (GPU). Claude labels on a ~8–12k stratified training set → distilled local classifier → applied to the full corpus (section 5) | Distilled model held-out agreement with Claude labels reported; agreement with Dylan's 100 and with TDW reported for RoBERTa, Claude-direct and the distilled model |
| 6 | Measurement results | Accuracy, kappa and confusion matrices by function and topic; label-stability check (re-label 200 sentences, measure agreement with the first pass); FinBERT caution table | Tables generated by script |
| 7 | Indices | Per-document indices: guidance, assessment, outlook (+ per topic), each document's level and its change from the previous release; optional version with macro information removed | Index plots sanity-checked against known episodes (2004 "measured pace", 2008–09, 2013 taper, 2019 pivot, 2022 hikes) |
| 8 | Main analysis | Event-study regressions; local projections; robustness: zero-rate era vs normal, excluding 2020, statements vs minutes, placebo (random non-event days and shuffled tone), alternative surprise measure | All tables and figures generated by script; results written up in plain language in PROGRESS.md; write a one-page "results story" brief under "For Dylan" and **go straight on to drafting** |
| 9 | Forecasting (secondary) | As in decision 3 | Clark–West p-values reported |
| 10 | Figures and tables | One script regenerates all of `results/` | Clean rerun from processed data succeeds |
| 11 | Paper | `paper/main.tex` in arXiv style; numbers via `\input` of generated `.tex` macros; `refs.bib` with **every reference checked against the actual source** (DOI or publisher page) | Compiles with tectonic; no hand-typed numbers; check that every claim matches a table |

---

## 5. LLM labelling with no API billing

- **Run labelling as Claude Code subagents** (Agent tool, `model: "haiku"` or
  `"sonnet"`). This runs on the Max subscription. Never use the Anthropic SDK or API.
- **Batching:** write `labels/batches/<set>/batch_NNNN.jsonl`, 200–400 sentences per
  batch, each line `{id, prev_text, text}` with no dates. A labeler's prompt contains only
  the codebook path, its batch path and its output path. It must not read anything else
  in the repo; that is how it stays blind.
- **Resumable:** skip batches whose outputs already exist. A validator script checks the
  schema, that every ID is present, and the value ranges, and re-queues failures.
- **Choosing the production labeler:** on the 350-sentence set, compare Haiku and Sonnet
  against the majority vote (and, once it exists, Dylan's 100). Use the cheapest model
  within ~5 points of the best. Log the decision.
- **Distillation** is the main token saver. Claude labels only a stratified ~8–12k
  sentences (after de-duplication). Fine-tune a multi-head classifier (function, topic,
  direction) on the RTX 2070 Super (8 GB), starting from `gtfintechlab/FOMC-RoBERTa` or
  `roberta-base` (try both if cheap). Apply it to the full corpus. This also makes the
  measure reproducible: a fixed local model, unlike sampled LLM output. Scoring the whole
  corpus directly with Claude is optional and only if the distilled model clearly falls
  short.
- **Blind validators** for phases 3–4: 3 agents. Mix them for some diversity, e.g.
  Opus at low effort, Sonnet, Haiku, or 2×Sonnet + 1×Opus. Their prompts are identical
  apart from the batch paths.
- **Rate limits:** the subscription has rolling usage limits. Pace the large label runs.
  If you hit a limit, record exactly where you stopped in PROGRESS.md so the next session
  can resume.
- **Workflow tool:** Dylan's kickoff message authorizes multi-agent orchestration, so you
  may use Workflow scripts for the fan-out labelling. Keep each run's agent count
  reasonable and resumable.

---

## 6. Repo conventions

```
CLAUDE.md  HANDOFF.md  PROGRESS.md  README.md  environment.yml  config.yaml  run_all.py
src/ftyields/{data,text,label,model,analysis,figures}/
codebook/codebook.md            (+ CHANGELOG)
labels/{pilot,gold,human,train,batches,out}/
data/{raw,interim,processed}/   (large intermediates in .gitignore; raw text and small tables committed)
results/{tables,figures,macros}/
paper/{main.tex,refs.bib,arxiv.sty or equivalent}
tests/
```
- Every parameter (dates, horizons, paths, seeds, models) lives in `config.yaml`.
- Results are in **basis points** and dates are **trading days**; document the event-day
  alignment rule (release on day t means the change from close t−1 to close t; statements
  come out at 2:00pm ET, before the close).
- Fix seeds everywhere, including `random_state` for every library.
- Time-series hygiene: anything fit (scalers, models) is fit on training data only; leave
  a gap equal to the horizon between training and test; macro data is aligned to its
  publication date.
- **The repo is public, as Dylan intends.** Commit and push on your own judgment, without
  asking. Good moments are verified milestones (at minimum the end of each phase), with
  clear messages.
- **Git workflow: branch → PR → merge. Never work on `main`.**
  - Every unit of work gets a branch: `phase-N-<topic>`, or `phase-N-<topic>-<subtask>`
    for a subagent's piece.
  - Commit and push the branch freely.
  - When the work is verified: `gh pr create` with a summary of the change and its
    verification, then `gh pr merge <n> --merge` (a normal merge commit). Afterwards
    `git switch main && git pull`.
  - Never squash, rebase, amend, force-push, delete branches or tags, or commit, merge or
    push on `main` directly. To undo something, add a commit (`git revert`). Merged
    branches stay as a record.
  - Parallel subagents use separate branches or worktrees (Agent `isolation: "worktree"`)
    so they never share a working tree. You merge their PRs one at a time, resolving
    conflicts with new commits.
- **Large-file guard (set up in phase 0, then automatic):**
  - Keep generated artifacts out of the repo: model checkpoints (`models/`, `*.pt`,
    `*.bin`, `*.safetensors`, `*.ckpt`), `data/interim/`, large parquet/feather files,
    training logs and checkpoints, and LaTeX build files (`*.aux`, `*.log`, …). Commit
    the compiled paper PDF.
  - Add a git **pre-commit hook** that refuses any staged file over 50 MB and any file
    matching the patterns above. Use the `pre-commit` framework's
    `check-added-large-files --maxkb=51200` plus a small local hook for the extension
    list. Test it once by staging a dummy large file and confirming the commit is
    refused.
  - Model weights are not versioned in git. `run_all` rebuilds them from the committed
    labels and code. If Dylan later wants the distilled model shared, publish it to the
    Hugging Face Hub (ask first, since that is outward-facing).
- Public repo: commit only data you are allowed to redistribute. FOMC texts and FRED
  series are fine. For third-party datasets (Trillion Dollar Words labels, the surprise
  series), check the license; by default commit a download script, not the data.

### Project environment (build in phase 0, then use it for everything)
- **Write `environment.yml`** at the repo root. Name `ftyields`, channel `conda-forge`,
  Python 3.11. Conda packages: numpy, pandas, scipy, scikit-learn, statsmodels,
  matplotlib, pyarrow, pyyaml, tqdm, requests, pandas-datareader, pytest, pre-commit,
  tectonic, and a `pip` section with
  `--extra-index-url https://download.pytorch.org/whl/cu121`, `torch==2.5.1`
  (2.5.1+cu121 is known to work with this GPU and driver), `transformers`, `datasets`,
  `huggingface-hub`, `accelerate`. Set lower bounds and let the solver choose versions;
  don't hand-pin everything.
- **Build it:** `conda env create -f environment.yml`, using
  `C:/Users/dylan/anaconda3/Scripts/conda.exe` if `conda` isn't on PATH. Run the phase-0
  import checks. If the solve or the CUDA check fails, fix `environment.yml` and run
  `conda env update -f environment.yml --prune`. Never delete or recreate the env, and
  never touch other envs.
- **Lock it:** after the checks pass, run `conda env export -n ftyields --no-builds` into
  `environment.lock.yml`, remove the machine-specific `prefix:` line, and commit both
  files. Whenever a dependency is added, edit `environment.yml`, run
  `conda env update ... --prune`, and regenerate the lock in the same PR.
- **Use only this env.** Every script and subagent calls
  `C:/Users/dylan/anaconda3/envs/ftyields/python.exe` (or
  `conda run -n ftyields <cmd>`). Never use the base env, `ML`, or bare `python`/`pip`.
  Install packages only by editing `environment.yml`, never with ad-hoc
  `pip install`. Put the interpreter path in `config.yaml` / README and in every
  subagent brief that runs code.

### Environment notes (Windows 10, Git Bash + PowerShell)
- Anaconda: `C:\Users\dylan\anaconda3`. The existing `ML` env (torch 2.5.1+cu121,
  transformers 4.40, numpy 2.5.4) **breaks on `from transformers import pipeline`**
  because of a numpy/C-extension mismatch. Create a fresh env (`ftyields`) with pinned,
  mutually compatible versions instead.
- GPU: RTX 2070 Super, 8 GB, CUDA works.
- The HF cache already holds `gtfintechlab/FOMC-RoBERTa`, `ProsusAI/finbert` and the
  dataset `vtasca/fomc-statements-minutes`.
- `gh` is authenticated as dylanzlatev.
- No LaTeX is installed. Install `tectonic` (conda-forge) in phase 0 or 11.
- Use the Bash tool with forward-slash paths; the shell has no `python` on PATH, so call
  the env's python.exe explicitly.

---

## 7. Delegation and verification policy

**Your job is judgment, integration and verification.** Delegate execution. Use the
lowest tier that can do the task correctly.

| Tier | Use for |
|---|---|
| **You (Opus, high)** | Phase plans, codebook design and revisions, reading disagreement cases, choosing models, reviewing every deliverable, interpreting results, writing the paper's argument |
| **Opus at medium or low effort** | A blind validator; hard debugging that Sonnet failed at twice; a careful second-opinion review of the main analysis code |
| **Sonnet** | Writing pipeline code (data, text cleaning, scorers, distillation training, analysis, figures); finding and checking literature with web access; drafting paper sections from your outline; labelling if Haiku fails the threshold |
| **Haiku** | Bulk labelling batches; mechanical edits; format conversions; running scripts and summarizing their output; collecting log tails |

Dylan explicitly authorized setting `model` and `effort` on subagents for this purpose.

**Briefing subagents.** Each brief states the goal, exact input and output paths, the
interface or schema, what not to touch, and the acceptance check. It also asks for a
short report: what changed, the commands run, the check output, and anything uncertain.
Don't paste this whole file into briefs; give only what the task needs.

**Verification. Don't take a subagent's word for anything.**
- Run the acceptance check yourself. Read the critical code paths yourself: event-day
  alignment, the training/test split, index construction, regression specifications.
- Spot-check data. Print shapes, date ranges, NaN counts and a few rows; compare against
  known facts (e.g. the 2022-03-16 hike to 0.25–0.50; the 2020-03-15 emergency cut).
- Check results for plausibility. Release-day yield moves are usually a few bp; anything
  claiming large R² or t > 6 needs a leakage hunt before you believe it.
- When a subagent's work fails review, send it back once with specific feedback via
  SendMessage. If it fails again, take it up a tier or do it yourself.
- Keep diffs small and phase-scoped; commit after verification.

**Never stop to wait for Dylan.** The only exception is the session-start warning that
`ANTHROPIC_API_KEY` is set. In that case, pause all subagent and labelling work and leave
a note, because continuing would bill the API. Everything else goes in `PROGRESS.md` under
"For Dylan", "Proposed deviations" or "Requests for Dylan", and you carry on.

### Prohibited actions (always) and what to do instead

These are never necessary for this project, so they are banned outright for you and every
subagent. Put the relevant line in each subagent brief.

| Never | Do instead |
|---|---|
| Force-push, rebase, amend, reset --hard, filter history, delete branches or tags | New commits on a branch; `git revert` to undo |
| Commit, merge or push on `main` | Feature branch, then `gh pr create`, then `gh pr merge --merge` |
| `gh pr merge --admin`, change repo settings, branch protection, rulesets, releases, gists, secrets | Log the request under "Requests for Dylan" |
| Recursive or bulk deletes (`rm -r`, `Remove-Item -Recurse`, `shutil.rmtree`, `git clean`) | Overwrite generated outputs in place; keep scratch output in a gitignored `tmp/` |
| Publish anywhere but this repo (Hugging Face Hub, PyPI, gists, uploads) | Keep artifacts local; log the idea for Dylan |
| Call the Anthropic API or install/import the `anthropic` SDK | Claude Code subagents |
| Modify the old repo, `.claude/settings*.json`, `.claude/hooks/`, `.git/hooks`; use `--no-verify` or `commit -n` | Read the old repo with Read/Grep/Glob; log config needs for Dylan; fix whatever the hook reports |

**These bans are enforced, not just requested.**
- `.claude/settings.json` holds deny rules and auto-mode hard-deny rules.
- `.claude/hooks/guard.sh` runs before every shell or file-writing tool call, for you and
  for all subagents, and rejects these actions with a message naming the alternative.
- GitHub branch protection on `main` requires PRs and refuses force-pushes and deletions,
  even with admin rights.

If the guard blocks you, treat that as a signal to change approach. Never try to work
around it with other spellings, scripts or wrapper tools.

---

## 8. Literature to read, check and cite (verify every one; don't cite from memory)
Gürkaynak, Sack & Swanson (2005, IJCB), "Do actions speak louder than words?";
Lucca & Moench (2015, JF), pre-FOMC drift; Nakamura & Steinsson (2018, QJE);
Hansen & McMahon (2016, JIE); Cieslak & Schrimpf (2019, JIE); Shah, Paturi & Chava (2023,
ACL), Trillion Dollar Words; Bauer & Swanson (2023, NBER Macro Annual / AER), reassessing
monetary-policy surprises; Gorodnichenko, Pham & Talavera (2023, AER), voice of monetary
policy; Aruoba & Drechsel (2024), NLP monetary-policy shocks; Jordà (2005, AER), local
projections; Clark & West (2007, J. Econometrics); Litterman & Scheinkman (1991).
Also look for work on look-ahead bias in LLMs used for financial text, and cite it when
justifying the date-stripping and the distilled local model.

## 9. What a good final paper argues
A careful measurement section (codebook, validation against humans, why FinBERT
measures the wrong thing). Then the main result: which kind of Fed language (guidance,
assessment or outlook) moves which part of the curve on release days, beyond the rate
surprise, and whether the effect persists (local projections). Robustness. Forecasting is
a short secondary section that reports honest results, null or not. Write plainly, give
uncertainty for every estimate, and don't claim causality beyond what the event design
supports.
