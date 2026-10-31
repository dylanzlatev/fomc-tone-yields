# PROGRESS — FOMC Tone & the Yield Curve

Running log kept by the orchestrator session. Read this at the start of every session
(see CLAUDE.md). Newest entries at the top of each section.

## Current state
- **Phase:** 5 (scoring) in progress. Phases 0–4 merged (PRs #1–#8); 5a (RoBERTa scores)
  merged (#9). Distiller + training-set sampler being built on `phase-5-distill`
  (smoke-tested on the 650 existing labels; real training needs the production labels).
- **Next session — start here:**
  1. The restart loads `.claude/agents/fomc-labeler.md` (Read/Write only, `omitClaudeMd:
     true`). Verify with a canary prompt (ask it to quote any CLAUDE.md lines / mention of
     yields in its context) before using it for labelling.
  2. Label the gold set with **Haiku** via `fomc-labeler`: prompt = codebook path +
     `labels/batches/gold/batch_000{1,2}.jsonl` + output
     `labels/out/gold/haiku/batch_000{1,2}.jsonl`. Then
     `python -m ftyields.label.validation`; choose the production labeler (cheapest within
     ~5 points of the best on full match / direction exact vs the leave-out consensus;
     Opus leave-out numbers are inflated by the same-model pair — weigh Sonnet vs Haiku
     on the Opus-pair consensus). Log the decision.
  3. Production labels: `labels/batches/train/` (≈34 batches × 300) with the chosen labeler
     → `labels/out/train/<labeler>/`. Resumable: `python -m ftyields.label pending --set
     train --labeler <labeler>`; validate each with `python -m ftyields.label validate`.
     Pace for rate limits (e.g. 4–6 parallel agents per wave) and record the last
     completed batch here if a limit hits.
  4. Claude-direct on TDW: label `labels/batches/tdw_test/` (479 sentences, gitignored,
     regenerate with `python -m ftyields.label.tdw`) with the production labeler.
  5. Train the distiller (`python -m ftyields.model.distill`, see phase-5-distill notes),
     evaluate year-held-out and vs gold, apply to the corpus.
- **Interrupted batch jobs:** none.

## Phase status

| # | Phase | Status | Verified | PR |
|---|---|---|---|---|
| 0 | Setup | done | yes | #1 |
| 1 | Data | done | yes | #2, #3, #4 |
| 2 | Text | done | yes | #6 |
| 3 | Codebook | done (v2 frozen) | yes | #5, #7 |
| 4 | Validation sets | done (human pending) | yes | #8 |
| 5 | Scoring | in progress (5a done) | | #9 |
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

### 2026-10-30 — Phase 1 (data) — done
Built by 3 Sonnet subagents (1A yields+surprises, 1B corpus+calendar in worktrees, 1C
integration); merged via PRs #2, #3 into `phase-1-data`. Outputs: `data/raw/fred/*`,
`data/raw/jk/*`, `data/raw/text/fomc_documents.parquet`, `data/raw/MANIFEST.csv`
(source, SHA-256, license, committed?), `data/processed/{yields_cmt_daily, yields_gsw_daily,
fomc_events, calendar_reconciliation, event_days}.csv`, `results/tables/coverage_events.{csv,tex}`.
`run_all.py --only data` rebuilds it all.
- **Events 2000–2025: 433** = 224 statements (207 scheduled, 17 unscheduled incl. 2020
  notation votes and the 2020-08-27 / 2025-08-22 framework statements) + 209 minutes.
  428 have text; no text for `min_19991221` (released 2000-02-03), `stmt_20200319`,
  `stmt_20200331`, `stmt_20200827`, `stmt_20250822`. 30 dataset minutes rows are exact
  duplicates of matched documents (conference-call stubs) and are excluded with reasons.
- **Calendar reconciliation:** 30 cases across Fed pages / USMPD / Bauer–Swanson / TDW /
  dataset, none unresolved — e.g. 2008-01-21 call → released 2008-01-22 08:20; the
  dataset's "2008-08-08" documents are the Aug 5 meeting; four 2009 docs dated a day
  early; 2007-10-09 one minutes release covering Sep 18 + Aug 10/16 calls.
- **Moved to the next trading day (3):** 2010-05-09 (Sun 21:15), 2020-03-15 (Sun 17:00),
  `min_20040921` released 2004-11-11 (Veterans Day). None after the close.
- **Surprises:** B–S MPS for 205/224 statements (missing: 2001-09-17 by design, 2010-05-09,
  2024–25); USMPD STMT 223, J-K 221; USMPD minutes 169/209 (from 2005). One-day/one-week
  date fallbacks logged (USMPD 2019-10-04 → 2019-10-11; USMPD minutes 2025-12-31 →
  2025-12-30). Pairwise corr of the surprise series 0.95–0.99.
- **Verified myself:** read `map_to_trading_day`, `event_days_table`, `change` (break
  rule (prev, event]); recomputed d2Y/d10Y/dslope from raw FRED for 6 events incl. the
  Sunday, MLK-weekend and Veterans-Day cases — all exact; spot-checked release times
  (2022-03-16 14:00, 2012-12-12 12:30, 2004-06-30 14:15). Plausibility: std d2Y 7.6bp on
  statement days vs 5.3 minutes days vs 5.5 all days; corr(d2Y, MPS) = 0.52 on scheduled
  statement days (n=191); largest |d2Y| days 2008-01-22 (−28), 2008-03-11 (+27),
  2023-12-13 (−27). 108 tests pass. No surprise values in committed files.

### 2026-10-30 — Phase 2 (text) — done (PR #6)
Sonnet subagent in a worktree; reviewed by me, sent back once (short-sentence rule).
`data/processed/sentences.parquet` (one row per sentence occurrence) and
`sentences_unique.parquet` (de-duplicated on masked, normalised text);
`results/tables/sentence_counts.{csv,tex}`; `run_all.py --only text`.
- **Counts:** 428 documents (220 statements, 208 minutes); **51,006 sentences, 41,974
  unique** (statements 3,007 → 1,558 unique; minutes 47,999 → 41,334 unique). 2,356
  minutes sentences flagged `is_procedural` (annual organisational/authorisation blocks);
  41,177 unique uids outside them. Median words/sentence 23 (statements), 26 (minutes).
- **Cleaning:** headers, attendance blocks, vote tallies, footnotes, web boilerplate,
  adjournment/next-meeting lines removed; dissent explanations, directive text and goals
  statements kept; mojibake repaired (36 sentences keep an unrecoverable `[FRAC]`).
- **Masking:** names (772-name gazetteer from attendance/vote lists + title/honorific
  patterns) → `[NAME]`, dates → `[DATE]`, years 1990–2029 → `[YEAR]`; numbers untouched.
  Known residual era cues not masked: bare month names, named institutions and episodes
  ("the pandemic", "Silicon Valley Bank", "the invasion of Ukraine"). The codebook forbids
  outside-knowledge reasoning; the paper will state this limitation.
- **Sections (minutes):** financial_markets, staff_economic, staff_financial,
  staff_outlook, participants_views, policy_action, other — headings from 2009, opening
  triggers before that.
- **Verified myself:** read a whole statement (S20220316) in order, the opening of
  M20220316 (no header/attendance residue), dissent sentences; the "�" I saw in a 2022
  sentence was a console rendering of a real em dash (U+2014). Fix I required: the
  fragment rule had merged complete short sentences ("Inflation remained elevated.");
  now only splitter artefacts merge (18 sentences of 2–3 words remain, all legitimate).
  Byte-identical reruns; 171 tests pass on main.

<details><summary>30 random statement sentences (masked text as labelers see it; seed 20261030)</summary>

- `S20030625_0006` [statement] The Committee perceives that the upside and downside risks to the attainment of sustainable growth for the next few quarters are roughly equal.
- `S20030916_0002` [statement] The evidence accumulated over the intermeeting period confirms that spending is firming, although the labor market has been weakening.
- `S20090624_0005` [statement] The prices of energy and other commodities have risen of late.
- `S20101214_0012` [statement] The Committee will regularly review the pace of its securities purchases and the overall size of the asset-purchase program in light of incoming information and will adjust the program as needed to best foster maximum employment and price stability.
- `S20110809_0015` [statement] The Committee also will maintain its existing policy of reinvesting principal payments from its securities holdings.
- `S20120313_0007` [statement] The Committee expects moderate economic growth over coming quarters and consequently anticipates that the unemployment rate will decline gradually toward levels that the Committee judges to be consistent with its dual mandate.
- `S20130320_0007` [statement] The Committee continues to see downside risks to the economic outlook.
- `S20131030_0011` [statement] Accordingly, the Committee decided to continue purchasing additional agency mortgage-backed securities at a pace of $40 billion per month and longer-term Treasury securities at a pace of $45 billion per month.
- `S20140129_0020` [statement] In determining how long to maintain a highly accommodative stance of monetary policy, the Committee will also consider other information, including additional measures of labor market conditions, indicators of inflation pressures and inflation expectations, and readings on financial developments.
- `S20140618_0009` [statement] The Committee recognizes that inflation persistently below its 2 percent objective could pose risks to economic performance, and it is monitoring inflation developments carefully for evidence that inflation will move back toward its objective over the medium term.
- `S20140917_0004` [statement] Inflation has been running below the Committee's longer-run objective.
- `S20150917_0015` [statement] The Committee anticipates that it will be appropriate to raise the target range for the federal funds rate when it has seen some further improvement in the labor market and is reasonably confident that inflation will move back to its 2 percent objective over the medium term.
- `S20150917_0016` [statement] The Committee is maintaining its existing policy of reinvesting principal payments from its holdings of agency debt and agency mortgage-backed securities in agency mortgage-backed securities and of rolling over maturing Treasury securities at auction.
- `S20161102_0000` [statement] Information received since the Federal Open Market Committee met in September indicates that the labor market has continued to strengthen and growth of economic activity has picked up from the modest pace seen in the first half of this year.
- `S20170315_0001` [statement] Job gains remained solid and the unemployment rate was little changed in recent months.
- `S20170726_0010` [statement] The stance of monetary policy remains accommodative, thereby supporting some further strengthening in labor market conditions and a sustained return to 2 percent inflation.
- `S20180131_0003` [statement] Market-based measures of inflation compensation have increased in recent months but remain low; survey-based measures of longer-term inflation expectations are little changed, on balance.
- `S20190130_0005` [statement] Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability.
- `S20190619_0008` [statement] In light of these uncertainties and muted inflation pressures, the Committee will closely monitor the implications of incoming information for the economic outlook and will act as appropriate to sustain the expansion, with a strong labor market and inflation near its symmetric 2 percent objective.
- `S20190918_0000` [statement] Information received since the Federal Open Market Committee met in July indicates that the labor market remains strong and that economic activity has been rising at a moderate rate.
- `S20200315_0022` [statement] Voting against this action was [NAME], who was fully supportive of all of the actions taken to promote the smooth functioning of markets and the flow of credit to households and businesses but preferred to reduce the target range for the federal funds rate to 1/2 to 3/4 percent at this meeting.
- `S20200323_0008` [statement] In connection with these plans, the Committee voted unanimously to authorize and direct the Federal Reserve Bank of New York, until instructed otherwise, to execute transactions in the System Open Market Account in accordance with the following domestic policy directive:
- `S20200916_0005` [statement] The path of the economy will depend significantly on the course of the virus.
- `S20210317_0011` [statement] In addition, the Federal Reserve will continue to increase its holdings of Treasury securities by at least $80 billion per month and of agency mortgage-backed securities by at least $40 billion per month until substantial further progress has been made toward the Committee's maximum employment and price stability goals.
- `S20210616_0016` [statement] The Committee's assessments will take into account a wide range of information, including readings on public health, labor market conditions, inflation pressures and inflation expectations, and financial and international developments.
- `S20220504_0012` [statement] In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook.
- `S20220727_0009` [statement] The Committee is strongly committed to returning inflation to its 2 percent objective.
- `S20230726_0005` [statement] The extent of these effects remains uncertain.
- `S20240731_0000` [statement] Recent indicators suggest that economic activity has continued to expand at a solid pace.
- `S20240918_0001` [statement] Job gains have slowed, and the unemployment rate has moved up but remains low.

</details>

<details><summary>30 random minutes sentences (masked text as labelers see it; seed 20261030)</summary>

- `M20000321_0000` [financial_markets] The Manager of the System Open Market Account reported on recent developments in foreign exchange markets.
- `M20000628_0045` [staff_financial] Open market operations during the intermeeting period were directed toward implementing the desired increased pressure on reserve positions, and the federal funds rate averaged very close to the Committee's 6-1/2 percent target.
- `M20000628_0096` [participants_views] Looking beyond the near term, a number of members, noting that core measures of consumer prices had been rising more rapidly this year, were concerned that these prices might well continue to accelerate gradually, even assuming that economic expansion would be sustained at a pace close to the economy's potential.
- `M20000822_0077` [participants_views] These developments had much improved the prospects for a sustainable economic expansion at the prevailing stance of monetary policy.
- `M20010821_0113` [participants_views] Some members expressed a degree of concern, however, about the longer-term outlook for inflation.
- `M20020813_0097` [participants_views] In addition, already enacted income tax cuts and the tax expensing provision for certain investment outlays would help to support both consumer and business expenditures.
- `M20040316_0130` [policy_action] While many viewed it as a close call, all members indicated that they could support a proposal to retain the existing wording involving a slight tilt toward the possibility of some further disinflation.
- `M20071031_0022` [staff_economic] Real outlays on consumer services were strong in August because of a weather-induced jump in energy services.
- `M20080808_0018` [staff_economic] The factory utilization rate held steady in June at a rate below its long-run average but was still well above its low rate from [YEAR] through [YEAR].
- `M20090811_0040` [staff_economic] However, household net worth received a boost from the rise in equity prices since their low in March.
- `M20091104_0194` [participants_views] Moreover, these participants noted that banks might seek to reduce appreciably their excess reserves as the economy improves by purchasing securities or by easing credit standards and expanding their lending substantially.
- `M20100316_0184` [policy_action] At the conclusion of the discussion, the Committee voted to authorize and direct the Federal Reserve Bank of New York, until it was instructed otherwise, to execute transactions in the System Account in accordance with the following domestic policy directive:
- `M20101103_0084` [staff_financial] In the secondary market for syndicated leveraged loans, prices of loans continued to move up and bid-asked spreads narrowed a bit further.
- `M20101214_0005` [financial_markets] In addition, the Manager updated the Committee on the SOMA's holdings of foreign-currency instruments.
- `M20110315_0101` [staff_financial] Currency continued to advance at a relatively fast rate in January and February, likely boosted by a strong expansion in foreign holdings of U.S. bank notes.
- `M20121212_0226` [policy_action] Inflation has been running somewhat below the Committee's longer-run objective, apart from temporary variations that largely reflect fluctuations in energy prices.
- `M20140129_0077` [other] Drawings must be approved by the Committee (or by the Subcommittee, if the Subcommittee believes that consultation with the full Committee is not feasible in the time available, or by the Chairman, if the Chairman believes that consultation with the Subcommittee is not feasible in the time available) if the swap drawing proposed by a foreign bank exceeds the larger of (i) $200 million or (ii) 15 percent of the size of the swap arrangement.
- `M20140129_0179` [staff_economic] Over that period, consumer energy prices declined, consumer food prices rose modestly, and core PCE prices--which exclude consumer food and energy prices--increased slightly more than 1 percent.
- `M20160127_0116` [other] Inflation, employment, and long-term interest rates fluctuate over time in response to economic and financial disturbances.
- `M20160727_0016` [other] Also, because episodes of financial stress can arise with little warning, policymakers noted the advantage of being operationally ready for such situations; however, they also recognized that such operational readiness could entail some costs.
- `M20160727_0168` [participants_views] Household spending had been growing strongly, but business fixed investment had been soft.
- `M20161102_0159` [staff_outlook] Consistent with the downside risks to aggregate demand, the staff viewed the risks to its outlook for the unemployment rate as tilted to the upside.
- `M20170315_0067` [staff_economic] Although state and local government payrolls expanded in January and February, nominal construction spending by these governments fell sharply in January.
- `M20180131_0167` [staff_economic] Consumer sentiment in early January, as measured by the University of Michigan Surveys of Consumers, remained upbeat.
- `M20191030_0213` [staff_outlook] The staff continued to view the uncertainty around its projections for real GDP growth, the unemployment rate, and inflation as generally similar to the average of the past 20 years.
- `M20200610_0112` [staff_economic] Output in the mining sector—which includes crude oil extraction—also decreased, reflecting the effects of low crude oil prices.
- `M20200916_0054` [financial_markets] On net, financial conditions eased over the intermeeting period.
- `M20211103_0211` [policy_action] Members agreed that the postmeeting statement should acknowledge that the sectors of the economy most adversely affected by the pandemic had improved in recent months, but that the summer's rise in COVID-19 cases had slowed their recovery.
- `M20220615_0083` [staff_financial] Sovereign yields in most AFEs rose over the intermeeting period amid investors' concerns about further inflationary pressures and major central banks' policy communications suggesting a firmer stance of policy.
- `M20230726_0217` [policy_action] The extent of these effects remains uncertain.

</details>

### 2026-10-30 — Phase 3 (codebook) — done
- **Pilot protocol:** each round = 100 fresh stratified unique sentences
  (`python -m ftyields.label.pilot --round k`; statements ×4, policy-like ×2 weight;
  excludes earlier rounds and flagged procedural blocks), 3 blind labelers (Explore
  agents: Haiku, Sonnet, Opus at low effort) with identical prompts except the output
  path; validate → `python -m ftyields.label agree` → I read every disagreement →
  revise. Tables in `labels/pilot/rK_agreement/`.
- **Write path:** Explore agents can't reliably use Bash heredocs here (quoting error);
  PowerShell `[IO.File]::WriteAllText(path, @'…'@)` works. In round 1 Haiku refused to
  write and returned its labels as text; I saved them verbatim (Write tool) — the
  validator passed. From the next session use the `fomc-labeler` agent (Write tool).
- **Round 1 (codebook v0):** function Fleiss κ 0.976 (98% all-agree), topic κ 0.932,
  direction all-exact 80%, α_interval 0.914 (0.917 excluding PROCEDURAL), full match 77%;
  pairwise direction within ±1: 98–100%. 23 disagreements, read individually; patterns
  and the resulting v1 rules are in `codebook/CHANGELOG.md`.
- Phase-2 bug found via the pilot: the splitter treats "E&S." as an abbreviation, so one
  unique item holds two sentences. To fix in a phase-2 follow-up (low impact).
- **Round 2 (v1):** function κ 0.922 (93% all-agree), topic κ 0.951, direction exact 81%,
  α 0.917, full match 74%; 26 disagreements → v2 (money-market observations, purchase
  pace, non-stance dissents, directionless patience, policy arguments, unevaluated
  numbers; see CHANGELOG).
- **Round 3 (v2):** function κ 0.960, topic κ 0.970, direction exact 82%, α 0.876, full
  match 78%; 22 disagreements, all ±1 strength calls on graded wording. Odd one out:
  Haiku 13, Opus-low 8, Sonnet 0; Opus–Sonnet full match 91%, direction QWK 0.952.
- **Codebook v2 frozen** (tag `codebook-v2`): agreement plateaued (exact direction
  80 → 81 → 82%, function/topic κ ≥ 0.92 throughout). Cross-round table generated by
  `python -m ftyields.label.pilot --summarize` → `results/tables/pilot_agreement.{csv,tex}`.
  Caveat for the paper: three same-family models; the high κ partly reflects correlated
  judgements, which is why Dylan's 100 labels and TDW are the independent checks.

### 2026-10-30 — Phase 4 (validation sets) — done
- **(a) Gold set:** 350 unique sentences (`python -m ftyields.label.gold`; statements ×3,
  policy-like ×2.5 weight; pilot items and procedural-only uids excluded; 134 policy-like).
  Labelled blind by **Sonnet, Opus-low, Opus-medium** (Explore agents, identical prompts).
  Agreement (`results/tables/gold_agreement.csv`): function κ 0.959, topic κ 0.958,
  direction exact 87.7%, α 0.949, full match 83.7%; 0 items without a function majority.
  Majority vote → `labels/gold/gold_majority.jsonl`. I read a random sample of 14 majority
  labels against their sentences: all sensible.
- **Why not Haiku in the gold trio:** Explore-role Haiku refuses to write files, and
  piping 350 lines through the orchestrator is wasteful; Haiku is labelled next session
  with `fomc-labeler` and scored as a *candidate* (it never enters the gold vote).
- **(b) Dylan's sheet:** `labels/human/human_sheet.csv` (100 items drawn from inside the
  gold set, stratified by doc type × policy-like, floor 5 per stratum; blank label
  columns) + `labels/human/INSTRUCTIONS.md` (condensed from codebook v2, ~1,200 words).
- **(c) TDW:** `gtfintechlab/fomc_communication` downloaded (CC BY-NC → not committed;
  manifest rows only); test 496 → 479 after dropping 17 sentences also in train
  (dovish 130 / hawkish 109 / neutral 240); written as labelling set `tdw_test` (2
  batches, gitignored) for the Claude-direct comparison in phase 5.
- **Validation tables** (`python -m ftyields.label.validation`): `gold_agreement`,
  `labeler_vs_reference` (vs gold majority and vs leave-out consensus),
  `human_vs_models` (status "pending" until the sheet is filled; reruns fill it in).
- **Production-labeler choice — provisional, not yet made.** Leave-out full match:
  Opus-low 97.0% (n=302), Opus-med 96.4% (304), Sonnet 89.6% (327). The Opus numbers are
  inflated because each Opus run's reference contains the other Opus run (same model).
  Final choice after (i) Haiku's gold labels and (ii) Dylan's 100 as an independent
  reference; default meanwhile: Sonnet, with Opus-low if Sonnet trails by > 5 points on
  the human reference.

### 2026-10-30 — Phase 5a (RoBERTa sentence scores) — done (PR #9)
- `python -m ftyields.model.score` (= `run_all.py --only score`): FOMC-RoBERTa on the raw
  text of all 51,006 sentence occurrences (42,289 unique raw texts); score =
  P(hawk) − P(dov), label map in config (LABEL_0 dovish, 1 hawkish, 2 neutral).
  → `data/processed/roberta_sentence_scores.parquet`. 19 occurrences truncated at 256 tokens.
- **Precision decision:** fp16 passed a 200-sentence check (max diff 0.0091) but on the
  full corpus it moved scores by up to 0.23 and flipped 12 labels vs fp32; even fp32 at
  batch 64 varied ~2e-4 with batch composition. Final: **fp32, batch size 1** (every score
  independent of batch composition; bit-identical reruns; ~10 min on the 2070S).
- **Probe table** `results/tables/sentence_probe.{csv,tex}` (FinBERT caution note)
  reproduces the handoff's 8 sentences exactly (RoBERTa +0.99, +0.99, −0.97, −0.15,
  +1.00, +1.00, 0.00, +0.65; FinBERT positive/negative/…).
- Label shares: minutes 24% dovish / 23% hawkish / 53% neutral; statements 36 / 25 / 39.
  Spot check: "Inflation remains elevated." scores 0.999 across its 12 occurrences.

## For Dylan
- **Codebook ready to review** — `codebook/codebook.md` (v2, frozen, tag `codebook-v2`) with
  the reasoning for every rule in `codebook/CHANGELOG.md` and per-round agreement in
  `results/tables/pilot_agreement.csv`. Decisions worth a look: the added GENERAL topic;
  minority views and dissents treated as hedges (strength 1); a wider trade deficit
  coded dovish (Shah et al. code it hawkish); implementation/market-functioning
  operations coded POLICY 0; same-topic contrasts netted into one aspect. I'm continuing
  with v2; if your feedback changes a definition, I'll bump the version and relabel.
- **Your 100 labels** — `labels/human/human_sheet.csv` with instructions in
  `labels/human/INSTRUCTIONS.md` (≈2 pages; full rules in the codebook). Fill
  function/topic/direction (second aspect optional) and confidence; save as CSV. Then
  `python -m ftyields.label.validation` regenerates `results/tables/human_vs_models.*`
  automatically. Nothing is waiting on it; it is the independent human anchor and will
  also settle the production-labeler choice.

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

- **Labeler blinding, residual leak (optional):** every agent's environment block shows
  the working directory `…\fomc-tone-yields`, so labelers can see the word "yields" in
  the folder name. If you want strict blinding for the production label run, you could
  run labelling from a neutrally named clone/worktree path (e.g. `C:\work\fomc-text`).
  Not blocking; labelling proceeds and the paper will note it.
- **New agent definition:** I added `.claude/agents/fomc-labeler.md` (Read/Write only,
  no CLAUDE.md). It isn't a settings or hook file, but it is under `.claude/`; delete it
  if you'd rather not have it.

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
- 2026-10-30 (phase 1): **Event-day alignment.** Trading days = FRED DGS10 business days.
  Release at ET datetime d → event day t = d's date if trading day and time < 15:30 ET
  (CMT quotes are taken ~3:30pm), else the next trading day. Change = y(t) − y(t−1).
  Recorded in `config.yaml` → `events`.
- 2026-10-30 (phase 1): **Event dates come from authoritative sources, not the text
  dataset.** The vtasca dataset's dates are wrong for some unscheduled actions (e.g.
  the 2008-01-21 call: the cut was announced 2008-01-22 08:20 ET). Statements: USMPD /
  Bauer–Swanson timestamps + Fed calendar pages; minutes: Fed pages, USMPD, TDW master
  file; every disagreement logged in `data/processed/calendar_reconciliation.csv`.
- 2026-10-30 (phase 1): **Rate surprise.** Main = Bauer–Swanson MPS (bp), which ends
  2023-12-13, so surprise-controlled statement regressions run 2000–2023 (the handoff's
  fallback rule: "only if unavailable" → it is unavailable for 2024–25). Robustness:
  MPS_ORTH (NA for 13 dates incl. most of 2020), USMPD STMT (through 2025; different
  construction), Jarociński–Karadi pc1. Minutes days: surprise = 0 by construction;
  USMPD minutes-window surprise available from 2005 as an optional control.
- 2026-10-30 (phase 1): **30Y suspension window (revised).** I expected DGS30 to be
  missing 2002-02-19 … 2006-02-08; verified that the current FRED file instead fills it
  with Treasury's extrapolated 30-year rate (a different construction), with breaks at
  the edges: 2002-02-19 Δ30Y = +17bp vs Δ20Y = +3; 2006-02-09 Δ30Y = −16bp vs Δ20Y = −3.
  Inside the window daily changes are smooth. Main line: keep the official series, and
  set to NaN any change whose interval crosses a break date (`fred.series_breaks`;
  neither break date is an FOMC event day, but 60-day local projections can cross them).
  Robustness: GSW zero-coupon factors (SVENY02/05/10/30; no gaps), and dropping
  `fred.suspended_30y_window`.
- 2026-10-30 (phase 1): **BLAS switched MKL → OpenBLAS.** conda-forge numpy came linked
  to MKL, whose delay-loaded DLLs crash (0xc06d007f) on any linalg call when python.exe is
  run without `Library/bin` on PATH (the way every script and agent runs it). Pinned
  `libblas=*=*openblas`; verified lstsq/polyfit/SVD/statsmodels HAC OLS, torch CUDA and
  the RoBERTa pipeline all work; `tests/test_env.py` guards it.
- 2026-10-30 (phase 3 prep): **Labeler blinding.** Verified from the Claude Code docs
  that general-purpose and custom subagents load CLAUDE.md (which states the research
  question) unless the agent definition sets `omitClaudeMd: true`; Explore/Plan skip it.
  Created `.claude/agents/fomc-labeler.md` (tools: Read, Write; `omitClaudeMd: true`) —
  an agent *definition*, not a settings/hook file; it only removes capabilities. Custom
  agents load at session start, so it is usable from the **next session**. Until then,
  pilot labelers run as built-in **Explore** agents with a `model` override; a canary
  Explore agent confirmed it sees no CLAUDE.md, no research question and no mention of
  yields, except the working-directory path `…\fomc-tone-yields` in its environment
  block. That residual leak (the folder name) can't be removed from inside a session;
  logged as a limitation and under "Requests for Dylan".
- 2026-10-30 (phase 1): `data/raw/** -text` in `.gitattributes` so committed raw files
  are byte-identical to the downloads (manifest SHA-256 stays valid).
- 2026-10-30: **Note for phases 5–6.** FOMC-RoBERTa was fine-tuned on the TDW labels,
  so RoBERTa-vs-TDW agreement is (partly) in-sample. Report it with that caveat; use the
  TDW test split and say we can't rule out that the released model saw it.
