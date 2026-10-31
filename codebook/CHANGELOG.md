# Codebook changelog

Not shown to labelers (they see only `codebook.md`).

## Frozen: v2 — 2026-10-30 (git tag `codebook-v2`)
Round 3 (100 fresh items, v2): function κ 0.960, topic κ 0.970, direction all-exact 82%,
α_interval 0.876, full match 78%; 22 disagreements. Exact direction agreement 80 → 81 →
82% and function/topic κ ≥ 0.92 in every round: a plateau. The remaining disagreements
are ±1 strength calls on graded wording ("modest pace", "considerable period") rather
than rule gaps; Haiku is the odd one out in 13 of 22 and Opus-low in 8, Sonnet in none.
The direction α dip in round 3 comes from the Haiku pairs (QWK 0.82–0.86 vs 0.95 for
Opus–Sonnet). Frozen without further changes. Any later change (e.g. after Dylan's
review) gets a new version and triggers relabelling.

## v2 — 2026-10-30 (after pilot round 2)
Round 2 (100 fresh items, v1): function κ 0.922 (93% all-agree), topic κ 0.951,
direction all-exact 81%, α_interval 0.917, full match 74%; 26 disagreements, most of
them ±1 strength calls. v1's same-topic netting rule worked (no labeler split same-topic
contrasts). Changes:
- **Observed money-market conditions** (policy rate vs target, facility/repo usage and
  why it changed) → ASSESSMENT/FINANCIAL 0; actions/decisions → POLICY (3 function splits).
- **Purchase pace**: changes in pace with an unchanged total, or for market functioning,
  → 0; only total size / holdings carry direction. POLICY row reworded accordingly.
- **Non-stance dissents** → 0. **Caution/patience without a stated direction** → 0
  (replaced the v0 anchor "will be patient as it determines future adjustments", whose
  direction was implicit, with "patient in beginning to normalize").
- **Arguments about a policy choice** → POLICY with the direction argued for.
- **Unevaluated numbers** → 0; **imports alone** → 0; **staff research/model findings**
  → PROCEDURAL; **premise clauses** are not separate aspects.
- GENERAL definition no longer lists "the expansion" (contradicted the v1 topic rule).

## v1 — 2026-10-30 (after pilot round 1)
Round 1 (100 items; Haiku, Sonnet, Opus-low): function Fleiss κ 0.976, topic κ 0.932,
direction all-exact 80%, Krippendorff α (interval) 0.914; 23 items with any primary-label
disagreement. Changes, each targeting an observed disagreement pattern:
- **No two aspects with the same function and topic**; same-topic contrasts get one
  aspect with the net direction (main clause dominates a concession). 8 of 23 cases were
  one labeler splitting "X grew strongly, while Y moderated" into +2 and −1.
- **Topic choice rules**: activity/growth/recovery/expansion → ACTIVITY; slack/resource
  utilization → LABOR; inflation compensation from inflation-indexed securities →
  INFLATION; GENERAL only for risk balance, uncertainty, both goals together.
- **Implementation/operations** (repo operations, reserve management, money-market
  functioning, balance-sheet mechanics/framework) → POLICY 0; staff/manager operational
  recommendations → PROCEDURAL; generic monitoring/attentiveness/"as appropriate" → POLICY 0.
- **Directional readiness** ("prepared to provide additional accommodation") → ±1;
  **neutral-rate views** (lower → dovish); **outcomes at/near goals** → 0.
- **Risk-statement strength**: tilted/weighted/increased → ±1; significant/substantial → ±2.

## v0 — 2026-10-30 (draft for pilot round 1)
Written by the orchestrator from the HANDOFF section 3 specification.

Departures from / additions to the specification, with reasons:
- **Added topic `GENERAL`** for sentences about the economy as a whole or both goals at
  once (balance of risks, overall uncertainty). The four specified topics have no home
  for these, and forcing them into ACTIVITY would contaminate the activity index.
- **Minority views as hedges.** "Some/a few participants" was listed as a hedge; v0 makes
  the quantifier list explicit and treats a dissent (a single member) the same way
  (strength 1). Majority quantifiers ("most", "many") are not hedged.
- **Reports of interest-rate moves / policy expectations → FINANCIAL 0.** Keeps the
  labels about the content of the text, not about financial-market pricing.

Topic and sign rules follow Shah, Paturi & Chava (2023, ACL), Appendix C.2 / Table 10
("Annotation Guide"), with these deliberate differences:
- Trade: Shah et al. code a larger trade deficit as hawkish. v0 codes a wider deficit /
  weaker exports as dovish (it subtracts from growth), consistent with the ACTIVITY rule.
- Bank reserves and money supply rows are not carried over (rare in the corpus; the
  balance-sheet direction is covered under POLICY).
- Kept from Shah et al.: dollar appreciation dovish; energy and house prices up hawkish;
  productivity growth dovish; "moderate"/"mixed" as neutral cues.
