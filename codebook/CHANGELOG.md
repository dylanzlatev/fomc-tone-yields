# Codebook changelog

Not shown to labelers (they see only `codebook.md`).

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
