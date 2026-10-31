# FOMC Tone & the Yield Curve: project rules

Sole author: Dylan Zlatev. Research question: does what the FOMC says move Treasury
yields beyond what it does with rates, and which kind of language moves which part of
the curve?

- **Read `HANDOFF.md` once** at the start of the first session; it holds the full briefing
  and the locked decisions. **Read `PROGRESS.md` at the start of every session**; it
  records where things stand. Update it at the end of each work block: what was done,
  what was verified, what's next, and exactly where any interrupted batch job stopped.
- The main session is the orchestrator. It delegates execution to subagents at the
  lowest capable tier (Haiku for bulk or mechanical work, Sonnet for code, Opus at lower
  effort for blind validation and hard cases) and verifies everything itself.
- **No Anthropic API usage.** LLM labelling runs only as Claude Code subagents on the
  subscription. If the session-start warning says `ANTHROPIC_API_KEY` is set, pause all
  subagent and labelling work and leave a note. This is the only thing that halts work.
- Labelers are **blind**: they see only `codebook/codebook.md` and their batch file.
  The codebook never mentions yields, markets or hypotheses.
- Units are basis points and trading days. All parameters live in `config.yaml`. Fix
  seeds everywhere. Fit on training data only, leave a gap of one horizon before the
  test data, and align macro data to its publication date.
- Every number in the paper comes from generated files in `results/`. Never hand-type
  results. Check every citation against its source.
- **Never wait for Dylan.** The codebook review, his 100 labels and the results story are
  notes under "For Dylan" in PROGRESS.md, not pauses. To deviate from a locked decision,
  run the alternative as an extra analysis on its own branch and log it under "Proposed
  deviations"; the locked design stays the main line.
- **Git: branch → PR → merge, never on `main`.** Work on `phase-N-<topic>` branches and
  push them freely. Merge verified work with `gh pr create` then
  `gh pr merge <n> --merge`, then `git switch main && git pull`. Undo with
  `git revert`. The repo is public; commit only data you're allowed to redistribute. The
  pre-commit hook blocks files over 50 MB and model or intermediate artifacts; fix
  `.gitignore` when it complains.
- **Banned for you and every subagent** (enforced by `.claude/hooks/guard.sh`, deny rules
  and GitHub branch protection; never try to get around a block):
  - force-push, rebase, amend, reset --hard, branch or tag deletion, `--no-verify` or
    `commit -n`, `gh pr merge --admin`
  - committing, merging or pushing on `main`
  - changing repo settings or protection
  - recursive deletes
  - publishing outside this repo
  - Anthropic API or SDK use
  - editing the old repo, `.claude/` config or `.git/hooks`

  Put this list in every subagent brief.
- **Python environment:** the project conda env `ftyields` is defined by `environment.yml`
  and locked in `environment.lock.yml`, both committed; it's built in phase 0. Always run
  code with `C:/Users/dylan/anaconda3/envs/ftyields/python.exe` (or
  `conda run -n ftyields ...`). Never use bare `python`/`pip` (not on PATH), the base
  env, or the old `ML` env, which has a numpy/transformers conflict. Add a dependency by
  editing `environment.yml`, then run `conda env update -f environment.yml --prune` and
  regenerate the lock file. Never remove or recreate envs.
- Windows: use the Bash tool with forward slashes.
- The old repo, `C:\Users\dylan\Documents\GT\CS\4641\FedSentiment-NLP`, is a read-only
  reference. Never modify it.
