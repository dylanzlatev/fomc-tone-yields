#!/usr/bin/env bash
# PreToolUse guard: blocks destructive / outward-facing actions for every agent in this repo.
# Reads the hook JSON on stdin. Exit 2 = block (stderr is shown to the agent); exit 0 = allow.
# No jq on this machine, so fields are pulled out with grep/sed.

payload="$(cat)"

tool="$(printf '%s' "$payload" | grep -oE '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"([^"]*)"$/\1/')"

block() {
  printf 'BLOCKED by project guard: %s\nSafe alternative: %s\n' "$1" "$2" >&2
  exit 2
}

# ---- File-writing tools: protect the old repo, Claude settings/hooks, and .git internals ----
case "$tool" in
  Write|Edit|NotebookEdit|MultiEdit)
    path="$(printf '%s' "$payload" | grep -oE '"(file_path|notebook_path)"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 \
            | sed -E 's/^"[a-z_]*"[[:space:]]*:[[:space:]]*"//; s/"$//' | sed 's/\\\\/\//g' | tr 'A-Z' 'a-z')"
    case "$path" in
      *fedsentiment-nlp*) block "the old repo is read-only" "read it with Read/Grep/Glob; write only inside this repo" ;;
      */.claude/settings*|*/.claude/hooks/*) block "agents may not change Claude Code settings or guard hooks" "log the needed change in PROGRESS.md under 'Requests for Dylan'" ;;
      */.git/*) block "agents may not edit git internals" "use normal git commands on a branch" ;;
    esac
    exit 0 ;;
  Bash|PowerShell) ;;
  *) exit 0 ;;
esac

# ---- Shell tools: extract and unescape the command string ----
cmd="$(printf '%s' "$payload" | grep -oE '"command"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 \
       | sed -E 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//' | sed 's/\\"/"/g; s/\\\\/\\/g')"
lc="$(printf '%s' "$cmd" | tr 'A-Z' 'a-z')"
# Same command with quoted strings (commit messages, PR bodies) removed, for git/gh checks
lcq="$(printf '%s' "$lc" | sed -E "s/\"[^\"]*\"//g; s/'[^']*'//g")"
m()  { printf '%s' "$lc"  | grep -qE -e "$1"; }
mq() { printf '%s' "$lcq" | grep -qE -e "$1"; }

# Old repo is read-only (shell access to it is blocked entirely; file tools can still read it)
m 'fedsentiment-nlp' && block "shell commands touching the old repo are not allowed" "read it with the Read/Grep/Glob tools"

# Guard files themselves
m '\.claude/(settings|hooks)|\.git/hooks|core\.hookspath' && block "guard/hook configuration is off-limits" "log the needed change in PROGRESS.md under 'Requests for Dylan'"
mq '--no-verify' && block "bypassing git hooks" "fix the problem the hook reports (usually .gitignore)"

# ---- git history / remote safety ----
if mq '(^|[^a-z])git([[:space:]]|$)'; then
  mq 'push.*(--force|--mirror|--delete|--prune|[[:space:]]-[a-z]*f[a-z]*([[:space:]]|$)|[[:space:]]\+[^[:space:]]|[[:space:]]:[^[:space:]/])' \
    && block "force-push / remote ref deletion" "push a new commit (git revert to undo) on a feature branch"
  mq 'push.*([[:space:]:](main|master)([[:space:]]|$))' \
    && block "pushing directly to main" "push your feature branch and merge it with: gh pr create ... && gh pr merge --merge"
  # git subcommands actually invoked (handles `git -C dir x` / `git -c k=v x`)
  subs="$(printf '%s' "$lcq" | grep -oE '(^|[^a-z])git([[:space:]]+-c[[:space:]]+[^[:space:]]+)*[[:space:]]+[a-z-]+' | awk '{print $NF}')"
  if printf '%s\n' "$subs" | grep -qxE 'commit|push|merge|cherry-pick|revert'; then
    cur="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    if [ "$cur" = "main" ] || [ "$cur" = "master" ]; then
      block "committing/merging/pushing on main" "git switch -c phase-N-<topic>, commit there, push that branch, then gh pr create + gh pr merge --merge"
    fi
  fi
  mq 'reset.*--hard' && block "git reset --hard discards work" "git revert <sha>, or commit/stash and switch branches"
  mq 'clean([[:space:]].*)?[[:space:]]-[a-z]*[fdx]' && block "git clean deletes untracked files" "leave files in place; add generated paths to .gitignore"
  mq '(rebase|filter-branch|filter-repo|replace|update-ref|reflog[[:space:]]+(expire|delete)|gc[[:space:]].*prune)' && block "history rewriting" "add new commits; use git revert to undo"
  mq 'commit.*--amend' && block "amending commits" "make a new follow-up commit"
  mq 'commit.*[[:space:]]-[a-z]*n[a-z]*([[:space:]]|$)' && block "git commit -n skips hooks" "commit without -n and fix what the hook reports"
  mq 'branch.*([[:space:]]-[a-z]*d|--delete)' && block "deleting branches" "leave merged branches in place"
  mq 'tag.*([[:space:]]-[a-z]*d|--delete)' && block "deleting tags" "create a new tag instead"
  mq '(checkout|switch).*([[:space:]]-f([[:space:]]|$)|--force|--discard-changes)' && block "forced checkout discards work" "commit or stash first"
  mq '(checkout|restore)[[:space:]].*(--[[:space:]]+\.|[[:space:]]\.)([[:space:]]|$)' && block "discarding all working-tree changes" "commit or stash first; restore individual files by name"
  mq 'stash[[:space:]]+(drop|clear)' && block "dropping stashes" "leave stashes; apply with git stash pop"
fi

# ---- GitHub CLI ----
if mq '(^|[^a-z])gh[[:space:]]'; then
  mq 'repo[[:space:]]+(delete|edit|rename|archive|unarchive)' && block "changing repo settings" "log it in PROGRESS.md for Dylan"
  mq '[[:space:]](release|gist|secret|variable|ruleset)[[:space:]]' && block "publishing/settings outside the repo contents" "log it in PROGRESS.md for Dylan"
  mq 'api.*(protection|rulesets|-x[[:space:]]*(delete|patch|put)|--method[[:space:]]*(delete|patch|put))' && block "mutating GitHub API calls" "use gh pr create / gh pr merge only"
  mq 'pr[[:space:]]+merge.*(--admin|--delete-branch|[[:space:]]-d([[:space:]]|$))' && block "admin-bypass merges or branch deletion" "gh pr merge <n> --merge"
fi

# ---- conda: only the project env `ftyields` may be created/changed; nothing is ever removed ----
if mq '(^|[^a-z])(conda|mamba)(\.exe)?([[:space:]]|$)'; then
  mq '(conda|mamba)(\.exe)?[[:space:]]+(env[[:space:]]+)?remove|[[:space:]]--all([[:space:]]|$)|[[:space:]]--force([[:space:]]|$)' \
    && block "removing or force-recreating conda environments" "conda env update -f environment.yml --prune"
  envname="$(printf '%s' "$lcq" | grep -oE '[[:space:]](-n|--name)[[:space:]=]+[^[:space:]]+' | head -1 | sed -E 's/.*[[:space:]=]//')"
  if [ -n "$envname" ] && [ "$envname" != "ftyields" ]; then
    mq '[[:space:]](create|install|update|uninstall|upgrade)([[:space:]]|$)' \
      && block "modifying conda env '$envname' (only 'ftyields' belongs to this project)" "edit environment.yml and update the ftyields env"
  fi
  mq '(-p|--prefix)[[:space:]=]' && mq '[[:space:]](create|install|update|uninstall)([[:space:]]|$)' \
    && block "modifying a conda env by path" "use -n ftyields"
  if mq '[[:space:]]create([[:space:]]|$)' && [ -d "/c/Users/dylan/anaconda3/envs/ftyields" ]; then
    block "the ftyields env already exists; re-creating would replace it" "conda env update -f environment.yml --prune"
  fi
  if [ -z "$envname" ] && mq '(conda|mamba)(\.exe)?[[:space:]]+(install|uninstall|update|upgrade)([[:space:]]|$)'; then
    block "conda install/update without -n targets the base env" "add -n ftyields (and record the package in environment.yml)"
  fi
fi

# ---- recursive / bulk deletion ----
m '(^|[[:space:];&|])rm[[:space:]]+(-[a-z]*r|--recursive)' && block "recursive delete" "overwrite generated files in place; put scratch output in a gitignored tmp/ dir"
m '(^|[[:space:];&|])(rmdir|rd)[[:space:]]' && block "directory removal" "leave the directory; gitignore it if it is scratch"
m 'remove-item.*-r(ecurse)?' && block "recursive delete" "overwrite generated files in place"
m 'del[[:space:]]+/s|shutil\.rmtree|find[[:space:]].*-delete' && block "bulk delete" "overwrite generated files in place"

# ---- publishing anywhere except this repo ----
m 'huggingface-cli[[:space:]]+(upload|repo)|(^|[[:space:]])hf[[:space:]]+(upload|repo)|push_to_hub|upload_folder|upload_file|create_repo|twine[[:space:]]+upload|npm[[:space:]]+publish' \
  && block "publishing outside the repo" "keep artifacts local; log the idea in PROGRESS.md for Dylan"

# ---- no Anthropic API usage (subscription only) ----
m 'api\.anthropic\.com|import[[:space:]]+anthropic|from[[:space:]]+anthropic|pip[[:space:]]+install[^;&|]*[[:space:]]anthropic([[:space:]=<>]|$)|anthropic_api_key=' \
  && block "Anthropic API usage (would bill the API, not the subscription)" "run LLM labelling as Claude Code subagents"

exit 0
