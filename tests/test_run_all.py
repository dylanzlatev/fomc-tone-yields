import subprocess
import sys
from pathlib import Path

RUN_ALL = Path(__file__).resolve().parents[1] / "run_all.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(RUN_ALL), *args],
        capture_output=True, text=True, cwd=RUN_ALL.parent,
    )


def test_help():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout


def test_dry_run():
    result = _run("--dry-run")
    assert result.returncode == 0, result.stderr
    for stage in ("data", "text", "label", "score", "indices", "analysis",
                  "forecast", "figures", "paper"):
        assert stage in result.stdout


def test_dry_run_from_and_only():
    assert _run("--dry-run", "--from", "score").returncode == 0
    assert _run("--dry-run", "--only", "data", "paper").returncode == 0
