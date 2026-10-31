"""Shared downloader: fetch a URL to a local file and record it in the manifest.

Every external file the pipeline uses goes through `fetch`, so the committed
manifest (`config.yaml` -> downloads.manifest) records where each file came from,
when, its SHA-256, its license, and whether the file itself is committed.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests

from ftyields.config import load_config, repo_root

USER_AGENT = "fomc-tone-yields research pipeline (github.com/dylanzlatev/fomc-tone-yields)"
MANIFEST_FIELDS = ["name", "url", "local_path", "retrieved_utc", "sha256", "bytes",
                   "license", "committed"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest_path(cfg: dict) -> Path:
    return repo_root() / cfg["downloads"]["manifest"]


def _read_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["name"]: row for row in csv.DictReader(fh)}


def _write_manifest(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow({k: rows[name].get(k, "") for k in MANIFEST_FIELDS})


def fetch(name: str, url: str, dest: str | Path, *, license: str, committed: bool,
          refresh: bool = False, cfg: dict | None = None, timeout: int = 60) -> Path:
    """Download `url` to `dest` (relative to the repo root) unless it already exists.

    `name` is the manifest key. The manifest row is (re)written on every download;
    an existing file is reused as-is (pass refresh=True to re-download).
    """
    cfg = cfg if cfg is not None else load_config()
    dest = Path(dest)
    if not dest.is_absolute():
        dest = repo_root() / dest
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)

    mpath = _manifest_path(cfg)
    rows = _read_manifest(mpath)
    rows[name] = {
        "name": name,
        "url": url,
        "local_path": dest.relative_to(repo_root()).as_posix(),
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256(dest),
        "bytes": dest.stat().st_size,
        "license": license,
        "committed": "yes" if committed else "no",
    }
    _write_manifest(mpath, rows)
    return dest
