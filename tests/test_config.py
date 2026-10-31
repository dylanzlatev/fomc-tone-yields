from pathlib import Path

import pytest

from ftyields import config


def test_config_loads():
    cfg = config.load_config()
    assert isinstance(cfg, dict)
    for key in ("python", "seed", "paths", "sample", "fred", "factors", "text",
                "events", "scoring", "labelling", "analysis"):
        assert key in cfg


def test_repo_root_has_config():
    assert (config.repo_root() / "config.yaml").is_file()


def test_paths_resolve_inside_repo():
    cfg = config.load_config()
    root = config.repo_root().resolve()
    for key in cfg["paths"]:
        p = config.path(key, cfg)
        assert isinstance(p, Path) and p.is_absolute()
        assert p == root or root in p.parents, f"{key} escapes the repo: {p}"


def test_unknown_path_key_raises():
    with pytest.raises(KeyError):
        config.path("no_such_key")


def test_seed_is_int():
    cfg = config.load_config()
    assert isinstance(cfg["seed"], int)
    assert config.set_seeds() == cfg["seed"]
