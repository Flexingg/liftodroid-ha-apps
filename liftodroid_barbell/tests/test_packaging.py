"""The app must stay installable from a GitHub app repository."""

from pathlib import Path

import yaml

from barbell import __version__

APP = Path(__file__).resolve().parents[1]
HA = APP.parent


def test_store_metadata():
    repo = yaml.safe_load((HA / "repository.yaml").read_text())
    assert repo["name"]  # the Supervisor skips a repository without a name
    cfg = yaml.safe_load((APP / "config.yaml").read_text())
    for key in ("name", "version", "slug", "description", "arch"):
        assert cfg.get(key), key
    assert set(cfg["arch"]) <= {"amd64", "aarch64", "armv7"}
    assert set(cfg["options"]) <= set(cfg["schema"])


def test_version_bumps_together():
    # Home Assistant offers an update only when config.yaml's version changes; the code reports __version__.
    cfg = yaml.safe_load((APP / "config.yaml").read_text())
    assert cfg["version"] == __version__
    assert f"## {__version__}" in (APP / "CHANGELOG.md").read_text()


def test_build_files_present():
    build = yaml.safe_load((APP / "build.yaml").read_text())
    assert set(build["build_from"]) == set(yaml.safe_load((APP / "config.yaml").read_text())["arch"])
    assert (APP / "Dockerfile").exists() and (APP / "requirements.txt").exists()
