"""The FLEET wrapper's dustmaps setup: OSPool's cleaned env means the wrapper must
point dustmaps (via DUSTMAPS_CONFIG_FNAME) at the SFD maps baked into the image."""

import json
import os

import fleet_wrapper


def test_configure_dustmaps_writes_config_and_sets_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUSTMAPS_CONFIG_FNAME", raising=False)
    monkeypatch.delenv("DUSTMAPS_DATA_DIR", raising=False)
    fleet_wrapper._configure_dustmaps()
    rc = tmp_path / ".dustmapsrc.json"
    assert os.environ["DUSTMAPS_CONFIG_FNAME"] == str(rc)
    assert json.loads(rc.read_text())["data_dir"] == "/opt/dustmaps"


def test_configure_dustmaps_honors_image_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUSTMAPS_CONFIG_FNAME", raising=False)
    monkeypatch.setenv("DUSTMAPS_DATA_DIR", "/custom/dust")
    fleet_wrapper._configure_dustmaps()
    assert json.loads((tmp_path / ".dustmapsrc.json").read_text())["data_dir"] == "/custom/dust"


def test_configure_dustmaps_respects_existing_config(tmp_path, monkeypatch):
    # An operator-provided config must not be clobbered.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DUSTMAPS_CONFIG_FNAME", "/preset/rc.json")
    fleet_wrapper._configure_dustmaps()
    assert os.environ["DUSTMAPS_CONFIG_FNAME"] == "/preset/rc.json"
    assert not (tmp_path / ".dustmapsrc.json").exists()
