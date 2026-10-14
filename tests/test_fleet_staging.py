"""Pure-logic tests for fleet_staging (no DB): the best-effort guards that skip
the fetch. The in-process BOOM query needs a live SkyPortal and is exercised
there, not here."""

import fleet_staging


def test_stage_host_skips_without_obj_id(tmp_path):
    assert fleet_staging.stage_host({}, {}, tmp_path, log=lambda *_: None) == []


def test_stage_host_non_fatal_without_skyportal(tmp_path):
    # Standalone (no SkyPortal/DB importable) degrades to hostless, never raises.
    out = fleet_staging.stage_host({}, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
    assert out == []
    assert not (tmp_path / fleet_staging.HOST_FILE).exists()


def test_host_file_matches_bridge():
    import fleet_bridge

    assert fleet_staging.HOST_FILE == fleet_bridge.HOST_FILE


def _inject_fake_boom(monkeypatch, fake_request):
    import sys
    import types

    boom = types.ModuleType("skyportal.broker_apis.boom")
    boom._request = fake_request
    pkg = types.ModuleType("skyportal.broker_apis")
    pkg.boom = boom
    monkeypatch.setitem(sys.modules, "skyportal.broker_apis", pkg)
    monkeypatch.setitem(sys.modules, "skyportal.broker_apis.boom", boom)


def test_boom_find_retries_transient_then_succeeds(monkeypatch):
    import requests

    import fleet_staging

    calls = []

    def fake_request(broker, method, path, json=None):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("blip")
        return {"data": [{"ok": True}]}

    _inject_fake_boom(monkeypatch, fake_request)
    monkeypatch.setattr(fleet_staging.time, "sleep", lambda s: None)
    assert fleet_staging._boom_find(object(), "ZTF_alerts_aux", "ZTF1") == {"data": [{"ok": True}]}
    assert len(calls) == 3  # two transient retries, then success


def test_boom_find_does_not_retry_4xx(monkeypatch):
    import types

    import pytest
    import requests

    import fleet_staging

    calls = []

    def fake_request(broker, method, path, json=None):
        calls.append(1)
        raise requests.HTTPError(response=types.SimpleNamespace(status_code=404))

    _inject_fake_boom(monkeypatch, fake_request)
    monkeypatch.setattr(fleet_staging.time, "sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        fleet_staging._boom_find(object(), "ZTF_alerts_aux", "ZTF1")
    assert len(calls) == 1  # a 4xx is a real error, not retried
