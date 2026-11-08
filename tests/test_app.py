from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_loads_demo_and_reacts_to_ensemble_weight(monkeypatch):
    monkeypatch.delenv("F1_REPORT_DIR", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.dataframe[0].value.shape[0] == 22
    app.slider(key="v1_weight").set_value(1.0).run()
    assert not app.exception
    scores = app.dataframe[0].value
    assert scores["Ensemble score (%)"].tolist() == pytest.approx(scores["V1 score (%)"].tolist())
    assert any("Uncalibrated" in warning.value for warning in app.warning)


def test_diagnostic_results_are_opt_in(monkeypatch):
    monkeypatch.delenv("F1_REPORT_DIR", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    table = app.dataframe[1].value
    assert 2026 not in table["season"].values
    app.checkbox(key="include_diagnostic").check().run()
    assert not app.exception
    assert 2026 in app.dataframe[1].value["season"].values


def test_missing_report_directory_shows_recovery_command(monkeypatch, tmp_path):
    monkeypatch.setenv("F1_REPORT_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.error
    assert any("f1_forecast run" in code.value for code in app.code)
