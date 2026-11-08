import json
import subprocess
import sys

import pandas as pd
import pytest

from f1_forecast.pipeline import load_config, run_pipeline


def test_full_offline_run_writes_reproducible_artifacts(history, grid, tmp_path):
    history.to_csv(tmp_path / "history.csv", index=False)
    grid.to_csv(tmp_path / "grid.csv", index=False)
    config_path = tmp_path / "race.toml"
    config_path.write_text("""
[data]
history = "history.csv"
grid = "grid.csv"
cutoff = "2023-01-01"
provenance = "Test fixture"
[race]
name = "Test GP"
date = "2023-01-01"
season = 2023
round = 1
[evaluation]
years = [2021, 2022]
diagnostic_from = 2022
""")
    result = run_pipeline(load_config(config_path), tmp_path / "output", n_estimators=3)
    assert result.is_dir()
    predictions = pd.read_csv(result / "predictions.csv")
    assert len(predictions) == 2
    assert predictions.Blend_50_50.sum() == pytest.approx(1)
    summary = pd.read_csv(result / "backtest_summary.csv")
    assert summary.loc[summary.season == 2022, "sample"].eq("diagnostic").all()
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["training_max_date"] == "2022-03-01"
    assert len(manifest["inputs"]["history"]["sha256"]) == 64
    assert manifest["models"]["V1"]["n_estimators"] == 3
    assert (result / "models/V2.joblib").is_file()
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "f1_forecast",
            "predict",
            "--config",
            str(config_path),
            "--output",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert len(pd.read_csv(result / "predictions.csv")) == 2


def test_config_refuses_training_cutoff_after_race_date(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("""[data]
history = "h.csv"
grid = "g.csv"
cutoff = "2026-11-09"
[race]
name = "GP"
date = "2026-11-08"
season = 2026
round = 20
[evaluation]
years = [2025]
diagnostic_from = 2026
""")
    with pytest.raises(ValueError, match="cutoff"):
        load_config(path)


def test_cli_reports_missing_input_without_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "f1_forecast", "run", "--config", str(tmp_path / "missing.toml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
