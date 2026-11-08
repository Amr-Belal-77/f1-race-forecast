# 🏁 F1 Race Forecast

An end-to-end Formula 1 winner forecasting project: historical data collection, leakage-aware features, XGBoost models, walk-forward evaluation, grid-risk audits and an interactive Streamlit dashboard.

The default example is a **frozen Sao Paulo 2026 research snapshot**, inherited from the original project. It is not a live race feed. Its grid and historical inputs have **not been independently verified** against current official sources. Model outputs are **uncalibrated normalized scores**, not established real-world win odds.

## Quick start

Requires Python 3.11 or later. Run commands from the repository root.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The dashboard initially reads the committed `reports/demo/` CSV/JSON files, so you can explore it without downloading data or loading model binaries.

After activating the environment, the following commands use `python`. On Windows, you can instead use `.\.venv\Scripts\python.exe` directly.

## Reproduce the complete workflow

```bash
python -m f1_forecast run --config config/sao_paulo_2026.toml --output artifacts
```

This offline run validates the bundled history, builds features, evaluates expanding season folds, fits V1/V2/V3, predicts the supplied grid, and writes a manifest with input hashes and software versions. It never downloads data implicitly.

| Artifact in `artifacts/` | Contents |
| --- | --- |
| `features.csv` | Historical lagged form, qualifying and displacement features |
| `target_features.csv` | Pre-race rows, with no target outcomes |
| `models/V1.joblib`, `V2.joblib`, `V3.joblib` | Locally trained models and provenance |
| `predictions.csv` | V1/V2/V3 scores and fixed 50/50 ensemble |
| `backtest_summary.csv` | Season metrics, baseline and training boundaries |
| `backtest_predictions.csv` | Out-of-season driver scores and observed winners |
| `grid_calibration.csv` | Expected versus observed wins by starting band |
| `manifest.json` | Cutoff, input/report hashes, parameters, versions and provenance |

To display your freshly generated results:

```powershell
# PowerShell
$env:F1_REPORT_DIR = (Resolve-Path artifacts).Path
python -m streamlit run app.py
```

```bash
# macOS / Linux
F1_REPORT_DIR="$PWD/artifacts" python -m streamlit run app.py
```

Report hashes detect incomplete or stale combinations of files. Use the full `run` command to create a complete, consistent dashboard report after changing inputs. Changing weights in the dashboard is exploratory; historical results always refer to the predeclared 50/50 blend.

## Pipeline stages

```bash
python -m f1_forecast features
python -m f1_forecast evaluate
python -m f1_forecast train
python -m f1_forecast predict
```

All stages accept `--config` and `--output`. `predict` requires trusted models created by `train` for the same history hash, cutoff and feature definitions. Only load joblib files that you created or otherwise trust; the dashboard does not deserialize them. Individual stages are useful for research; `run` produces the complete report and manifest.

Optional network collection uses the [Jolpica Ergast-compatible API](https://api.jolpi.ca/ergast/):

```bash
python -m f1_forecast collect --output artifacts/download
python -m f1_forecast collect --output artifacts/download --refresh
```

Collection paginates results and qualifying, caches responses, retries transient HTTP failures and excludes target-date/later outcomes. It writes `collected_history.csv`; it does not replace the committed research snapshot. To use downloaded history, copy the race config and point `data.history` to that file. Cached partial-season data can become stale; `--refresh` explicitly replaces the API cache. Collection needs internet access and is not exercised against the live API in CI.

## Model comparison

Fixed expanding-window backtests on the bundled data, **2022–2025 only**:

| Model | Selected winners | Accuracy |
| --- | ---: | ---: |
| Pole-sitter baseline | 51 / 92 | 55.4% |
| V1: grid + recent form | 53 / 92 | 57.6% |
| V2: V1 + qualifying gaps | 52 / 92 | 56.5% |
| V3: V2 + grid displacement | 52 / 92 | 56.5% |
| Fixed V1/V2 50/50 ensemble | 53 / 92 | 57.6% |

These are retrospective research comparisons on repeatedly explored data. The small difference from the baseline does not establish reliable superiority or statistical significance. V3 remains an experimental candidate. The dashboard also reports race log loss, multiclass Brier score and season-by-season variation.

**2026 is diagnostic:** the original research already inspected it repeatedly. It is excluded from the default comparison and can be enabled explicitly in the dashboard. Earlier seasons are also research evidence, not a preregistered final test set.

## Use another race

1. Copy `config/sao_paulo_2026.toml` and edit the race name, date, season, round, input paths and evaluation years. Paths resolve relative to the config file. Every configured evaluation season must exist in history and have earlier training seasons.
2. Supply history with the schema in `data/raw/historical_f1_2018_2026.csv`. The cutoff is exclusive and must be on or before the race date. History containing outcomes on/after the cutoff is rejected rather than silently included.
3. Supply an outcome-free grid CSV using the schema below. Match stable driver/team IDs to history. For a debutant, supply a new driver ID and the actual team ID; missing prior form is handled by the training-fold imputer.
4. Run `python -m f1_forecast run --config path/to/race.toml --output artifacts/my-race`.

Grid columns:

```text
driver_id,driver,team_id,team,grid_position,qualifying_position,q1,q2,q3
```

Positions are integers; grid `0` denotes a pit-lane start, qualifying `0` denotes no recorded position. Positive positions must be unique. Lap times use seconds or `minutes:seconds`; leave missing segments blank. Supply the entire field. Driver and team IDs, grid amendments and snapshot timing are the caller's responsibility.

## Project structure

```text
app.py                       Streamlit dashboard
config/                      Editable race configuration
src/f1_forecast/              Maintained pipeline package
src/*.py                     Preserved original research experiments
data/raw/                    Small frozen historical CSV
data/snapshots/               Explicit outcome-free target grid
reports/demo/                Portable dashboard example reports
tests/                       Data, leakage, model, pipeline and dashboard tests
docs/methodology.md          Modeling contracts and limitations
.github/workflows/ci.yml     Installation, lint, tests and offline pipeline
```

The original scripts remain available for comparison. They retain their original file dependencies and fixed dates and are not the supported entry point. Their generated files, model binaries, API caches and local virtual environments are ignored by Git. Optional `pip install -e '.[research]'` adds FastF1/matplotlib for those experiments. The old environment freeze is preserved in `docs/reference/requirements-original.txt`; the package defines the maintained dependencies.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The tests exercise current-outcome leakage, shared historical/target formulas, unseen drivers, input validation, race normalization, fold boundaries, actual training/prediction artifacts and dashboard interactions using [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

## Provenance and limitations

- Historical CSV: inherited local data originally collected with the Jolpica API scripts. The preserved source covers 2018–2026 before the configured cutoff.
- Target grid: extracted from the existing research prediction file, retaining only IDs, driver/team identities, starting positions, qualifying positions and lap times. Past model scores and rolling features were discarded and recomputed.
- The older scripts cite FIA qualifying/grid documents; those references have not been independently revalidated for this packaged release. Treat the bundled grid as a research example until verified.
- The shared ChatGPT conversation could not be retrieved. Implementation decisions are grounded in the local scripts and saved outputs, not a claimed reading of that conversation.
- Weather, crashes, strategy, component reliability and regulation changes are not modeled explicitly. Qualifying gaps can be misleading in changing conditions. No live updating, real-time race prediction or calibrated betting odds are provided.
- Versions and parameters are recorded per run; exact numerical reproducibility across dependency versions/platforms is not guaranteed. Runtime dependencies use compatible ranges rather than a full lock file.
- No software license has been selected. Third-party data retain their upstream terms; verify those terms before redistributing data publicly.
