# F1 winner forecasting: end-to-end design

The existing project is a research workflow for the Sao Paulo 2026 race. The user asked to turn it into an end-to-end project and upload it to GitHub, then delegated scope decisions with “choose and proceed.”

## Chosen scope

Build a local Python package and CLI that collect historical results, validate data, construct chronological features, evaluate models on expanding season folds, train final models, and predict a supplied pre-race grid. Add a Streamlit dashboard, meaningful tests, CI and onboarding documentation. Preserve all original research scripts.

## Approaches

1. Package the scripts only: small scope but still difficult to explore results.
2. Package plus Streamlit dashboard (chosen by the implementer under delegated scope): one language, easy local use, interactive predictions and honest evaluation.
3. Separate web frontend and API: adds deployment and maintenance without a stated need for multi-user service.

## Constraints and boundaries

- Work in the user's supplied folder; it has no existing Git history to isolate.
- Do not change original experiments or overwrite their outputs/models.
- Python >=3.11; support Windows and Linux.
- Default cutoff is strictly before 2026-11-08; default evaluation is 2022–2025. Report 2026 separately as previously inspected diagnostics.
- Use stable driver/team IDs and the same feature functions for historical and target rows.
- New drivers have missing history handled by a training-fold imputer; do not invent past results.
- Normalized scores sum to one per race but are uncalibrated. V3 is an experimental candidate, not automatically selected.
- Snapshot and bundled history are inherited local inputs, not independently verified live FIA facts. State this in the UI, metadata and docs.
- No secrets, virtual environments, caches, or model binaries in Git. Include the small historical CSV and an explicit target input snapshot so a fresh checkout runs offline.
- Save reproducibility metadata and hashes. Network collection is optional and reports failures clearly.
- Publish a new GitHub repository after local verification; use private visibility by default because visibility was unspecified.

## Architecture

`src/f1_forecast/data.py` validates and optionally collects history. `features.py` builds rolling and qualifying features. `modeling.py` contains fixed model definitions, race normalization, training, prediction and season backtests. `pipeline.py` handles paths, artifacts and provenance. `cli.py` exposes each stage and a full run. `app.py` reads CSV/JSON reports without loading model binaries and offers exploratory ensemble weights, model comparisons, grid calibration and source context.

## Acceptance

A fresh editable install can run the full pipeline without network access using bundled inputs. Tests prove that current/future outcomes do not leak into features, folds use earlier seasons only, malformed race/grid inputs are rejected, and probabilities normalize by race. The dashboard renders and responds to controls without exceptions. GitHub Actions installs from scratch, runs lint/tests and the full offline workflow. README contains copyable Windows/POSIX commands, methodology, limitations and data provenance.
