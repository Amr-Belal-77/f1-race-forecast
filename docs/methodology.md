# Methodology and reproducibility

## Prediction task

Choose a race winner from a complete pre-race field using completed historical races plus target qualifying/grid inputs. This is a driver-row binary classification experiment normalized within each race; it is not a native conditional race-choice model.

## Chronology

The configured cutoff excludes all outcomes on that calendar date or later. Race dates, driver/team identities, keys and labels are validated before feature construction. Each race must have exactly one winner; winner labels must agree with finishing position. Historical positions must be nonnegative integers (or missing) and points must be nonnegative. Race seasons must match their calendar years. Distinct races on the same calendar date are rejected because this date-only workflow cannot establish their information order.

Driver features use `shift(1)` followed by a five-start rolling mean for finish, points and win rate. Prior starts are a count within the available historical window, not necessarily career starts. Team points first aggregate all drivers per team/race, then use the previous five team races. A driver's team at inference comes from the explicit target input, supporting transfers and debutants.

Target rows have no outcomes. They are appended to earlier historical rows and passed through the same feature function. This avoids divergent training and inference formulas. Target feature columns supplied by a caller are discarded; form is rebuilt from history. Later historical records are excluded when constructing target features, while pipeline history as a whole is strictly validated against the configured training cutoff.

## Feature families

| Model | Features |
| --- | --- |
| V1 | Grid, qualifying position, recent driver finish/points/wins, prior starts, recent team points, pit-lane flag |
| V2 | V1 plus Q1/Q2/Q3 gap to that segment's fastest lap and Q2/Q3 time-availability flags |
| V3 | V2 plus places lost/gained versus qualifying and P14-or-worse indicator |

Qualifying gaps compare drivers within the same segment of the same race. Missing sessions remain missing. Time-availability flags do not imply a driver's official advancement status. Invalid/missing historical grids are represented as one position behind the maximum positive grid with a pit-lane flag. Multiple pit-lane starters can share that synthetic position. Qualifying position `0` becomes missing.

Grid displacement is a derived position difference. It is not an FIA penalty count or causal estimate of a penalty's effect.

## Estimation

Each model uses a median imputer fitted only on its training fold, with missingness indicators and empty-feature retention, followed by XGBoost. Fixed parameters: 250 trees, depth 3, learning rate 0.03, row sample 0.85, column sample 0.9, minimum child weight 5, L2 regularization 10, seed 42, four worker threads. No automated hyperparameter search or target-based ensemble optimization is performed.

If `s_i` is a binary model score, race-normalized `p_i = s_i / sum_j(s_j)`. The ensemble is `0.5 * p_i(V1) + 0.5 * p_i(V2)`. Normalization gives a distribution summing to one but does not calibrate it. V3 is saved and shown as a candidate without being automatically selected for production.

## Evaluation

For evaluation year Y, fit on seasons strictly before Y and predict every completed race in Y. The model is fixed throughout the held-out season. Rolling form before each held-out race uses earlier completed races, including earlier races from that season. This matches a model trained annually with updated pre-race inputs; it does not represent all predictions being made before the season starts.

Default historical comparison: 2022–2025. Default diagnostic sample: 2026 before the target cutoff. Because research repeatedly inspected data, these results are not an untouched, preregistered test set. Comparisons must be described as retrospective. No statistical significance claim is made.

Metrics:

- Top-1 accuracy: fraction of races where the largest score belongs to the winner.
- Race log loss: mean `-log(p_actual_winner)`; probabilities clip below `1e-12` for numerical safety.
- Multiclass Brier: average across races of `sum_driver((p_i - winner_i)^2)`. This is the sum convention, not the row-averaged binary Brier convention.
- Pole baseline: select the minimum represented starting position. It reports top-1 accuracy only, not probabilistic loss for hard 0/1 scores.

Ties select the first row in stable race/driver-ID order. Aggregate accuracy uses total correct divided by total races; loss metrics weight each season by its race count.

## Grid audit

Calibration reports split driver-race rows into P1, P2–P3, P4–P6, P7–P10, P11–P13 and P14+. Summed model scores are expected wins; observed winner counts are actual wins. Gaps are diagnostics rather than calibration fixes. Very small samples have high uncertainty.

## Artifacts and provenance

The full run records input SHA-256 hashes, the configuration hash, software versions, cutoff, actual last training date, model feature sets and hyperparameters. Text hashes normalize CRLF to LF before hashing so Git's Windows/Linux line-ending conversion does not invalidate reports. Dashboard report hashes prevent silently combining inconsistent report files. Training bundles record history hashes, cutoff and expected features; prediction refuses mismatches. Joblib artifacts are executable serialization and must remain trusted local files.

`reports/demo/` includes only the small report files required for initial dashboard exploration. Rebuildable model files and the full output directory stay out of Git. To regenerate the demo, run the full pipeline, then copy `predictions.csv`, `backtest_summary.csv`, `grid_calibration.csv` and `manifest.json` from the output directory to `reports/demo/` together.

## Operational limits

The collector uses an external API with retry and pagination. Cached pages are snapshots, not proof of freshness. Run collection with `--refresh` when intentionally updating history, and use a new config to preserve the frozen example. API failures propagate as nonzero command-line errors. The live API and FIA document provenance are not verified by offline tests.

The dashboard reads generated reports only. It starts from the bundled demo unless `F1_REPORT_DIR` is explicitly set. It neither retrains models nor silently switches to whatever happens to be in the local artifacts directory.
