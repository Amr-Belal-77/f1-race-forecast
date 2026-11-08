"""Walk-forward comparison of V1, V2, and a 50/50 probability ensemble.

Run from the project root:
    python src/backtest_ensemble.py

Required input:
    data/processed/f1_features_v2.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'processed' / 'f1_features_v2.csv'
OUTDIR = ROOT / 'data' / 'processed'
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)
df['race_date'] = pd.to_datetime(df['race_date'])
df = df.sort_values(['race_date', 'season', 'round', 'driver_id']).reset_index(drop=True)

# Recreate precisely the grid preprocessing used by the earlier V1/V2 experiments.
nonstandard = df['grid_position'].le(0)
df['pit_lane_start'] = nonstandard.astype(int)
max_grid = df.groupby(['season', 'round'])['grid_position'].transform('max')
df.loc[nonstandard, 'grid_position'] = max_grid[nonstandard] + 1

V1 = [
    'grid_position', 'qualifying_position', 'driver_avg_finish_last5',
    'driver_avg_points_last5', 'driver_win_rate_last5',
    'driver_prior_starts', 'team_avg_points_last5', 'pit_lane_start'
]
V2 = V1 + ['q1_gap_s', 'q2_gap_s', 'q3_gap_s', 'has_q2_time', 'has_q3_time']
missing = sorted(set(V2) - set(df.columns))
if missing:
    raise ValueError(f'Missing columns: {missing}')

# No future race results may enter the dataset.
assert df['race_date'].max() < pd.Timestamp('2026-11-08')
assert not df.duplicated(['season', 'round', 'driver_id']).any()
assert df.groupby(['season', 'round'])['winner'].sum().eq(1).all()


def build_model():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('classifier', XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=10,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=4,
        )),
    ])


def race_normalize(scores, races):
    scores = pd.Series(scores, index=races.index)
    totals = scores.groupby([races['season'], races['round']]).transform('sum')
    if not (totals > 0).all():
        raise ValueError('Zero or negative race score total')
    return scores / totals


def get_metrics(races, col):
    keys = ['season', 'round']
    winners = races.loc[races['winner'].eq(1), col].clip(lower=1e-12)
    selected = races.loc[races.groupby(keys)[col].idxmax()]
    squared = (races[col] - races['winner']) ** 2
    per_race_brier = squared.groupby([races['season'], races['round']]).sum()
    return {
        'races': len(selected),
        'correct': int(selected['winner'].sum()),
        'top1_accuracy': float(selected['winner'].mean()),
        'race_log_loss': float(-np.log(winners).mean()),
        'brier': float(per_race_brier.mean()),
    }

rows = []
all_preds = []
for year in range(2022, 2027):
    train = df.loc[df['season'] < year]
    test = df.loc[df['season'] == year].copy()
    if train.empty or test.empty:
        continue

    result = test[['season', 'round', 'driver', 'grid_position', 'winner']].copy()
    for label, cols in [('V1', V1), ('V2', V2)]:
        model = build_model()
        model.fit(train[cols], train['winner'])
        raw = model.predict_proba(test[cols])[:, 1]
        result[label] = race_normalize(raw, result)

    # Simple, pre-declared, non-tuned ensemble of the two probability vectors.
    result['Blend_50_50'] = 0.5 * result['V1'] + 0.5 * result['V2']
    assert np.allclose(result.groupby(['season', 'round'])['Blend_50_50'].sum(), 1.0)

    # Top-1 pole baseline has no probabilistic log loss because its scores are hard 0/1.
    pole = result.loc[result.groupby(['season', 'round'])['grid_position'].idxmin()]
    rows.append({'season': year, 'model': 'Pole_Top1', 'races': len(pole),
                 'correct': int(pole['winner'].sum()),
                 'top1_accuracy': float(pole['winner'].mean()),
                 'race_log_loss': np.nan, 'brier': np.nan})

    for label in ['V1', 'V2', 'Blend_50_50']:
        rows.append({'season': year, 'model': label, **get_metrics(result, label)})
    all_preds.append(result)

summary = pd.DataFrame(rows)
print('\nYEAR-BY-YEAR RESULTS:')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print('\nAGGREGATE 2022–2025 (do not use 2026 for model selection):')
for label, group in summary[summary['season'] <= 2025].groupby('model', sort=False):
    n = group['races'].sum()
    acc = group['correct'].sum() / n
    loss = np.average(group['race_log_loss'], weights=group['races']) if label != 'Pole_Top1' else np.nan
    brier = np.average(group['brier'], weights=group['races']) if label != 'Pole_Top1' else np.nan
    print(f'{label:13s} | {group["correct"].sum():2d}/{n} | accuracy {acc:.2%} | '
          f'log loss {loss:.4f} | Brier {brier:.4f}')

summary.to_csv(OUTDIR / 'ensemble_backtest_summary.csv', index=False)
pd.concat(all_preds).to_csv(OUTDIR / 'ensemble_backtest_driver_predictions.csv', index=False)
print('\nSaved 2 output CSV files to:', OUTDIR)
