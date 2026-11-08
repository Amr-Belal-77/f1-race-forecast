import numpy as np
import pandas as pd
import pytest

from f1_forecast.features import build_features
from f1_forecast.modeling import backtest, normalize_scores, score_races


def test_normalization_is_per_race():
    races = pd.DataFrame({"season": [2022] * 4, "round": [1, 1, 2, 2]})
    actual = normalize_scores([0.2, 0.6, 0.3, 0.3], races)
    np.testing.assert_allclose(actual, [0.25, 0.75, 0.5, 0.5])


@pytest.mark.parametrize("scores", [[0, 0], [np.nan, 1], [-1, 2], [np.inf, 1], [1.5, 0.5]])
def test_invalid_scores_are_rejected(scores):
    races = pd.DataFrame({"season": [2022, 2022], "round": [1, 1]})
    with pytest.raises(ValueError):
        normalize_scores(scores, races)


def test_race_metrics_are_hand_checked():
    races = pd.DataFrame(
        {
            "season": [2022] * 4,
            "round": [1, 1, 2, 2],
            "winner": [1, 0, 0, 1],
            "score": [0.75, 0.25, 0.6, 0.4],
        }
    )
    metrics = score_races(races, "score")
    assert metrics["correct"] == 1
    assert metrics["races"] == 2
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["brier"] == pytest.approx(0.4225)
    assert metrics["race_log_loss"] == pytest.approx(0.6019864022)


def test_backtest_records_strictly_earlier_training_seasons(history):
    summary, predictions = backtest(build_features(history), [2021, 2022], n_estimators=3)
    assert set(summary.model) == {"V1", "V2", "V3", "Blend_50_50", "Pole_Top1"}
    assert summary.loc[summary.season == 2021, "training_last_season"].eq(2020).all()
    assert summary.loc[summary.season == 2022, "training_last_season"].eq(2021).all()
    np.testing.assert_allclose(predictions.groupby(["season", "round"]).Blend_50_50.sum(), 1)


def test_empty_evaluation_fold_is_not_silently_skipped(history):
    with pytest.raises(ValueError, match="fold"):
        backtest(build_features(history), [2025], n_estimators=3)
