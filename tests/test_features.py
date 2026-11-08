import pandas as pd
import pytest

from f1_forecast.features import build_features, build_target_features


def test_current_outcomes_do_not_change_current_features(history):
    before = build_features(history)
    history.loc[history.season == 2021, "points"] = 999
    after = build_features(history)
    cols = ["driver_avg_points_last5", "team_avg_points_last5", "driver_win_rate_last5"]
    pd.testing.assert_frame_equal(before.loc[before.season == 2021, cols],
                                  after.loc[after.season == 2021, cols])
    assert after.loc[(after.season == 2022) & (after.driver_id == "a"),
                     "driver_avg_points_last5"].item() == 512


def test_rolling_history_and_qualifying_gaps(history):
    result = build_features(history.sample(frac=1, random_state=1))
    row = result.loc[(result.season == 2022) & (result.driver_id == "a")].iloc[0]
    assert row.driver_avg_points_last5 == 21.5
    assert row.team_avg_points_last5 == 43
    assert row.driver_prior_starts == 2
    assert row.driver_win_rate_last5 == 0.5
    assert result.loc[result.driver_id == "b", "q1_gap_s"].tolist() == [1.5, 1.5, 1.5]
    assert result.has_q2_time.sum() == 0


def test_target_features_share_historical_formula_and_exclude_future(history, grid):
    target = build_target_features(history, grid, "2022-03-01", 2022, 1)
    row = target.loc[target.driver_id == "a"].iloc[0]
    historical = build_features(history)
    reference = historical.loc[(historical.season == 2022) & (historical.driver_id == "a")].iloc[0]
    for col in ["driver_avg_finish_last5", "driver_avg_points_last5", "team_avg_points_last5"]:
        assert row[col] == reference[col]
    assert row.grid_places_lost_vs_qualifying == 1
    assert target.loc[target.driver_id == "new", "driver_avg_points_last5"].isna().all()
    assert target.loc[target.driver_id == "new", "driver_prior_starts"].item() == 0


@pytest.mark.parametrize("value", [1.5, -1, float("inf")])
def test_invalid_grid_values_are_rejected(history, grid, value):
    grid["grid_position"] = grid.grid_position.astype(float)
    grid.loc[0, "grid_position"] = value
    with pytest.raises(ValueError, match="grid_position"):
        build_target_features(history, grid, "2023-01-01", 2023, 1)


def test_duplicate_grid_is_rejected(history, grid):
    grid["grid_position"] = 1
    with pytest.raises(ValueError, match="grid"):
        build_target_features(history, grid, "2023-01-01", 2023, 1)


def test_pit_lane_start_is_preserved(history, grid):
    grid.loc[0, "grid_position"] = 0
    target = build_target_features(history, grid, "2023-01-01", 2023, 1)
    assert target.loc[target.driver_id == "a", "pit_lane_start"].item() == 1
    assert target.loc[target.driver_id == "a", "grid_position"].item() == 2
