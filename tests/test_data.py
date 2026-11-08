import json

import pandas as pd
import pytest

from f1_forecast.data import collect_history, validate_history


def test_cutoff_is_exclusive(history):
    with pytest.raises(ValueError, match="cutoff"):
        validate_history(history, "2022-03-01")


@pytest.mark.parametrize("breakage", ["duplicate", "winner", "date", "label", "identity"])
def test_malformed_history_fails_before_training(history, breakage):
    if breakage == "duplicate":
        history.loc[1] = history.loc[0]
    elif breakage == "winner":
        history.loc[0, "winner"] = 0
    elif breakage == "date":
        history.loc[0, "race_date"] = "invalid"
    elif breakage == "label":
        history.loc[0, "winner"] = 2
    else:
        history.loc[0, "driver_id"] = None
    with pytest.raises(ValueError):
        validate_history(history, "2023-01-01")


def test_collection_from_cached_api_pages_respects_cutoff(tmp_path):
    driver = {"driverId": "a", "givenName": "A", "familyName": "Driver"}
    team = {"constructorId": "team", "name": "Team"}
    race = {
        "season": "2020",
        "round": "1",
        "date": "2020-03-01",
        "raceName": "GP",
        "Circuit": {"circuitId": "test"},
    }
    result = {
        "Driver": driver,
        "Constructor": team,
        "grid": "1",
        "position": "1",
        "points": "25",
        "status": "Finished",
    }
    quali = {"Driver": driver, "position": "1", "Q1": "1:20.000"}
    for kind, key, record in [
        ("results", "Results", result),
        ("qualifying", "QualifyingResults", quali),
    ]:
        races = [
            dict(race, **{key: [record]}),
            dict(race, round="2", date="2021-01-01", **{key: [record]}),
        ]
        payload = {"MRData": {"total": "2", "limit": "100", "RaceTable": {"Races": races}}}
        (tmp_path / f"2020_{kind}_0.json").write_text(json.dumps(payload))
    collected = collect_history(2020, 2020, "2021-01-01", tmp_path)
    assert len(collected) == 1
    assert collected.iloc[0].driver_id == "a"
    assert collected.iloc[0].winner == 1
    assert collected.iloc[0].q1 == "1:20.000"


@pytest.mark.parametrize(
    "column,value",
    [
        ("grid_position", 1.5),
        ("grid_position", -1),
        ("qualifying_position", 1.5),
        ("qualifying_position", -1),
        ("points", -1),
    ],
)
def test_invalid_historical_positions_and_points_are_rejected(history, column, value):
    history[column] = history[column].astype(float)
    history.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        validate_history(history)


def test_multiple_races_on_same_date_are_rejected_by_date_only_chronology(history):
    duplicate = history.loc[history.season == 2020].copy()
    duplicate["round"] = 2
    with pytest.raises(ValueError, match="date"):
        validate_history(pd.concat([history, duplicate], ignore_index=True))


def test_historical_season_matches_date_year(history):
    history.loc[history.season == 2020, "race_date"] = "2019-03-01"
    with pytest.raises(ValueError, match="season"):
        validate_history(history)
