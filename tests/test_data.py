import json

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
