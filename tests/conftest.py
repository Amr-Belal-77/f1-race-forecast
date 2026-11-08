import pandas as pd
import pytest


@pytest.fixture
def history():
    rows = []
    for year, finishes in [(2020, [1, 2]), (2021, [2, 1]), (2022, [1, 2])]:
        for driver, finish in zip(["a", "b"], finishes):
            rows.append(
                {
                    "season": year,
                    "round": 1,
                    "race_date": f"{year}-03-01",
                    "race_name": "Test GP",
                    "driver_id": driver,
                    "driver": driver.upper(),
                    "team_id": "team",
                    "team": "Team",
                    "grid_position": 1 if driver == "a" else 2,
                    "qualifying_position": 1 if driver == "a" else 2,
                    "finish_position": finish,
                    "points": 25 if finish == 1 else 18,
                    "winner": int(finish == 1),
                    "q1": "1:20.000" if driver == "a" else "1:21.500",
                    "q2": None,
                    "q3": None,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def grid():
    return pd.DataFrame(
        [
            {
                "driver_id": "a",
                "driver": "A",
                "team_id": "team",
                "team": "Team",
                "grid_position": 2,
                "qualifying_position": 1,
                "q1": "1:20.000",
                "q2": None,
                "q3": None,
            },
            {
                "driver_id": "new",
                "driver": "New Driver",
                "team_id": "team",
                "team": "Team",
                "grid_position": 1,
                "qualifying_position": 2,
                "q1": "1:21.500",
                "q2": None,
                "q3": None,
            },
        ]
    )
