from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from outboxml.core.utils import (ResultPickle, calculate_correlations, check_prepared_dataset, find_drop_values,
                                 find_drop_values_pl, td_to_hms)


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame({
        "AGE": [22.0, 38.0, None, 35.0, 54.0, 2.0],
        "FARE": [7.25, 71.28, 7.92, 53.1, 51.86, 21.07],
        "SEX": ["male", "female", "female", "female", "male", "rare"],
    })


def test_check_prepared_dataset_counts_unique_values(data, capsys):
    unique = check_prepared_dataset(data, ["AGE", "FARE", "SEX"], min_group_rate=0.2)

    assert unique == {"AGE": 5, "FARE": 6, "SEX": 3}
    output = capsys.readouterr().out
    assert "1 nan values" in output
    assert "min group rate" in output


def test_calculate_correlations_reports_strong_pairs(data, capsys):
    data["FARE_COPY"] = data["FARE"] * 2

    corr = calculate_correlations(data, ["FARE", "FARE_COPY", "SEX"], max_corr_coef=0.9)

    assert {"FARE", "FARE_COPY", "SEX_male", "SEX_female"} <= set(corr.columns)
    assert corr.loc["FARE", "FARE_COPY"] == pytest.approx(1.0)
    assert "FARE <-> FARE_COPY" in capsys.readouterr().out


def test_find_drop_values_returns_config_values_missing_in_train():
    serie = pd.Series(["1", "2", "3"], index=[10, 11, 12])

    assert find_drop_values(serie, {"a": "1", "b": "4"}, train_ind=pd.Index([10, 11])) == ["4"]
    assert find_drop_values_pl(pl.Series(["1", "2"]), {"a": "1", "b": "4"}) == ["4"]


def test_td_to_hms():
    assert td_to_hms(timedelta(hours=1, minutes=2, seconds=3)) == "1h 2m 3s"


def write_pickles(path, names):
    for name in names:
        (path / name).write_bytes(b"")


def test_last_group_name_takes_latest_past_pickle_of_group(tmp_path):
    now = datetime.now()
    result_pickle = ResultPickle(SimpleNamespace(prod_models_path=tmp_path))
    write_pickles(tmp_path, [
        result_pickle.generate_name("titanic", now - timedelta(days=2)),
        result_pickle.generate_name("titanic", now - timedelta(days=1)),
        result_pickle.generate_name("titanic", now + timedelta(days=1)),  # not released yet
        result_pickle.generate_name("housing", now - timedelta(hours=1)),
        "notes.txt",
    ])

    assert result_pickle.get_last_group_name("titanic") == \
        result_pickle.generate_name("titanic", now - timedelta(days=1)).replace(".pickle", "")
    assert result_pickle.get_last_group_name().startswith("housing_")


def test_last_group_name_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ResultPickle(SimpleNamespace(prod_models_path=tmp_path / "missing")).get_last_group_name()
    with pytest.raises(FileNotFoundError, match="is empty"):
        ResultPickle(SimpleNamespace(prod_models_path=tmp_path)).get_last_group_name()

    write_pickles(tmp_path, ["titanic_without_date.pickle"])
    with pytest.raises(FileNotFoundError, match="no actual models"):
        ResultPickle(SimpleNamespace(prod_models_path=tmp_path)).get_last_group_name("titanic")
