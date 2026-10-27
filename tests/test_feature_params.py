import pandas as pd
import pytest

from outboxml.core.config_builders import feature_params


@pytest.mark.parametrize("depth, expected", [(0.005, ["A", "B", "C"]), (0.05, ["A", "B", "OTHER"]),
                                             (0.2, ["A", "OTHER"])])
def test_rare_categories_are_merged_into_other(depth, expected):
    serie = pd.Series(["A"] * 90 + ["B"] * 9 + ["C"], name="CAT")

    params = feature_params(serie, max_category_num=20, cutoff_nan=0.6, cutoff_1_category=0.99, depth=depth,
                            encoding_cat="WoE_cat_to_num")

    assert sorted(params["feature_values"].unique()) == expected
