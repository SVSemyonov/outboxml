from types import SimpleNamespace

import pandas as pd
import pytest

import outboxml.feature_selection as feature_selection
from outboxml.core.pydantic_models import FeatureSelectionConfig
from outboxml.feature_selection import FeatureSelectionInterface

ELIMINATION_ORDER = [f"F{i}" for i in range(8)]  # worst first, best last


class FakeShapAnalysis:
    def __init__(self, **kwargs):
        pass

    def result(self):
        return {"eliminated_features_names": ELIMINATION_ORDER[:4], "selected_features_names": ELIMINATION_ORDER[4:],
                "loss_graph": {"loss_values": list(range(len(ELIMINATION_ORDER)))}}


class NoCorrelations:
    def __init__(self, **kwargs):
        pass

    def result(self):
        return []


@pytest.mark.parametrize("top", [1, 4, 8])
def test_top_features_to_select_selects_exactly_top(monkeypatch, top):
    monkeypatch.setattr(feature_selection, "CatboostShapAnalysis", FakeShapAnalysis)
    monkeypatch.setattr(feature_selection, "CorrelationMatrix", NoCorrelations)
    config = FeatureSelectionConfig(top_features_to_select=top, cv_diff_value=None)
    data_subset = SimpleNamespace(X=pd.DataFrame(), features_numerical=[])

    selected = FeatureSelectionInterface(feature_selection_config=config, objective="binomial") \
        .feature_selection(data_subset, [], {})

    assert selected == ELIMINATION_ORDER[-top:]
