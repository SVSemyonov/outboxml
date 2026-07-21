"""Tests for ``DSManagerResult.dict_for_export`` and ``DSManagerResult.from_dict``.

These two methods are the (de)serialization pair used to persist a trained model as
a plain ``dict`` and rebuild a ``DSManagerResult`` (with a unified
``GLMCatboostCombineModel`` predict interface) from it.

The fixture ``test_data/kasko_ul_2026_02_09_car_2026_06_22_11_49_34.pickle`` is a real
production export: a list of three model dicts (frequency/catboost, severity/catboost,
total/glm), each in ``dict_for_prod_export`` layout. We rebuild a ``DSManagerResult``
from every entry and assert the ``dict_for_export`` -> ``from_dict`` round-trip:

* ``dict_for_export`` emits the documented keys and stores the *inner* estimator under
  ``"model"`` with a ``"wrapper"`` that matches the config;
* ``from_dict`` rebuilds a ``GLMCatboostCombineModel`` whose ``_wrapper`` matches the
  config and whose inner estimator is the very same object (so predictions are identical);
* the round-trip is idempotent (export -> from_dict -> export yields an equal dict).
"""
import pickle
import sys
import types
from pathlib import Path
from unittest import TestCase, main

# polars is imported at module level across outboxml but unused on this path; fall back
# to a lightweight stub when it is not installed (mirrors test_ensemble_references.py).
try:  # pragma: no cover - environment dependent
    import polars  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without polars
    _polars_stub = types.ModuleType("polars")
    _polars_stub.__version__ = "0.0.0"
    _polars_attrs: dict = {}

    def _polars_getattr(name):
        if name not in _polars_attrs:
            _polars_attrs[name] = type(f"polars.{name}", (), {})
        return _polars_attrs[name]

    _polars_stub.__getattr__ = _polars_getattr
    sys.modules["polars"] = _polars_stub

import outboxml.models as _models

# The fixture was pickled when this package was named ``mldataworker``; alias the legacy
# module path to the current one so the stored ``GLMCatboostCombineModel`` resolves.
if "mldataworker" not in sys.modules:
    _legacy = types.ModuleType("mldataworker")
    _legacy.models = _models
    sys.modules["mldataworker"] = _legacy
    sys.modules["mldataworker.models"] = _models

from outboxml.datasets_manager import DSManagerResult
from outboxml.data_subsets import ModelDataSubset
from outboxml.core.pydantic_models import ModelConfig
from outboxml.models import GLMCatboostCombineModel

TESTS_DIR = Path(__file__).resolve().parent
PICKLE_PATH = TESTS_DIR / "test_data" / "kasko_ul_2026_02_09_car_2026_06_22_11_49_34.pickle"

EXPORT_KEYS = {
    "model_config", "wrapper", "model", "glm_model", "catbosot_model",
    "xgm_model", "min_max_scaler", "features_numerical", "features_categorical",
}


def _result_from_prod_entry(entry: dict) -> DSManagerResult:
    """Build a ``DSManagerResult`` from one ``dict_for_prod_export`` entry."""
    model_config = ModelConfig.model_validate(entry["model_config"])
    return DSManagerResult(
        model_name=model_config.name,
        model=entry["model"],
        model_config=model_config,
        data_subset=ModelDataSubset(
            model_name=model_config.name,
            features_numerical=entry["features_numerical"],
            features_categorical=entry["features_categorical"],
        ),
    )


class TestDictForExportFromDict(TestCase):
    """Round-trip ``dict_for_export`` <-> ``from_dict`` on a real production export."""

    @classmethod
    def setUpClass(cls):
        with open(PICKLE_PATH, "rb") as f:
            cls.group = pickle.load(f)
        # the fixture is the frequency/severity/total kasko bundle
        cls.results = [_result_from_prod_entry(entry) for entry in cls.group]

    def test_fixture_layout(self):
        """The fixture is a non-empty list of GLMCatboostCombineModel entries."""
        self.assertIsInstance(self.group, list)
        self.assertGreater(len(self.group), 0)
        for entry in self.group:
            self.assertIsInstance(entry["model"], GLMCatboostCombineModel)

    def test_dict_for_export_keys_and_inner_model(self):
        """``dict_for_export`` emits the documented keys and stores the inner estimator."""
        for result in self.results:
            exported = result.dict_for_export()
            self.assertEqual(set(exported), EXPORT_KEYS)
            # wrapper label and feature lists carry through unchanged
            self.assertEqual(exported["wrapper"], result.model_config.wrapper)
            self.assertEqual(exported["features_numerical"], result.data_subset.features_numerical)
            self.assertEqual(exported["features_categorical"], result.data_subset.features_categorical)
            # for a GLMCatboostCombineModel the inner estimator goes under "model";
            # the glm/catboost/xgb-specific slots stay empty
            self.assertIs(exported["model"], result.model.model)
            self.assertIsNone(exported["glm_model"])
            self.assertIsNone(exported["catbosot_model"])
            self.assertIsNone(exported["xgm_model"])

    def test_from_dict_rebuilds_combine_model(self):
        """``from_dict`` rebuilds a combine model with the right wrapper and same estimator."""
        for result in self.results:
            exported = result.dict_for_export()
            rebuilt = DSManagerResult.from_dict(exported)

            self.assertIsInstance(rebuilt.model, GLMCatboostCombineModel)
            # _wrapper must equal the config wrapper so predict() dispatches correctly
            self.assertEqual(rebuilt.model._wrapper, result.model_config.wrapper)
            # the inner estimator is passed through by reference -> identical predictions
            self.assertIs(rebuilt.model.model, result.model.model)
            # config and feature lists survive the round-trip
            self.assertEqual(rebuilt.model_config.name, result.model_config.name)
            self.assertEqual(rebuilt.model_config.wrapper, result.model_config.wrapper)
            self.assertEqual(rebuilt.data_subset.features_numerical, result.data_subset.features_numerical)
            self.assertEqual(rebuilt.data_subset.features_categorical, result.data_subset.features_categorical)

    def test_roundtrip_is_idempotent(self):
        """export -> from_dict -> export yields an equivalent dict."""
        for result in self.results:
            first = result.dict_for_export()
            second = DSManagerResult.from_dict(first).dict_for_export()

            self.assertEqual(set(first), set(second))
            self.assertEqual(first["wrapper"], second["wrapper"])
            self.assertEqual(first["model_config"], second["model_config"])
            self.assertEqual(first["features_numerical"], second["features_numerical"])
            self.assertEqual(first["features_categorical"], second["features_categorical"])
            # the underlying estimator stays the same object across the round-trip
            self.assertIs(first["model"], second["model"])


if __name__ == "__main__":
    main()
