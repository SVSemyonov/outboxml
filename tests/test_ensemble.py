import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

from outboxml.ensemble import Ensemble, EnsembleResult
from outboxml.core.errors import EnsembleError


class TestEnsemble(TestCase):

    def setUp(self):
        groups = {
            "x": [
                {
                    "model_config": {
                        "name": "K",
                        "objective": "poisson",
                        "column_target": "kx",
                        "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                    }
                },
                {
                    "model_config": {
                        "name": "M",
                        "objective": "gamma",
                        "column_target": "mx",
                        "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                    }
                },
            ],
            "y": [
                {
                    "model_config": {
                        "name": "K",
                        "objective": "poisson",
                        "column_target": "ky",
                        "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                    }
                },
                {
                    "model_config": {
                        "name": "M",
                        "objective": "gamma",
                        "column_target": "my",
                        "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                    }
                },
            ],
        }
        # make_ensemble loads groups from prod_models_path
        prod_models_dir = tempfile.TemporaryDirectory()
        self.addCleanup(prod_models_dir.cleanup)
        prod_models_path = Path(prod_models_dir.name)
        for group_name, group in groups.items():
            with open(prod_models_path / f"{group_name}.pickle", "wb") as f:
                pickle.dump(group, f)
        self.ensemble = Ensemble(config=SimpleNamespace(prod_models_path=prod_models_path,
                                                        results_path=prod_models_path))

    def test_make_ensemble_error_group_not_found(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=[("a>b", "x"), ("c<d", "z")])

    def test_make_ensemble_error_ensemble_name_1(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name=1, models_names=["K", "M"], groups=[("a>b", "x"), ("c<d", "y")])

    def test_make_ensemble_error_ensemble_name_2(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="", models_names=["K", "M"], groups=[("a>b", "x"), ("c<d", "y")])

    def test_make_ensemble_error_models_names_1(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names="K, M", groups=[("a>b", "x"), ("c<d", "y")])

    def test_make_ensemble_error_models_names_2(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "K", "M"],
                                        groups=[("a>b", "x"), ("c<d", "y")])

    def test_make_ensemble_error_groups_1(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups="x, y")

    def test_make_ensemble_error_groups_2(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=[])

    def test_make_ensemble_error_groups_3(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=["a>b", "c<d"])

    def test_make_ensemble_error_groups_4(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"],
                                        groups=[("a>b", "x"), ("c<d", "y", "yy")])

    def test_make_ensemble_error_groups_5(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=[(1, "x"), ("c<d", "y")])

    def test_make_ensemble_error_groups_6(self):
        with self.assertRaises(EnsembleError):
            self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=[("a>b", 1), ("c<d", "y")])

    def test_make_ensemble(self):
        self.ensemble.make_ensemble(ensemble_name="1", models_names=["K", "M"], groups=[("a>b", "x"), ("c<d", "y")])
        self.assertIsInstance(self.ensemble._result_pickle, list)
        self.assertEqual(len(self.ensemble._result_pickle), 2)
        self.assertIsInstance(self.ensemble._result_pickle[0], EnsembleResult)

        self.assertEqual(self.ensemble._result_pickle[0].model_name, "K")
        self.assertEqual(self.ensemble._result_pickle[0].models, [
            ("a>b", "x", {
                "model_config": {
                    "name": "K",
                    "objective": "poisson",
                    "column_target": "kx",
                    "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                }
            }),
            ("c<d", "y", {
                "model_config": {
                    "name": "K",
                    "objective": "poisson",
                    "column_target": "ky",
                    "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                }
            }),
        ])
        self.assertEqual(self.ensemble._result_pickle[1].models, [
            ("a>b", "x", {
                "model_config": {
                    "name": "M",
                    "objective": "gamma",
                    "column_target": "mx",
                    "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                }
            }),
            ("c<d", "y", {
                "model_config": {
                    "name": "M",
                    "objective": "gamma",
                    "column_target": "my",
                    "features": [{"name": "A", "default": 1, "replace": {"_TYPE_": "_NUM_"}}]
                }
            }),
        ])


if __name__ == '__main__':
    main()