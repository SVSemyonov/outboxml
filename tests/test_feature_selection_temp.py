import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml import config
from outboxml.core.prepared_datasets import FeatureSelectionPrepareDataset
from outboxml.core.pydantic_models import FeatureSelectionConfig
from outboxml.datasets_manager import DataSetsManager
from outboxml.feature_selection import BaseFS, FeatureSelectionInterface

CANDIDATES = ['PCLASS', 'NAME', 'TICKET', 'CABIN', 'EMBARKED']


def select(use_temp_data: bool):
    fs_config = FeatureSelectionConfig(
        metric_eval={"first": "accuracy"}, top_features_to_select=4, count_category=100, cutoff_1_category=0.9,
        cutoff_nan=0.7, max_corr_value=0.6, cv_diff_value=0.1, use_temp_data=use_temp_data,
        encoding_cat='WoE_cat_to_num', encoding_num='WoE_num_to_num', default_cat='_NAN_', default_num='_MEDIAN_')
    ds_manager = DataSetsManager(config_name=str(TEST_CONFIGS_PATH / "config-example-titanic.json"))
    ds_manager._data_preprocessor._retro = True
    base_fs = BaseFS(new_features_list=CANDIDATES, parameters=fs_config,
                     data_preprocessor=ds_manager._data_preprocessor,
                     prepare_data_interface=FeatureSelectionPrepareDataset(model_config=ds_manager._models_configs[0]),
                     feature_selection_interface=FeatureSelectionInterface(feature_selection_config=fs_config,
                                                                           objective='binomial'))
    data = base_fs.select_features(params={"iterations": 30}, model_name='first')
    return sorted(data.features_categorical + data.features_numerical), ds_manager


def test_temp_data_after_a_run_that_saved_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "results_path", tmp_path)
    select(use_temp_data=False)  # saves a subset that already contains the candidate features

    features, ds_manager = select(use_temp_data=True)

    assert features == ['AGE', 'EMBARKED', 'PCLASS', 'SEX', 'SIBSP']
    model_features = [f.name for f in ds_manager._data_preprocessor._prepare_datasets['first'].get_model_config().features]
    assert len(model_features) == len(set(model_features))
