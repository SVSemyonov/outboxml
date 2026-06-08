import json
from array import array
from copy import deepcopy
from unittest import TestCase
from unittest import main

import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import statsmodels.formula.api as sf

from outboxml.core.data_prepare import OptiBinningEncoder, PrepareDatasetResult
from outboxml.core.enums import ModelsParams
from outboxml.core.predict import one_model_predict
from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.core.pydantic_models import DataModelConfig, DataConfig, AllModelsConfig
from outboxml.dataset_retro import RetroDataset
from outboxml.datasets_manager import DataSetsManager, DSManagerResult
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path

from outboxml.extractors import BaseExtractor, SimpleExtractor
from outboxml.feature_importance import FeatureImportance
from outboxml.models import BaselineModels, ModelsWrapper, StatsmodelsModel, StatsModelsEstimator, CatboostModel, \
    CatboostOverGLMModel, XgboostModel, GLMCatboostCombineModel, BaseWrapperModel

test_configs_path = Path(__file__).resolve().parent/ "test_configs"
test_data_path = Path(__file__).resolve().parent/"test_data"

config_name = str(test_configs_path / 'config-example-titanic.json')
#config_name = str(test_configs_path / 'config-example-titanic_xgboost.json')


path_to_data = test_data_path / 'titanic.csv'
path_to_parquet = test_data_path / 'titanic.csv'


class TestTitanicDS(TestCase):

    def setUp(self) -> None:
        def data_post_prep_func(data: pd.DataFrame):
            data["SEX"] = pd.to_numeric(data["SEX"])
            return data

        self.dsManager = DataSetsManager(config_name=config_name,
                                         )

        self.dsManager_base = DataSetsManager(config_name=config_name,
                                              prepared_datasets={
                                                            'first': PrepareDataset(group_name='survived1',
                                                                                        data_post_prep_func=data_post_prep_func),
                                                            'second': PrepareDataset(group_name='survived2',)
                                                                 },

                                              use_baseline_model=1)

    def test_config_extractor(self):
        self.dsManager.load_dataset()
        self.assertIsNotNone(self.dsManager.dataset)
        self.assertEqual(self.dsManager.dataset.shape, (891, 7))

    def test_db_extractor(self):
        self.assertIsInstance(BaseExtractor(data_config=DataModelConfig(source='database',
                                                                        table_name_source='public."TitanicExample"',
                                                                        data=DataConfig())).extract_dataset(),
                              pd.DataFrame)

    def test_DFs(self):
        subset = self.dsManager.get_subset(model_name='first')
        self.assertEqual(subset.X.shape, (891, 3))
        self.assertEqual(subset.y_train.shape, (712, ))
     #   self.assertEqual(len(set(self.dsManager.index_test) & set(self.dsManager.index_train)), 0)

    def test_encoding(self):
        subset = self.dsManager.get_subset(model_name='first')
        X = subset.X_train
        y = subset.y_train
        mapping, bins = OptiBinningEncoder(X=X['SEX'], y=y, train_ind=X.index, type='numerical', name='first').encode_data()
    def test_getsubset(self):
        subset = self.dsManager.get_subset('second')
        X_train = subset.X_train
        y_train = subset.y_train
        self.assertEqual(X_train.shape, (712, 4))
        self.assertEqual(y_train.shape, (712,))

    def test_getTrainResults(self):
        results1 = self.dsManager.fit_models()
        self.assertIsInstance(results1, dict)
        self.assertEqual(len(results1['first']['train']), self.dsManager.data_config.data.targetslices[0]['slices'] + 1)
        rf = RandomForestRegressor()
        subset = self.dsManager.get_subset('first')
        X_train = subset.X_train
        y_train = subset.y_train
        rf.fit(X_train, y_train)
        resultDics = {'first': rf}
        results2 = self.dsManager.fit_models(models_dict=resultDics)
        self.assertIsInstance(results2, dict)
        self.assertIsInstance(self.dsManager.get_result()['first'], DSManagerResult)

    def test_api_models_and_metrics(self):

        self.dsManager.index_train = pd.Index([i for i in range(891) if i % 2 == 0])
        self.indexTest = pd.Index([i for i in range(300) if i % 2 == 1])
        lgr = RandomForestClassifier()
        rf = RandomForestClassifier()
        subset = self.dsManager.get_subset('first')
        X_train = subset.X_train
        y_train = subset.y_train
        lgr.fit(X_train, y_train)
        subset = self.dsManager.get_subset('second')
        X_train = subset.X_train
        y_train = subset.y_train
        rf.fit(X_train, y_train)
        resultDics = {'first': lgr, 'second': rf}
        results = self.dsManager.fit_models(resultDics,)
        self.assertIsInstance(results, dict)

    def test_feature_importance(self):
        with open(file=config_name, mode="r") as f:
            all_models_config_val = AllModelsConfig.model_validate(json.load(f))
            all_models_config_dict = all_models_config_val.model_dump()
            all_models_config_dict["models_configs"][0]["features"][0]["replace"] = {"MALE": "MALE", "FEMALE": "MALE"}

        dsManager_fi = DataSetsManager(config_name=all_models_config_dict)
        dsManager_fi.fit_models(calc_feature_importance=True)
        result = dsManager_fi.get_result()
        self.assertIsInstance(result['first'].feature_importance, FeatureImportance)
        self.assertIsInstance(result['second'].feature_importance, FeatureImportance)
        self.assertEqual(len(result['first'].feature_importance.importance_data), 3)
        self.assertEqual(len(result['second'].feature_importance.importance_data), 4)

        with open(file=config_name, mode="r") as f:
            all_models_config_val = AllModelsConfig.model_validate(json.load(f))
            all_models_config_dict = all_models_config_val.model_dump()
            all_models_config_dict["data_config"]["separation"]["kind"] = "none"

        dsManager_fi = DataSetsManager(config_name=all_models_config_dict)
        dsManager_fi.fit_models(calc_feature_importance=True)
        result = dsManager_fi.get_result()
        self.assertEqual(len(result['first'].feature_importance.importance_data), 3)
        self.assertEqual(len(result['second'].feature_importance.importance_data), 4)

    def test_baseline_classification(self):
        self.assertEqual(len(self.dsManager_base.fit_models().keys()),2)
        self.assertEqual(len(self.dsManager_base.fit_models()['first'].keys()), 2)
        self.assertEqual(len(self.dsManager_base.fit_models()['first']['train'].keys()), 6)

    def test_check_datadrift(self):
        self.assertIsInstance(self.dsManager.check_datadrift(model_name='first'), pd.DataFrame)


    def test_predict(self):
        data = pd.read_csv(path_to_parquet)
        result = self.dsManager.fit_models()
        result = self.dsManager.get_result()
        model_res = self.dsManager.model_predict(data, model_result=result, model_name='second')
        self.assertIsInstance(model_res, DSManagerResult)
        self.assertEqual(len(model_res.predictions['train']), 712)
        self.assertEqual(model_res.data_subset.X_train.shape, (712, 4))
        self.assertEqual(len(model_res.data_subset.features_categorical), 0)
        self.assertEqual(len(model_res.data_subset.features_numerical), 4)

        result = one_model_predict(group_name='test', model_result=model_res,
                          features_values=data.iloc[0].to_dict())
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['result'], dict)
        self.assertIsInstance(result['df'], dict)
        self.assertIsInstance(result['version_model'], dict)

    def test_default_models(self):
        self.dsManager.get_subset(model_name='first')
        datasubset = self.dsManager.get_subset('first')
        self.assertIsInstance(BaselineModels(dataset=datasubset,
                                             model_name='first', model_number=1).choose_model(), BaseEstimator)
        self.assertIsInstance(BaselineModels(dataset=datasubset,
                                             model_name='first', model_number=2).choose_model(), BaseEstimator)
        self.assertIsInstance(BaselineModels(dataset=datasubset,
                                             model_name='first', model_number=3).choose_model(), BaseEstimator)
        self.assertIsInstance(BaselineModels(dataset=datasubset,
                                             model_name='first', model_number=4).choose_model(), BaseEstimator)
        self.assertIsInstance(ModelsWrapper(data_subsets=self.dsManager.data_subsets,
                                            models_configs=self.dsManager._models_configs).models_dict(), dict)

    def test_retro(self):
        DataSetsManager(config_name=config_name, retro_changes=DSRetro(path_to_parquet=str(path_to_parquet))).fit_models()


class DSRetro(RetroDataset):
    def __init__(self, path_to_parquet: str):
        super().__init__()
        self._path_to_parquet = path_to_parquet

    def load_retro_data(self):
        self.retro_data = pd.read_csv(self._path_to_parquet)



class ModelsTest(TestCase):
    def setUp(self):
        self.dsManager = DataSetsManager(config_name=config_name,
                                         )
        self.subset = self.dsManager.get_subset(model_name='first')
        self.model_config = self.dsManager._models_configs[0]

    def test_stats_models(self):
        model = StatsmodelsModel(data_subset=self.subset,
                         model_config=self.model_config,
                         ).fit()
        model._wrapper = 'glm'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.params, pd.Series)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_catboost_model(self):
        model = CatboostModel(data_subset=self.subset,
                                 model_config=self.model_config,
                                 ).fit()
        model._wrapper = 'catboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_objective_catboost(self):
        model_config = deepcopy(self.model_config)
        model_config.objective = ModelsParams.rmsewithuncertainty
        model = CatboostModel(data_subset=self.subset,
                              model_config=model_config,
                              ).fit()

        model._wrapper = 'catboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config = deepcopy(self.model_config)

        model_config.objective = ModelsParams.gamma
        model = CatboostModel(data_subset=self.subset,
                              model_config=model_config,
                              ).fit()
        model._wrapper = 'catboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config = deepcopy(self.model_config)

        model_config.objective = ModelsParams.rmse
        model = CatboostModel(data_subset=self.subset,
                              model_config=model_config,
                              ).fit()
        model._wrapper = 'catboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config.objective = ModelsParams.binary
        model = CatboostModel(data_subset=self.subset,
                              model_config=model_config,
                              ).fit()
        model._wrapper = 'catboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)


    def test_catboostoverglm_model(self):
        model = CatboostOverGLMModel(data_subset=self.subset,
                                     sm_model=StatsmodelsModel(data_subset=self.subset,
                                                                 model_config=self.model_config,
                                                                 ).fit(),
                                 model_config=self.model_config,
                                 )
        model.fit()
        model._wrapper = 'catboost_over_glm'
        self.assertIsInstance(model, CatboostOverGLMModel)
        self.assertIsInstance(model._model_sm, GLMResultsWrapper )
        self.assertIsInstance(model._model_ctb, CatBoostRegressor)
        self.assertIsInstance(model._model_ctb.get_feature_importance(), np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_statsmodels_estimator_model(self):
        model = StatsModelsEstimator(datasubset=self.subset,
                                     sm_model=sf.glm,
                                     model_config=self.model_config,
                                     )
        sm=model.fit(self.subset.X_train, self.subset.y_train)
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(sm.params, pd.Series )
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_xgboost_model(self):
        model = XgboostModel(data_subset=self.subset,
                                 model_config=self.model_config,
                                work_type_fit='cpu' ).fit()
        model._wrapper = 'xgboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.feature_importances_, np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_objective_xgboost(self):
        model_config = deepcopy(self.model_config)
        model_config.objective = ModelsParams.rmsewithuncertainty
        model = XgboostModel(data_subset=self.subset,
                              model_config=model_config,
                             work_type_fit='cpu'
                             ).fit()

        model._wrapper = 'xgboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.feature_importances_, np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config = deepcopy(self.model_config)

        model_config.objective = ModelsParams.gamma
        model = XgboostModel(data_subset=self.subset,
                              model_config=model_config,
                             work_type_fit='cpu'
                             ).fit()
        model._wrapper = 'xgboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.feature_importances_, np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config = deepcopy(self.model_config)

        model_config.objective = ModelsParams.rmse
        model = XgboostModel(data_subset=self.subset,
                              model_config=model_config,
                              work_type_fit='cpu'
                              ).fit()
        model._wrapper = 'xgboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.feature_importances_, np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model_config.objective = ModelsParams.binary
        model = XgboostModel(data_subset=self.subset,
                              model_config=model_config,
                             work_type_fit='cpu'
                              ).fit()
        model._wrapper = 'xgboost'
        self.assertIsInstance(model, GLMCatboostCombineModel)
        self.assertIsInstance(model.model.feature_importances_, np.ndarray)
        self.assertIsInstance(model.predict(X=self.subset.X_test), pd.Series)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_baseline_models(self):
        model = BaselineModels(dataset=self.subset,
                               model_name='first',
                               model_number=1).choose_model()
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model.predict(X=self.subset.X_test), np.ndarray)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model = BaselineModels(dataset=self.subset,
                               model_name='first',
                               model_number=2).choose_model()
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model.predict(X=self.subset.X_test), np.ndarray)
        self.assertEqual(model.predict(X=self.subset.X_test).sum(), 0) #mode

        model = BaselineModels(dataset=self.subset,
                               model_name='first',
                               model_number=3).choose_model()
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model.predict(X=self.subset.X_test), np.ndarray)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

        model = BaselineModels(dataset=self.subset,
                               model_name='first',
                               model_number=4).choose_model()
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model.predict(X=self.subset.X_test), np.ndarray)
        self.assertGreater(model.predict(X=self.subset.X_test).sum(), 0)

    def test_basewrapper_models(self):

        models = ModelsWrapper(data_subsets={'first': self.subset},
                               models_configs=[self.model_config],
                               work_type_fit='CPU').models_dict()
        self.assertIsInstance(models, dict)


class TestCatboostSampleWeights(TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "SURVIVED": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            "SEX": ["male", "female", "female", "male", "female", "male", "female", "male", "female", "male", "female", "male"],
            "AGE": [22, 38, 26, 35, 28, 2, 54, 27, 14, 4, 58, 20],
            "ROW_WEIGHT": np.linspace(0.5, 1.6, 12),
        })
        self.config = {
            "group_name": "example",
            "project": "weighted",
            "version": "weights",
            "data_config": {
                "source": "csv",
                "local_name_source": "unused.csv",
                "separation": {
                    "kind": "random",
                    "random_state": 1,
                    "test_train_proportion": 0.25,
                    "period_column": ["AGE"]
                },
                "data": {
                    "targetcolumns": [],
                    "targetslices": []
                }
            },
            "models_configs": [
                {
                    "name": "weighted_model",
                    "column_target": "SURVIVED",
                    "column_weight": "ROW_WEIGHT",
                    "objective": "binary",
                    "wrapper": "catboost",
                    "features": [
                        {
                            "name": "SEX",
                            "default": "0",
                            "replace": {
                                "male": "1",
                                "female": "2"
                            }
                        },
                        {
                            "name": "AGE",
                            "default": 0,
                            "replace": {"_TYPE_": "_NUM_"}
                        }
                    ]
                }
            ]
        }
        self.ds_manager = DataSetsManager(
            config_name=self.config,
            extractor=SimpleExtractor(data=self.data)
        )

    def test_catboost_with_sample_weights(self):
        subset = self.ds_manager.get_subset("weighted_model")
        self.assertIsNotNone(subset.sample_weight_train)
        self.assertGreater(subset.sample_weight_train.nunique(), 1)

        model = CatboostModel(
            data_subset=subset,
            model_config=self.ds_manager._models_configs[0],
        ).fit()

        self.assertIsInstance(model.predict(X=subset.X_test), pd.Series)


class TestPrepareDatasets(TestCase):
    def setUp(self):
        self.dsManager = DataSetsManager(config_name=config_name,
                                         )
        self.subset = self.dsManager.get_subset(model_name='first')
        self.model_config = self.dsManager._models_configs[0]

    def test_prepare_dataset(self):
        pr_d = PrepareDataset(check_prepared=True,
                       calc_corr=True,
                       corr_threshold=0.8,
                       save_data=False,
                       model_config=self.model_config,
                       group_name='test')
        self.assertIsInstance(pr_d.prepare_dataset(
            data=self.dsManager.dataset,
            test_ind=self.dsManager._data_preprocessor.index_test,
            train_ind=self.dsManager._data_preprocessor.index_train,

        ), PrepareDatasetResult)
        model_config = deepcopy(self.model_config)
        pr_d.update_model_config(features_to_drop=['SEX'])
        self.assertEqual(len(pr_d.get_model_config().features),2)
        pr_d.update_model_config(features_to_append=[model_config.features[2]])
        self.assertEqual(len(pr_d.get_model_config().features),3)

if __name__ == '__main__':
    main()
