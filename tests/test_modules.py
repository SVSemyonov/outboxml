from unittest import TestCase
from unittest import main

from sklearn.base import BaseEstimator

from outboxml.core.data_prepare import OptiBinningEncoder
from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.core.pydantic_models import DataModelConfig
from outboxml.dataset_retro import RetroDataset
from outboxml.datasets_manager import DataSetsManager, DSManagerResult
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path

from outboxml.extractors import BaseExtractor
from outboxml.models import BaselineModels, ModelsWrapper

test_configs_path = Path(__file__).resolve().parent.parent / "examples"/"titanic"/"configs"
test_data_path = Path(__file__).resolve().parent.parent /"examples"/ "titanic"/"data"

config_name = str(test_configs_path / 'config-example-titanic.json')


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
        self.assertEqual(self.dsManager.dataset.shape, (891, 12))

    def test_db_extractor(self):
        self.assertIsInstance(BaseExtractor(data_config=DataModelConfig(source='database',
                                                  table_name_source='public."TitanicExample"')).extract_dataset(), pd.DataFrame)

    def test_DFs(self):
        self.dsManager.get_TrainDfs(model_name='first')
        self.assertEqual(self.dsManager.X.shape, (891, 11))
        self.assertEqual(self.dsManager.Y.shape, (891, 1))
        self.assertEqual(len(set(self.dsManager.index_test) & set(self.dsManager.index_train)), 0)

    def test_encoding(self):
        X, y = self.dsManager.get_TrainDfs(model_name='first')
        mapping, bins = OptiBinningEncoder(X=X['SEX'], y=y, train_ind=X.index, type='numerical', name='first').encode_data()
    def test_getTrainDfs(self):

        X_train, y_train = self.dsManager.get_TrainDfs('second')
        self.assertEqual(X_train.shape, (712, 4))
        self.assertEqual(y_train.shape, (712,))

    def test_getTrainResults(self):
        results1 = self.dsManager.fit_models()
        self.assertIsInstance(results1, dict)
        self.assertEqual(len(results1['first']['train']), self.dsManager.data_config.data.targetslices[0]['slices'] + 1)
        rf = RandomForestRegressor()
        X_train, y_train = self.dsManager.get_TrainDfs('first')
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
        X_train, y_train = self.dsManager.get_TrainDfs('first')
        lgr.fit(X_train, y_train)
        X_train, y_train = self.dsManager.get_TrainDfs('second')
        rf.fit(X_train, y_train)
        resultDics = {'first': lgr, 'second': rf}
        results = self.dsManager.fit_models(resultDics,)
        self.assertIsInstance(results, dict)


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
        self.assertEqual(len(model_res.predictions['test']), 891)
        self.assertEqual(model_res.data_subset.X_test.shape, ( 891, 4))
        self.assertEqual(len(model_res.data_subset.features_categorical), 0)
        self.assertEqual(len(model_res.data_subset.features_numerical), 4)


    def test_default_models(self):
        self.dsManager.get_TrainDfs(model_name='first')
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





if __name__ == '__main__':
    main()
