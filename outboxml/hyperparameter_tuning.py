from typing import Callable, Union
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from loguru import logger
from optuna import create_study
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from sklearn.model_selection import cross_val_score, KFold
import statsmodels.formula.api as sf
from outboxml.core.enums import ModelsParams
from outboxml.data_subsets import DataPreprocessor

from outboxml.datasets_manager import DataSetsManager, ModelDataSubset
from outboxml.models import StatsModelsEstimator


class OptunaModel:
    def __init__(self,):
        pass
class HPTuningData:
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 exposure_train: pd.Series,
                 exposure_test: pd.Series,
                 features_numerical,
                 features_categorical):
        self.exposure_test = exposure_test
        self.exposure_train = exposure_train
        self.y_train = y_train
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.model_name = None

    @classmethod
    def load_from_ds_manager_subset(cls, data_subset: ModelDataSubset):

        features_numerical = data_subset.features_numerical
        features_categorical = data_subset.features_categorical
        exposure_train = data_subset.exposure_train# if data_subset.exposure_train else 1
        return cls(X_train=data_subset.X_train,
                   y_train=data_subset.y_train,
                   X_test=data_subset.X_test,
                   y_test=data_subset.y_test,
                   exposure_train=exposure_train,
                   exposure_test=data_subset.exposure_test,
                   features_numerical=features_numerical,
                   features_categorical=features_categorical)


class HPTuning:
    """Класс для подбора гиперпараметров на основе optuna решателя.
    Для задания параметров необходима функция вида:

    def parameters_for_optuna(trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 1200, step=100),
            'depth': trial.suggest_int('depth', 1, 15, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
            'subsample': trial.suggest_float("subsample", 0.05, 1.0),
            'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 101, step=10),
        }

    Parameters:
    ____________
    ds_manager: Initilized DS Manager with preprared_datasets

    sampler: Name of optuna sampler 'TPE', 'RandomSampler' or'CmaEsSampler':

    model: external model

    scoring_fun: function for calculating score for CV callback of str name from sklearn.metrics

    folds_num_for_cv: number of KFolds for CV

    objective = objective of model

    random_state = seed for sampler
    ___________
    Methods
    _________
    best_params( model_name: str, scoring_fun: Callable = None, trials: int = 100, direction: str = 'maximize')
        - returns best dict of params values

    parameters_for_optuna(trial) : parmeters setting method for optuna API
    """

    def __init__(self,
                 data_preprocessor: DataPreprocessor,
                 sampler='TPE',
                 model=None,
                 scoring_fun: Union[str, Callable] = 'neg_mean_absolute_error',
                 folds_num_for_cv: int = 3,
                 objective = 'RMSE',
                 random_state: int = 42
                 ):

        self._glm_model = None
        self._folds_num = folds_num_for_cv
        self._data_preprocessor = data_preprocessor
        self._sampler = sampler
        self._model = model
        self.scoring_fun = scoring_fun
        self._best_params = {}
        self.result_configs = {}
        self.objective = objective
        self._random_state = random_state
        self.__wrapper_model = False

    @staticmethod
    def parameters_for_optuna(trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 1200, step=100),
            'depth': trial.suggest_int('depth', 1, 15, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
            'subsample': trial.suggest_float("subsample", 0.05, 1.0),
            'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 101, step=10),
        }


    def best_params(self, model_name: str, scoring_fun: Callable = None, trials: int = 100, direction: str = 'maximize',
                    parameters_for_optuna_func: Callable = None, timeout=None):
        logger.debug('HP Tuning||Preparing data')
        hp_tuning_data = self._prepare_data(model_name)
        if scoring_fun is None:
            scoring_fun = self.scoring_fun
        self._sample_parameters()
        if self._model is None:
            self._model = self._load_model_from_config(model_name, hp_tuning_data)
        logger.debug('Optimizing...')
        best_params = self._optimize(model_name, hp_tuning_data,  scoring_fun, trials, direction,
                                     parameters_for_optuna_func, timeout=timeout)
        self._write_parameters(model_name, best_params)
        return best_params

    def _write_parameters(self, model_name: str, best_params: dict):
        self._best_params[model_name] = best_params

    def _prepare_data(self, model_name) -> HPTuningData:

        hp_tining_data = HPTuningData.load_from_ds_manager_subset(data_subset=self._data_preprocessor.get_subset(model_name),
                                                                  )
        return hp_tining_data

    def _sample_parameters(self):
        if self._sampler == 'TPE':
            self._sampler = TPESampler(seed=self._random_state)
        elif self._sampler == 'RandomSampler':
            self._sampler = RandomSampler(seed=self._random_state)
        elif self._sampler == 'CmaEsSampler':
            self._sampler = CmaEsSampler(seed=self._random_state)
        else:
            logger.info('Using default sampler')
            self._sampler = None

    def _prepare_catboost(self, model_name: str, hp_tuning_data, parameters):
        if self.result_configs[model_name].objective is not None:
            objective = self.result_configs[model_name].objective
            if objective == ModelsParams.poisson:
                objective = "Poisson"
            elif objective == ModelsParams.gamma:
                objective = "Tweedie:variance_power=1.9999999"
            elif objective == ModelsParams.binary:
                objective = "Logloss"
            else:
                objective = self.objective
        else:
            logger.info('No objective in config||Using from init')
            objective = self.objective
        model = self._model(cat_features=hp_tuning_data.features_categorical,
                            thread_count=-1,
                            objective=objective, verbose=False,
                            **parameters)

        return model, None

    def _prepare_xgb(self, model_name: str, parameters):
        if self.result_configs[model_name].objective is not None:
            objective = self.result_configs[model_name].objective
            if objective == ModelsParams.poisson:
                parameters['objective'] = "count:poisson"
            elif objective == ModelsParams.gamma:
                parameters['objective'] = "reg:gamma"
            elif objective == ModelsParams.binary:
                parameters['objective'] = "binary:logistic"
            else:
                parameters['objective'] = self.objective
        else:
            logger.info('No objective in config||Using from init')
            parameters['objective'] = self.objective

        model = self._model.set_params(**parameters)
        return model, None

    def _prepare_model(self, model_name: str, hp_tuning_data, parameters ):
        model = self._model
        if isinstance(model, (XGBRegressor, XGBClassifier)):
            return self._prepare_xgb(model_name, parameters)

        elif model.__name__ == 'CatBoostRegressor' or model.__name__ == 'CatBoostClassifier':
            return self._prepare_catboost(model_name, hp_tuning_data, parameters)

        elif  model.__name__ == 'from_formula':
            return self._prepare_glm(model_name, hp_tuning_data, parameters)

        else:
            return model, parameters

    def _optimize(self, model_name: str,
                  hp_tuning_data: HPTuningData,
                  scoring_fun: Callable,
                  trials: int,
                  direction: str='maximize',
                  parameters_for_optuna_func: Callable=None,
                  timeout=None):


        X = hp_tuning_data.X_train
        y_train = hp_tuning_data.y_train
        if hp_tuning_data.exposure_train is not None:
            y = y_train / hp_tuning_data.exposure_train
        else:
            y = y_train
        y.name = y_train.name

        def objective(trial):
            if parameters_for_optuna_func is not None:
              #  logger.info('User-defined paramters for optuna')
                parameters_for_optuna = parameters_for_optuna_func(trial)
            else:
              #  logger.info('Default paramters for optuna||Catboost')
                parameters_for_optuna = self.parameters_for_optuna(trial)
            model, parameters = self._prepare_model(model_name, hp_tuning_data, parameters_for_optuna)
            #FIXME catboost over glm
            score = cross_val_score(
                estimator=model,
                X=X[hp_tuning_data.features_numerical + hp_tuning_data.features_categorical],
                y=y,
                scoring=scoring_fun,
                params=parameters,
                cv=KFold(n_splits=self._folds_num),
            ).mean()
            return score

        # Подбор гиперпараметров
        study = create_study(sampler=self._sampler, direction=direction)
        study.optimize(objective, n_trials=trials, show_progress_bar=True, timeout=timeout, n_jobs=-1)
        return study.best_params

    def _load_model_from_config(self, model_name: str, hp_tuning_data):
        self.__wrapper_model = True
        model_config = self._data_preprocessor._prepare_datasets[model_name]._model_config
        self.result_configs[model_name] = model_config
        logger.info('Loading model and objective from config')
        wrapper = model_config.wrapper
        model = None
        if wrapper == ModelsParams.catboost and self.objective != ModelsParams.binary:
            model = CatBoostRegressor

        elif wrapper == ModelsParams.catboost and self.objective == ModelsParams.binary:
            model = CatBoostClassifier

        elif wrapper == ModelsParams.xgboost and self.objective != ModelsParams.binary:
            model = XGBRegressor(verbose=False, enable_categorical=True)

        elif wrapper == ModelsParams.xgboost and self.objective == ModelsParams.binary:
            model = XGBClassifier(verbose=False, enable_categorical=True)

        elif wrapper == ModelsParams.glm:
            model = sf.glm

        else:
            logger.error('No model in library for HP Tuning||Use external model')

        return model

    def _prepare_glm(self, model_name, hp_tuning_data, parameters):
        logger.info('Preparing glm')
        model = StatsModelsEstimator(sm_model=self._model, model_config=self.result_configs[model_name],
                                     datasubset=hp_tuning_data)

        return model, parameters


