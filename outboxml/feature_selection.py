from copy import deepcopy

import numpy as np
import pandas as pd
from abc import ABC
from datetime import datetime as dt
import phik
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType, Pool, CatBoostClassifier, CatBoostRegressor
from loguru import logger
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_val_score

from tqdm import tqdm

from outboxml.core.data_prepare import PrepareDatasetResult
from outboxml.core.prepared_datasets import BasePrepareDataset
from outboxml.core.pydantic_models import FeatureSelectionConfig, ModelConfig
from outboxml.data_subsets import DataPreprocessor, ModelDataSubset


class SelectionInterface(ABC):
    """Selection interface"""
    def __init__(self)->None:
        pass
    def feature_selection(self, *params) -> list:
        """Main selection method"""
        pass



class FeatureSelectionInterface(SelectionInterface):
    """Основной интерфейс для алгоритма выбора фичей
    Parameters:
    ______________
        feature_selection_config: конфиг для выбора фичей (см. pydantic_models.py)
        train_ind: индексы для train
        test_ind: индексы для test
        target: target Serie
        objective: objective type according to catboost objective (by default reading from config)

    Methods:
        feature_selection() - Метод отброра фичей. Возвращает список отобранных
    
    """
    def __init__(self, feature_selection_config: FeatureSelectionConfig, objective: str = 'RMSE'):

        super().__init__()
        self.to_drop = None
        self.last = None
        self.params = {}
        self._config = feature_selection_config

        self.__objective_map = {'poisson': "Poisson",
                                'gamma': "Tweedie:variance_power=1.9999999",
                                'binary': "Logloss",
                                'binomial': "Logloss",
                                }
        try:
            self.objective = self.__objective_map[objective]
        except KeyError:
            self.objective = objective

    def feature_selection(self,data_subset: ModelDataSubset, params: dict = {}, new_features_list: list = []):

        summary = self.__fit_catboost(data_subset, params)
        res = pd.DataFrame([summary['eliminated_features_names'] + summary['selected_features_names'],
                            summary['loss_graph']['loss_values']]).T  # .plot()
        rank = self._config.top_feautures_to_select
        self.last = list(res[res.index >= (res.index.max() - rank)][0].values)
        logger.info('Choosing top ' + str(rank) + str(' features') + '||' + str(self.last))
        self.to_drop = self.calculate_correlation(X=data_subset.X,
                                                  threshold=self._config.max_corr_value,
                                                  features_numerical=data_subset.features_numerical,
                                                  features_categorical=data_subset.features_categorical)
        logger.info('Features to drop||' + str(self.to_drop))

        selected_features = []
     #   self.calculate_stability(data_subset, features=new_features_list, params=params)
        for feature in self.last:
            if feature not in self.to_drop: selected_features.append(feature)
        return selected_features

    def __fit_catboost(self, data_subset: ModelDataSubset, params={}):
        logger.debug('Feature selection||Fitting catboost')
        X_train, X_test, y_train, y_test, cat_features = self.__train_data(data_subset)
        train_pool = Pool(X_train, y_train, feature_names=list(X_train.columns),
                          cat_features=cat_features)
        test_pool = Pool(X_test, y_test, feature_names=list(X_train.columns),
                         cat_features=cat_features)
        steps = X_train.shape[1]
        model = self.__load_model(params)
        summary = model.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=f'0-{steps - 1}',
            num_features_to_select=1,
            #     steps=train_X.shape[1] - 1,
            steps=steps - 1,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=False
        )
        return summary

    def calculate_correlation(self,
                              X,
                              features_numerical: list,
                              features_categorical: list,
                              threshold: float = 0.9):
        X = X[reversed(self.last)]  # упорядочен по значимости
        logger.debug('Feature selection||Calculating correlations')
        phik_matrix = X.phik_matrix(interval_cols=features_numerical)
        upper = phik_matrix.where(np.triu(np.ones(phik_matrix.shape), k=1).astype(
            bool))  # берем из набора скоррелированных только самую значимую

        # Найти признаки с корреляцией выше порогового значения
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if len(to_drop) > 0:
            logger.info('Dropping ' + str(to_drop))
        return to_drop

    def calculate_stability(self, data_subset: ModelDataSubset, features=[], params={}):
        """Calculation of stability using phik matrix"""
        #TODO разобраться со списками
        if features == []:
            return features
        else:
            features_for_calc = features.copy()
            cat_features = data_subset.features_categorical.copy()
            cat_features_for_calc = cat_features.copy()
            for feature in features:
                if feature in self.to_drop:
                    features_for_calc.remove(feature)
            for cat_feature in cat_features:
                if cat_feature in self.to_drop:
                    cat_features_for_calc.remove(cat_feature)
            X_train, X_test, y_train, y_test, _ = self.__train_data(data_subset)
            for feature in features_for_calc:
                catboost_features = cat_features_for_calc.copy()
                if len(features_for_calc) > 1:
                    features_for_cv = features_for_calc.copy()
                    features_for_cv.remove(feature)
                    X = X_train[X_train.columns[~X_train.columns.isin(features_for_cv)]]
                else:
                    X = X_train
                logger.debug('CV for feature ' + str(feature))
                for cat_feature in cat_features_for_calc:
                    if cat_feature not in X.columns:
                        catboost_features.remove(cat_feature)

                model = self.__load_model(params, catboost_features)

                try:
                    scoring = self.__choose_scoring_fun(model_name=data_subset.model_name)
                    scores = cross_val_score(model, X, y_train, cv=3, scoring=scoring)
                    logger.info('CV dif for feature||'+ str(np.max(scores)/np.min(scores)))
                    if (np.max(scores) / np.min(scores) - 1) > self._config.cv_diff_value:
                        logger.info('Dropping non-stable feature')
                        self.to_drop.append(feature)

                except Exception as exc:
                    logger.error(exc)
                    logger.info('No CV for feature')

    def __train_data(self, data_subset:ModelDataSubset)->tuple:
        X_train = data_subset.X_train
        X_test = data_subset.X_test
        cat_features = data_subset.features_categorical
        y_train = data_subset.y_train / data_subset.exposure_train if data_subset.exposure_train is not None else data_subset.y_train
        y_test = data_subset.y_test / data_subset.exposure_test if data_subset.exposure_test is not None else data_subset.y_test
        return X_train, X_test, y_train, y_test, cat_features

    def __choose_scoring_fun(self, model_name: str):
        if self._config.metric_eval[model_name] in get_scorer_names():
            return self._config.metric_eval[model_name]
        else:
            logger.error('Unknown metric for cv||Returning neg_mean_absolute_error')
            return 'neg_mean_absolute_error'

    def __load_model(self, params, cat_features=None):
        if self.objective == "Logloss":
            logger.info('Classification')
            model = CatBoostClassifier(objective=self.objective, cat_features=cat_features, verbose=False, **params)

        else:
            logger.info('Regression')
            model = CatBoostRegressor(objective=self.objective, cat_features=cat_features, verbose=False, **params)

        return model


class BaseFS:
    """Класс выбора новых фичей для датасета по настройкам конфига. Исползует базоыве интерфейсы подготовки данных и выбора фичей

    Parameters:

        parameters: конифг для выбора фичей (см. pydantic models)
        feature_selection_interface: интерфейс для выбора фичей с методом feature_selection()
        prepare_data_interface: интерфейс подготовки данных с методом data_prepare
        new_features_list: список фичей для проверки


    Methods:
        select_features
    """
    def __init__(self,
                 data_preprocessor: DataPreprocessor,
                 parameters: FeatureSelectionConfig,
                 feature_selection_interface: SelectionInterface,
                 prepare_data_interface: BasePrepareDataset,
                 new_features_list: list = None,
                 ):

        self._data_preprocessor = data_preprocessor
        self._feature_selection_interface = feature_selection_interface
        self._data_prepare_interface = prepare_data_interface
        self.parameters = parameters
        self._new_features_list = new_features_list
        self.old_data_list = list(set(list(self._data_preprocessor.dataset.columns.copy())) - set(self._new_features_list))
        self.types_dict = {}
        self.features_for_model = []
        self.columns_to_drop = []
        self.result_features = []

    def select_features(self, model_name: str=None, params={}):
        logger.debug('Feature selection||Prepare of new_features for research')
        self.value_type()
        data_for_research = self._prepare_data(model_name=model_name)
        logger.debug('Feature selection||Preparation finished')
        try:
            selected_features = self._feature_selection_interface.feature_selection(data_for_research,
                                                                                    params,
                                                                                    self.features_for_model,
                                                                                    )

        except Exception as exc:
            logger.error(str(exc) + '||Return origin')
            selected_features = []

        final_data = self._filter_data(data_for_research, selected_features)

        return final_data

    def _prepare_data(self, model_name: str=None)->ModelDataSubset:
        feature_params = {}
        full_data = self._data_preprocessor.dataset
        for feature in self.features_for_model:
            if feature in self.types_dict['NUMERIC']:
                type = 'numerical'
            else:
                type = 'categorical'
            feature_params[feature] = self._prepare_feature(serie=full_data[feature], type=type)

        return self._data_preprocessor.get_subset(model_name=model_name,
                                                  prepare_func=self._data_prepare_interface.prepare_dataset,
                                                  args={'features_params': feature_params,
                                                        'new_features': self.types_dict}
                                                    )


    def value_type(self):
        """
        Функция для разделения признаков по количеству значений данных в них

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм, из которого будут получены данные
        isprint : bool
            Флаг, отвечающий за то, будет ли выводиться строка после распределения по каждому признаку

        Returns
        -------
        (bin_list, cat_list, num_list, drop_list, obj_list): Cortage of 5 [list of str]
            (
            Список бинарных признаков (2 значения),
            Список категориальных признаков (от 3 до 20 уникальных значений в столбцах),
            Список числовых признаков (всё, что не object с большим количеством значений),
            Список признаков на удаление (1 значение),
            Список признаков типа object (обязательны к рассмотрению),
            )

        Examples
        --------
        #>>> (bin_list, cat_list, num_list, drop_list, obj_list) = value_type(df, isprint=False)
            BINARY: ['EventCreatedByGIBDDFlag', 'E-Garant', <...> ]
            CATEGORIAL: ['CustomerImportance', 'DTPOSAGOType', <...>]
            NUMERIC: ['LossNumber', 'InsuredSum', 'LossDateTime', <...>]
            TO_DROP: ['EventTypeDescription', 'InsuranceTypeName', <...>]
            OBJECT: ['ContractNumber', 'VictimContractNumber', <...>]
        """
        # Инициализация списков
        bin_list, cat_list, num_list, drop_list, date_list, obj_list = [], [], [], [], [], []
        cutoff_1_category = self.parameters.cutoff_1_category
        cutoff_nan = self.parameters.cutoff_nan
        count_category = self.parameters.count_category
        data = self._data_preprocessor.dataset
        # Цикл по колонкам датафрейма
        for col in tqdm(self._new_features_list):
            try:
                VC = data[col].nunique(dropna=False)
            except:
                logger.error(col + ' не хэшируемый тип')
                continue
            # Если только 1 значение
            if VC == 1 or \
                    data[col].value_counts(normalize=True, dropna=False).values[0] > cutoff_1_category or \
                    data[col].isna().mean() > cutoff_nan:
                drop_list.append(col)
            # Если только 2 значения
            elif VC == 2:
                bin_list.append(col)
            # Если значений в столбце от 3 до count_category
            elif 2 < VC <= count_category and data[col].dtype == object:
                cat_list.append(col)
            elif data[col].dtype == object:
                obj_list.append(col)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                date_list.append(col)
            else:
                num_list.append(col)

        self.types_dict = {"BINARY": bin_list,
                           "CATEGORIAL": cat_list,
                           "NUMERIC": num_list,
                           "TO_DROP": drop_list,
                           "OBJECT": obj_list,
                           "DATE": date_list
                           }
        self.features_for_model = self.types_dict['NUMERIC'] + self.types_dict['CATEGORIAL'] + self.types_dict['BINARY']
        for key, value in self.types_dict.items():
            logger.info(f"{key}:" + str(value))
        return self.types_dict

    def _prepare_feature(self, serie: pd.Series, method='label', type: str = 'categorical', depth: float = 0.01,
                         q1: float = 0.001, q2: float = 0.999, cut_outliers=True):
        """ Функция среднего уровня, работает с серией из датафрейма, обучает и применяет энкодер

            Parameters
            ----------
            Serie : pd.Series
                Серия для применения к ней преобразований
            method : str or None
                "std" for StandardScaler
                "minmax" for MinMaxScaler
                "label" for LabelEncoderPro
                None for None
            depth: float
                [0-1] для отсечения по долям. Если какого-то значения меньше 0.01 (1%), то его строки
                попадут в отдельную объединённую категорию
                > 1: int, для отсечения по количеству в value_counts
            q1, q2 : float
                границы для отсечения выбросов
            cut_outliers : bool
                Если True, значения выбросов будут отправлены в None для дальнейшей работы
                Если False, значения выбросов будут заменены на границы отсечения (винсоризация)
            name_of_feature: str
                Имя серии (ключ в словаре кодировщиков)

            Returns
            -------
            self.fit_transform_Encoder(Serie, method, name_of_feature): pd.Series
                Изменённая серия
            """
        feature_params = {}
        logger.info('Prepare feature||' + str(serie.name))
        if type == 'categorical':
            if 0 < depth < 1:
                VC = serie.value_counts(dropna=False, normalize=True).reset_index()
                try:
                    #                     VC = VC[VC[Serie.name] > depth]["index"]
                    VC = VC[VC['proportion'] > depth][serie.name]
                    # VC = serie.value_counts(dropna=False).reset_index()[:int(depth)]["index"]
                    feature_params['default'] = '_NAN_'  # проверить
                    serie.apply(lambda x: x if (x in set(VC)) or (pd.isnull(x)) else "OTHER")
                    feature_params['encoding'] = self.parameters.encoding_cat
                except:
                    #                     VC = VC[VC[0] > depth]["index"]
                    self.features_for_model.remove(serie.name)

        elif type == 'numerical':
            if q1 or q2:
                if cut_outliers:
                    feature_params['clip'] = {'min_value': serie.quantile(q1),
                                              # winsorize(serie, limits=[q1, q2], nan_policy='omit').data.min(),
                                              'max_value': serie.quantile(
                                                  q2)}  # winsorize(serie, limits=[q1, q2], nan_policy='omit').data.max()}
                    feature_params['default'] = serie.fillna(0).median()  # 0 #медиана или средняя в конфиге _MIN_ or _MEAN_ можно оставить пропуски
                    feature_params['encoding'] = self.parameters.encoding_num
        logger.info(feature_params)
        return feature_params

    def _filter_data(self, data_subset: ModelDataSubset, selected_features: list):
        logger.debug('Feature selection||Preparing results')
        result_features = []
        for selected_feature in selected_features:
            if selected_feature not in self.old_data_list:
                result_features.append(selected_feature)
        logger.info('Selected features: ' + str(result_features))
        self.result_features = result_features
        columns_to_drop = []

        for feature in data_subset.X.columns:
            if feature not in result_features and feature not in self.old_data_list:
                columns_to_drop.append(feature)
        logger.info('Columns to drop||'+ str(columns_to_drop))

        data_subset.X_train = data_subset.X_train.drop(columns=columns_to_drop).copy()
        data_subset.X_test = data_subset.X_test.drop(columns=columns_to_drop).copy() if data_subset.X_test is not None else None
        data_subset.X = data_subset.X.drop(columns=columns_to_drop).copy()
        for feature in columns_to_drop:
            if feature in data_subset.features_numerical:
                data_subset.features_numerical.remove(feature)
            elif feature in data_subset.features_categorical:
                data_subset.features_categorical.remove(feature)
        logger.info('Features for model||' + str(data_subset.X_train.columns.to_list()))
        self._data_preprocessor._prepare_datasets[data_subset.model_name]._model_config = self.get_updated_model_config(
            self._data_prepare_interface._new_model_config, columns_to_drop)
        return data_subset

    @staticmethod
    def get_updated_model_config(model_config: ModelConfig, features_to_drop: list) -> ModelConfig:

        model_config_to_return = deepcopy(model_config)
        if model_config_to_return.features is not None:
            model_config_to_return.features = [obj for obj in model_config_to_return.features if
                                               obj.name not in features_to_drop]
        return model_config_to_return
