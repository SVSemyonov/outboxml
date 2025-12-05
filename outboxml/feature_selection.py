from copy import deepcopy

import pandas as pd
from abc import ABC
from loguru import logger

from tqdm import tqdm

from outboxml.analysis_tools import CorrelationMatrix, CatboostShapAnalysis,CVStability
from outboxml.core.data_prepare import PrepareDatasetResult
from outboxml.core.enums import FeaturesTypes, FeatureEngineering
from outboxml.core.prepared_datasets import BasePrepareDataset
from outboxml.core.pydantic_models import FeatureSelectionConfig, ModelConfig, FeatureModelConfig
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
        self.to_drop = []
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


    def feature_selection(self,data_subset: ModelDataSubset, new_features_list: list, params: dict = None ):
        catboost_shap_analysis = CatboostShapAnalysis(data_subset=data_subset,
                                                      config=self._config,
                                                      objective=self.objective,
                                                      params=params)
        summary = catboost_shap_analysis.result()
        res = pd.DataFrame([summary['eliminated_features_names'] + summary['selected_features_names'],
                            summary['loss_graph']['loss_values']]).T  # .plot()
        rank = self._config.top_feautures_to_select
        self.last = list(res[res.index >= (res.index.max() - rank)][0].values)
        logger.info('Choosing top ' + str(rank) + str(' features') + '||' + str(self.last))
        if self._config.max_corr_value is not None:
            self.to_drop = CorrelationMatrix(data=data_subset.X,
                              threshold=self._config.max_corr_value,
                              feature_importance_list=self.last,
                              features_numerical=data_subset.features_numerical
                              ).result()

        logger.info('Features to drop||' + str(self.to_drop))

        selected_features = []
        if self._config.cv_diff_value is not None:
            self.to_drop = CVStability(list_to_exclude=self.to_drop,
                                       data_subset=data_subset,
                                       config=self._config,
                                       objective=self.objective,
                                       catboost_params=params,
                                       features=new_features_list).result()
        for feature in self.last:
            if feature not in self.to_drop: selected_features.append(feature)
        return selected_features


class BaseFS:
    """Class for selecting new features for a dataset based on configuration settings.
    Utilizes base interfaces for data preparation and feature selection.
    Parameters:

        parameters : FeatureSelectionConfig
            parameters for feature selection (see pydantic models for schema).
        feature_selection_interface : SelectionInterface
            Interface for feature selection with a `feature_selection()` method.
        prepare_data_interface : BasePrepareDataset
            Interface for data preparation with a `data_prepare()` method.
        new_features_list : list of str
            List of candidate features to evaluate.


    Methods:
       select_features()
            Executes the feature selection pipeline.
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
        """Method for executing the feature selection pipeline. Returns a list of names of selected features"""
        logger.debug('Feature selection||Prepare of new_features for research')
        self.value_type()
        if not self.parameters.use_temp_data:
            data_for_research = self._prepare_data(model_name=model_name)
        else:
            logger.info('Using temp data for preparing new features')
            data_for_research = self._prepare_data_using_temp(model_name=model_name)
        logger.debug('Feature selection||Preparation finished')
        try:
            selected_features = self._feature_selection_interface.feature_selection(data_for_research,
                                                                                    self.features_for_model,
                                                                                    params,
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
                                                        'new_features': self.types_dict},
                                                  )


    def value_type(self)->dict:
        """
        Splits features into groups based on the number of unique values they contain.

        Returns
        dict of 5 lists
        A dict containing five lists of strings:
        (
            bin_list:   List of binary features (exactly 2 unique values),
            cat_list:   List of categorical features (3 to 20 unique values),
            num_list:   List of numeric features (non-object dtype with many unique values),
            drop_list:  List of features to drop (only 1 unique value),
            obj_list:   List of object-type features (require special attention)
        )
        """
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
                         q1: float = 0.001, q2: float = 0.999, cut_outliers=True)->dict:
        """
            Mid-level function that processes a dict of params for data prepare

            Parameters
            ----------
            series : pd.Series to be transformed.
            type: str 'categorical' or 'numerical'
            depth : float or int, default=None
                Threshold for rare category grouping:
                - If float [0-1]: categories with frequency < depth are combined
                  (e.g., 0.01 groups values with <1% frequency)
                - If int > 1: categories with count < depth are combined
            q1, q2 : float, default=0.001, 0.999
                Quantile boundaries for outlier detection.
            cut_outliers : bool, default=True
                - If True: outlier values are set to None for later handling
                - If False: outlier values are clipped to the quantile boundaries (winsorization)

            Returns
            -------
            {'default': '_NAN_', 'encoding': 'WoE_cat_to_num'}
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
                    feature_params['replace'] = dict(
                        (value, FeatureEngineering.not_changed) for value in list(serie.unique()))
                except:
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
                    feature_params['replace'] = {"_TYPE_": "_NUM_"}

        logger.info(feature_params)
        return feature_params

    def _filter_data(self, data_subset: ModelDataSubset, selected_features: list)->ModelDataSubset:
        """Method for creating of the result list of selected features"""
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
        ModelDataSubset.drop_columns(data_subset, columns_to_drop)
        logger.info('Features for model||' + str(data_subset.X_train.columns.to_list()))

        if self._data_prepare_interface._new_model_config is not None:
            self._data_preprocessor._prepare_datasets[data_subset.model_name].update_model_config(
                features_to_drop=columns_to_drop,
            )


        return data_subset

    def _prepare_data_using_temp(self, model_name: str=None)->ModelDataSubset:
        init_version = deepcopy(self._data_preprocessor._version)
        version = self._data_preprocessor._version.split('_new')[0]
        self._data_preprocessor._pickle_subset.version = version
        self._data_preprocessor._model_config_pickle.version = version
        self._data_preprocessor._version = version

        logger.debug('Loading previously saved subsets')
        self._data_preprocessor._use_saved_files = True
        subset = self._data_preprocessor.get_subset(model_name)
        self._data_preprocessor._use_saved_files = False

        logger.debug('New data prepare')
        self._data_preprocessor._pickle_subset.version = init_version
        self._data_preprocessor._version = init_version
        self._data_preprocessor._model_config_pickle.version = init_version

        new_preproc = self._preprocessor_for_using_temp_files(model_name)
        new_features_subset = new_preproc.get_subset(model_name)

        self._data_preprocessor._prepare_datasets[model_name].update_model_config(
            features_to_drop=[],
            features_to_append=new_preproc.model_config(model_name).features,
        )

        return subset + new_features_subset


    def _preprocessor_for_using_temp_files(self, model_name):
        full_data = self._data_preprocessor.dataset
        feature_params = {}
        new_model_config = deepcopy(self._data_preprocessor.model_config(model_name))
        new_model_config.features = []

        for feature in self.features_for_model:
            if feature in self.types_dict['NUMERIC']:
                type = 'numerical'
            else:
                type = 'categorical'
            feature_params[feature] = self._prepare_feature(serie=full_data[feature], type=type)
            new_model_config.features.append(FeatureModelConfig(name=feature, **feature_params[feature]))
        new_prepare_datasets = deepcopy(self._data_preprocessor._prepare_datasets)
        new_prepare_datasets[model_name].load_model_config(new_model_config)
        new_data_config = deepcopy(self._data_preprocessor._data_config)
        new_data_config.extra_columns = None
        data_new = full_data[self.features_for_model +
                             [new_model_config.column_target]
                             ]
        self._data_prepare_interface.load_model_config(new_model_config)

        return DataPreprocessor(prepare_engine=self._data_preprocessor._prepare_engine,
                                version=self._data_preprocessor._version + '_new',
                                prepare_dataset_interface_dict=new_prepare_datasets,
                                data_config=new_data_config,
                                dataset=data_new,
                                retro=True,
                                use_saved_files=False,
                                external_config=self._data_preprocessor.config,
                                )
