from copy import deepcopy

import pandas as pd
from abc import ABC
from loguru import logger

from tqdm import tqdm

from outboxml.analysis_tools import CorrelationMatrix, CatboostShapAnalysis,CVStability
from outboxml.core.config_builders import feature_params, feature_type
from outboxml.core.data_prepare import PrepareDatasetResult
from outboxml.core.enums import FeaturesTypes, FeatureEngineering, FeatureTypesForSelection
from outboxml.core.prepared_datasets import BasePrepareDataset
from outboxml.core.pydantic_models import FeatureSelectionConfig, ModelConfig, FeatureModelConfig
from outboxml.data_subsets import DataPreprocessor, ModelDataSubset

"""
Feature selection module.

Provides interfaces and base implementations for feature selection
pipelines, including feature preparation, SHAP-based selection,
correlation filtering, and CV stability checks.
"""


class SelectionInterface(ABC):
    """Base interface for feature selection algorithms."""
    def feature_selection(self, *params) -> list:
        """Main feature selection method.

        :return: List of selected feature names.
        :rtype: list
        """
        pass


class FeatureSelection(ABC):
    """Abstract base class for feature selection implementations."""
    def select_features(self, *params)->ModelDataSubset:
        """Executes feature selection.

                :return: Dataset subset with selected features.
                :rtype: ModelDataSubset
                """
        pass


class FeatureSelectionInterface(SelectionInterface):
    """Main interface for feature selection algorithms.

    Uses SHAP values, correlation filtering, and cross-validation
    stability analysis to select the most relevant features.

    :param feature_selection_config: Configuration for feature selection.
    :type feature_selection_config: FeatureSelectionConfig

    :param objective: CatBoost objective name.
    :type objective: str
    """

    def __init__(self, feature_selection_config: FeatureSelectionConfig, objective: str = 'RMSE'):
        """Initializes feature selection interface."""
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
        """Selects features based on SHAP importance and stability criteria.

                :param data_subset: Prepared dataset subset for model training.
                :type data_subset: ModelDataSubset

                :param new_features_list: List of candidate features.
                :type new_features_list: list

                :param params: Additional CatBoost parameters.
                :type params: dict, optional

                :return: List of selected feature names.
                :rtype: list
                """
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


class BaseFS(FeatureSelection):
    """Base feature selection pipeline.

    Orchestrates feature preparation, selection, and dataset filtering.

    :param data_preprocessor: Dataset preprocessor.
    :type data_preprocessor: DataPreprocessor

    :param parameters: Feature selection configuration.
    :type parameters: FeatureSelectionConfig

    :param feature_selection_interface: Feature selection algorithm.
    :type feature_selection_interface: SelectionInterface

    :param prepare_data_interface: Dataset preparation interface.
    :type prepare_data_interface: BasePrepareDataset

    :param new_features_list: Candidate features.
    :type new_features_list: list, optional
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
        """Executes the full feature selection pipeline.

        :param model_name: Model name.
        :type model_name: str, optional

        :param params: Model parameters.
        :type params: dict

        :return: Dataset subset with selected features.
        :rtype: ModelDataSubset
        """
        logger.debug('Feature selection||Prepare of new_features for research')
        if not self.parameters.use_temp_data:
            data_for_research = self.prepare_data(model_name=model_name)
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

    def prepare_data(self, model_name: str=None)->ModelDataSubset:
        """Prepares dataset for feature selection.

                :param model_name: Model name.
                :type model_name: str, optional

                :return: Prepared dataset subset.
                :rtype: ModelDataSubset
                """
        feature_params = {}
        full_data = self._data_preprocessor.dataset
        self.features_for_model = self.feature_types(full_data)
        for feature in self.features_for_model:
            feature_params[feature] = self._prepare_feature(serie=full_data[feature])

        return self._data_preprocessor.get_subset(model_name=model_name,
                                                  prepare_func=self._data_prepare_interface.prepare_dataset,
                                                  args={'features_params': feature_params,
                                                        'new_features': self.types_dict},
                                                  )

    def feature_types(self, data: pd.DataFrame)->dict:
        """Determines feature types for selection.

                :param data: Input dataset.
                :type data: pandas.DataFrame

                :return: List of feature names suitable for modeling.
                :rtype: list
                """
        cutoff_1_category = self.parameters.cutoff_1_category
        cutoff_nan = self.parameters.cutoff_nan
        count_category = self.parameters.count_category
        self.types_dict[FeatureTypesForSelection.numeric] = []
        self.types_dict[FeatureTypesForSelection.categorical] = []
        # Цикл по колонкам датафрейма
        for col in tqdm(self._new_features_list):
            try:
                VC = data[col].nunique(dropna=False)
            except:
                logger.error(col + ' не хэшируемый тип')
                continue
            type = feature_type(serie=data[col], max_category_num=count_category, cutoff_1_category=cutoff_1_category,
                         cutoff_nan=cutoff_nan)
            if type == 'numerical':
                self.types_dict[FeatureTypesForSelection.numeric].append(col)
            elif type == 'categorical':
                self.types_dict[FeatureTypesForSelection.categorical].append(col)
            else:
                if type not in self.types_dict.keys():
                    self.types_dict[type] = []
                self.types_dict[type].append(col)
        features_for_model = self.types_dict[FeatureTypesForSelection.numeric] + self.types_dict[FeatureTypesForSelection.categorical]
        for key, value in self.types_dict.items():
            logger.info(f"{key}:" + str(value))
        return features_for_model

    def _prepare_feature(self, serie: pd.Series,  depth: float = 0.01,
                         q1: float = 0.001, q2: float = 0.999)->dict:
        """Prepares parameters for a single feature.

                Internal helper method.

                :param serie: Feature data.
                :type serie: pandas.Series

                :param depth: Category proportion cutoff.
                :type depth: float

                :param q1: Lower quantile.
                :type q1: float

                :param q2: Upper quantile.
                :type q2: float

                :return: Feature parameters dictionary.
                :rtype: dict
                """
        return feature_params(serie=serie, max_category_num=self.parameters.count_category,
                              cutoff_nan=self.parameters.cutoff_nan,
                              cutoff_1_category=self.parameters.cutoff_1_category,
                              default_num=self.parameters.default_num,
                              default_cat=self.parameters.default_cat,
                              depth=self.parameters.depth,
                              q1=q1, q2=q2,
                              encoding_cat=self.parameters.encoding_cat,
                              encoding_num=self.parameters.encoding_num)

    def _filter_data(self, data_subset: ModelDataSubset, selected_features: list)->ModelDataSubset:
        """Filters dataset based on selected features.

        Removes non-selected features and updates model configuration.

        :param data_subset: Dataset subset to filter.
        :type data_subset: ModelDataSubset

        :param selected_features: Selected feature names.
        :type selected_features: list

        :return: Filtered dataset subset.
        :rtype: ModelDataSubset
        """
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
        """Prepares dataset using temporary saved subsets.

                Used when `use_temp_data=True` in configuration.

                :param model_name: Model name.
                :type model_name: str, optional

                :return: Combined dataset with original and new features.
                :rtype: ModelDataSubset
                """
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
        """Creates a new DataPreprocessor for temporary feature preparation.

                Internal helper for reusing saved dataset subsets.

                :param model_name: Model name.
                :type model_name: str

                :return: Configured DataPreprocessor instance.
                :rtype: DataPreprocessor
                """
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
