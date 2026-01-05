from copy import deepcopy

import pandas as pd
from abc import ABC
from loguru import logger
from tqdm import tqdm

from outboxml.analysis_tools import (
    CorrelationMatrix,
    CatboostShapAnalysis,
    CVStability
)
from outboxml.core.config_builders import feature_params, feature_type
from outboxml.core.prepared_datasets import BasePrepareDataset
from outboxml.core.pydantic_models import (
    FeatureSelectionConfig,
    FeatureModelConfig
)
from outboxml.core.enums import FeatureTypesForSelection
from outboxml.data_subsets import DataPreprocessor, ModelDataSubset


class SelectionInterface(ABC):
    """Base interface for feature selection algorithms.

    Defines a common contract for all feature selection strategies
    used inside the framework.
    """

    def feature_selection(self, *params) -> list:
        """Run feature selection algorithm.

        :param params: Algorithm-specific parameters
        :return: List of selected feature names
        :rtype: list
        """
        pass


class FeatureSelection(ABC):
    """Abstract base class for feature selection pipelines."""

    def select_features(self, *params) -> ModelDataSubset:
        """Execute feature selection pipeline.

        :param params: Pipeline-specific parameters
        :return: Dataset with selected features
        :rtype: ModelDataSubset
        """
        pass


class FeatureSelectionInterface(SelectionInterface):
    """SHAP-based feature selection implementation.

    Uses CatBoost SHAP values to rank features and applies
    optional post-processing filters:

    - correlation threshold filtering
    - cross-validation stability filtering
    """

    def __init__(
        self,
        feature_selection_config: FeatureSelectionConfig,
        objective: str = 'RMSE'
    ):
        """Initialize feature selection interface.

        :param feature_selection_config: Feature selection configuration
        :type feature_selection_config: FeatureSelectionConfig
        :param objective: CatBoost objective name
        :type objective: str
        """

        self.to_drop = []
        self.last = None
        self.params = {}
        self._config = feature_selection_config

        self.__objective_map = {
            'poisson': "Poisson",
            'gamma': "Tweedie:variance_power=1.9999999",
            'binary': "Logloss",
            'binomial': "Logloss",
        }

        try:
            self.objective = self.__objective_map[objective]
        except KeyError:
            self.objective = objective

    def feature_selection(
        self,
        data_subset: ModelDataSubset,
        new_features_list: list,
        params: dict = None
    ) -> list:
        """Perform SHAP-based feature selection.

        Workflow:
            1. Train CatBoost model
            2. Calculate SHAP feature importance
            3. Select top-N features
            4. Remove correlated features
            5. Validate stability across CV folds

        :param data_subset: Prepared dataset
        :type data_subset: ModelDataSubset
        :param new_features_list: Candidate features
        :type new_features_list: list
        :param params: CatBoost parameters
        :type params: dict, optional
        :return: Selected feature names
        :rtype: list
        """

        catboost_shap_analysis = CatboostShapAnalysis(
            data_subset=data_subset,
            config=self._config,
            objective=self.objective,
            params=params
        )

        summary = catboost_shap_analysis.result()

        res = pd.DataFrame([
            summary['eliminated_features_names']
            + summary['selected_features_names'],
            summary['loss_graph']['loss_values']
        ]).T

        rank = self._config.top_feautures_to_select
        self.last = list(
            res[res.index >= (res.index.max() - rank)][0].values
        )

        logger.info(
            f'Choosing top {rank} features || {self.last}'
        )

        if self._config.max_corr_value is not None:
            self.to_drop = CorrelationMatrix(
                data=data_subset.X,
                threshold=self._config.max_corr_value,
                feature_importance_list=self.last,
                features_numerical=data_subset.features_numerical
            ).result()

        logger.info(f'Features to drop || {self.to_drop}')

        if self._config.cv_diff_value is not None:
            self.to_drop = CVStability(
                list_to_exclude=self.to_drop,
                data_subset=data_subset,
                config=self._config,
                objective=self.objective,
                catboost_params=params,
                features=new_features_list
            ).result()

        return [
            feature for feature in self.last
            if feature not in self.to_drop
        ]


class BaseFS(FeatureSelection):
    """Main feature selection pipeline.

    Integrates feature typing, data preparation, model training
    and post-processing into a single reproducible workflow.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        parameters: FeatureSelectionConfig,
        feature_selection_interface: SelectionInterface,
        prepare_data_interface: BasePrepareDataset,
        new_features_list: list = None,
    ):
        """Initialize feature selection pipeline.

        :param data_preprocessor: Data preprocessing engine
        :type data_preprocessor: DataPreprocessor
        :param parameters: Feature selection configuration
        :type parameters: FeatureSelectionConfig
        :param feature_selection_interface: Feature selection strategy
        :type feature_selection_interface: SelectionInterface
        :param prepare_data_interface: Dataset preparation interface
        :type prepare_data_interface: BasePrepareDataset
        :param new_features_list: New candidate features
        :type new_features_list: list, optional
        """

        self._data_preprocessor = data_preprocessor
        self._feature_selection_interface = feature_selection_interface
        self._data_prepare_interface = prepare_data_interface
        self.parameters = parameters
        self._new_features_list = new_features_list

        self.old_data_list = list(
            set(self._data_preprocessor.dataset.columns)
            - set(self._new_features_list)
        )

        self.types_dict = {}
        self.features_for_model = []
        self.columns_to_drop = []
        self.result_features = []

    def select_features(
        self,
        model_name: str = None,
        params: dict = {}
    ) -> ModelDataSubset:
        """Run full feature selection pipeline.

        :param model_name: Model identifier
        :type model_name: str, optional
        :param params: Model training parameters
        :type params: dict
        :return: Dataset with selected features
        :rtype: ModelDataSubset
        """

        logger.debug('Feature selection || Data preparation')

        if not self.parameters.use_temp_data:
            data_for_research = self.prepare_data(model_name)
        else:
            data_for_research = self._prepare_data_using_temp(
                model_name
            )

        try:
            selected_features = (
                self._feature_selection_interface.feature_selection(
                    data_for_research,
                    self.features_for_model,
                    params
                )
            )
        except Exception as exc:
            logger.error(f'{exc} || Returning original dataset')
            selected_features = []

        return self._filter_data(
            data_for_research,
            selected_features
        )

    def prepare_data(
        self,
        model_name: str = None
    ) -> ModelDataSubset:
        """Prepare dataset for feature selection.

        :param model_name: Model identifier
        :type model_name: str, optional
        :return: Prepared dataset
        :rtype: ModelDataSubset
        """

        feature_params_dict = {}
        full_data = self._data_preprocessor.dataset

        self.features_for_model = self.feature_types(full_data)

        for feature in self.features_for_model:
            feature_params_dict[feature] = self._prepare_feature(
                serie=full_data[feature]
            )

        return self._data_preprocessor.get_subset(
            model_name=model_name,
            prepare_func=self._data_prepare_interface.prepare_dataset,
            args={
                'features_params': feature_params_dict,
                'new_features': self.types_dict
            }
        )

    def feature_types(self, data: pd.DataFrame) -> list:
        """Detect feature types for candidate features.

        :param data: Source dataframe
        :type data: pd.DataFrame
        :return: List of features used for modeling
        :rtype: list
        """

        self.types_dict = {
            FeatureTypesForSelection.numeric: [],
            FeatureTypesForSelection.categorical: []
        }

        for col in tqdm(self._new_features_list):
            try:
                _ = data[col].nunique(dropna=False)
            except Exception:
                logger.error(f'{col} is not hashable')
                continue

            detected_type = feature_type(
                serie=data[col],
                max_category_num=self.parameters.count_category,
                cutoff_1_category=self.parameters.cutoff_1_category,
                cutoff_nan=self.parameters.cutoff_nan
            )

            if detected_type == 'numerical':
                self.types_dict[
                    FeatureTypesForSelection.numeric
                ].append(col)
            elif detected_type == 'categorical':
                self.types_dict[
                    FeatureTypesForSelection.categorical
                ].append(col)
            else:
                self.types_dict.setdefault(detected_type, []).append(col)

        features_for_model = (
            self.types_dict[FeatureTypesForSelection.numeric]
            + self.types_dict[FeatureTypesForSelection.categorical]
        )

        for key, value in self.types_dict.items():
            logger.info(f'{key}: {value}')

        return features_for_model

    def _prepare_feature(
        self,
        serie: pd.Series,
        depth: float = 0.01,
        q1: float = 0.001,
        q2: float = 0.999
    ) -> dict:
        """Build feature configuration for a single feature.

        :param serie: Feature series
        :type serie: pd.Series
        :param depth: Quantization depth
        :type depth: float
        :param q1: Lower quantile
        :type q1: float
        :param q2: Upper quantile
        :type q2: float
        :return: Feature configuration dictionary
        :rtype: dict
        """

        return feature_params(
            serie=serie,
            max_category_num=self.parameters.count_category,
            cutoff_nan=self.parameters.cutoff_nan,
            cutoff_1_category=self.parameters.cutoff_1_category,
            default_num=self.parameters.default_num,
            default_cat=self.parameters.default_cat,
            depth=self.parameters.depth,
            q1=q1,
            q2=q2,
            encoding_cat=self.parameters.encoding_cat,
            encoding_num=self.parameters.encoding_num
        )

    def _filter_data(
        self,
        data_subset: ModelDataSubset,
        selected_features: list
    ) -> ModelDataSubset:
        """Filter dataset to selected features only.

        :param data_subset: Prepared dataset
        :type data_subset: ModelDataSubset
        :param selected_features: Selected feature names
        :type selected_features: list
        :return: Filtered dataset
        :rtype: ModelDataSubset
        """

        result_features = [
            f for f in selected_features
            if f not in self.old_data_list
        ]

        self.result_features = result_features
        logger.info(f'Selected features || {result_features}')

        columns_to_drop = [
            f for f in data_subset.X.columns
            if f not in result_features
            and f not in self.old_data_list
        ]

        logger.info(f'Columns to drop || {columns_to_drop}')
        ModelDataSubset.drop_columns(data_subset, columns_to_drop)

        if self._data_prepare_interface._new_model_config is not None:
            self._data_preprocessor._prepare_datasets[
                data_subset.model_name
            ].update_model_config(
                features_to_drop=columns_to_drop
            )

        return data_subset

    def _prepare_data_using_temp(
        self,
        model_name: str = None
    ) -> ModelDataSubset:
        """Prepare data using temporary saved subsets.

        :param model_name: Model identifier
        :type model_name: str, optional
        :return: Combined dataset with old and new features
        :rtype: ModelDataSubset
        """

        init_version = deepcopy(self._data_preprocessor._version)
        base_version = init_version.split('_new')[0]

        self._data_preprocessor._version = base_version
        self._data_preprocessor._use_saved_files = True

        subset = self._data_preprocessor.get_subset(model_name)

        self._data_preprocessor._use_saved_files = False
        self._data_preprocessor._version = init_version

        new_preprocessor = self._preprocessor_for_using_temp_files(
            model_name
        )

        new_features_subset = new_preprocessor.get_subset(
            model_name
        )

        return subset + new_features_subset

    def _preprocessor_for_using_temp_files(
        self,
        model_name: str
    ) -> DataPreprocessor:
        """Create a temporary preprocessor for new features.

        :param model_name: Model identifier
        :type model_name: str
        :return: Temporary data preprocessor
        :rtype: DataPreprocessor
        """

        full_data = self._data_preprocessor.dataset
        feature_params_dict = {}

        new_model_config = deepcopy(
            self._data_preprocessor.model_config(model_name)
        )
        new_model_config.features = []

        for feature in self.features_for_model:
            feature_params_dict[feature] = self._prepare_feature(
                serie=full_data[feature]
            )
            new_model_config.features.append(
                FeatureModelConfig(
                    name=feature,
                    **feature_params_dict[feature]
                )
            )

        new_prepare_datasets = deepcopy(
            self._data_preprocessor._prepare_datasets
        )
        new_prepare_datasets[model_name].load_model_config(
            new_model_config
        )

        return DataPreprocessor(
            prepare_engine=self._data_preprocessor._prepare_engine,
            version=self._data_preprocessor._version + '_new',
            prepare_dataset_interface_dict=new_prepare_datasets,
            data_config=self._data_preprocessor._data_config,
            dataset=full_data[self.features_for_model +
                              [new_model_config.column_target]],
            retro=True,
            use_saved_files=False,
            external_config=self._data_preprocessor.config
        )
