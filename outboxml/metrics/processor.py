from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from outboxml.core.enums import ModelsParams
from outboxml.core.pydantic_models import DataModelConfig, ModelConfig
from outboxml.data_subsets import ModelDataSubset
from outboxml.metrics.base_metrics import BaseMetrics


class ModelMetrics:
    """Class for calculating model metrics for different model types.
    
    Supports classification, regression, and clustering model types.
    Calculates metrics for full datasets and data slices if configured.
    """
    
    def __init__(self,
                 model_config: ModelConfig,
                 data_subset: ModelDataSubset,
                 data_config: DataModelConfig=None):
        """Initialize ModelMetrics instance.
        
        :param model_config: Configuration object for the model.
        :type model_config: ModelConfig
        :param data_subset: Object containing train/test data subsets.
        :type data_subset: ModelDataSubset
        :param data_config: Optional configuration for data slices and preprocessing.
        :type data_config: DataModelConfig, optional
        """
        self._data_config = data_config
        self._model_config = model_config
        self._data_subset = data_subset
        if self._model_config.objective == ModelsParams.binary:
            self.model_type = 'classification'
        elif self._model_config.objective == ModelsParams.clustering:
            self.model_type = 'clustering'
        else:
            self.model_type = 'regression'


    def result_dict(self, predictions: Dict[str, pd.Series])->dict:
        """Calculate metrics dictionary for train and test predictions.
        
        :param predictions: Dictionary with 'train' and 'test' keys containing prediction Series.
        :type predictions: Dict[str, pd.Series]
        :return: Dictionary with metrics for train and test sets, including slice metrics if configured.
        :rtype: dict
        
        .. rubric:: Examples
        
        >>> predictions = {'train': train_pred, 'test': test_pred}
        >>> metrics = model_metrics.result_dict(predictions)
        >>> metrics['train']['full']
        {'mae': 0.1234, 'rmse': 0.5678, 'r2': 0.9012}
        """
        logger.debug(f'Model metrics {self.model_type}||{self._model_config.name}')

        result_metrics = {'train': {}, 'test':{}}

        if self._data_config is not None:
            if len(self._data_config.data.targetslices) > 0:
                logger.info('Train metrics for slices')
                result_metrics['train'] = self._metric_loop(X=self._data_subset.X, y_pred=predictions['train'],
                                                            y_true=self._data_subset.y_train)
                logger.info('Test metrics for slices')
                result_metrics['test'] = self._metric_loop(X=self._data_subset.X, y_pred=predictions['test'],
                                                            y_true=self._data_subset.y_test)

        logger.info('Model metrics||Full train metrics')
        result_metrics['train']['full'] = self.calculate_metrics(y_true=self._data_subset.y_train,
                                                                 y_pred=predictions['train'],
                                                                 weights=self._data_subset.exposure_train)
        logger.info('Model metrics||Full test metrics')
        result_metrics['test']['full'] = self.calculate_metrics(y_true=self._data_subset.y_test,
                                                                y_pred=predictions['test'],
                                                                weights=self._data_subset.exposure_test)

        logger.info(result_metrics)
        return result_metrics

    def calculate_metrics(self, y_pred: pd.Series, y_true: pd.Series=None, weights: pd.Series=None):
        """Calculate metrics for given predictions and true values.
        
        :param y_pred: Series of predicted values.
        :type y_pred: pd.Series
        :param y_true: Series of true values. Optional for clustering models.
        :type y_true: pd.Series, optional
        :param weights: Series of exposure/weight values for weighted metrics.
        :type weights: pd.Series, optional
        :return: Dictionary containing calculated metrics based on model type.
        :rtype: dict
        
        .. rubric:: Examples
        
        >>> metrics = model_metrics.calculate_metrics(y_pred=pred, y_true=actual, weights=exposure)
        >>> metrics
        {'mae': 0.1234, 'rmse': 0.5678, 'r2': 0.9012}
        """

        logger.debug('Calculating metrics')
        metrics_dict = BaseMetrics(y_true=y_true,y_pred=y_pred, exposure=weights,
                                   ).calculate_metric(model_type=self.model_type)
        logger.info(metrics_dict)
        return metrics_dict

    def _metric_loop(self, X: pd.DataFrame, y_pred: pd.Series, y_true: pd.Series=None, weights: pd.Series=None) -> Dict[str, dict]:
        """Calculate metrics for each data slice defined in data_config.
        
        :param X: DataFrame with feature data used for slicing.
        :type X: pd.DataFrame
        :param y_pred: Series of predicted values.
        :type y_pred: pd.Series
        :param y_true: Series of true values. Optional for clustering models.
        :type y_true: pd.Series, optional
        :param weights: Series of exposure/weight values for weighted metrics.
        :type weights: pd.Series, optional
        :return: Dictionary with slice names as keys and metric dictionaries as values.
        :rtype: Dict[str, dict]
        :raises Exception: If slice type is not 'numerical' or 'categorical'.
        """
        results = {}
        slicedDf = pd.DataFrame()
        for data_slice in self._data_config.data.targetslices:

            logger.info('Model metrics||Collecting slices')
            logger.info('Metrics for slice ' + data_slice['column'])
            if data_slice['column'] not in X.columns:
                logger.error('No target slice column in X')
                continue
            if data_slice['type'] == 'numerical':
                slicedDf['slice'] = data_slice['column'] + '_' + pd.cut(X[data_slice['column']],
                                                                       data_slice['slices']).astype(str)
            elif data_slice['type'] == 'categorical':
                slicedDf['slice'] =data_slice['column'] + '_' + X[data_slice['column']].astype(str)
            else:
                raise Exception("Unknown slice type")
            for name in slicedDf['slice'].unique():
                logger.info('Slice ' + name)
                slice_index = slicedDf.loc[slicedDf['slice'] == name].index
                y_true_indexed = y_true[y_true.index.isin(slice_index)] if y_true is not None else None
                y_pred_indexed = y_pred[y_pred.index.isin(slice_index)]
                weights_indexed = weights[weights.index.isin(slice_index)] if weights is not None else None
                if y_pred_indexed.empty:
                    results[name] = None
                else:
                    results[name] = self.calculate_metrics(y_pred=y_pred_indexed,
                                                           y_true=y_true_indexed,
                                                           weights=weights_indexed)


        return results


