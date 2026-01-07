from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance,
    f1_score, precision_score, recall_score, roc_auc_score, root_mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

from loguru import logger

from outboxml.core.enums import ModelTypes
from outboxml.core.pydantic_models import DataModelConfig


class BaseMetric(ABC):
    """Abstract base class for metric calculation.
    
    All metric classes should inherit from this class and implement calculate_metric method.
    """

    @abstractmethod
    def calculate_metric(self, *params) -> dict:
        """Calculate metrics based on provided parameters.
        
        :param *params: Variable number of parameters depending on implementation.
        :return: Dictionary containing calculated metrics.
        :rtype: dict
        """
        pass


class BaseMetrics(BaseMetric):
    """Class for calculating standard metrics for different model types.
    
    Supports regression, classification, and clustering model types.
    Calculates metrics with optional exposure weighting.
    """
    
    def __init__(self,
                 y_pred: np.array,
                 y_true: np.array = None,
                 exposure=None):
        """Initialize BaseMetrics instance.
        
        :param y_pred: Array of predicted values.
        :type y_pred: np.array
        :param y_true: Array of true values. Optional for clustering models.
        :type y_true: np.array, optional
        :param exposure: Array of exposure/weight values for weighted metrics.
        :type exposure: np.array, optional
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._exposure = exposure

    def calculate_metric(self, model_type: str='regression' ) -> dict:
        """Calculate metrics based on model type.
        
        :param model_type: Type of model ('regression', 'classification', or 'clustering').
        :type model_type: str
        :return: Dictionary containing metrics specific to model type.
        :rtype: dict
        
        .. rubric:: Examples
        
        >>> base_metrics = BaseMetrics(y_pred=pred, y_true=actual, exposure=weights)
        >>> metrics = base_metrics.calculate_metric(model_type='regression')
        >>> metrics
        {'mae': 0.1234, 'rmse': 0.5678, 'r2': 0.9012}
        """
        if self._exposure is not None:
            y_pred_exp = self._y_pred * self._exposure
        else:
            y_pred_exp = self._y_pred

        try:
            if model_type == ModelTypes.regression:
                return {
                    "mae": round(mean_absolute_error(self._y_true, y_pred_exp, sample_weight=self._exposure), 4),
                    "rmse": round(root_mean_squared_error(self._y_true, y_pred_exp, sample_weight=self._exposure),
                                  4),
                    "r2": round(r2_score(self._y_true, y_pred_exp, sample_weight=self._exposure), 4),

                }
            elif model_type == ModelTypes.classification:
                logger.info('Metrics for classification||cut_off = 0.5')
                cutoff = 0.5
                return {
                        'f1_score': round(f1_score(self._y_true, (y_pred_exp > cutoff).astype(int),
                                                   sample_weight=self._exposure), 4),
                        'precision_score': round(precision_score(self._y_true,
                                                                 (y_pred_exp > cutoff).astype(int),
                                                                 sample_weight=self._exposure), 4),
                        'recall_score': round(recall_score(self._y_true,
                                                           y_pred_exp,
                                                           sample_weight=self._exposure), 4),
                        'gini': round(2 * roc_auc_score(self._y_true, y_pred_exp,
                                                   sample_weight=self._exposure) - 1, 4)
                        }
            elif model_type == ModelTypes.clustering:
                return {
                    'silhouette_score': round(silhouette_score(y_pred_exp), 4),
                    'davies_bouldin_score': round(davies_bouldin_score(y_pred_exp), 4),
                    'calinski_harabasz_score': round(calinski_harabasz_score(y_pred_exp), 4),
                }
            else:
                logger.error('Unknown model_type||Returning {}')
                return {}

        except ValueError as e:
            logger.error(e)
            return {}
