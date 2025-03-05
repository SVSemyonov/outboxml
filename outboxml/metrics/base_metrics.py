from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance,
    f1_score, precision_score, recall_score, roc_auc_score, root_mean_squared_error
)

from loguru import logger


class BaseMetric(ABC):

    @abstractmethod
    def calculate_metric(self, *params) -> dict:
        pass


class BaseMetrics(BaseMetric):
    def __init__(self,
                 y_true: np.array,
                 y_pred: np.array,
                 exposure=None):
        self._y_true = y_true
        self._y_pred = y_pred
        self._exposure = exposure

    def calculate_metric(self, classification: bool = False) -> dict:
        if self._exposure is not None:
            y_pred_exp = self._y_pred * self._exposure
        else:
            y_pred_exp = self._y_pred

        try:
            if not classification:
                return {
                    "mae": round(mean_absolute_error(self._y_true, y_pred_exp, sample_weight=self._exposure), 4),
                    "rmse": round(root_mean_squared_error(self._y_true, y_pred_exp, sample_weight=self._exposure),
                                  4),
                    "r2": round(r2_score(self._y_true, y_pred_exp, sample_weight=self._exposure), 4),

                }
            else:
                logger.info('Metrics for classification||cut_off = 0.5')
                cutoff = 0.5
                return {
                        'f1_score': round(f1_score(self._y_true, (y_pred_exp > cutoff).astype(int),
                                                   sample_weight=self._exposure), 4),
                        'precision_score': round(precision_score(self._y_true,
                                                                 (y_pred_exp > cutoff).astype(int),
                                                                 sample_weight=self._exposure), 4),
                        'recall_score': round(recall_score(self._y_true,
                                                           (y_pred_exp > cutoff).astype(int),
                                                           sample_weight=self._exposure), 4),
                        'gini': round(2 * roc_auc_score(self._y_true, (y_pred_exp > cutoff).astype(int),
                                                   sample_weight=self._exposure) - 1, 4)
                        }
        except ValueError as e:
            logger.error(e)
            return {}

