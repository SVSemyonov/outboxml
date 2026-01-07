from copy import deepcopy
from typing import Callable

import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error

from outboxml.metrics.base_metrics import BaseMetric


class BaseBusinessMetricConverter:
    """Converter class for preparing data for business metric calculations.
    
    Handles data preparation including exposure weighting and DataFrame conversion.
    """
    
    def __init__(self,
                 use_exposure: bool=True):
        """Initialize BaseBusinessMetricConverter instance.
        
        :param use_exposure: Whether to use exposure weighting in calculations.
        :type use_exposure: bool
        """
        self.use_exposure = use_exposure

    def _model_data(self, data, ):
        """Extract and prepare model data for metric calculation.
        
        :param data: DSManagerResult object containing model predictions and data subsets.
        :type data: DSManagerResult
        :return: DataFrame with features, true values, predictions, and optionally exposure.
        :rtype: pd.DataFrame
        """
        exposure_test = data.data_subset.exposure_test
        exposure_train = data.data_subset.exposure_train
        logger.info('Collecting data for plots')
        X = pd.concat([data.data_subset.X_test, data.data_subset.X_train])
        y_true = pd.concat([data.data_subset.y_test, data.data_subset.y_train])
        y_true.name = 'y_true'
        y_pred = pd.concat([data.predictions['test'], data.predictions['train']])
        y_pred.name = 'y_prediction'
        y_graph = pd.concat([X, y_true.fillna(0), y_pred.fillna(0)], axis=1)
        if exposure_test is not None and exposure_train is not None and self.use_exposure:
            exposure = pd.concat([exposure_test, exposure_train])
            exposure.name = 'exposure'
            y_graph = pd.concat([y_graph, exposure], axis=1)
            y_graph['y_prediction'] = y_graph['y_prediction'] * y_graph['exposure']
        else:
            y_graph['exposure'] = 1
        return y_graph

    def convert_to_df(self, result1: dict, result2: dict = None, model_name: str = None):
        """Convert model results to DataFrame for comparison.
        
        :param result1: Dictionary with model results (DSManagerResult objects).
        :type result1: dict
        :param result2: Optional second dictionary with model results for comparison.
        :type result2: dict, optional
        :param model_name: Name of the model to extract. If None, uses first key from result1.
        :type model_name: str, optional
        :return: DataFrame with predictions and true values for comparison.
        :rtype: pd.DataFrame
        
        .. rubric:: Examples
        
        >>> converter = BaseBusinessMetricConverter(use_exposure=True)
        >>> df = converter.convert_to_df(result1, result2, model_name='model1')
        >>> df.columns
        Index(['first_model_prediction', 'y_true', 'second_model_prediction'], dtype='object')
        """
        if model_name is None:
            model_name = list(result1.keys())[0]
        data = deepcopy(result1[model_name])
        y1 = self._model_data(data)
        y1 = y1.rename(columns={'y_prediction': 'first_model_prediction'})
        if result2 is not None:
            y2 = self._model_data(result2[model_name]).rename(columns={'y_prediction': 'second_model_prediction'})
            df = pd.concat([y1[['first_model_prediction', 'y_true']], y2['second_model_prediction']], axis=1)
        else:
            df = y1[['first_model_prediction', 'y_true']]
        return df


class BaseCompareBusinessMetric(BaseMetric):
    """Class for comparing business metrics between two models.
    
    Calculates metrics with optional threshold optimization and model comparison.
    """
    
    def __init__(self,
                 metric_function: Callable = mean_absolute_error,
                 metric_converter: BaseBusinessMetricConverter = None,
                 calculate_threshold=True,
                 use_exposure: bool=True,
                 direction: str='minimize'):
        """Initialize BaseCompareBusinessMetric instance.
        
        :param metric_function: Function to calculate the metric (e.g., mean_absolute_error).
        :type metric_function: Callable
        :param metric_converter: Converter object for data preparation. If None, creates default converter.
        :type metric_converter: BaseBusinessMetricConverter, optional
        :param calculate_threshold: Whether to automatically calculate optimal threshold.
        :type calculate_threshold: bool
        :param use_exposure: Whether to use exposure weighting in calculations.
        :type use_exposure: bool
        :param direction: Optimization direction - 'minimize' or 'maximize'.
        :type direction: str
        """

        self.metric_function = metric_function
        self._calculate_threshold = calculate_threshold
        self.use_exposure = use_exposure
        if metric_converter is not None:
            self.metric_converter = metric_converter
        else:
            self.metric_converter = BaseBusinessMetricConverter(use_exposure=self.use_exposure )

        if direction not in ['minimize', 'maximize']:
            self.direction = 'minimize'
        else:
            self.direction = direction

    def calculate_metric(self, result1: dict, result2: dict=None, threshold=[0.8, 1.2]) -> dict:
        """Calculate business metric for model comparison.
        
        :param result1: Dictionary with first model results (DSManagerResult objects).
        :type result1: dict
        :param result2: Optional dictionary with second model results for comparison.
        :type result2: dict, optional
        :param threshold: Threshold values for filtering predictions. Can be list/tuple of two values or None.
        :type threshold: list, tuple, or None
        :return: Dictionary with metric results for both models and their difference.
        :rtype: dict
        :raises Exception: If threshold format is incorrect.
        
        .. rubric:: Examples
        
        >>> metric = BaseCompareBusinessMetric(metric_function=mean_absolute_error)
        >>> result = metric.calculate_metric(result1, result2, threshold=[0.8, 1.2])
        >>> result['difference']
        0.05
        """
        logger.debug('Compare business metric||Calculating')
        logger.debug('Compare business metric||'+ self.direction)
        second_model_metric_results = None
        second_model_threshold = None
        if not (isinstance(threshold, (list, tuple)) and len(threshold) == 2) and threshold is not None:

            raise Exception("Select correct threshold")
        df_for_model_comparison = self.metric_converter.convert_to_df(result1, result2)
        first_model_index = df_for_model_comparison.index
        first_model_threshold = threshold[0]
        first_model_data = df_for_model_comparison.loc[first_model_index, ["y_true", "first_model_prediction"]]
        if result2 is not None:
            second_model_index = df_for_model_comparison.index
            second_model_threshold = threshold[1]
            second_model_data = df_for_model_comparison.loc[second_model_index, ["y_true", "second_model_prediction"]]

        if (self._calculate_threshold):
            first_model_threshold = self.find_threshold(first_model_data, "first_model_prediction",
                                                        self.metric_function)
            first_model_index = df_for_model_comparison["first_model_prediction"] < first_model_threshold
            if result2 is not None:
                second_model_threshold = self.find_threshold(second_model_data, "second_model_prediction",
                                                             self.metric_function)
                second_model_index = df_for_model_comparison["second_model_prediction"] < second_model_threshold

        else:
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                if (threshold[0] is None):
                    first_model_index = df_for_model_comparison["first_model_prediction"] < threshold[1]
                    if result2 is not None:
                        second_model_index = df_for_model_comparison["second_model_prediction"] < threshold[1]
                else:
                    first_model_index = df_for_model_comparison["first_model_prediction"].between(threshold[0],
                                                                                                  threshold[1])
                    if result2 is not None:
                        second_model_index = df_for_model_comparison["second_model_prediction"].between(threshold[0],
                                                                                                        threshold[1])

        first_model_data = df_for_model_comparison.loc[first_model_index, ["y_true", "first_model_prediction"]]
        if result2 is not None:
            second_model_data = df_for_model_comparison.loc[second_model_index, ["y_true", "second_model_prediction"]]
        first_model_metric_results = self.metric_function(first_model_data["y_true"],
                                                                           first_model_data["first_model_prediction"])

        if result2 is not None:
            second_model_metric_results = self.metric_function(second_model_data["y_true"],
                                                                                second_model_data[
                                                                                    "second_model_prediction"])
        self.result ={'first_model': {'metric': first_model_metric_results, 'threshold': first_model_threshold},
                      'second_model': {'metric': second_model_metric_results, 'threshold': second_model_threshold}}
        if result2 is not None:
            if self.direction == 'maximize':
                self.result['difference'] = self.result['first_model']['metric'] - self.result['second_model']['metric']
            else:
                self.result['difference'] = self.result['second_model']['metric']  - self.result['first_model']['metric']
        else:
            self.result['difference'] = None
        return self.result

    def find_threshold(self, df: pd.DataFrame, model_name: str, metric_function: Callable):
        """Find optimal threshold value for metric calculation.
        
        Tests multiple threshold values and returns the one that minimizes the metric.
        
        :param df: DataFrame with true values and predictions.
        :type df: pd.DataFrame
        :param model_name: Name of the column containing predictions.
        :type model_name: str
        :param metric_function: Function to calculate the metric.
        :type metric_function: Callable
        :return: Optimal threshold value that minimizes the metric.
        :rtype: float
        
        .. rubric:: Examples
        
        >>> threshold = metric.find_threshold(df, 'first_model_prediction', mean_absolute_error)
        >>> threshold
        1.15
        """
        buckets = 100
        first_model_data = df[["y_true", model_name]]

        max_value = first_model_data[model_name].max()
        min_value = first_model_data[model_name].min()

        step = (max_value - min_value) / buckets

        metrics_dict = {}

        for iteration in range(buckets):
            threshold = min_value + step * (iteration + 1)
            filtered_data = first_model_data[first_model_data[model_name] < threshold]
            # print('threshold', threshold, 'len(filtered_data)', len(filtered_data))

            metric_value = metric_function(filtered_data["y_true"], filtered_data[model_name])
            metrics_dict[threshold] = metric_value

        max_metric_threshold = min(metrics_dict, key=metrics_dict.get)
        max_metric_value = metrics_dict[max_metric_threshold]

        # print(f'Max metric: {max_metric_value} Threshold: {max_metric_threshold}')

        return max_metric_threshold
