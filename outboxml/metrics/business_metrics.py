from copy import deepcopy
from typing import Callable

import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error

from outboxml.metrics.base_metrics import BaseMetric


class BaseBusinessMetricConverter:
    def __init__(self,
                 use_exposure: bool=True):
        self.use_exposure = use_exposure

    def _model_data(self, data, ):
        exposure_test = data.datasubset.exposure_test
        exposure_train = data.datasubset.exposure_train
        logger.info('Collecting data for plots')
        X = pd.concat([data.datasubset.X_test, data.datasubset.X_train])
        y_true = pd.concat([data.datasubset.y_test, data.datasubset.y_train])
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
        if model_name is None:
            model_name = list(result1.keys())[0]
        data = deepcopy(result1[model_name])
        y1 = self._model_data(data)
        y1 = y1.rename(columns={'y_prediction': 'first_model_prediction'})
        y2 = self._model_data(result2[model_name]).rename(columns={'y_prediction': 'second_model_prediction'})
        df = pd.concat([y1[['first_model_prediction', 'y_true']], y2['second_model_prediction']], axis=1)

        return df


class BaseCompareBusinessMetric(BaseMetric):
    def __init__(self,
                 metric_function: Callable = mean_absolute_error,
                 metric_converter: BaseBusinessMetricConverter = None,
                 calculate_threshold=True,
                 use_exposure: bool=True,
                 direction: str='minimize'):

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

    def calculate_metric(self, result1: dict, result2: dict, threshold=[0.8, 1.2]) -> dict:
        logger.debug('Compare business metric||Calculating')
        logger.debug('Compare business metric||'+ self.direction)

        if not (isinstance(threshold, (list, tuple)) and len(threshold) == 2) and threshold is not None:

            raise Exception("Select correct threshold")
        df_for_model_comparison = self.metric_converter.convert_to_df(result1, result2)
        first_model_index = df_for_model_comparison.index
        second_model_index = df_for_model_comparison.index
        first_model_threshold = threshold[0]
        second_model_threshold = threshold[1]

        first_model_data = df_for_model_comparison.loc[first_model_index, ["y_true", "first_model_prediction"]]
        second_model_data = df_for_model_comparison.loc[second_model_index, ["y_true", "second_model_prediction"]]

        if (self._calculate_threshold):
            first_model_threshold = self.find_threshold(first_model_data, "first_model_prediction",
                                                        self.metric_function)
            first_model_index = df_for_model_comparison["first_model_prediction"] < first_model_threshold
            second_model_threshold = self.find_threshold(second_model_data, "second_model_prediction",
                                                         self.metric_function)
            second_model_index = df_for_model_comparison["second_model_prediction"] < second_model_threshold

        else:
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                if (threshold[0] is None):
                    first_model_index = df_for_model_comparison["first_model_prediction"] < threshold[1]
                    second_model_index = df_for_model_comparison["second_model_prediction"] < threshold[1]
                else:
                    first_model_index = df_for_model_comparison["first_model_prediction"].between(threshold[0],
                                                                                                  threshold[1])
                    second_model_index = df_for_model_comparison["second_model_prediction"].between(threshold[0],
                                                                                                    threshold[1])

        first_model_data = df_for_model_comparison.loc[first_model_index, ["y_true", "first_model_prediction"]]
        second_model_data = df_for_model_comparison.loc[second_model_index, ["y_true", "second_model_prediction"]]
        first_model_metric_results = self.metric_function(first_model_data["y_true"],
                                                                           first_model_data["first_model_prediction"])
        second_model_metric_results = self.metric_function(second_model_data["y_true"],
                                                                            second_model_data[
                                                                                "second_model_prediction"])
        self.result ={'first_model': {'metric': first_model_metric_results, 'threshold': first_model_threshold},
                      'second_model': {'metric': second_model_metric_results, 'threshold': second_model_threshold}}
        if self.direction == 'maximize':
            self.result['difference'] = self.result['first_model']['metric'] - self.result['second_model']['metric']
        else:
            self.result['difference'] = self.result['second_model']['metric']  - self.result['first_model']['metric']
        return self.result

    def find_threshold(self, df: pd.DataFrame, model_name: str, metric_function: Callable):
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
