import csv
import os
import pickle
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
from loguru import logger
from matplotlib.pyplot import show
from sqlalchemy import create_engine, TIMESTAMP

from outboxml import config as env_config
from outboxml.plots import MLPlot, CompareModelsPlot, CompareModelsMetrics, DataframeForPlots

from outboxml.metrics.base_metrics import BaseMetric
from outboxml.datasets_manager import DataSetsManager, DSManagerResult
from outboxml.core.utils import save_results


class ResultExport:
    """
    Module for results export.

    Основной модуль выгрузки и обработки основных результатов расчёта и моделирования.
    Может использоваться для сравнения моделей, в т.ч. для ретро. Использует компоненты из библиотеки plots

    from outboxml.plots import MLPlot, CompareModelsPlot, PlotlyWrapper, CompareModelsMetrics


    Parameters:
    -------------
    ds_manager: DataSetsManager object with results (after called fit_models() method)
    ds_manager_to_compare: another DataSetsManager object with results (after called fit_models() method)
    config: external config with passwords, logins, path to save and folders. By default .env is used
    ------------

    Methods:
    ------------
    save(to_pickle: bool=False,  path_to_save: pathlib.Path,  to_mlflow: bool = False, save_ds_manager: bool = False):
                method for saving main results for all models and targets.
                Saving metrics in excel for train and test, predictions in parquet,
                 model configs in json, model in pickle, ds_manager in pickle, artefacts to mlflow

    df_for_graphs(DSManagerResult, model_name: str, features: list[str]=['All'], plot_category: int = 1, bins_for_numerical_features: int = 5,
            use_exposure: bool = False)

    plots(model_name: str, features: list[str]=['All'], plot_category: int = 1, bins_for_numerical_features: int = 5,
            use_exposure: bool = False, user_plot_func=None):

            method for plotting fact/model result for chosen model with exposure for chosen features.
            If plot category = 1: factors plot
            if plot category = 0: main metrics plot
            If plot category = 2: cohort plot

            You can use custom function for plot. See plots() info for details

    compare_metrics(model_name: str, ds_manager_result: DSManagerResult=None)
    compare_models_plot( model_name: str,
                            features: list[str] = None,
                            plot_type: int = 1,
                            bins_for_numerical_features: int = 5,
                            use_exposure: bool = True,
                            show: bool = True,
                            user_plot_func=None,
                            cut_min_value: float = 0.01,
                            cut_max_value: float = 0.8,
                            samples: float = 100,
                            cohort_base: str = 'model1',
                            ds_manager_result: DSManagerResult = None)
    loss_plot - график убыточности
    compare_loss_plot - графики убыточности моделей

    """

    def __init__(self,
                 ds_manager: DataSetsManager,
                 ds_manager_to_compare: DataSetsManager = None,
                 config=None,
                 ):

        self._ds_manager = ds_manager
        self.result = None
        self.project_name = None  # self._ds_manager.config.project
        if not config:
            self.config = env_config
            logger.warning('Default ENV config for export results')
        else:
            self.config = config

        self.__base_path = self.config.base_path
        self.__results_path = self.config.results_path
        self.__results_path.mkdir(exist_ok=True)
        self._ds_manager_to_compare = ds_manager_to_compare

        try:
            self.result = self._ds_manager.get_result()
            if not isinstance(self.result, dict):
                raise ('Wrong format of results')
        #   logger.debug(str(self.project_name) + '||Results from DS_manager read')
        except TypeError:
            logger.error('No result found')
        if self._ds_manager_to_compare is not None:
            self.result_to_compare = self._ds_manager_to_compare.get_result()
            self.project_name_to_compare = self._ds_manager_to_compare.config.group_name
            if not isinstance(self.result_to_compare, dict):
                raise ('Wrong format of results')
            logger.debug(str(self.project_name_to_compare) + '||Results from DS_manager read')

    def __prepare_results(self, result: DSManagerResult):
        return result

    def save(self, to_pickle: bool = False,
             path_to_save: Path = None,
             to_mlflow: bool = False,
             save_ds_manager: bool = False,
             ds_manager_name: str = 'ds_manager'):
        """Save results for production. Metrics in xlsx, preidctions in parquet, models in pickle, configs in json
        use to_pickle for model saving
        path_to_save - path in Path() format
       """
        if path_to_save is None:
            path_to_save = self.config.results_path
            logger.info('Saving due to config file||Check config carefully')
        else:
            if isinstance(path_to_save, str): path_to_save = Path(path_to_save)
        path_to_save.mkdir(exist_ok=True)
        logger.debug('Saving started')

        metrics_test = None
        if save_ds_manager:
            logger.info('Saving ds_manager')
            path_ds_manager = os.path.join(path_to_save, ds_manager_name + ".pickle")
            with open(path_ds_manager, "wb") as f:
                pickle.dump(self._ds_manager, f)
        for result in self.result.values():

            res_dict = self.__prepare_results(result=result)
            if res_dict.model_name == 'general': continue
            logger.info('Results for model ' + res_dict.model_name)
            min_max_scaler = None
            try:
                min_max_scaler = res_dict.model.min_max_scaler
            except:
                logger.error('No min max scaler')
            predictions = pd.DataFrame()
            for key in res_dict.predictions:

                if key == 'train':
                    y_true = res_dict.data_subset.y_train
                    x = res_dict.data_subset.X_train

                elif key == 'test' and res_dict.predictions[key] is None:
                    logger.error('Test data in None')
                    metrics_test = pd.DataFrame([])
                    continue

                elif key == 'test' and res_dict.predictions[key] is not None:
                    metrics_test = self.metrics_df(model_name=res_dict.model_name, train_test='test')['full']
                    y_true = res_dict.data_subset.y_test
                    x = res_dict.data_subset.X_test

                predictions = pd.concat([predictions,
                                         self.__save_predictions(prediction=res_dict.predictions[key],
                                                                 y_true=y_true,
                                                                 model_name=res_dict.model_name,
                                                                 key=key,
                                                                 x=x,
                                                                 )])

            save_results(config=self.config,
                         min_max_scaler=min_max_scaler,
                         group_name=self._ds_manager.group_name,
                         model_config=res_dict.model_config,
                         features_numerical=res_dict.data_subset.features_numerical,
                         features_categorical=res_dict.data_subset.features_categorical,
                         model=res_dict.model,
                         to_mlflow=to_mlflow,
                         metrics_train=self.metrics_df(model_name=res_dict.model_name, train_test='train')['full'],
                         metrics_test=metrics_test,
                         predictions=predictions,
                         to_pickle=to_pickle)

    def compare_metrics(self, model_name: str,
                        ds_manager_result: dict = None,
                        business_metric: BaseMetric = None,
                        only_main: bool = True,
                        ) -> pd.DataFrame:
        """
        Return dataframe with metrics of models and show it on plot
        model_name: target name
        ds_manager_result: DSManagerResult object after DataSetsManager.get_results() method.
                            If not ds_manager_to_compare in init of class
        business_metric: User-defined metric to calculate. (from outboxml.metrics.metrics import BaseMetric)
        """

        if ds_manager_result is None and self.result_to_compare is None:
            raise ('No results to compare!')
        if ds_manager_result:
            result2 = ds_manager_result
        else:
            result2 = self.result_to_compare

        df = CompareModelsMetrics(result1=self.result,
                                  result2=result2,
                                  show=False
                                  ).compare_metrics(model_name=model_name,
                                                    business_metric=business_metric,
                                                    only_main=only_main)
        return df

    def compare_models_plot(self,
                            model_name: str,
                            features: list = None,
                            plot_type: int = 1,
                            bins_for_numerical_features: int = 5,
                            use_exposure: bool = True,
                            user_plot_func=None,
                            cut_min_value: float = 0.01,
                            cut_max_value: float = 0.9,
                            samples: float = 100,
                            cohort_base: str = 'model1',
                            ds_manager_result: dict = None,
                            plotly_params=None,
                            only_test: bool = True):
        """Return figures with y fact, yPredict for two models and selected features

             model_name: target for plotting
            plot_category:   0=metrics plot; 1=factors plot; 2=cohort plot; 3=relative models plot
            bins_for_numerical_features = number of cutting buns for numerical feature
            use_exposure: use exposure vector for calculating frequency and severity models and showing on plots
            user_plot_func: user-defined function in form of func(df1, df2, model_name, features_categorical, features_numerical, bins_for_numerical_features)
                            where df if full dataframe with features, 'y_prediction', 'y_true' and 'exposure'

            cut_min_value: quantile lower number for cutting for cohort plot
            cut_max_value: quantile max number for cutting for cohort plot
            samples: number of samples for grouping in cohort plot
            cohort_base: 'model1', 'model2' or 'fact' for default line in cohort plot
        """
        if ds_manager_result is None and self.result_to_compare is None:
            raise ('No results to compare!')
        if ds_manager_result:
            result2 = ds_manager_result
        else:
            result2 = self.result_to_compare[model_name]
        bins = bins_for_numerical_features
        if features is None:
            features, bins = self.__read_config_target_slices(model_name)
        if not features and plot_type != 2:
            raise 'No features to compare models!'
        df1 = self.df_for_graphs(result=self.result[model_name], features=features, use_exposure=use_exposure,
                                 only_test=only_test)
        df2 = self.df_for_graphs(result=result2, features=features, use_exposure=use_exposure, only_test=only_test)
        if user_plot_func:
            user_plot_func(df1, df2, model_name, features, bins)
        else:
            if plot_type == 0:
                figure = CompareModelsMetrics(result1=self.result,
                                              result2=result2,
                                              show=False
                                              ).compare_metrics(model_name=model_name)
            else:
                figure = CompareModelsPlot(model_name=model_name,
                                           df1=df1[0],
                                           df2=df2[0],
                                           features_categorical=df1[1],
                                           features_numerical=df1[2],
                                           show=False,
                                           bins=bins,
                                           plotly_params=plotly_params
                                           ).make(plot_type=plot_type,
                                                  cut_min_value=cut_min_value,
                                                  cut_max_value=cut_max_value,
                                                  samples=samples,
                                                  cohort_base=cohort_base)

            return figure

    def grafana_export(self, project_name: str = None, date_time=datetime.now()):
        """DataFrame with all metrics of model for grafana export.
         Use project_name for formating table (casco by default). Add date_time optionally"""
        df_for_grafana = pd.DataFrame()

        logger.info('Collecting data for Grafana')
        for key in self.result.keys():
            df_train = self.metrics_df(model_name=key, train_test='train').transpose().reset_index()
            df_train['TYPE'] = 'train'
            df_test = self.metrics_df(model_name=key, train_test='test').transpose().reset_index()
            df_test['TYPE'] = 'test'
            df = pd.concat([df_train, df_test])
            df['MODEL'] = key
            df_for_grafana = pd.concat([df_for_grafana, df], axis=0)

        df_for_grafana['CALCULATION_DATETIME'] = date_time
        df_for_grafana = df_for_grafana.rename(columns={'index': 'TARGET_SLICE'})
        return df_for_grafana
    @staticmethod
    def df_for_graphs(result: DSManagerResult, features: list = None, use_exposure: bool = False,
                      only_test: bool = True) -> \
            (pd.DataFrame, list, list):
        """
        method for constructing dataframe with results, and lists of features names (features, y_true and y_prediction, exposure)
        """
        y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(result=result,
                                                                                             features=features,
                                                                                             use_exposure=use_exposure,
                                                                                             only_test=only_test)
        return y_graph, features_categorical, features_numerical

    def metrics_df(self, model_name: str, metrics_dict: dict = None, train_test: str = 'train',
                  business_metric: BaseMetric = None) -> pd.DataFrame:
        """Method for constricting dataframe with metrics. By default usinf ds_manager result.
         Also you can you external metrics_dict.
         """
        logger.info('Preparing metrics for ' + model_name)
        if metrics_dict is None:
            metrics_dict = self.result[model_name].metrics[train_test]
        else:
            metrics_dict = metrics_dict[train_test]

        df1 = pd.DataFrame(metrics_dict).transpose()
        df = df1.fillna(0)
        df = df.transpose()
        return df

    def plots(self, model_name: str,
              features: list = None,
              plot_type: int = 1,
              bins_for_numerical_features: int = 5,
              use_exposure: bool = True,
              user_plot_func=None,
              cut_min_value: float = 0.01,
              cut_max_value: float = 0.9,
              samples: int = 100,
              cohort_base: str = 'model',
              only_test: bool = True,
              plotly_params: dict = None,
              ):
        """Plot results for chosen model and features
        model_name: target for plotting
        plot_type:  0= metrics plot;  1=factors plot; 2=cohort plot
        bins_for_numerical_features = number of cutting buns for numerical feature
        use_exposure: use exposure vector for calculating frequency and severity models and showing on plots
        user_plot_func: user-defined function in form of func(df, model_name, features, bins_for_numerical_features)
                        where df if full dataframe with features, 'y_prediction', 'y_true' and 'exposure'

        cut_min_value: quantile lower number for cutting for cohort plot
        cut_max_value: quantile max number for cutting for cohort plot
        samples: number of samples for grouping in cohort plot
        cohort_base: 'model' or 'fact' for default line in cohort plot
        """
        bins = bins_for_numerical_features
        if bins_for_numerical_features is not None and features is not None:
            bins = bins_for_numerical_features
        if features is None:
            features, bins = self.__read_config_target_slices(model_name)
            if features == [] and plot_type == 1:
                logger.error('No features for plots')
                return

        y_graph, features_categorical, features_numerical = self.df_for_graphs(result=self.result[model_name],
                                                                               features=features,
                                                                               use_exposure=use_exposure,
                                                                               only_test=only_test)
        if user_plot_func is not None:
            user_plot_func(y_graph, model_name, features, bins)
        else:
            figure = MLPlot(model_name_1=model_name,
                            y_graph=y_graph,
                            features_categorical=features_categorical,
                            features_numerical=features_numerical,
                            show=False,
                            bins=bins,
                            use_exposure=use_exposure,
                            plotly_params=plotly_params).make(plot_type=plot_type,
                                                              cut_min_value=cut_min_value,
                                                              cut_max_value=cut_max_value,
                                                              samples=samples,
                                                              cohort_base=cohort_base,
                                                              )

            return figure

    def __save_to_pickle(self, path_to_save, model, model_name: str):
        model_path = os.path.join(path_to_save, f"{model_name}.pickle")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            logger.info(model_name + ' to pickle')

    def __save_metrics_to_excel(self, metrics: dict, model_name: str, key: str, path, to_mlflow: bool = False):
        df = {}
        if metrics is None:
            logger.info('No metrics for model||' + model_name)
        else:
            with pd.ExcelWriter(str(path) + '/' f"{model_name}_metrics_{key}.xlsx") as writer:
                df['Full results'] = pd.DataFrame(metrics).to_excel(writer, sheet_name='Full results')
                for key2 in metrics.keys():
                    df[key2] = pd.DataFrame([metrics[key2][model_name + '_model_1']])
                    sheet_name = key2.replace("]", ")")
                    df[key2].to_excel(writer, sheet_name=sheet_name)
                    writer._save()

    def __save_predictions(self, y_true, prediction: pd.DataFrame, model_name: str, key: str, path=None,
                           x: pd.DataFrame = None):

        df = pd.concat([x, y_true, prediction], axis=1)
        return df

    def __read_config_target_slices(self, model_name: str) -> tuple:
        try:
            features = []
            features_list = self.result[model_name].config.data_config.data.targetslices
            for item in features_list:
                features.append(item['column'])
                bins = item['slices']
            logger.info('Reading feature from config||' + str(features) + ' Bins:' + str(bins))
            return features, bins
        except:
            logger.info('No features for plots in config||User features')
            return [], []


class GrafanaExport:
    """
    Class for export data to grafana db (Motor.grafana.table_name)

    Parameters:
    -----------
        df: data to export
        table_name: table_name to replace data
        connection: user connection engine. By default sqlalchemy MSSQL engine
    ----------
    Methods:
        load_data_to_db: loading dataframe to grafana source
    """

    def __init__(self,
                 df: pd.DataFrame,
                 table_name: str = 'FrameworkTest',
                 schema: str = 'public',
                 connection=None):
        self.df = df
        if self.df.empty:
            raise logger.error('Empty dataframe to load')
        self.table_name = table_name
        self.schema = schema
        self.__connection = None
        logger.debug('Connecting to  db..')
        if connection is not None:
            self.__connection = connection
        else:
            self.__connection = create_engine(env_config.connection_params)
        logger.debug('Connection completed')

    def load_data_to_db(self):
        """"Method for loading data to grafana database using params from .env config"""
        logger.debug('Loading data to db..')
        self.df.to_sql(self.table_name, schema=self.schema, con=self.__connection, if_exists='append', index=False)
        logger.debug('Loading finished')
