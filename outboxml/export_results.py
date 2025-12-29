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
    Main module for exporting and processing calculation and model results, including metrics and plots.
    Can be used for model comparison and retrospective analysis. 
    
    :param ds_manager: DataSetsManager object with results (after calling fit_models() method)
    :type ds_manager: DataSetsManager
    :param ds_manager_to_compare: Another DataSetsManager object with results for comparison
    :type ds_manager_to_compare: DataSetsManager, optional
    :param config: External config file with passwords, logins, paths to save and folders.
                   Uses .env by default
    :type config: .py, optional
    """

    def __init__(self,
                 ds_manager: DataSetsManager,
                 ds_manager_to_compare: DataSetsManager = None,
                 config=None):
        """
        Initialize ResultExport class.

        :param ds_manager: DataSetsManager object with results (after calling fit_models() method)
        :type ds_manager: DataSetsManager
        :param ds_manager_to_compare: Another DataSetsManager object with results for comparison
        :type ds_manager_to_compare: DataSetsManager, optional
        :param config: External config file with passwords, logins, paths to save and folders.
                       Uses .env by default
        :type config: .py, optional
        """
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
        """
        Prepare results for processing.

        :param result: DSManagerResult object to prepare
        :type result: DSManagerResult
        :return: Prepared result
        :rtype: DSManagerResult
        """
        return result

    def save(self, to_pickle: bool = False,
             path_to_save: Path = None,
             to_mlflow: bool = False,
             save_ds_manager: bool = False,
             ds_manager_name: str = 'ds_manager'):
        """
        Save results for production.

        Saves metrics in Excel, predictions in Parquet, models in pickle, and configs in JSON.

        :param to_pickle: Whether to save models in pickle format, defaults to False
        :type to_pickle: bool, optional
        :param path_to_save: Path for saving results, defaults to None (uses config path)
        :type path_to_save: Path, optional
        :param to_mlflow: Whether to save artifacts to MLflow, defaults to False
        :type to_mlflow: bool, optional
        :param save_ds_manager: Whether to save ds_manager object, defaults to False
        :type save_ds_manager: bool, optional
        :param ds_manager_name: Name for ds_manager file, defaults to 'ds_manager'
        :type ds_manager_name: str, optional

        :raises FileNotFoundError: If saving path does not exist
        :raises PermissionError: If no write permissions to the path
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
        Return dataframe with metrics of models and show it on plot.

        :param model_name: Model name from models_configs
        :type model_name: str
        :param ds_manager_result: Dict with results after DataSetsManager.get_results() method.
                                  If not provided, uses ds_manager_to_compare from initialization
        :type ds_manager_result: dict, optional
        :param business_metric: User-defined metric to calculate
        :type business_metric: BaseMetric, optional
        :param only_main: Whether to show only main metrics, defaults to True
        :type only_main: bool, optional

        :return: DataFrame with comparison metrics
        :rtype: pd.DataFrame

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
        """
        Return figures with y True and y Predictions for two models and selected features.

        :param model_name: Model name from models_configs
        :type model_name: str
        :param features: List of features to analyze, defaults to None
        :type features: list, optional
        :param plot_type: Plot type: 0=metrics plot; 1=factors plot; 2=cohort plot; 3=relative models plot,
                         defaults to 1
        :type plot_type: int, optional
        :param bins_for_numerical_features: Number of bins for numerical features, defaults to 5
        :type bins_for_numerical_features: int, optional
        :param use_exposure: Use exposure vector from column_exposure, defaults to True
        :type use_exposure: bool, optional
        :param user_plot_func: User-defined plotting function, defaults to None
        :type user_plot_func: callable, optional
        :param cut_min_value: Lower quantile for cutting in cohort plot, defaults to 0.01
        :type cut_min_value: float, optional
        :param cut_max_value: Upper quantile for cutting in cohort plot, defaults to 0.9
        :type cut_max_value: float, optional
        :param samples: Number of samples for grouping in cohort plot, defaults to 100
        :type samples: float, optional
        :param cohort_base: Base line for cohort plot ('model1', 'model2' or 'fact'), defaults to 'model1'
        :type cohort_base: str, optional
        :param ds_manager_result: DataSetsManager.get_results() dict for comparison, defaults to None
        :type ds_manager_result: dict, optional
        :param plotly_params: Parameters for Plotly visualization, defaults to None
        :type plotly_params: dict, optional
        :param only_test: Use only test data, defaults to True
        :type only_test: bool, optional

        :return: Plotly figure object
        :rtype: plotly.graph_objects.Figure

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
        """
        Create DataFrame with all model metrics for Grafana export.

        :param project_name: Project name for formatting table, defaults to None
        :type project_name: str, optional
        :param date_time: Date and time for the export, defaults to current datetime
        :type date_time: datetime, optional

        :return: DataFrame formatted for Grafana
        :rtype: pd.DataFrame
        """
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
                      only_test: bool = True) -> tuple:
        """
        Construct dataframe with results and lists of feature names.

        :param result: DSManagerResult object
        :type result: DSManagerResult
        :param features: List of features to include, defaults to None
        :type features: list, optional
        :param use_exposure: Use exposure vector, defaults to False
        :type use_exposure: bool, optional
        :param only_test: Use only test data, defaults to True
        :type only_test: bool, optional

        :return: Tuple containing (dataframe, categorical_features, numerical_features)
        :rtype: tuple[pd.DataFrame, list, list]
        """
        y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(result=result,
                                                                                             features=features,
                                                                                             use_exposure=use_exposure,
                                                                                             only_test=only_test)
        return y_graph, features_categorical, features_numerical

    def metrics_df(self, model_name: str, metrics_dict: dict = None, train_test: str = 'train',
                  business_metric: BaseMetric = None) -> pd.DataFrame:
        """
        Construct dataframe with metrics.

        :param model_name: Model name for metrics extraction
        :type model_name: str
        :param metrics_dict: External metrics dictionary, defaults to None
        :type metrics_dict: dict, optional
        :param train_test: Test or train data, defaults to 'train'
        :type train_test: str, optional
        :param business_metric: User-defined business metric, defaults to None
        :type business_metric: BaseMetric, optional

        :return: DataFrame with metrics
        :rtype: pd.DataFrame
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
              plotly_params: dict = None):
        """
        Plot results for chosen model and features.

        :param model_name: Model name from models_configs
        :type model_name: str
        :param features: List of features to analyze, defaults to None
        :type features: list, optional
        :param plot_type: Plot type: 0=metrics plot; 1=factors plot; 2=cohort plot, defaults to 1
        :type plot_type: int, optional
        :param bins_for_numerical_features: Number of bins for numerical features, defaults to 5
        :type bins_for_numerical_features: int, optional
        :param use_exposure: Use exposure vector from column_exposure, defaults to True
        :type use_exposure: bool, optional
        :param user_plot_func: User-defined plotting function, defaults to None
        :type user_plot_func: callable, optional
        :param cut_min_value: Lower quantile for cutting in cohort plot, defaults to 0.01
        :type cut_min_value: float, optional
        :param cut_max_value: Upper quantile for cutting in cohort plot, defaults to 0.9
        :type cut_max_value: float, optional
        :param samples: Number of samples for grouping in cohort plot, defaults to 100
        :type samples: int, optional
        :param cohort_base: Base line for cohort plot ('model' or 'fact'), defaults to 'model'
        :type cohort_base: str, optional
        :param only_test: Use only test data, defaults to True
        :type only_test: bool, optional
        :param plotly_params: Parameters for Plotly visualization, defaults to None
        :type plotly_params: dict, optional

        :return: Plotly figure object
        :rtype: plotly.graph_objects.Figure
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
        """
        Save model to pickle file.

        :param path_to_save: Path to save the model
        :type path_to_save: str
        :param model: Model object to save
        :type model: object
        :param model_name: Name of the model
        :type model_name: str
        """
        model_path = os.path.join(path_to_save, f"{model_name}.pickle")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            logger.info(model_name + ' to pickle')

    def __save_metrics_to_excel(self, metrics: dict, model_name: str, key: str, path, to_mlflow: bool = False):
        """
        Save metrics to Excel file.

        :param metrics: Metrics dictionary
        :type metrics: dict
        :param model_name: Model name
        :type model_name: str
        :param key: Key identifier
        :type key: str
        :param path: Path to save the file
        :type path: str
        :param to_mlflow: Whether to save to MLflow, defaults to False
        :type to_mlflow: bool, optional
        """
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
        """
        Save predictions to dataframe.

        :param y_true: True values
        :type y_true: pd.Series
        :param prediction: Predictions
        :type prediction: pd.DataFrame
        :param model_name: Model name
        :type model_name: str
        :param key: Key identifier
        :type key: str
        :param path: Path to save, defaults to None
        :type path: str, optional
        :param x: Features, defaults to None
        :type x: pd.DataFrame, optional

        :return: Combined dataframe with features, true values, and predictions
        :rtype: pd.DataFrame
        """
        df = pd.concat([x, y_true, prediction], axis=1)
        return df

    def __read_config_target_slices(self, model_name: str) -> tuple:
        """
        Read targetslices features from model config.

        :param model_name: Model name to get configuration for
        :type model_name: str

        :return: Tuple containing (features list, bins)
        :rtype: tuple[list, int]
        """
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
    Class for exporting data to Grafana database.

    This class facilitates exporting pandas DataFrames to a specified table
    in a Grafana-compatible database

    :param df: DataFrame containing data to export
    :type df: pd.DataFrame
    :param table_name: Name of the database table to replace/append data, defaults to 'FrameworkTest'
    :type table_name: str, optional
    :param schema: Database schema name, defaults to 'public'
    :type schema: str, optional
    :param connection: Custom SQLAlchemy engine connection, defaults to None (uses env config)
    :type connection: sqlalchemy.engine.Engine, optional

    :raises ValueError: If the input DataFrame is empty
    :raises Exception: If database connection fails
    """

    def __init__(self,
                 df: pd.DataFrame,
                 table_name: str = 'FrameworkTest',
                 schema: str = 'public',  # TODO: Consider making None
                 connection=None):
        """
        Initialize GrafanaExport instance.

        :param df: DataFrame containing data to export
        :type df: pd.DataFrame
        :param table_name: Name of the database table to replace/append data, defaults to 'FrameworkTest'
        :type table_name: str, optional
        :param schema: Database schema name, defaults to 'public'
        :type schema: str, optional
        :param connection: Custom SQLAlchemy engine connection, defaults to None (uses env config)
        :type connection: sqlalchemy.engine.Engine, optional

        :raises ValueError: If the input DataFrame is empty
        :raises Exception: If database connection fails
        """
        self.df = df
        if self.df.empty:
            raise ValueError('Empty dataframe to load')
        self.table_name = table_name
        self.schema = schema
        self.__connection = None
        
        logger.debug('Connecting to db..')
        if connection is not None:
            self.__connection = connection
        else:
            self.__connection = create_engine(env_config.connection_params)
        logger.debug('Connection completed')

    def load_data_to_db(self):
        """
        Load data to Grafana database using parameters from external config.

        This method appends the DataFrame data to the specified table in the database.
        If the table doesn't exist, it will be created automatically.

        .. warning::
            Uses `if_exists='append'` which will add data to existing tables without
            clearing previous data. Consider table size implications.

        :return: None
        :rtype: None

        :raises sqlalchemy.exc.SQLAlchemyError: If database operation fails
        :raises ValueError: If DataFrame contains unsupported data types
        :raises Exception: For other unexpected errors during database operations


        Example:
            >>> df = pd.DataFrame({'metric': [1, 2, 3], 'value': [10, 20, 30]})
            >>> exporter = GrafanaExport(df, table_name='metrics')
            >>> exporter.load_data_to_db()
            Data loaded successfully
        """
        logger.debug('Loading data to db..')
        self.df.to_sql(self.table_name, 
                      schema=self.schema, 
                      con=self.__connection, 
                      if_exists='append', 
                      index=False)
        logger.debug('Loading finished')
