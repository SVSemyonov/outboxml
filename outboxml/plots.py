from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from outboxml.core.enums import ResultNames
from outboxml.metrics.base_metrics import BaseMetrics
from outboxml.datasets_manager import DSManagerResult
from loguru import logger
from matplotlib.pyplot import show


class DataframeForPlots:
    """
    Class for preparing dataframes for visualization plots.
    
    This class provides methods to construct and format dataframes
    from model results for plotting purposes, handling both categorical
    and numerical features with optional exposure adjustments.
    """

    def __init__(self):
        """
        Initialize the DataframeForPlots instance.
        """
        pass

    def df_for_plots(self, result: DSManagerResult, features: list = None, 
                     use_exposure: bool = False, only_test: bool = False) -> (pd.DataFrame, list, list):
        """
        Construct a dataframe with results for plotting.
        
        This method creates a consolidated dataframe containing features,
        true values, predictions, and optionally exposure values. It also
        returns lists of categorical and numerical features.
        
        :param result: DSManagerResult object containing model predictions and data subsets
        :type result: DSManagerResult
        :param features: Specific features to include in the output dataframe.
                         If None, all features are included. Default is None.
        :type features: list, optional
        :param use_exposure: Whether to include and apply exposure to predictions.
                            Default is False.
        :type use_exposure: bool, optional
        :param only_test: Whether to include only test data or both test and train data.
                         Default is False (includes both).
        :type only_test: bool, optional
        
        :return: A tuple containing three elements:
                 - pd.DataFrame: Consolidated dataframe with features, true values,
                   predictions, and optionally exposure
                 - list: List of categorical feature names
                 - list: List of numerical feature names
        :rtype: tuple
        
        :note: If no categorical/numerical features exist, the corresponding lists
               will contain ['No features']
        :note: When `use_exposure` is True, predictions are multiplied by exposure values
        :note: Missing values in y_true and y_prediction are filled with 0
        """
        data = deepcopy(result)
        features_categorical = data.data_subset.features_categorical
        if features_categorical == []: 
            features_categorical = ['No features']
        features_numerical = data.data_subset.features_numerical
        if features_numerical == []: 
            features_numerical = ['No features']
        exposure_test = data.data_subset.exposure_test
        exposure_train = data.data_subset.exposure_train
        logger.info('Collecting data for plots')
        
        if features is not None:
            features_categorical = [a for a in features if a in features_categorical]
            features_numerical = [a for a in features if a in features_numerical]
            
        if not only_test:
            X = pd.concat([data.data_subset.X_test, data.data_subset.X_train])
            y_true = pd.concat([data.data_subset.y_test, data.data_subset.y_train])
            y_pred = pd.concat([data.predictions['test'], data.predictions['train']])
        else:
            X = data.data_subset.X_test
            y_true = data.data_subset.y_test
            y_pred = data.predictions['test']
            
        y_true.name = 'y_true'
        y_pred.name = 'y_prediction'
        y_graph = pd.concat([X, y_true.fillna(0), y_pred.fillna(0)], axis=1)
        
        if exposure_test is not None and exposure_train is not None and use_exposure:
            exposure = pd.concat([exposure_test, exposure_train])
            exposure.name = 'exposure'
            y_graph = pd.concat([y_graph, exposure], axis=1)
            y_graph['y_prediction'] = y_graph['y_prediction'] * y_graph['exposure']
            
        return y_graph, features_categorical, features_numerical

class MLPlot:
    """
    A class for creating various machine learning visualization plots.

    This class provides functionality to generate different types of plots for model evaluation,
    including cohort analysis, feature importance, and model performance metrics. The plots are
    created using Plotly for interactive visualization.
    """

    def __init__(self,
                 model_name_1: str,
                 y_graph: pd.DataFrame,
                 features_numerical: list,
                 features_categorical: list,
                 model_name_2: str = None,
                 bins: int = None,
                 show: bool = True,
                 use_exposure: bool = True,
                 plotly_params: dict = None
                 ):
        """
        Initialize the MLPlot instance.

        :param model_name_1: Name of the main model for plotting
        :type model_name_1: str
        :param y_graph: DataFrame containing features, true values, and predictions
        :type y_graph: pd.DataFrame
        :param features_numerical: List of numerical feature names
        :type features_numerical: list
        :param features_categorical: List of categorical feature names
        :type features_categorical: list
        :param model_name_2: Name of the secondary model for comparison (optional)
        :type model_name_2: str, optional
        :param bins: Number of bins for numerical feature discretization (optional)
        :type bins: int, optional
        :param show: Whether to immediately display plots when created
        :type show: bool, optional
        :param use_exposure: Whether to use exposure for calculations
        :type use_exposure: bool, optional
        :param plotly_params: Custom parameters for Plotly figures (optional)
        :type plotly_params: dict, optional

        :ivar plotly_params: Parameters to customize Plotly figures
        :ivar comparative_mode: Flag indicating if comparison between two models is enabled
        :ivar show: Whether to immediately display the plots
        """
        self.plotly_params = plotly_params
        self._model1_name = model_name_1
        self._model2_name = model_name_2
        self._features_categorical = features_categorical
        self._features_numerical = features_numerical
        self.comparative_mode = False
        self._filter_features = False
        self._y_graph = y_graph.copy()
        self.show = show
        if self._model2_name is not None:
            self.comparative_mode = True
        self._bins = bins
        self._use_exposure = use_exposure

    def make(self, plot_type, cut_min_value, cut_max_value, samples, cohort_base):
        """
        Main method to create specified plot type.

        :param plot_type: Type of plot to create:
                          - 0: Metrics plot
                          - 1: Feature plot
                          - 2: Cohort plot
        :type plot_type: int
        :param cut_min_value: Minimum quantile value for clipping in cohort plot
        :type cut_min_value: float
        :param cut_max_value: Maximum quantile value for clipping in cohort plot
        :type cut_max_value: float
        :param samples: Sampling parameter for cohort plot discretization
        :type samples: float
        :param cohort_base: Base for cohort analysis ('model' or 'fact')
        :type cohort_base: str

        :return: Generated Plotly figure
        :rtype: plotly.graph_objs.Figure

        """
        if plot_type == 1:
            logger.info('Factors plot for ' + self._model1_name)
            figure = self.feature_plot(self._use_exposure)
        elif plot_type == 0:
            logger.info('Metrics for ' + self._model1_name)
            figure = MetricsPlot(self.show).make_plots(self._y_graph, self._model1_name, self._features_categorical,
                                                       self._features_numerical, self._bins)
        elif plot_type == 2:
            logger.info('Cohort plot ' + self._model1_name)
            figure = self.cohort_plot(cut_min_value, cut_max_value, samples, cohort_base)
        else:
            raise 'Unknown plot type!'
        return figure

    def cohort_plot(self, cut_min_value: float, cut_max_value: float, samples=100.0, cohort_base: str = 'model'):
        """
        Creates a cohort analysis plot comparing predictions vs actuals.

        :param cut_min_value: Minimum quantile value for clipping predictions
        :type cut_min_value: float
        :param cut_max_value: Maximum quantile value for clipping predictions
        :type cut_max_value: float
        :param samples: Sampling parameter for discretization (default: 100.0)
        :type samples: float, optional
        :param cohort_base: Base for cohort analysis:
                           - 'model': Group by model predictions, compare with actuals
                           - 'fact': Group by actual values, compare with predictions
        :type cohort_base: str, optional

        :return: Cohort analysis plot
        :rtype: plotly.graph_objs.Figure

        :note: If exposure column is not found, a warning is logged and exposure=1 is used
        :note: The method automatically adjusts sampling if number of unique groups is too high or too low
        """
        samples = samples
        df = self._y_graph.copy()
        if 'exposure' not in df.columns:
            df['exposure'] = 1
            logger.warning('No exposure for cohort plot||Exposure=1 vector added')
        df["predict"] = df['y_prediction'] / df['exposure']
        if cohort_base == 'model':
            column_to_group = "predict"
            column_for_target_count = 'y_true'
            names = ['fact', 'model']
        elif cohort_base == 'fact':
            names = ['model', 'fact']
            column_to_group = "y_true"
            column_for_target_count = 'y_prediction'
        else:
            logger.error('Unknown cohort base!||Chosen Model type ')
            column_to_group = "predict"
            column_for_target_count = 'y_true'
            names = ['fact', 'model']
        df["predict_gr"] = round(df[column_to_group] * samples) / samples
        df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                              upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
        if len(df["predict_gr"].unique()) > 20 and samples == 100.0:
            for i in range(8):
                if len(df["predict_gr"].unique()) < 20:
                    break
                samples = samples / 5
                df["predict_gr"] = round(df[column_to_group] * samples) / samples
                df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                      upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
        elif len(df["predict_gr"].unique()) < 5 and cut_max_value == 0.9:
            for i in range(8):
                if len(df["predict_gr"].unique()) >= 5:
                    break
                cut_max_value = cut_max_value * 1.1
                if cut_max_value > 1: cut_max_value = 1
                samples = samples * 2
                df["predict_gr"] = round(df[column_to_group] * samples) / samples
                df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                      upper=df["predict_gr"].quantile(cut_max_value), inplace=True)

        y_test_freq_gr = (
            df
            [[column_for_target_count, 'exposure', "predict_gr"]]
            .groupby("predict_gr")
            .sum()
            .reset_index()
        )
        y_test_freq_gr["predict_gr_"] = y_test_freq_gr["predict_gr"]
        y_test_freq_gr['target'] = y_test_freq_gr[column_for_target_count] / y_test_freq_gr['exposure']
        cohort_plot = PlotlyWrapper(model_name=self._model1_name, plotly_params=self.plotly_params)
        cohort_plot.add_figure(x=y_test_freq_gr["predict_gr"].values,
                               y=y_test_freq_gr["target"].values, name=names[0])
        cohort_plot.add_figure(name=names[1],
                               x=y_test_freq_gr["predict_gr"].values,
                               y=y_test_freq_gr["predict_gr_"].values, )
        cohort_plot.add_figure(name="exposure",
                               x=y_test_freq_gr["predict_gr"].values,
                               y=y_test_freq_gr['exposure'].values, type='bar')
        figure = cohort_plot.prepare_plot()
        if self.show:
            figure.show()
        return figure

    def feature_plot(self, use_exposure: bool = True):
        """
        Creates feature analysis plots showing model predictions vs actuals by feature.

        :param use_exposure: Whether to use exposure for calculations
        :type use_exposure: bool, optional

        :return: Last generated feature plot
        :rtype: plotly.graph_objs.Figure

        :note: For categorical features, groups are created by unique values
        :note: For numerical features, bins are created based on the `bins` parameter
        :note: If exposure column is missing and use_exposure=True, exposure=1 is used with error logging
        """
        X_test = self._y_graph
        column_exposure = 'exposure'
        if not use_exposure: X_test[column_exposure] = 1
        try:
            X_test[column_exposure]
        except:
            X_test[column_exposure] = 1
            logger.error('No exposure in model||Added exposure = 1')
        column_claims_count = 'y_true'
        bins_test_fact = {}
        bins_test_pred = {}
        features = self._features_numerical + self._features_categorical
        for feature in features:
            if feature in self._features_categorical:
                feature_vals = X_test[feature]
                bins_test_fact[feature] = X_test[[column_claims_count, column_exposure]].groupby(feature_vals,
                                                                                                 dropna=False,
                                                                                                 observed=False).sum()
                bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                                  bins_test_fact[feature][column_exposure]
                bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(feature_vals, dropna=False,
                                                                                            observed=False).sum()
                bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                    column_exposure]
            else:
                feature_vals = X_test[feature]
                n_bins = self._bins
                if feature_vals.nunique() > n_bins and self._bins is not None:
                    breakpoints = np.arange(0, n_bins + 1) / (n_bins) * 100
                    breakpoints = [np.percentile(X_test[feature], bp) for bp in breakpoints]
                    breakpoints[0] = breakpoints[0] - 0.1
                else:
                    breakpoints = self._bins

                bins_test_fact[feature] = X_test[[column_claims_count, column_exposure]].groupby(
                    pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                    observed=False).sum()
                bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                                  bins_test_fact[feature][column_exposure]
                bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(
                    pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                    observed=False).sum()
                bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                    column_exposure]
            fig = PlotlyWrapper(model_name=self._model1_name, plotly_params=self.plotly_params)
            fig.add_figure(type='bar', name="exposure",
                           x=bins_test_fact[feature].index.astype(str),
                           y=bins_test_fact[feature][column_exposure].values)
            fig.add_figure(name="fact",
                           x=bins_test_fact[feature].index.astype(str),
                           y=bins_test_fact[feature]["freq"].values)
            fig.add_figure(
                name="model",
                x=bins_test_pred[feature].index.astype(str),
                y=bins_test_pred[feature]["freq"].values)

            figure = fig.prepare_plot(x_name='Target groups for ' + feature)
            if self.show:
                figure.show()

        return figure


class FactorsPlot:
    """Class for creating factor analysis plots."""
    
    def __init__(self,
                 show: bool = True
                 ):
        """
        Initialize the FactorsPlot instance.
        
        :param show: Whether to immediately display plots when created
        :type show: bool, optional
        """
        self.show = show

    def make_plots(self, y_graph, model_name: str, features_categorical, features_numerical, bins: int = 5,
                   use_exposure: bool = False):
        """
        Create factor analysis plots for categorical and numerical features.
        
        This method generates bar plots showing the relationship between features
        and model predictions vs actual values, grouped by feature values.
        
        :param y_graph: DataFrame containing features, predictions, and actual values
        :type y_graph: pd.DataFrame
        :param model_name: Name of the model for plot titles
        :type model_name: str
        :param features_categorical: List of categorical feature names to plot
        :type features_categorical: list
        :param features_numerical: List of numerical feature names to plot
        :type features_numerical: list
        :param bins: Number of bins for discretizing numerical features, defaults to 5
        :type bins: int, optional
        :param use_exposure: Whether to include exposure in calculations, defaults to False
        :type use_exposure: bool, optional
        
        :return: DataFrame containing features, predictions, and actual values
        :rtype: pd.DataFrame
        
        :note: For categorical features, plots show sums grouped by unique feature values
        :note: For numerical features, plots show sums grouped into bins
        """
        y_graph = y_graph.copy()
        
        for feature in features_categorical:
            if use_exposure:
                plot_columns = [feature, 'exposure', 'y_prediction', 'y_true']
            else:
                plot_columns = [feature, 'y_prediction', 'y_true']
            y = y_graph[plot_columns].groupby(feature, observed=False).sum().sort_values(by=feature)
            y.plot(kind='bar', title=model_name, xlabel=feature)

        for feature in features_numerical:
            if use_exposure:
                plot_columns = [feature, 'y_prediction', 'y_true']
            else:
                plot_columns = [feature, 'y_prediction', 'y_true']
            y = y_graph[plot_columns].groupby(pd.cut(y_graph[feature], bins=bins)).sum()
            y.plot(kind='bar', title=model_name, xlabel=feature)

        if self.show:
            show()
        return y_graph


class MetricsPlot:
    """Class for creating metric plots for model evaluation."""
    
    def __init__(self,
                 show: bool = True
                 ):
        """
        Initialize the MetricsPlot instance.
        
        :param show: Whether to immediately display plots when created
        :type show: bool, optional
        """
        self.show = show

    def make_plots(self, y_graph, model_name: str, features_categorical, features_numerical, bins: int = 5):
        """
        Create metric plots for the model across different feature slices.
        
        This method calculates and plots various metrics (MAE, MSE, RMSE, etc.) 
        grouped by feature values or bins, allowing for analysis of model performance
        across different segments of the data.
        
        :param y_graph: DataFrame containing features, predictions, and actual values
        :type y_graph: pd.DataFrame
        :param model_name: Model name
        :type model_name: str
        :param features_categorical: List of categorical feature names to analyze
        :type features_categorical: list
        :param features_numerical: List of numerical feature names to analyze
        :type features_numerical: list
        :param bins: Number of bins for discretizing numerical features, defaults to 5
        :type bins: int, optional
        
        :return: DataFrame containing calculated metrics for all feature slices
        :rtype: pd.DataFrame
        
        :note: For categorical features, metrics are calculated per unique value
        :note: For numerical features, metrics are calculated per bin
        :note: Metrics are calculated using BaseMetrics class with optional exposure

        """
        metrics_res = pd.DataFrame()
        features = features_numerical + features_categorical

        for feature in features:
            if 'exposure' in y_graph.columns:
                plot_columns = [feature, 'exposure', 'y_prediction', 'y_true']
            else:
                plot_columns = [feature, 'y_prediction', 'y_true']
            
            if feature in features_categorical:
                y = y_graph[plot_columns].groupby(feature, observed=False)
            else:
                y = y_graph[plot_columns].groupby(pd.cut(y_graph[feature], bins=bins), observed=False)
            
            metrics = {}
            for group, name in y:
                if 'exposure' not in plot_columns:
                    exposure = None
                else:
                    exposure = name['exposure']
                metrics[group] = BaseMetrics(name['y_true'], name['y_prediction'], exposure).calculate_metric()
            
            metrics_df = pd.DataFrame(metrics).transpose()
            metrics_df.index.name = str(feature)
            metrics_df.plot(title=model_name, xlabel=feature)
            
            metrics_res = pd.concat([metrics_res, metrics_df])
            metrics_res = pd.concat([metrics_res, metrics_df])
        
        if self.show:
            show()
        return metrics_res


class CompareModelsMetrics:
    """Class for comparing metrics between two different models."""
    
    def __init__(self,
                 result1: dict,
                 result2: dict,
                 show: bool = True,
                 plot_columns=[]
                 ):
        """
        Initialize the CompareModelsMetrics instance.
        
        :param result1: Export results dictionary containing the primary model results
        :type result1: dict
        :param result2: Export results dictionary containing the secondary model results for comparison
        :type result2: dict
        :param show: Whether to immediately display plots when created, defaults to True
        :type show: bool, optional
        :param plot_columns: Specific columns to include in plots, defaults to empty list
        :type plot_columns: list, optional
        """
        self.result = result1
        self.result_to_compare = result2
        self.show = show
        self.plot_columns = plot_columns

    def compare_metrics(self, model_name: str, business_metric=None, only_main: bool = False):
        """
        Compare metrics between two models for a specific model configuration.
        
        This method extracts and compares training and testing metrics from two different
        model results, aggregating them into a structured DataFrame for analysis.
        
        :param model_name: Name of the specific model which metrics going to be compared
        :type model_name: str
        :param business_metric: Optional business metric calculator for additional comparisons
        :type business_metric: object, optional
        :param only_main: If True, only compares the 'full' metric group; if False, compares all metric groups
        :type only_main: bool, optional
        
        :return: DataFrame containing compared metrics for both models across train/test splits
        :rtype: pd.DataFrame
        
        :note: The method handles cases where secondary model might not have all metric groups
        :note: When a metric group is missing in the secondary model, 'full' metrics are used as fallback
        :note: Results are labeled using ResultNames constants for clear identification
        :note: The output DataFrame can be used for visualization or further analysis
        
        """
        res_dict_main = self.result[model_name]
        res_dict_to_compare = self.result_to_compare[model_name]
        metrics1 = {}
        metrics2 = {}
        
        for key in res_dict_main.metrics['train'].keys():
            if only_main and key != 'full': 
                continue

            metrics1[key] = {}
            metrics2[key] = {}

            metrics1[key]['first model train'] = res_dict_main.metrics['train'][key]
            metrics1[key]['first model test'] = res_dict_main.metrics['test'][key]
            
            try:
                metrics2[key]['second model train'] = res_dict_to_compare.metrics['train'][key]
                metrics2[key]['second model test'] = res_dict_to_compare.metrics['test'][key]
            except KeyError:
                logger.info('No key in metrics||Calculating for full')
                metrics2[key]['second model train'] = res_dict_to_compare.metrics['train']['full']
                metrics2[key]['second model test'] = res_dict_to_compare.metrics['test']['full']

            df1 = pd.DataFrame(metrics1).transpose()['first model train'].apply(pd.Series).reset_index()
            df1['Model'] = ResultNames.new_result_train
            df2 = pd.DataFrame(metrics1).transpose()['first model test'].apply(pd.Series).reset_index()
            df2['Model'] = ResultNames.new_result_test
            df3 = pd.DataFrame(metrics2).transpose()['second model train'].apply(pd.Series).reset_index()
            df3['Model'] = ResultNames.old_result_train
            df4 = pd.DataFrame(metrics2).transpose()['second model test'].apply(pd.Series).reset_index()
            df4['Model'] = ResultNames.old_result_test

        df = pd.concat([df1, df2, df3, df4]).reset_index()
        df = df.rename(columns={'index': 'Metric group'})
        df = df.sort_values(by=['Metric group'])
        df = df.fillna(0)

        for column in df.columns:
            if column in ['Metric group', 'Model']:
                continue
            df[['Metric group', 'Model', column]].set_index('Metric group').groupby('Model')[column]

        if business_metric is not None:
            business_value = business_metric.calculate_metric(self.result, self.result_to_compare)
            logger.info('Business metric||' + str(business_value))
            for key in business_value.keys():
                df[key] = business_value[key]

        return df


class CompareModelsPlot:
    """Main class for creating comparison plots between two models."""
    
    def __init__(self,
                 model_name: str,
                 df1: pd.DataFrame,
                 df2: pd.DataFrame,
                 features_categorical: list,
                 features_numerical: list,
                 name1: str = 'Model 1',
                 name2: str = 'Model 2',
                 show: bool = True,
                 bins: int = 5,
                 plotly_params=None):
        """
        Initialize the CompareModelsPlot instance.
        
        :param model_name: Name of the model for plot titles
        :type model_name: str
        :param df1: DataFrame containing data for the first model
        :type df1: pd.DataFrame
        :param df2: DataFrame containing data for the second model
        :type df2: pd.DataFrame
        :param features_categorical: List of categorical feature names
        :type features_categorical: list
        :param features_numerical: List of numerical feature names
        :type features_numerical: list
        :param name1: Display name for the first model, defaults to 'Model 1'
        :type name1: str, optional
        :param name2: Display name for the second model, defaults to 'Model 2'
        :type name2: str, optional
        :param show: Whether to immediately display plots when created, defaults to True
        :type show: bool, optional
        :param bins: Number of bins for numerical feature discretization, defaults to 5
        :type bins: int, optional
        :param plotly_params: Custom parameters for Plotly figures, defaults to None
        :type plotly_params: dict, optional
        """
        self.model_name = model_name
        self._features_numerical = features_numerical
        self._features_categorical = features_categorical
        self.df1 = df1
        self.df2 = df2
        self.name1 = name1
        self.name2 = name2
        self.show = show
        self._bins = bins
        self.plotly_params = plotly_params

    def make(self, plot_type: int, cut_min_value: float = 0.01, cut_max_value: float = 0.9, 
             samples: float = 100, cohort_base: str = 'model1'):
        """
        Main method to create specified comparison plot type.
        
        :param plot_type: Type of plot to create:
                          - 1: Factor comparison plots
                          - 2: Cohort comparison plot
                          - 3: Relative models plot
        :type plot_type: int
        :param cut_min_value: Minimum quantile value for clipping in cohort plots, defaults to 0.01
        :type cut_min_value: float, optional
        :param cut_max_value: Maximum quantile value for clipping in cohort plots, defaults to 0.9
        :type cut_max_value: float, optional
        :param samples: Sampling parameter for discretization, defaults to 100
        :type samples: float, optional
        :param cohort_base: Base for cohort analysis ('model1' or 'fact'), defaults to 'model1'
        :type cohort_base: str, optional
        
        :return: Generated Plotly figure
        :rtype: plotly.graph_objs.Figure
        
        :raises ValueError: If unknown plot type is specified
        
        :example:
            >>> plotter = CompareModelsPlot('frequency', df1, df2, cat_features, num_features)
            >>> fig = plotter.make(plot_type=2, cohort_base='fact')
        """
        if plot_type == 1:
            figure = self._factors_plots()

        elif plot_type == 2:
            figure = self._compare_models_cohort_plot(cut_min_value=cut_min_value,
                                                      cut_max_value=cut_max_value,
                                                      samples=samples,
                                                      cohort_base=cohort_base, )
        elif plot_type == 3:
            figure = self._relative_models_plot(cut_min_value=cut_min_value,
                                                cut_max_value=cut_max_value,
                                                samples=samples, )
        else:
            raise logger.error('Unknown plot type')
        return figure

    def _factors_plots(self):
        """
        Create factor comparison plots showing model predictions vs actuals by feature.
        
        Generates plots for both categorical and numerical features, comparing predictions
        from two models against actual values.
        
        :return: Last generated feature plot
        :rtype: plotly.graph_objs.Figure
        
        :note: If exposure column is missing, it's added with value 1 and a warning is logged
        :note: For categorical features, groups are created by unique values
        :note: For numerical features, bins are created based on percentiles
        """
        X_test = self.df1.copy()
        X_test2 = self.df2.copy()
        column_exposure = 'exposure'
        if column_exposure not in X_test:
            X_test[column_exposure] = 1
            logger.error('No exposure for model 1||Added exposure = 1')
        if column_exposure not in X_test2:
            X_test2[column_exposure] = 1
            logger.error('No exposure for model 2||Added exposure = 1')
        column_claims_count = 'y_true'
        bins_test_fact = {}
        bins_test_pred = {}
        bins_test_pred2 = {}
        features = self._features_categorical + self._features_numerical

        for feature in features:
            if feature in self._features_categorical:
                feature_vals = X_test[feature]
                bins_test_fact[feature] = X_test[[column_claims_count, column_exposure]].groupby(feature_vals,
                                                                                                 dropna=False,
                                                                                                 observed=False).sum()
                bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                                  bins_test_fact[feature][
                                                      column_exposure]
                bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(feature_vals, dropna=False,
                                                                                            observed=False).sum()
                bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                    column_exposure]
                feature_vals2 = X_test2[feature]
                bins_test_pred2[feature] = X_test2[['y_prediction', column_exposure]].groupby(feature_vals2,
                                                                                              dropna=False,
                                                                                              observed=False).sum()
                bins_test_pred2[feature]["freq"] = bins_test_pred2[feature]['y_prediction'] / bins_test_pred2[feature][
                    column_exposure]
            else:
                feature_vals = X_test[feature]
                feature_vals2 = X_test2[feature]
                n_bins = self._bins
                if feature_vals.nunique() > n_bins and self._bins is not None:
                    breakpoints = np.arange(0, n_bins + 1) / (n_bins) * 100
                    breakpoints = [np.percentile(X_test[feature], bp) for bp in breakpoints]
                    breakpoints[0] = breakpoints[0] - 0.1
                else:
                    breakpoints = self._bins

                bins_test_fact[feature] = X_test[[column_claims_count, column_exposure]].groupby(
                    pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                    observed=False).sum()
                bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                                  bins_test_fact[feature][
                                                      column_exposure]
                bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(
                    pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                    observed=False).sum()
                bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                    column_exposure]

                bins_test_pred2[feature] = X_test2[['y_prediction', column_exposure]].groupby(
                    pd.cut(X_test2[feature], breakpoints, duplicates="drop"),
                    observed=False).sum()
                bins_test_pred2[feature]["freq"] = bins_test_pred2[feature]['y_prediction'] / bins_test_pred2[feature][
                    column_exposure]

            figure = PlotlyWrapper(model_name=self.model_name, plotly_params=self.plotly_params)
            figure.add_figure(x=bins_test_fact[feature].index.astype(str),
                              y=bins_test_fact[feature][column_exposure].values,
                              name="exposure", type="bar")

            figure.add_figure(name="fact",
                              x=bins_test_fact[feature].index.astype(str),
                              y=bins_test_fact[feature]["freq"].values, )

            figure.add_figure(name="model1",
                              x=bins_test_pred[feature].index.astype(str),
                              y=bins_test_pred[feature]["freq"].values,
                              )
            figure.add_figure(name='model2',
                              x=bins_test_pred2[feature].index.astype(str),
                              y=bins_test_pred2[feature]["freq"].values,
                              )
            fig = figure.prepare_plot(x_name=feature)
            df = pd.concat([bins_test_pred[feature], bins_test_pred2[feature], bins_test_fact[feature]], axis=1)
            if self.show:
                fig.show()

        return fig

    def _compare_models_cohort_plot(self, cut_min_value: float = 0.001, cut_max_value: float = 0.9, samples=100.0,
                                    cohort_base: str = 'model1'):
        """
        Create cohort comparison plot between two models.
        
        :param cut_min_value: Minimum quantile value for clipping predictions, defaults to 0.001
        :type cut_min_value: float, optional
        :param cut_max_value: Maximum quantile value for clipping predictions, defaults to 0.9
        :type cut_max_value: float, optional
        :param samples: Sampling parameter for discretization, defaults to 100.0
        :type samples: float, optional
        :param cohort_base: Base for cohort analysis ('model1' or 'fact'), defaults to 'model1'
        :type cohort_base: str, optional
        
        :return: Cohort comparison plot
        :rtype: plotly.graph_objs.Figure
        
        :note: If exposure column is missing, it's added with value 1 and a warning is logged
        :note: The method automatically adjusts sampling if number of unique groups is too high or too low
        """
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        data = [df1, df2]
        for df in data:
            if 'exposure' not in df.columns:
                df['exposure'] = 1
                logger.warning('No exposure for cohort plot||Exposure=1 vector added')
            df["predict"] = df['y_prediction'] / df['exposure']

        fig = PlotlyWrapper(model_name=self.model_name, plotly_params=self.plotly_params)

        if cohort_base == 'model1':
            y = self.__compare_cohort_model_base(df1=df1, df2=df2, samples=samples, cut_min_value=cut_min_value,
                                                 cut_max_value=cut_max_value)
            names = ['fact', 'model1', 'model2', 'exposure']


        elif cohort_base == 'fact':
            y = self.__compare_cohort_fact_base(df1=df1, df2=df2, samples=samples, cut_min_value=cut_min_value,
                                                 cut_max_value=cut_max_value)
            names = ['model1', 'fact', 'model2', 'exposure']

        else:
            logger.error('Unknown cohort base num||Chosen model for default line')
            y = self.__compare_cohort_model_base(df1=df1, df2=df2, samples=samples, cut_min_value=cut_min_value,
                                                 cut_max_value=cut_max_value)
            names = ['fact', 'model1', 'model2', 'exposure']

        fig.add_figure(x=y["predict_gr"].values, y=y["target"].values, type='scatter', name=names[0])
        fig.add_figure(x=y["predict_gr"].values, y=y["predict_gr_"].values, name=names[1])
        fig.add_figure(x=y["predict_gr"].values, y=y["prediction_old_model"].values, name=names[2])
        fig.add_figure(x=y["predict_gr"].values, y=y["exposure"].values, type='bar', name=names[3])

        figure = fig.prepare_plot()

        if self.show:
            figure.show()
        return figure

    def _relative_models_plot(self, cut_min_value: float = 0.001, cut_max_value: float = 0.8, samples=100.0, ):
        """
        Create relative comparison plot showing ratio between model predictions.
        
        :param cut_min_value: Minimum quantile value for clipping ratio, defaults to 0.001
        :type cut_min_value: float, optional
        :param cut_max_value: Maximum quantile value for clipping ratio, defaults to 0.8
        :type cut_max_value: float, optional
        :param samples: Sampling parameter for discretization, defaults to 100.0
        :type samples: float, optional
        
        :return: Relative models comparison plot
        :rtype: plotly.graph_objs.Figure
        
        :note: Plots the ratio of model1 predictions to model2 predictions
        :note: If exposure column is missing, it's added with value 1 and a warning is logged
        """
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        data = [df1, df2]
        for df in data:
            if 'exposure' not in df.columns:
                df['exposure'] = 1
                logger.warning('No exposure for cohort plot||Exposure=1 vector added')
            df["predict"] = df['y_prediction'] / df['exposure']
        y_test_freq_gr = []
        fig = PlotlyWrapper(model_name=self.model_name, plotly_params=self.plotly_params)

        names = ['fact', 'model1', 'model2', 'exposure']
        column_to_group = "predict"
        column_to_plot = 'y_true'

        for df in data:
            df["predict_gr"] = round(data[0][column_to_group] / data[1][column_to_group] * samples) / samples
            df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                  upper=df["predict_gr"].quantile(cut_max_value), inplace=True)

            y_test_freq_gr.append(
                df
                [['y_true', 'y_prediction', 'exposure', "predict_gr"]]
                .groupby("predict_gr")
                .sum()
                .reset_index()
            )

        i = 1
        for y in y_test_freq_gr:
            y['target'] = y['y_prediction']
            fig.add_figure(x=y["predict_gr"].values, y=y["target"].values, type='scatter', name=names[i])
            fig.add_figure(x=y["predict_gr"].values, y=y["y_true"].values, type='scatter', name=names[0])
            fig.add_figure(x=y["predict_gr"].values, y=y["exposure"].values, type='bar', name=names[3])
            i += 1
        figure = fig.prepare_plot(x_name='Model 1/ Model 2')

        if self.show:
            figure.show()
        return figure

    def __compare_cohort_fact_base(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                   samples:float, cut_min_value: float, cut_max_value: float):
        """
        Prepare data for cohort comparison with 'fact' as the grouping base.
        
        :param df1: DataFrame for the first model
        :type df1: pd.DataFrame
        :param df2: DataFrame for the second model
        :type df2: pd.DataFrame
        :param samples: Sampling parameter for discretization
        :type samples: float
        :param cut_min_value: Minimum quantile value for clipping
        :type cut_min_value: float
        :param cut_max_value: Maximum quantile value for clipping
        :type cut_max_value: float
        
        :return: Aggregated data for cohort plotting
        :rtype: pd.DataFrame
        """
        column_to_group ="y_true"
        column_to_plot = "y_true"
        df2 = df2.rename(columns={"predict": "predict_old_model", "y_prediction": "prediction_old_model"})
        change_samples = False
        df = pd.concat([df1, df2[['predict_old_model', "prediction_old_model"]]], axis=1)

        df["predict_gr"] = round(df[column_to_group] * samples) / samples
        df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                              upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
        df = self.__auto_scale(df, column_to_group, samples, cut_min_value, cut_max_value, change_samples)

        y = df[['y_prediction', "prediction_old_model", 'exposure', "predict_gr", 'y_true']].groupby(
            "predict_gr").sum().reset_index()
        y['predict_gr_'] = y['predict_gr']
        y['target'] = y[column_to_plot] / y['exposure']
        y["prediction_old_model"] = y["prediction_old_model"] / y['exposure']
        return y

    def __compare_cohort_model_base(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                    samples:float, cut_min_value: float, cut_max_value: float):
        """
        Prepare data for cohort comparison with 'model1' as the grouping base.
        
        :param df1: DataFrame for the first model
        :type df1: pd.DataFrame
        :param df2: DataFrame for the second model
        :type df2: pd.DataFrame
        :param samples: Sampling parameter for discretization
        :type samples: float
        :param cut_min_value: Minimum quantile value for clipping
        :type cut_min_value: float
        :param cut_max_value: Maximum quantile value for clipping
        :type cut_max_value: float
        
        :return: Aggregated data for cohort plotting
        :rtype: pd.DataFrame
        """
        column_to_group = "predict"
        column_to_plot = 'y_true'
        df2 = df2.rename(columns={"predict": "predict_old_model", "y_prediction": "prediction_old_model"})
        change_samples = False
        df = pd.concat([df1, df2[['predict_old_model',"prediction_old_model" ]]], axis=1)
        df["predict_gr"] = round(df[column_to_group] * samples) / samples
        df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                              upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
        df = self.__auto_scale(df, column_to_group, samples, cut_min_value, cut_max_value, change_samples)
        y = df[['y_prediction', "prediction_old_model",'exposure', "predict_gr", 'y_true']].groupby("predict_gr").sum().reset_index()
        y['predict_gr_'] = y['predict_gr']
        y['target'] = y[column_to_plot] / y['exposure']
        y["prediction_old_model"] = y["prediction_old_model"] / y['exposure']
        return y

    def __auto_scale(self, df: pd.DataFrame,  column_to_group: str, samples: float, cut_min_value: float,
                cut_max_value: float, change_samples: bool = True):
        """
        Automatically adjust sampling parameters for optimal grouping.
        
        :param df: DataFrame to adjust
        :type df: pd.DataFrame
        :param column_to_group: Column name used for grouping
        :type column_to_group: str
        :param samples: Current sampling parameter
        :type samples: float
        :param cut_min_value: Current minimum quantile value
        :type cut_min_value: float
        :param cut_max_value: Current maximum quantile value
        :type cut_max_value: float
        :param change_samples: Whether samples parameter can be changed, defaults to True
        :type change_samples: bool, optional
        
        :return: DataFrame with adjusted grouping
        :rtype: pd.DataFrame
        
        :note: Reduces sampling if too many unique groups (>20)
        :note: Increases sampling if too few unique groups (<5)
        """
        if len(df["predict_gr"].unique()) > 20 and samples == 100.0 and not change_samples:

            for i in range(8):
                if len(df["predict_gr"].unique()) < 20:
                    break
                samples = samples / 5
                df["predict_gr"] = round(df[column_to_group] * samples) / samples
                df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                      upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
            change_samples = False
        if len(df["predict_gr"].unique()) < 5 and cut_max_value == 0.9 and not change_samples:
            for i in range(8):
                if len(df["predict_gr"].unique()) >= 5:
                    break
                cut_max_value = cut_max_value * 1.05
                if cut_max_value > 1: cut_max_value = 1
                samples = samples * 1.5
                df["predict_gr"] = round(df[column_to_group] * samples) / samples
                df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                      upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
        return df

class PlotlyWrapper:
    """Wrapper class for standardizing Plotly graph styles and configurations."""
   
    def __init__(self, model_name: str,
                 plotly_params: dict = None):
        """
        Initialize the PlotlyWrapper instance.
        
        :param model_name: Name of the model for plot titles and labels
        :type model_name: str
        :param plotly_params: Custom parameters to override default plot styles, defaults to None
        :type plotly_params: dict, optional
        
        :ivar figure: The underlying Plotly figure object
        :ivar colors: Default color scheme for different plot elements
        :ivar mode: Default plot modes (lines, markers) for different elements
        :ivar names: Default display names for plot legends
        :ivar params: Consolidated parameter dictionary containing all styling options
        
        :note: Initializes a subplot with secondary y-axis for exposure display
        :note: Default styles can be overridden via plotly_params dictionary
        """
        self.model_name = model_name
        self.figure = make_subplots(specs=[[{"secondary_y": True}]])
        self.plotly_params = plotly_params
        self.colors = {'fact': "rgba(255,0,0,1)",
                       'model': "rgba(60,255,60,1)",
                       'model1': "rgba(60,255,60,1)",
                       'model2': "rgba(60,60,255,1)",
                       "exposure": "rgba(255,180,180,0.6)"}
        self.mode = {'fact': "lines",
                     'model': "lines+markers",
                     'model1': "lines+markers",
                     'model2': "lines+markers",
                     "exposure": "lines+markers"}

        self.names = {'fact': 'Fact',
                      'model1': 'New model',
                      'model': 'New model',
                      'model2': 'Old model',
                      'exposure': 'Exposure(Size)'
                      }
        self.y_name = self.model_name

        self._plot_count = {'fact': 0,
                            'model1': 0,
                            'model': 0,
                            'model2': 0,
                            'exposure': 0}

        self.params = {'colors': self.colors,
                       'mode': self.mode,
                       'names': self.names,
                       'y_name': self.y_name,
                       'title': self.model_name}
        self._plotly_params_loader()

    def add_figure(self, x, y, name: str = 'fact', style_number: int = 0, type: str = 'scatter'):
        """
        Add a trace to the plot.
        
        :param x: X-axis data values
        :type x: array-like
        :param y: Y-axis data values
        :type y: array-like
        :param name: Type of trace to add ('fact', 'model', 'model1', 'model2', 'exposure'), defaults to 'fact'
        :type name: str, optional
        :param style_number: Style variant number (currently unused), defaults to 0
        :type style_number: int, optional
        :param type: Type of plot to add ('scatter' or 'bar'), defaults to 'scatter'
        :type type: str, optional
        
        :note: Each trace type can only be added once; subsequent calls with the same name are ignored
        :note: Exposure traces are automatically placed on the secondary y-axis
        """
        self._plot_count[name] = self._plot_count[name] + 1
        if self._plot_count[name] < 2:
            if type == 'scatter':
                self.__add_scatter(x, y, name)
            if type == 'bar':
                self.__add_bar(x, y, name)

    def prepare_plot(self, x_name: str = "Target groups"):
        """
        Finalize the plot with labels and layout settings.
        
        :param x_name: Title for the x-axis, defaults to "Target groups"
        :type x_name: str, optional
        
        :return: Configured Plotly figure ready for display
        :rtype: plotly.graph_objs.Figure
        
        :note: Sets the primary y-axis label to the model name
        :note: Sets the secondary y-axis label to "Exposure(Size)"
        :note: Applies the title from parameters
        """
        self.figure.update_layout(title=self.params['title'])
        self.figure.update_xaxes(title_text=x_name)

        self.figure.update_yaxes(title_text=self.params['y_name'], secondary_y=False)
        self.figure.update_yaxes(title_text=self.names['exposure'], side="right", secondary_y=True)
        return self.figure

    def __add_scatter(self, x, y, name: str = 'fact'):
        """
        Internal method to add a scatter trace.
        
        :param x: X-axis data values
        :type x: array-like
        :param y: Y-axis data values
        :type y: array-like
        :param name: Type of scatter trace to add
        :type name: str
        
        :note: Uses colors and modes from the params dictionary
        :note: Adds trace to primary y-axis
        """
        self.figure.add_trace(
            go.Scatter(
                name=self.names[name],
                x=x,
                y=y,
                mode=self.mode[name],
                marker=dict(color=self.colors[name]),
            )
        )

    def __add_bar(self, x, y, name: str = 'exposure'):
        """
        Internal method to add a bar trace.
        
        :param x: X-axis data values
        :type x: array-like
        :param y: Y-axis data values
        :type y: array-like
        :param name: Type of bar trace to add (typically 'exposure')
        :type name: str
        
        :note: Bar traces are always added to the secondary y-axis
        :note: Uses colors from the params dictionary
        """
        self.figure.add_trace(
            go.Bar(
                name=self.names[name],
                x=x,
                y=y,
                marker=dict(color=self.colors[name])
            ),
            secondary_y=True
        )

    def _plotly_params_loader(self):
        """
        Load and apply custom plot parameters.
        
        :note: Updates the params dictionary with values from plotly_params
        :note: Logs updates for successfully loaded parameters
        :note: Logs errors for invalid parameter keys
        :note: Uses deepcopy to avoid modifying the original parameter dictionary
        """
        if self.plotly_params is not None:
            logger.info('loading plotly parmas||PlotlyWrapper')
            iteration = deepcopy(self.plotly_params)
            for key in iteration.keys():
                try:
                    new_value = deepcopy(self.plotly_params[key])
                    self.params[key] = new_value
                    logger.info('Updating ' + str(key) + '||PlotlyWrapper')
                except:
                    logger.error('Wrong plotly param' + key)
