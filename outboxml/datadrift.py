from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from outboxml.monitoring_result import DataContext
from outboxml.core.monitoring_factory import (
    DataReviewerRegistry,
    DataReviewerComponent,
)


@DataReviewerRegistry.register("datadrift")
class DataDrift(DataReviewerComponent):
    """Data drift detection component for monitoring model performance.
    
    Calculates various drift metrics (PSI, KL divergence, JS divergence) to detect
    changes in data distribution between training and test datasets. Supports both
    numerical and categorical features.
    
    :param full_calc: Whether to perform full calculation including base/control
        data comparison. Defaults to True.
    :type full_calc: bool
    :param columns_to_exclude: List of column names to exclude from drift calculation.
        Defaults to empty list.
    :type columns_to_exclude: list
    :param n_bins: Number of bins to use for histogram-based calculations.
        Defaults to 100.
    :type n_bins: int
    :param dif_len_string: Maximum length of difference string in full calculation.
        Defaults to 100.
    :type dif_len_string: int
    
    :var dif_len_string: Maximum length of difference string in full calculation.
    :var n_bins: Number of bins to use for histogram-based calculations.
    :var full_calc: Whether to perform full calculation.
    :var types_dict: Dictionary mapping column names to their types ('NUMERICAL' or 'CATEGORICAL').
    :var full_report: List for storing full calculation reports.
    :var columns_to_exclude: List of column names to exclude from drift calculation.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.datadrift import DataDrift
        from outboxml.monitoring_result import DataContext
        drift_detector = DataDrift(full_calc=True, n_bins=50)
        result = drift_detector.review(data_context)
        # Returns DataFrame with PSI, KL, JS metrics for each column
    """
    def __init__(self, full_calc: bool = True, columns_to_exclude: list = [], n_bins: int = 100,
                 dif_len_string: int = 100):
        """Initialize DataDrift instance.
        
        :param full_calc: Whether to perform full calculation including base/control
            data comparison. Defaults to True.
        :type full_calc: bool
        :param columns_to_exclude: List of column names to exclude from drift calculation.
            Defaults to empty list.
        :type columns_to_exclude: list
        :param n_bins: Number of bins to use for histogram-based calculations.
            Defaults to 100.
        :type n_bins: int
        :param dif_len_string: Maximum length of difference string in full calculation.
            Defaults to 100.
        :type dif_len_string: int
        """
        super().__init__()
        self.dif_len_string = dif_len_string
        self.n_bins = n_bins
        self.full_calc = full_calc
        self.types_dict = {}
        self.full_report = []
        self.columns_to_exclude = columns_to_exclude

    def review(self, data_context: DataContext)-> pd.DataFrame:
        """Perform data drift review on the provided data context.
        
        Calculates PSI (Population Stability Index), KL divergence, and JS divergence
        for each column in the dataset. Automatically detects column types (numerical
        or categorical) and applies appropriate transformations. Optionally performs
        full calculation with base/control data comparison.
        
        :param data_context: DataContext object containing:
            - X_train: Training dataset (pandas.DataFrame)
            - X_test: Test dataset (pandas.DataFrame)
            - base: Base dataset for full calculation (optional, pandas.DataFrame)
            - actual: Control/actual dataset for full calculation (optional, pandas.DataFrame)
        :type data_context: DataContext
        :return: DataFrame with drift metrics for each column. Columns are features,
            rows are metrics (PSI, KL, JS). If full_calc=True, additional rows
            with detailed statistics are included.
        :rtype: pandas.DataFrame
        
        .. note::
            - Categorical columns are automatically encoded using LabelEncoder
            - Columns in columns_to_exclude are skipped
            - Errors during calculation are logged but don't stop the process
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            from outboxml.monitoring_result import DataContext
            data_context = DataContext(X_train=train_df, X_test=test_df)
            drift_detector = DataDrift(full_calc=True)
            result = drift_detector.review(data_context)
            # Returns DataFrame with PSI, KL, JS for each column
            print(result['feature_name']['PSI'])  # Get PSI for a specific feature
        """
        train_data = data_context.X_train
        test_data = data_context.X_test

        result = pd.DataFrame()

        for column in train_data.columns:
            if column in self.columns_to_exclude: continue
            self.types_dict[column] = 'NUMERICAL'
            X_train = train_data[column].copy()
            X_test = test_data[column].copy()
            if X_train.dtype == 'category' or X_train.dtype == 'object':
                self.types_dict[column] = 'CATEGORICAL'
                LE = LabelEncoder()
                LE.fit(X_train)
                X_train = LE.transform(X_train)
                X_test = LE.transform(X_test)
            try:
                psi = self._calculate_psi(X_train, X_test, self.n_bins)
                kl_divergence = self._calculate_kl_divirgence(X_train, X_test, self.n_bins)
                js_divergence = self._calculate_js_divirgence(X_train, X_test, self.n_bins)
                result[column] = [psi, kl_divergence, js_divergence]
            except Exception as exc:
                logger.error('No results for '+ column + '||' + str(exc))
        result.index = pd.Index(['PSI', 'KL', 'JS'])
        if self.full_calc:
            base_data = data_context.base
            control_data = data_context.actual
            if base_data is None or control_data is None:
                logger.error('No base/control data')

            full_result = self._full_calculation(base_data, control_data)
            result = pd.concat([result, full_result])
        return result.transpose()

    def _calculate_psi(self, train_sample, test_sample, n_bins):
        """Calculate Population Stability Index (PSI) between two samples.
        
        PSI measures the stability of a population over time. It compares the
        distribution of a feature between training and test datasets.
        
        :param train_sample: Training sample data (array-like).
        :type train_sample: array-like
        :param test_sample: Test sample data (array-like).
        :type test_sample: array-like
        :param n_bins: Number of bins to use for probability calculation.
        :type n_bins: int
        :return: PSI value. Lower values indicate more stability.
            - PSI < 0.1: No significant change
            - PSI 0.1-0.25: Some minor change
            - PSI > 0.25: Significant change
        :rtype: float
        
        .. note::
            This is a private method, typically called internally by review().
            Infinite values are excluded from the sum.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            drift_detector = DataDrift()
            psi = drift_detector._calculate_psi(train_data, test_data, n_bins=50)
            print(f"PSI: {psi}")
        """
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e[:-1])
        res = pd.Series((p - q) * np.log(p / q))
        PSI = res[res != np.inf].sum()
        return PSI

    def _calculate_js_divirgence(self, train_sample, test_sample, n_bins):
        """Calculate Jensen-Shannon (JS) divergence between two samples.
        
        JS divergence is a symmetric version of KL divergence that measures the
        difference between two probability distributions. It ranges from 0 to 1,
        where 0 means identical distributions.
        
        :param train_sample: Training sample data (array-like).
        :type train_sample: array-like
        :param test_sample: Test sample data (array-like).
        :type test_sample: array-like
        :param n_bins: Number of bins to use for probability calculation.
        :type n_bins: int
        :return: JS divergence value. Range: [0, 1]. Lower values indicate
            more similar distributions.
        :rtype: float
        
        .. note::
            This is a private method, typically called internally by review().
            JS divergence is symmetric and always finite.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            drift_detector = DataDrift()
            js = drift_detector._calculate_js_divirgence(train_data, test_data, n_bins=50)
            print(f"JS divergence: {js}")
        """
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)
        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)

        m = (1. / 2.) * (p + q)
        return (1. / 2.) * np.sum(p * np.log(p / m)) + (1. / 2.) * np.sum(q * np.log(q / m))

    def _calculate_kl_divirgence(self, train_sample, test_sample, n_bins):
        """Calculate Kullback-Leibler (KL) divergence between two samples.
        
        KL divergence measures how one probability distribution diverges from
        another. It is asymmetric and non-negative. Lower values indicate
        more similar distributions.
        
        :param train_sample: Training sample data (array-like).
        :type train_sample: array-like
        :param test_sample: Test sample data (array-like).
        :type test_sample: array-like
        :param n_bins: Number of bins to use for probability calculation.
        :type n_bins: int
        :return: KL divergence value. Range: [0, inf). Lower values indicate
            more similar distributions.
        :rtype: float
        
        .. note::
            This is a private method, typically called internally by review().
            Only bins where both distributions have non-zero probability are used.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            drift_detector = DataDrift()
            kl = drift_detector._calculate_kl_divirgence(train_data, test_data, n_bins=50)
            print(f"KL divergence: {kl}")
        """
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)

        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)

        return np.sum(p * np.log(p / q))

    def get_probs(self, list_of_tuples):
        """Extract probability arrays from a list of tuples.
        
        Converts a list of (p, q) tuples into separate numpy arrays for
        probability calculations.
        
        :param list_of_tuples: List of tuples where each tuple contains
            (probability_p, probability_q) pairs.
        :type list_of_tuples: list of tuples
        :return: Tuple of (p, q) numpy arrays containing probabilities.
        :rtype: tuple of numpy.ndarray
        
        .. note::
            This is a private method, typically called internally by divergence
            calculation methods.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            tuples = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
            p, q = drift_detector.get_probs(tuples)
            # p = array([0.1, 0.3, 0.5])
            # q = array([0.2, 0.4, 0.6])
        """
        p = np.array([p[0] for p in list_of_tuples])
        q = np.array([p[1] for p in list_of_tuples])
        return p, q

    def support_intersection(self, p, q):
        """Find intersection of probability distributions where both are non-zero.
        
        Filters out bins where either probability distribution has zero probability,
        which is necessary for stable KL and JS divergence calculations.
        
        :param p: First probability distribution array.
        :type p: array-like
        :param q: Second probability distribution array.
        :type q: array-like
        :return: List of tuples (p_i, q_i) where both p_i and q_i are non-zero.
        :rtype: list of tuples
        
        .. note::
            This is a private method, typically called internally by divergence
            calculation methods. Prevents division by zero in log calculations.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            p = np.array([0.1, 0.2, 0.0, 0.3])
            q = np.array([0.15, 0.0, 0.25, 0.35])
            result = drift_detector.support_intersection(p, q)
            # Returns: [(0.1, 0.15), (0.3, 0.35)]
        """
        sup_int = (
            list(
                filter(
                    lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)
                )
            )
        )
        return sup_int

    def compute_probs(self, data, n=10):
        """Calculate probabilities for feature bins.
        
        Computes histogram-based probability distribution for a feature vector.
        The data is divided into bins, and probabilities are calculated for each bin.
        Also includes a special bin for NaN values.
        
        :param data: Feature vector (pandas Series or numpy array).
        :type data: pandas.Series or numpy.ndarray
        :param n: Number of bins to divide the vector into. Defaults to 10.
        :type n: int
        :return: Tuple of (e, p) where:
            - e: Array of bin edges (left boundaries) plus NaN marker
            - p: Array of probabilities for each bin plus NaN probability
        :rtype: tuple of numpy.ndarray
        
        .. note::
            - Object dtype data is automatically converted to numeric
            - NaN values are handled separately and included as the last bin
            - The last element of e is NaN, and the last element of p is the
              proportion of NaN values
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None])
            e, p = drift_detector.compute_probs(data, n=5)
            # e contains bin edges plus NaN
            # p contains probabilities for each bin plus NaN probability
        """
        if isinstance(data, np.ndarray): data = pd.Series(data)
        if data.dtype == 'object': data = pd.to_numeric(data, errors='coerce')
        h, e = np.histogram(data[data.notna()], n)

        p = h / data.shape[0]

        e = np.append(e, np.nan)
        p = np.append(p, data.sum() / data.shape[0])
        return e, p

    def _full_calculation(self, base_data: pd.DataFrame, control_data: pd.DataFrame)->pd.DataFrame:
        """Perform full calculation with detailed statistics for all columns.
        
        Calculates detailed drift statistics for each column including NaN rates,
        unique value counts, modes, means (for numerical), and value differences
        between base and control datasets.
        
        :param base_data: Base/reference dataset for comparison.
        :type base_data: pandas.DataFrame
        :param control_data: Control/actual dataset to compare against base.
        :type control_data: pandas.DataFrame
        :return: DataFrame with detailed statistics for each column. Each column
            of the result corresponds to a feature, rows contain statistics like
            TYPE, col, date, NaN_train, NaN_test, uniq_train, uniq_test, etc.
        :rtype: pandas.DataFrame
        
        .. note::
            This is a private method, typically called internally by review()
            when full_calc=True. Uses the types_dict populated during review()
            to determine column types.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            base_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': ['a', 'b', 'c']})
            control_df = pd.DataFrame({'feature1': [1, 2, 4], 'feature2': ['a', 'b', 'd']})
            result = drift_detector._full_calculation(base_df, control_df)
            # Returns DataFrame with detailed statistics for each feature
        """
        result = pd.DataFrame()
        for column in self.types_dict.keys():
            try:
                if self.types_dict[column] == 'CATEGORICAL':
                    res = self._drift_calc(base_data[column], control_data[column], label=column, type='CATEGORICAL')
                    res = pd.Series(res, name=column)
                elif self.types_dict[column] == 'NUMERICAL':
                    res = self._drift_calc(base_data[column], control_data[column], label=column, type='NUMERICAL')
                    res = pd.Series(res, name=column)
                else:
                    logger.error('Unknown type for full calc')
                    res = pd.DataFrame()

                result = pd.concat([result, res], axis=1 )
            except Exception as exc:
                logger.error('Error while calculating datadrift for '+ str(column) + '||'+str(exc))
        return result
        
    def _drift_calc(self, base: pd.Series, control: pd.Series, label: str, type: str = 'CATEGORICAL')->dict:
        """Calculate detailed drift statistics for a single column.
        
        Computes comprehensive statistics comparing base and control datasets
        for a single feature, including NaN rates, unique counts, modes, means,
        and value differences.
        
        :param base: Base/reference series for comparison.
        :type base: pandas.Series
        :param control: Control/actual series to compare against base.
        :type control: pandas.Series
        :param label: Column name/label for the feature.
        :type label: str
        :param type: Feature type, either 'CATEGORICAL' or 'NUMERICAL'.
            Defaults to 'CATEGORICAL'.
        :type type: str
        :return: Dictionary containing drift statistics:
            - TYPE: Feature type
            - col: Column name
            - date: Current date
            - NaN_train: Proportion of NaN values in base
            - NaN_test: Proportion of NaN values in control
            - uniq_train: Number of unique values in base
            - uniq_test: Number of unique values in control
            - mode_train: Most frequent value in base
            - mode_test: Most frequent value in control
            - mean_train: Mean value in base (NaN for categorical)
            - mean_test: Mean value in control (NaN for categorical)
            - dif_train: Values in base but not in control (truncated)
            - dif_test: Values in control but not in base (truncated)
        :rtype: dict
        
        .. note::
            This is a private method, typically called internally by _full_calculation().
            Difference strings are truncated to dif_len_string characters.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            base_series = pd.Series([1, 2, 3, 4, 5])
            control_series = pd.Series([1, 2, 3, 6, 7])
            stats = drift_detector._drift_calc(
                base_series, control_series, label='feature1', type='NUMERICAL'
            )
            print(stats['dif_train'])  # Values in base but not in control
        """
        return {

            'TYPE': type,
            "col": label,
            'date': datetime.now().date(),
            "NaN_train": base.isna().mean(),
            "NaN_test": control.isna().mean(),
            "uniq_train": base.nunique(dropna=False),
            "uniq_test": control.nunique(dropna=False),
            "mode_train": base.mode(dropna=False)[0],
            "mode_test": control.mode(dropna=False)[0],
            "mean_train": np.nan if type == 'CATEGORICAL' else base.mean(),  # ,
            "mean_test":  np.nan if type == 'CATEGORICAL' else control.mean(),
            "dif_train": str(list(set(base.unique()) - set(control.unique())))[:self.dif_len_string],
            "dif_test": str(list(set(control.unique()) - set(base.unique())))[:self.dif_len_string]
        }

