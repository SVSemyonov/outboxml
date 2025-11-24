from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.frequencies import to_offset

def parse_time_offset(offset_str):
    if offset_str.endswith(('D', 'W', 'd', 'w')):
        return pd.to_timedelta(offset_str)
    else:
        return to_offset(offset_str)

def psi(expected_perc, actual_perc):
    expected_perc = np.array(expected_perc, dtype=float)
    actual_perc = np.array(actual_perc, dtype=float)

    # (actual - expected) * ln(actual / expected)
    return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))


def kl_divergence(expected_perc, actual_perc):
    expected_perc = np.array(expected_perc, dtype=float)
    actual_perc = np.array(actual_perc, dtype=float)

    # P * ln(P / Q)
    return np.sum(expected_perc * np.log(expected_perc / actual_perc))


def js_divergence(expected_perc, actual_perc):
    expected_perc = np.array(expected_perc, dtype=float)
    actual_perc = np.array(actual_perc, dtype=float)

    M = 0.5 * (expected_perc + actual_perc)

    # JS = 1/2 * KL(P || M) + 1/2 * KL(Q || M)
    kl_pm = np.sum(expected_perc * np.log(expected_perc / M))
    kl_qm = np.sum(actual_perc * np.log(actual_perc / M))

    return 0.5 * (kl_pm + kl_qm)

def calc_percents(expected, actual, buckets=10):
    """
    Calculate percentage distributions for two datasets (expected vs actual).

    Parameters
    ----------
    expected : pd.Series
        Base dataset (historical distribution).
    actual : pd.Series
        Current dataset to compare against the base.
    buckets : int, optional
        Number of bins for numeric data. Default is 10.

    Returns
    -------
    tuple of pd.Series
        (expected_percents, actual_percents) — percentage distributions
        of values in expected and actual datasets.

    Notes
    -----
    - Drops NaN values from both datasets.
    - For numeric data, values are binned into equal-width intervals.
      First and last bins are extended to -inf and inf.
    - For categorical data, categories are aligned across both datasets.
    - Adds a small epsilon (1e-6) to avoid division by zero in subsequent calculations.
    - Returns np.nan if either dataset is empty after removing NaNs or an error occurs.
    """
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    try:
        if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
            _, bins = pd.cut(expected, bins=buckets, retbins=True, duplicates='drop')

            bins[0] = -np.inf
            bins[-1] = np.inf

            expected_counts = pd.cut(expected, bins=bins).value_counts(sort=False)
            actual_counts = pd.cut(actual, bins=bins).value_counts(sort=False)

        else:
            expected_counts = expected.value_counts()
            actual_counts = actual.value_counts()

            all_categories = expected_counts.index.union(actual_counts.index)

            expected_counts = expected_counts.reindex(all_categories, fill_value=0)
            actual_counts = actual_counts.reindex(all_categories, fill_value=0)

        expected_percents = expected_counts / expected_counts.sum()
        actual_percents = actual_counts / actual_counts.sum()

        epsilon = 1e-6
        expected_percents = expected_percents + epsilon
        actual_percents = actual_percents + epsilon
    except:
        return np.nan

    return expected_percents, actual_percents


class RollingWindowDataDrift:
    """
    RollingWindowDataDrift computes data drift metrics using a rolling-window
    evaluation approach. For each report date, the method compares feature
    distributions between:
      - a historical base period, and
      - a recent target period.

    The drift metric used is PSI (Population Stability Index).

    Parameters
    ----------
    date_col : str
        Name of the column containing datetime information.
    base_period : str, optional
        Length of the historical comparison window (e.g. '365D', '90D').
        Default is '365D'.
    target_period : str, optional
        Length of the target comparison window (e.g. '1M', '30D').
        Default is '1M'.

    Methods
    -------
    report(dataset_name, data: pd.DataFrame) -> pd.DataFrame
        Generates a rolling-window data drift report. The output contains PSI
        values for all features for each report date.

    Notes
    -----
    - The method converts the date column to datetime format.
    - Report dates are generated starting from (min_date + base_period) until max_date,
      with a frequency equal to the target period.
    - For each report date, the method extracts:
         * base window: (report_date - base_period) to (report_date - target_period)
         * target window: (report_date - target_period) to report_date
    - PSI is calculated for each feature using distributions in the base and target windows.
    - The final result is returned in long format using `DataFrame.melt()`.
    - Metadata is added to result.attrs['meta'] to support downstream validation.
    """
    def __init__(self,
                 date_col,
                 base_period: str = '365D',
                 target_period: str = '1M'):
        self.date_col = date_col
        self.base_period = base_period
        self.target_period = target_period

    def report(self, dataset_name, data: pd.DataFrame = None) -> pd.DataFrame:
        df = data.copy()
        features = df.columns
        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except:
            logger.error("date_col cannot be None")
            return np.nan

        min_date = df[self.date_col].min()
        max_date = df[self.date_col].max()

        base_offset = parse_time_offset(self.base_period)
        target_offset = parse_time_offset(self.target_period)


        start_calculating_from = min_date + base_offset

        if start_calculating_from > max_date:
            logger.warning("Attention: The data history is shorter than the base period. Metric cannot be calculated.")
            return pd.DataFrame()

        report_dates = pd.date_range(start=start_calculating_from, end=max_date, freq=self.target_period)

        results = []
        logger.info(f"Calculation of Rolling Window DataDrift. Base: {self.base_period}, Target: {self.target_period}. Report points: {len(report_dates)}")

        for report_dt in report_dates:
            target_start = report_dt - target_offset
            target_end = report_dt

            base_end = target_start
            base_start = base_end - base_offset

            target_mask = (df[self.date_col] > target_start) & (df[self.date_col] <= target_end)
            base_mask = (df[self.date_col] > base_start) & (df[self.date_col] <= base_end)

            current_data = df[target_mask]
            base_data = df[base_mask]

            if len(current_data) == 0 or len(base_data) == 0:
                continue

            row = {
                'REPORT_DATE': report_dt,
                'N_BASE': len(base_data),
                'N_TARGET': len(current_data)
            }

            for feature in features:
                expected, actual = calc_percents(base_data[feature], current_data[feature])
                psi_value = psi(expected, actual)
                row[feature] = psi_value

            results.append(row)

        result = pd.DataFrame(results).melt(
            id_vars=['REPORT_DATE', 'N_BASE', 'N_TARGET'],
            var_name='FEATURE_NAME',
            value_name='METRIC_VALUE'
        )

        result['DATASET_NAME'] = dataset_name
        result.attrs['meta'] = {
            'DATASET_NAME': {'uniq': True}
        }
        return result


class DataDrift:
    """
    DataDrift calculates data drift metrics between two datasets (base vs control)
    for each feature. The following metrics are computed:

      - PSI (Population Stability Index)
      - KL divergence (Kullback-Leibler divergence)
      - JS divergence (Jensen-Shannon divergence)

    Parameters
    ----------
    columns_to_exclude : list, optional
        List of column names to exclude from analysis. Default is [].
    n_bins : int, optional
        Number of bins used to discretize feature distributions. Default is 10.

    Methods
    -------
    report(base_data: pd.DataFrame, control_data: pd.DataFrame, dataset_name: str = 'default') -> pd.DataFrame
        Generates a report of data drift metrics between base_data and control_data.
        Returns a DataFrame with columns:
            - FEATURE_NAME
            - PSI
            - KL
            - JS
            - DATASET_NAME

    Notes
    -----
    - Columns specified in columns_to_exclude are skipped.
    - If metrics cannot be calculated for a feature, an error is logged and that feature is skipped.
    - The resulting DataFrame includes metadata in `result.attrs['meta']` to support downstream validation.
    """
    def __init__(self,
                 columns_to_exclude: list = [],
                 n_bins: int = 10,
                 ):
        self.n_bins = n_bins
        self.columns_to_exclude = columns_to_exclude

    def report(self, base_data: pd.DataFrame, control_data: pd.DataFrame, dataset_name: str = 'default')-> pd.DataFrame:
        result = pd.DataFrame()

        for column in base_data.columns:
            if column in self.columns_to_exclude: continue
            base_percents, control_percents = calc_percents(base_data[column], control_data[column], self.n_bins)
            try:
                psi_value = psi(base_percents, control_percents)
                kl_divergence_value = kl_divergence(base_percents, control_percents)
                js_divergence_value = js_divergence(base_percents, control_percents)
                result[column] = [column, psi_value, kl_divergence_value, js_divergence_value]
            except Exception as exc:
                logger.error('No results for '+ column + '||' + str(exc))
        result.index = pd.Index(['FEATURE_NAME', 'PSI', 'KL', 'JS'])
        result = result.transpose().reset_index(drop=True)
        result['DATASET_NAME'] = dataset_name
        result.attrs['meta'] = {
            'DATASET_NAME': {'uniq': True}
        }
        return result

