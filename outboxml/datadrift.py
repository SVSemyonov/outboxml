from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


class DataDrift:
    def __init__(self,
                 full_calc: bool = False,
                 columns_to_exclude: list = [],
                 n_bins: int = 100,
                 dif_len_string: int = 100):
        self.dif_len_string = dif_len_string
        self.n_bins = n_bins
        self.full_calc = full_calc
        self.types_dict = {}
        self.full_report = []
        self.columns_to_exclude = columns_to_exclude

    def report(self, train_data: pd. DataFrame, test_data: pd.DataFrame, base_data: pd.DataFrame=None,
                                        control_data: pd.DataFrame=None)-> pd.DataFrame:
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
            full_result = self._full_calculation(base_data, control_data)
            result = pd.concat([result, full_result])
        return result.transpose()

    def _calculate_psi(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e[:-1])
        res = pd.Series((p - q) * np.log(p / q))
        PSI = res[res != np.inf].sum()
        return PSI

    def _calculate_js_divirgence(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)
        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)

        m = (1. / 2.) * (p + q)
        return (1. / 2.) * np.sum(p * np.log(p / m)) + (1. / 2.) * np.sum(q * np.log(q / m))

    def _calculate_kl_divirgence(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)

        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)

        return np.sum(p * np.log(p / q))

    def get_probs(self, list_of_tuples):
        p = np.array([p[0] for p in list_of_tuples])
        q = np.array([p[1] for p in list_of_tuples])
        return p, q

    def support_intersection(self, p, q):
        sup_int = (
            list(
                filter(
                    lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)
                )
            )
        )
        return sup_int

    def compute_probs(self, data, n=10):
        """
        Функция для расчёта вероятностей по бинам фактора

        Parameters
        ----------
        data : pd.Series
            вектор признака.
        n : int
            число бинов, на которые надо разбивать вектор

        Returns
        -------
        (e, p), где:
        e - массив из левых границ бинов
        p - вероятности встретить значения из этих бинов

        Examples
        --------
        compute_probs(df_Last['EMRValue'].fillna(0), n=10)
        (array([      0. ,  903454.5, 1806909. , 2710363.5, 3613818. , 4517272.5,
                5420727. , 6324181.5, 7227636. , 8131090.5, 9034545. ]),
         array([9.99986848e-01, 3.75759739e-06, 1.87879870e-06, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                5.63639609e-06, 1.87879870e-06]))
        """
        if isinstance(data, np.ndarray): data = pd.Series(data)
        if data.dtype == 'object': data = pd.to_numeric(data, errors='coerce')
        h, e = np.histogram(data[data.notna()], n)

        p = h / data.shape[0]

        e = np.append(e, np.nan)
        p = np.append(p, data.sum() / data.shape[0])
        return e, p

    def _full_calculation(self, base_data: pd.DataFrame, control_data: pd.DataFrame)->pd.DataFrame:
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

