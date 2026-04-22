import pandas as pd
import numpy as np
import shap
import plotly.express as px
from typing import Optional, Dict, List

from pandas.core.dtypes.common import is_numeric_dtype


class FeatureImportanceAnalyzer:
    def __init__(self, features_description: Optional[Dict] = None):
        self.features_description = features_description or {}

    def calculate_importance(
            self,
            model,
            data: pd.DataFrame,
            target: pd.Series,
            features: Optional[List[str]],
            add_random: bool = False,
            calculate_directions: bool = True,
            exposure: Optional[pd.Series] = None,
            zero_corr_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Основной метод: считает SHAP и (опционально) направления влияния.
        """
        data = self._prepare_data_for_shap(data)

        # 1. Расчет SHAP
        importance_df = self._calc_shap_values(model, data[features])

        # 2. Расчет направлений (корреляций)
        if calculate_directions:
            directions = self._calc_features_directions(
                data=data,
                target=target,
                exposure=exposure,
                zero_corr_threshold=zero_corr_threshold
            )
            # Объединяем SHAP и направления
            importance_df = importance_df.merge(directions, on='Признак', how='left')

        # 3. Добавляем описания
        importance_df['Описание'] = importance_df['Признак'].map(self.features_description).fillna(
            importance_df['Признак'])

        # Если нужно пометить рандомный признак
        if add_random and 'RANDOM' in importance_df['Признак'].values:
            importance_df.loc[importance_df['Признак'] == 'RANDOM', 'Описание'] = 'Случайный признак (контроль)'
        return importance_df

    def _prepare_data_for_shap(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Конвертирует данные в числовой формат.
        Если колонка категориальная, но содержит числа (после LabelEncoder/WoE),
        она будет приведена к float.
        """
        df = data.copy()
        for col in df.columns:
            # Если это категория или объект - пробуем перевести в числа
            if not is_numeric_dtype(df[col]):
                # errors='coerce' превратит реальные строки в NaN,
                # но ваши закодированные '1', '2' станут 1.0, 2.0
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Заполняем пропуски нулями, если они появились (SHAP не любит NaN в некоторых моделях)
        return df.fillna(0)

    def _calc_shap_values(self, model, data: pd.DataFrame) -> pd.DataFrame:
        """Внутренний метод для расчета SHAP."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data)

        if isinstance(shap_values, list):  # Для мультикласса
            values = np.abs(np.array([v.values for v in shap_values])).mean(axis=(0, 1))
        else:
            values = np.abs(shap_values.values).mean(axis=0)

        return pd.DataFrame({'Признак': data.columns, 'SHAP': values})

    def _calc_features_directions(
            self,
            data: pd.DataFrame,
            target: pd.Series,
            exposure: Optional[pd.Series] = None,
            zero_corr_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Рефакторинг вашей функции calc_features_directions.
        Работает через Spearman correlation.
        """
        X = data.copy()
        y = target.copy()

        if exposure is not None:
            # Используем фильтр по экспозиции (аналог вашего exp_filter)
            valid_idx = exposure > 0
            X = X[valid_idx]
            y = y[valid_idx] / exposure[valid_idx]

        results = []
        for col in X.columns:
            corr = X[col].corr(y, method='spearman')
            results.append({'Признак': col, 'Корр': corr})

        df_corr = pd.DataFrame(results)
        df_corr['Знак'] = 0
        df_corr.loc[df_corr['Корр'] > zero_corr_threshold, 'Знак'] = 1
        df_corr.loc[df_corr['Корр'] < -zero_corr_threshold, 'Знак'] = -1

        return df_corr[['Признак', 'Знак']]

    def plot(self, importance_df: pd.DataFrame, title: str = "Feature Importance"):
        """Визуализация через Plotly."""
        df = importance_df.sort_values('SHAP', ascending=True)

        # Определяем цвета на основе направлений, если они есть
        color_map = {1: 'Прямое', -1: 'Обратное', 0: 'Нейтральное/Категория'}
        if 'Знак' in df.columns:
            df['Влияние'] = df['Знак'].map(color_map)
            color_col = 'Влияние'
            color_discrete_map = {'Прямое': 'indianred', 'Обратное': 'royalblue', 'Нейтральное/Категория': 'gray'}
        else:
            color_col = None
            color_discrete_map = None

        fig = px.bar(
            df,
            x='SHAP',
            y='Описание',
            orientation='h',
            color=color_col,
            color_discrete_map=color_discrete_map,
            title=title,
            labels={'SHAP': 'Mean |SHAP|', 'Описание': 'Признак'}
        )

        fig.update_layout(
            height=max(400, len(df) * 20),
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )
        return fig