from itertools import chain
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from typing import Optional, List, Any

from loguru import logger


class FeatureImportance:
    def __init__(self, model_name: str, model: Any,):
        self.model_name = model_name
        self.model = model
        self.importance_data = None

    def calculate_importance(
            self,
            data: pd.DataFrame,
            target: pd.Series,
            calculate_directions: bool = True,
            exposure: Optional[pd.Series] = None,
            zero_corr_threshold: float = 0.0
    ):
        logger.debug(f"Calculating feature importance for model: {self.model_name}...")
        try:
            features = list(chain(self.model.features_numerical, self.model.features_categorical))
            shap_dict = self._calc_shap_values(self.model.model, data[features])
        except Exception as e:
            logger.error(f"Cannot calculate SHAP values for {self.model_name} || {e}")
            return

        directions_dict = {}
        if calculate_directions:
            directions_dict = self._calc_features_directions(
                data=data[features],
                target=target,
                exposure=exposure,
                zero_corr_threshold=zero_corr_threshold
            )

        result = []
        for feature, shap_value in shap_dict.items():
            row = {
                'FEATURE': feature,
                'SHAP': shap_value
            }
            if calculate_directions:
                row['SIGN'] = directions_dict.get(feature, 0)

            result.append(row)

        self.importance_data = result

    def _calc_shap_values(self, model, data: pd.DataFrame) -> dict:
        logger.info("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data)

        if isinstance(shap_values, list):  # Для мультикласса
            values = np.abs(np.array([v.values for v in shap_values])).mean(axis=(0, 1))
        else:
            values = np.abs(shap_values.values).mean(axis=0)

        return dict(zip(data.columns, values))

    def _calc_features_directions(
            self,
            data: pd.DataFrame,
            target: pd.Series,
            exposure: Optional[pd.Series] = None,
            zero_corr_threshold: float = 0.0
    ) -> dict:
        logger.info("Calculating features directions...")
        X = data.copy()
        y = target.copy()

        if exposure is not None:
            valid_idx = exposure > 0
            X = X[valid_idx]
            y = y[valid_idx] / exposure[valid_idx]

        directions = {}
        for col in X.columns:
            corr = X[col].corr(y, method='spearman')

            if corr > zero_corr_threshold:
                sign = 1
            elif corr < -zero_corr_threshold:
                sign = -1
            else:
                sign = 0

            directions[col] = sign

        return directions

    def plot(self, show: bool = True):
        logger.info("Plotting feature importance...")
        df = pd.DataFrame(self.importance_data)

        df = df.sort_values('SHAP', ascending=True)

        color_map_names = {
            1: 'Direct',
            -1: 'Inverse',
            0: 'Neutral/Category'
        }

        color_discrete_map = {
            'Direct': 'indianred',
            'Inverse': 'royalblue',
            'Neutral/Category': 'gray'
        }

        if 'SIGN' in df.columns:
            # Map numeric values to readable labels
            df['IMPACT'] = df['SIGN'].map(color_map_names)
            color_col = 'IMPACT'
        else:
            color_col = None
            color_discrete_map = None


        fig = px.bar(
            df,
            x='SHAP',
            y='FEATURE',
            orientation='h',
            color=color_col,
            color_discrete_map=color_discrete_map,
            title=f"Importance: {self.model_name}",
            labels={'SHAP': 'Mean |SHAP|', 'FEATURE': 'FEATURE', 'SIGN': 'Тип влияния'}
        )

        # 5. Настройка внешнего вида
        fig.update_layout(
            height=max(400, len(df) * 30),
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            margin=dict(l=150)
        )
        if show:
            fig.show()
        return fig