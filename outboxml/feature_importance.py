from itertools import chain
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from typing import Optional, List, Any

from loguru import logger
from shap.utils._exceptions import InvalidModelError
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from outboxml.models import GLMCatboostCombineModel, CatboostOverGLMModel


class FeatureImportance:
    def __init__(
            self,
            model_name: str,
            model: Any,
            data: pd.DataFrame,
            target: pd.Series,
            exposure: Optional[pd.Series] = None,
    ):
        self.model_name = model_name
        self.model = model
        self.data = data
        self.target = target
        self.exposure = exposure
        self._features = []
        self.importance_data = None

    def _init_model(self):
        if isinstance(self.model, GLMCatboostCombineModel):
            self._features = list(chain(self.model.features_numerical, self.model.features_categorical))
            self.model = self.model.model
        elif isinstance(self.model, (CatboostOverGLMModel, GLMResultsWrapper)):
            #TODO
            logger.warning(f"Can't calculate SHAP values for model class || {type(self.model)}")
        else:
            self.model = self.model.predict
            self._features = self.data.columns.tolist()

    def calculate_importance(
            self,
            calculate_directions: bool = True,
            zero_corr_threshold: float = 0.0
    ):
        logger.debug(f"Calculating feature importance for model: {self.model_name}...")
        self._init_model()
        try:
            shap_dict = self._calc_shap_values(self.model, self.data[self._features])
        except Exception as e:
            logger.error(f"Can't calculate SHAP values for {self.model_name} with model class {type(self.model)}|| {e}")
            return

        directions_dict = {}
        if calculate_directions:
            directions_dict = self._calc_features_directions(
                data=self.data[self._features],
                target=self.target,
                exposure=self.exposure,
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
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(data)
        except InvalidModelError:
            explainer = shap.Explainer(model, data)
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
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = X[col].corr(y, method='spearman')

                if corr > zero_corr_threshold:
                    sign = 1
                elif corr < -zero_corr_threshold:
                    sign = -1
                else:
                    sign = 0
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