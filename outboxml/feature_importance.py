from itertools import chain
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from typing import Optional, List, Any, Dict, Tuple

from loguru import logger
from shap.utils._exceptions import InvalidModelError
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from outboxml.data_subsets import ModelDataSubset
from outboxml.models import GLMCatboostCombineModel, CatboostOverGLMModel


class FeatureImportance:
    def __init__(
            self,
            model_name: str,
            model: Any,
            data_subset: ModelDataSubset,
    ):
        self.model_name = model_name
        self.model = model
        self.data_subset = data_subset
        self.importance_data: Optional[List[Dict]] = None

    def _get_explainer_items(self) -> Tuple[Any, List[str]]:

        if isinstance(self.model, GLMCatboostCombineModel):
            features = list(chain(self.model.features_numerical,
                                  self.model.features_categorical))
            return self.model.model, features

        elif isinstance(self.model, (CatboostOverGLMModel, GLMResultsWrapper)):
            # TODO
            logger.warning(f"Can't calculate SHAP values for model class || {type(self.model)}")
            return None, None
        else:
            features = self.data_subset.X_train.columns.tolist()
            return self.model, features

    def calculate_importance(
            self,
            use_test: bool = True,
            calculate_directions: bool = True,
            zero_corr_threshold: float = 0.0
    ) -> List[Dict]:
        logger.debug(f"Calculating feature importance for model: {self.model_name}...")

        model_to_explain, features = self._get_explainer_items()

        if use_test and not self.data_subset.X_test.empty:
            data = self.data_subset.X_test[features]
            target = self.data_subset.y_test
            exposure = getattr(self.data_subset, 'exposure_test', None)
        else:
            if use_test:
                logger.warning("X_test is empty || Use X_train")
            data = self.data_subset.X_train[features]
            target = self.data_subset.y_train
            exposure = getattr(self.data_subset, 'exposure_train', None)

        try:
            shap_dict = self._calc_shap_values(model_to_explain, data)
        except Exception as e:
            logger.error(f"Can't calculate SHAP values for {self.model_name}: {e}")
            return []

        directions_dict = {}
        if calculate_directions:
            directions_dict = self._calc_features_directions(
                data=data,
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
        return result

    def _calc_shap_values(self, model, data: pd.DataFrame) -> Dict[str, float]:
        logger.info(f"Running SHAP explainer for {len(data)} samples...")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(data)
        except (InvalidModelError, Exception):
            predict_fn = model.predict if hasattr(model, 'predict') else model
            explainer = shap.Explainer(predict_fn, data)
            shap_values = explainer(data)

        if hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = shap_values

        if isinstance(vals, list):  # Мультикласс
            values = np.abs(np.array(vals)).mean(axis=(0, 1))
        else:
            values = np.abs(vals).mean(axis=0)

        return dict(zip(data.columns, values))

    def _calc_features_directions(
            self,
            data: pd.DataFrame,
            target: pd.Series,
            exposure: Optional[pd.Series] = None,
            zero_corr_threshold: float = 0.0
    ) -> Dict[str, int]:
        logger.info("Calculating features directions via Spearman correlation...")

        working_data = data.copy()
        working_target = target.copy()

        if exposure is not None and not exposure.empty:
            valid_idx = exposure > 0
            working_data = working_data[valid_idx]
            working_target = working_target[valid_idx] / exposure[valid_idx]

        directions = {}
        for col in working_data.columns:
            if pd.api.types.is_numeric_dtype(working_data[col]):
                corr = working_data[col].corr(working_target, method='spearman')

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
        if not self.importance_data:
            logger.error("No importance data to plot. Run calculate_importance() first.")
            return None

        df = pd.DataFrame(self.importance_data)
        df = df.sort_values('SHAP', ascending=True)

        # Маппинг для легенды
        color_map_names = {1: 'Direct', -1: 'Inverse', 0: 'Neutral/Category'}
        color_discrete_map = {'Direct': 'indianred', 'Inverse': 'royalblue', 'Neutral/Category': 'gray'}

        if 'SIGN' in df.columns:
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
            title=f"Feature Importance (SHAP): {self.model_name}",
            labels={'SHAP': 'Mean |SHAP|', 'FEATURE': 'Feature', 'IMPACT': 'Impact Type'}
        )

        fig.update_layout(
            height=max(400, len(df) * 25),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )

        if show:
            fig.show()
        return fig