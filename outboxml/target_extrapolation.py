import pandas as pd
from loguru import logger
#import GPy

from outboxml.datasets_manager import DSManagerResult, ModelDataSubset
from outboxml.models import BaseWrapperModel


class TargetModel(BaseWrapperModel):
    """TargetModelInterface. Use fit and extrapolate_target methods."""
    def __init__(self, 
                 model,
                 input_dim: int = 1,
                 variance: float=1.0,
                 lengthscale: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.input_dim = input_dim
        self.model = model
    """
    def fit(self, X_train, y_train, extrapolation_period):
        logger.debug('Fitting GPy')
        seasonal_kernel = GPy.kern.PeriodicExponential(input_dim=self.input_dim, period=extrapolation_period, variance=self.variance,
                                                       lengthscale=self.lengthscale,)
        policy_kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=self.variance, lengthscale=self.lengthscale)
        error_kernel = GPy.kern.White(input_dim=self.input_dim, variance=self.variance)
        combined_kernel = seasonal_kernel + policy_kernel + error_kernel
        self.model = GPy.models.GPRegression(X_train.values, y_train.values, combined_kernel)
        self.model.optimize(messages=True)
    """
    def extrapolate_target(self, model_name, X_train, y_train, exposure=None, extrapolation_period=12)->DSManagerResult:
        self.model.fit( X_train, y_train, extrapolation_period)
        y_train_pred, y_train_pred_variance = self.model.predict(X_train.values)

       # logger.info(f'Training MAE: {mean_absolute_error(y_train, y_train_pred)}')
        return DSManagerResult(model_name=model_name,
                               model=self.model,
                               data_subset=ModelDataSubset(model_name=model_name,
                                                          X_train=X_train,
                                                          y_train=y_train,
                                                          y_test=y_train_pred,
                                                          extra_columns=pd.DataFrame(columns=['VARIANCE'],
                                                                                     data=y_train_pred_variance),
                                                          ),
                               )
