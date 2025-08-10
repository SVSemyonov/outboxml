from statistics import LinearRegression

import pandas as pd

from outboxml.automl_manager import AutoMLManager
from outboxml.core.email import AutoMLReviewEMail
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric

from outboxml.metrics.business_metrics import BaseCompareBusinessMetric
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sqlalchemy import create_engine

import config

config_name = 'configs/config-example-titanic.json'
auto_ml_config = 'configs/automl-titanic.json'
path_to_data = 'data/titanic.csv'


class TitanicExampleExtractor(Extractor):

    def __init__(self,
                 path_to_file: str
                 ):
        self.__path_to_file = path_to_file
        super().__init__()

    def extract_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.__path_to_file)
    #    data['survived1'] = data['SURVIVED']
    #    data['survived2'] = data['SURVIVED']
        return data


class TitanicMetric(BaseMetric):
    def __init__(self):
        pass

    def calculate_metric(self, result1: dict, result2: dict=None) -> dict:
        y1 = (result1['first'].y_pred + result1['second'].y_pred) / 2
        y = result1['first'].y
        score1 = (y - y1).sum()
        score2 = 0
        if result2 is not None:
            y2 = (result2['first'].y_pred + result2['second'].y_pred) / 2
            score2 = (y - y2).sum()
        return {'impact': score1-score2}


class TitanicExampleEMail(AutoMLReviewEMail):
    def __init__(self, config):
        super().__init__(config)

    def success_mail(self, auto_ml_result):
        self.base_mail(header_name='AutoML ' + str(auto_ml_result.group_name),
                       text='Auto ML Review')
        self.mail.add_text(text='Tested features: ' + str(list(auto_ml_result.new_features.items())), n_line_breaks=1, )

        self._decision_info(auto_ml_result.deployment)
        self.mail.add_text(text='All results were exported to MLFlow', n_line_breaks=1, )
        self._metrics_description(auto_ml_result.compare_metrics_df)
        self._plots(auto_ml_result.figures)
        self.create_time_table(pd.DataFrame(pd.Series(auto_ml_result.run_time)))
        with open('EMail.html', 'w+') as f:
            f.write(self.mail.msg.as_string())


params = f"postgresql+psycopg2://mlflow:mlflowpassword@127.0.0.1:5433/mlflow"
grafana_db_connection = create_engine(params)


def parameters_for_optuna_all_models(trial):
    return {
        'iterations': trial.suggest_int('iterations', 10, 12, step=1),
        'depth': trial.suggest_int('depth', 1, 15, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
        'subsample': trial.suggest_float("subsample", 0.05, 1.0),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 101, step=10),
    }


def main():
    lgr = LogisticRegression()
    rf = RandomForestRegressor()
    models_dict = {'first': rf, 'second': lgr}
    auto_ml = AutoMLManager(auto_ml_config=auto_ml_config,
                            models_config=config_name,
                            business_metric=TitanicMetric(),
                            external_config=config,
                            extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                            compare_business_metric=BaseCompareBusinessMetric(calculate_threshold=True,
                                                                              use_exposure=False,
                                                                              metric_function=mean_absolute_error,
                                                                              direction='minimize'),
                            save_temp=False,
                            grafana_connection=grafana_db_connection,
                            hp_tune=True,
                            models_dict=models_dict,
                            retro=True
                            )
    auto_ml.update_models(send_mail=False, parameters_for_optuna={'first': parameters_for_optuna_all_models,
                                                                  'second': parameters_for_optuna_all_models})
    auto_ml.review(email=TitanicExampleEMail(config), send_mail=False)


if __name__ == "__main__":
    main()
