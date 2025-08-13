import os
from datetime import datetime

import pandas as pd
from loguru import logger

#from outboxml.automl_manager import AutoMLResult
#from outboxml.automl_manager import AutoMLResult
from outboxml.datasets_manager import DataSetsManager
from outboxml.dsml.mailing import Mail
from outboxml.export_results import ResultExport
from outboxml.plots import MLPlot, DataframeForPlots, CompareModelsPlot
from outboxml.core.enums import ResultNames


class EMail:
    def __init__(self,
                 config):
        self.config = config
        self.mail = Mail(config=self.config)
        self.email_receivers = self.config.email_receivers

    def save_mail_as_html(self, ):
        """
        Save a MIMEMultipart message as HTML

        """
        # Get HTML content (prefer HTML over plain text)
        html_content = None
        logger.info('Saving mail as html in results path')
        for part in self.mail.msg.walk():
            if part.get_content_type() == 'text/html':
                html_content = part.get_payload(decode=True).decode(errors='replace')
                break
        logger.info('Saving mail as html in results path')
        # Fallback to plain text if no HTML available
        if html_content is None:
            for part in self.mail.msg.walk():
                if part.get_content_type() == 'text/plain':
                    plain_content = part.get_payload(decode=True).decode(errors='replace')
                    html_content = f"<pre>{plain_content}</pre>"
                    break
        logger.info('Saving mail as html in results path')
        # If still no content found
        if html_content is None:
            html_content = "<p>No readable content found in email</p>"

        # Save to file
        with open(os.path.join(self.config.results_path, 'email.html'), 'w', encoding='utf-8') as f:
            logger.info('Saving mail as html in results path')
            f.write(html_content)

    def header(self, group_name: str = 'Test Email'):
        self.mail.add_email_subject(
            f"{group_name}",
        )
        self.mail.add_text(
            "Добрый день, это ИИ.",
            n_line_breaks=2,
        )
    def create_time_table(self, time_table):
        self.mail.add_text(
            "Затраченное время:",
            n_line_breaks=1,
        )
        self.mail.add_pandas_table(
            time_table.reset_index(),
            params=dict(text_align='right', font_family='sans-serif', width="250px"),
        )

        self.mail.add_line_breaks(1)

    def base_mail(self, header_name: str='Test Email', text: str = "Добрый день, это ИИ."):
        self.header(header_name)
        self.mail.add_text(
            text,
            n_line_breaks=2,
        )

    def add_image_to_mail(self, figure):
        self.mail.add_image(figure, size_pixel=(750, 500), n_line_breaks=1)

    def success_mail(self, **params):
        pass

    def send(self):
        self.save_mail_as_html()
        self.mail.send_mail(self.email_receivers)

    def common_error_mail(self, group_name: str, error):
        self.mail.add_email_subject(
            f"{group_name}. Автообновление моделей. Статус: ошибка.",
        )
        self.mail.add_text(
            "Добрый день, это ИИ.",
            n_line_breaks=2,
        )
        self.mail.add_text(
            f"Произошла техническая ошибка:",
            properties=['bold'],
            n_line_breaks=2,
        )
        self.mail.add_text(
            str(error),
            n_line_breaks=1,
        )

        self.send()

    def success_release_mail(self, group_name: str, new_features: dict=None):
        self.mail.add_email_subject(
            f"{group_name}. Автообновление моделей. Статус: залиты в gitlab.",
        )
        self.mail.add_text(
            "Добрый день, это ИИ.",
            n_line_breaks=2,
        )
        self.mail.add_text(
            "Модели залиты в gitlab.",
            properties=['bold'],
            n_line_breaks=1,
        )
        self.mail.add_text(
            f"{group_name}",
            n_line_breaks=1,
        )
        self.mail.add_line_breaks(1)
        if new_features is not None:
            for key in new_features.keys():
                if new_features[key] is not None or new_features[key] != []:
                    self.mail.add_text(f"В модель {key} добавлены новые фичи: " + str(new_features[key]),
                        n_line_breaks=1,
                    )
        self.mail.send_mail(self.email_receivers)


class EMailDSResult(EMail):
    def __init__(self, config,
                 ds_manager_result: dict):
        super().__init__(config)
        self._ds_manager_result = ds_manager_result


    def _metrics_description(self, ):
        #  result_export = ResultExport()
        self.mail.add_text(
            "Характеристики моделей:",
            n_line_breaks=1,
        )
        df = pd.DataFrame()

        for key in self._ds_manager_result.keys():
            model_config = self._ds_manager_result[key].config
            ds = DataSetsManager(config_name=model_config)
            ds._all_models_config = model_config
            res_export = ResultExport(ds_manager=ds)
            res_export.result = self._ds_manager_result[key]
            try:
                df1 = res_export.metrics_df(model_name=key,
                                            train_test='train',
                                            metrics_dict=
                                            self._ds_manager_result[key].metrics)
                df1 = df1.reset_index()[['index', 'full']]
                df1.columns = [ResultNames.metric, ResultNames.new_result_train]
                df2 = ResultExport(ds_manager=ds).metrics_df(model_name=key,
                                                             train_test='test',
                                                             metrics_dict=
                                                             self._ds_manager_result[
                                                                 key].metrics)
                df2 = df2.reset_index()[['index', 'full']]
                df2.columns = [ResultNames.metric, ResultNames.new_result_test]

                metrics_df = pd.concat([df1, df2['Новая модель||Тестовая выборка']], axis=1)
                metrics_df['Имя модели'] = key
                df = pd.concat([df, metrics_df])
            except Exception as exc:
                logger.error(exc)

        self.mail.add_pandas_table(df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )

    def _plots(self, ):

        self.mail.add_text('Графики когорт по новой модели:', n_line_breaks=1)

        for key in self._ds_manager_result.keys():
            y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(
                result=self._ds_manager_result[key],
                use_exposure=True)
            figure_cohort = MLPlot(model_name_1=key,
                                   y_graph=y_graph,
                                   features_categorical=features_categorical,
                                   features_numerical=features_numerical,
                                   show=False,
                                   use_exposure=True).make(plot_type=2,
                                                           cut_min_value=0.1,
                                                           cut_max_value=0.9,
                                                           samples=100,
                                                           cohort_base='model',
                                                           )

            figure_cohort.write_image(os.path.join(self.config.results_path, key + ' cohort.png'))
            with open(os.path.join(self.config.results_path, key + ' cohort.png'), "rb") as f:
                fig_cohort_png = f.read()
            self.mail.add_image(fig_cohort_png, size_pixel=(750, 500), n_line_breaks=1)

    def success_mail(self, group_name: str = 'Test Email'):
        self.header(group_name=group_name)
        self._metrics_description()
        self._plots()
        self.send()


class EMailDSCompareResult(EMailDSResult):
    def __init__(self, config,
                 ds_manager_result: dict,
                 ds_result_to_compare: dict):
        super().__init__(config, ds_manager_result)
        self._ds_result_to_compare = ds_result_to_compare

    def _metrics_description(self, ):
        self.mail.add_text(
            "Характеристики моделей:",
            n_line_breaks=1,
        )
        df = pd.DataFrame()

        for key in self._ds_manager_result.keys():
            model_config = self._ds_manager_result[key].config
            ds = DataSetsManager(config_name=model_config)
            ds._all_models_config = model_config
            res_export = ResultExport(ds_manager=ds)
            res_export.result = self._ds_manager_result
            try:
                metrics_df = res_export.compare_metrics(model_name=key,
                                                        ds_manager_result=self._ds_result_to_compare,
                                                        show=False, only_main=True)
                metrics_df['Имя модели'] = key
                df = pd.concat([df, metrics_df])
            except Exception as exc:
                logger.error(exc)
        self.mail.add_pandas_table(df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )

    def _plots(self, ):
        self.mail.add_text('Графики когорт по двум моделям:', n_line_breaks=1)

        for key in self._ds_manager_result.keys():
            y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(
                result=self._ds_manager_result[key],
                use_exposure=True)
            y_graph2, features_categorical2, features_numerical2 = DataframeForPlots().df_for_plots(
                result=self._ds_result_to_compare[key],
                use_exposure=True)

            figure_cohort = CompareModelsPlot(model_name=key,
                                              df1=y_graph,
                                              df2=y_graph2,
                                              features_categorical=features_categorical,
                                              features_numerical=features_numerical,
                                              show=False).make(plot_type=2,
                                                               cut_min_value=0.1,
                                                               cut_max_value=0.9,
                                                               samples=100,
                                                               cohort_base='model1',
                                                               )

            figure_cohort.write_image(os.path.join(self.config.results_path, key + ' cohort.png'))
            with open(os.path.join(self.config.results_path, key + ' cohort.png'), "rb") as f:
                fig_cohort_png = f.read()
            self.mail.add_image(fig_cohort_png, size_pixel=(750, 500), n_line_breaks=1)


class AutoMLReviewEMail(EMail):
    def __init__(self, config):
        super().__init__(config)

    def success_mail(self, auto_ml_result):
        self.base_mail(header_name='AutoML '+ str(auto_ml_result.group_name),
                       text='Отчёт по запуску самообучения')
        self.mail.add_text(text='Проверены фичи: ' + str(list(auto_ml_result.new_features.items())),  n_line_breaks=1,)

        self._decision_info(auto_ml_result.deployment)
        self.mail.add_text(text='Результаты выложены в MLFlow', n_line_breaks=1,)

        self._metrics_description(auto_ml_result.compare_metrics_df)
        self._plots(auto_ml_result.figures)
        self.create_time_table(pd.DataFrame(pd.Series(auto_ml_result.run_time)))
        self.send()


    def error_mail(self, group_name: str, error, status: dict):

        self.common_error_mail(group_name, error)
        self.mail.add_text(
            'Статус задач:',
            n_line_breaks=2,
        )
        self.mail.add_pandas_table(pd.DataFrame(pd.Series(status)).reset_index(),
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )
        self.send()

    def _decision_info(self, decision):
        if decision:
            self.mail.add_text(
                "Модель выведена в фон.",
                n_line_breaks=2,
            )
        else:
            self.mail.add_text(
                "Модель не обеспечила заданный критерий качества.",
                n_line_breaks=2,
            )

    def _metrics_description(self, compare_metrics_df):
        self.mail.add_text(
            "Характеристики моделей:",
            n_line_breaks=1,
        )

        self.mail.add_pandas_table(compare_metrics_df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )

    def _plots(self, figures):
        if figures is not None and figures != []:
            self.mail.add_text(
                'Графики по моделям:',
                n_line_breaks=2,
            )

            for key in figures.keys():
                figures[key].write_image(os.path.join(self.config.results_path, key + ' figure.png'))
                with open(os.path.join(self.config.results_path, key + ' figure.png'), "rb") as f:
                    fig_cohort_png = f.read()
                self.mail.add_image(fig_cohort_png, size_pixel=(750, 500), n_line_breaks=1)


class EMailMonitoring(EMail):
    def __init__(self, config):
        super().__init__(config)

    def success_mail(self, monitoring_result):
        self.base_mail(header_name=monitoring_result.group_name + str(' Monitoring'), text='Отчет по запуску мониторинга')
        self.mail.add_text(
            "Обнаружен дрифт в фичах:",
            n_line_breaks=1,
        )
        drift_df = self._prepare_drift_df(monitoring_result.report)

        self.mail.add_pandas_table(drift_df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )
        self.mail.add_text(
            "Полные результаты выложены в Grafana: " + str(monitoring_result.grafana_dashboard),
            n_line_breaks=1,
        )
        self.send()

    def error_mail(self, group_name: str, error):
        self.common_error_mail(group_name, error)

    def _prepare_drift_df(self, df):
        if df is None:
            print('Нет данных для отчета')
            return pd.DataFrame()
        else:
            alarm_df = df.loc[df['PSI'] > 0.3]
            df_to_send = alarm_df[['model_name', 'col', 'PSI', 'KL', 'JS', 'model_version']].sort_values(by='PSI', ascending=False)
            return df_to_send


class HTMLReport:
    def __init__(self,  config):
        self.config = config
        self.html_content = []
        self.report_path = os.path.join(config.results_path, "automl_report.html")

    def _add_section(self, title=None, text=None, n_line_breaks=1):
        """Add a section to the HTML report"""
        if title:
            self.html_content.append(f"<h2>{title}</h2>")
        if text:
            self.html_content.append(f"<p>{text}</p>")
        self.html_content.extend(["<br/>"] * n_line_breaks)

    def _add_table(self, df):
        """Add a Pandas DataFrame table to the report"""
        self.html_content.append(df.to_html(classes='dataframe', border=0,
                                            justify='right', index=False))

    def _add_plot(self, figure, plot_name):
        """Add a Plotly figure to the report"""
        if figure:
            plot_path = os.path.join(self.config.results_path, f"{plot_name}.html")
            figure.write_html(plot_path)
            self.html_content.append(f'<iframe src="{plot_path}" width="800" height="500"></iframe>')

    def save_report(self):
        """Save the compiled report to an HTML file"""
        full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AutoML Report</title>
                <style>
                    body {{ font-family: sans-serif; margin: 20px; }}
                    .dataframe {{ margin: 10px 0; }}
                    iframe {{ margin: 15px 0; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>AutoML Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
                {"".join(self.html_content)}
            </body>
            </html>
            """

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Report saved to: {self.report_path}")

    def success_report(self, auto_ml_result):
        """Generate a success report"""
        self._add_section(title=f'AutoML {auto_ml_result.group_name}',
                          text='Automated training run report')

        self._add_section(text='Features checked: ' + str(list(auto_ml_result.new_features.items())))

        self._decision_info(auto_ml_result.deployment)
        self._add_section(text='Results published to MLFlow')

        self._metrics_description(auto_ml_result.compare_metrics_df)
        self._plots(auto_ml_result.figures)
        self._add_table(pd.DataFrame(auto_ml_result.run_time, columns=["Run Time"]))

        self.save_report()

    def error_report(self, group_name: str, error, status: dict):
        """Generate an error report"""
        self._add_section(title=f'AutoML {group_name} Error',
                          text=str(error))

        self._add_section(text='Task status:')
        self._add_table(pd.DataFrame.from_dict(status, orient='index').reset_index())

        self.save_report()

    def _decision_info(self, decision):
        if decision:
            self._add_section(text="Model deployed to production.")
        else:
            self._add_section(text="Model didn't meet the required quality criteria.")

    def _metrics_description(self, compare_metrics_df):
        self._add_section(text="Model metrics comparison:")
        self._add_table(compare_metrics_df)

    def _plots(self, figures):
        if figures:
            self._add_section(text='Model visualizations:')
            for key, fig in figures.items():
                self._add_plot(fig, key)