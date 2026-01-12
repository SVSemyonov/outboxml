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
    """Base class for email notifications in AutoML processes.
    
    Provides core functionality for creating and sending HTML emails with tables,
    images, and formatted text. All emails are saved as HTML files before sending.
    
    :param config: Configuration object with email settings and paths.
    :type config: object
    
    :var config: Configuration object with email settings and paths.
    :var mail: Mail instance for low-level email operations.
    :var email_receivers: List of recipient email addresses.
    
    .. note::
        All emails are automatically saved as HTML files in the results_path
        directory before being sent.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        import config
        from outboxml.core.email import EMail
        email = EMail(config=config)
        email.header("Test Email")
        email.mail.add_text("This is a test message.")
        email.send()
    """
    def __init__(self,
                 config):
        """Initialize EMail instance.
        
        :param config: Configuration object containing:
            - email_smtp_server: SMTP server address
            - email_port: SMTP server port
            - email_sender: Sender email address
            - email_login: SMTP login username
            - email_pass: SMTP password
            - email_receivers: List of recipient email addresses
            - results_path: Path to save email HTML files
        :type config: object
        """
        self.config = config
        self.mail = Mail(config=self.config)
        self.email_receivers = self.config.email_receivers

    def save_mail_as_html(self, ):
        """Save the current email message as an HTML file.
        
        Extracts HTML content from the email message and saves it to disk.
        Falls back to plain text if HTML is not available. The file is saved
        as 'email.html' in the results_path directory.
        
        :return: None
        :rtype: None
        
        .. note::
            Creates or overwrites 'email.html' in config.results_path.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            email.header("Test Email")
            email.mail.add_text("This is a test message.")
            email.save_mail_as_html()
            # Creates: <results_path>/email.html
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
        """Add a header section to the email with subject and greeting.
        
        Sets the email subject to the group name and adds a standard greeting
        text in Russian.
        
        :param group_name: Name of the model group or experiment.
        :type group_name: str
        :param group_name: Defaults to 'Test Email'
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            email.header(group_name="Titanic_Model_v1")
            # Sets email subject to "Titanic_Model_v1"
            # Adds greeting text "Добрый день, это ИИ."
        """
        self.mail.add_email_subject(
            f"{group_name}",
        )
        self.mail.add_text(
            "Добрый день, это ИИ.",
            n_line_breaks=2,
        )
    def create_time_table(self, time_table):
        """Add a time table to the email showing execution times.
        
        Adds a formatted table displaying execution times for different stages
        of the AutoML process. The table is right-aligned with sans-serif font.
        
        :param time_table: DataFrame with time information. The index will be
            reset to include it as a column in the table.
        :type time_table: pandas.DataFrame
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            from datetime import datetime
            time_data = pd.DataFrame({
                'Time': [datetime.now(), datetime.now()]
            }, index=['Loading dataset', 'Fitting'])
            email = EMail(config=config)
            email.create_time_table(time_data)
            # Adds "Затраченное время:" text and formatted table
        """
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
        """Create a basic email with header and custom text.
        
        Convenience method that combines header() and text addition in one call.
        
        :param header_name: Email subject/header. Defaults to 'Test Email'.
        :type header_name: str
        :param text: Main text content. Defaults to "Добрый день, это ИИ.".
        :type text: str
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            email.base_mail(
                header_name="AutoML Training Complete",
                text="The model training has finished successfully."
            )
        """
        self.header(header_name)
        self.mail.add_text(
            text,
            n_line_breaks=2,
        )

    def add_image_to_mail(self, figure):
        """Add an image to the email body.
        
        Embeds an image in the email with a fixed size of 750x500 pixels.
        The image must be provided as bytes.
        
        :param figure: Image data as bytes. Can be obtained from reading an image
            file in binary mode or from a Plotly figure's write_image() method.
        :type figure: bytes
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            with open("plot.png", "rb") as f:
                image_bytes = f.read()
            email = EMail(config=config)
            email.add_image_to_mail(image_bytes)
            # Adds image to email with size 750x500 pixels
        """
        self.mail.add_image(figure, size_pixel=(750, 500), n_line_breaks=1)

    def success_mail(self, **params):
        """Placeholder method for success email notifications.
        
        This method should be overridden in subclasses to provide specific
        success email implementations.
        
        :param **params: Variable keyword arguments (not used in base implementation).
        :return: None
        :rtype: None
        
        .. note::
            This is a placeholder method. Subclasses should override it with
            specific implementations.
        """
        pass

    def send(self):
        """Save the email as HTML and send it to configured recipients.
        
        First saves the email as an HTML file, then sends it via SMTP to all
        recipients specified in config.email_receivers.
        
        :return: None
        :rtype: None
        
        .. note::
            - Saves email as HTML file in results_path
            - Sends email via SMTP to all recipients
            
        :raises smtplib.SMTPException: If email sending fails (HTML file is still saved).
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            email.header("Test Email")
            email.mail.add_text("This is a test message.")
            email.send()
            # Saves email.html and sends email to recipients
        """
        self.save_mail_as_html()
        self.mail.send_mail(self.email_receivers)

    def common_error_mail(self, group_name: str, error):
        """Send an error notification email with error details.
        
        Creates and sends an error notification email with the error message
        and a standard error subject line.
        
        :param group_name: Name of the model group or experiment that failed.
        :type group_name: str
        :param error: Error object or error message string. Will be converted to
            string for display.
        :type error: Exception or str
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            try:
                result = 1 / 0
            except Exception as e:
                email.common_error_mail(
                    group_name="Titanic_Model_v1",
                    error=e
                )
            # Sends email with subject: "Titanic_Model_v1. Автообновление моделей. Статус: ошибка."
        """
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
        """Send a notification email when models are successfully released to GitLab.
        
        Creates and sends an email notification when models have been successfully
        released to GitLab, optionally including information about new features
        added to each model.
        
        :param group_name: Name of the model group that was released.
        :type group_name: str
        :param new_features: Optional dictionary mapping model names to lists of new
            features. Format: {model_name: [feature1, feature2, ...]}. Defaults to None.
        :type new_features: dict, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMail(config=config)
            new_features = {
                "model1": ["feature_a", "feature_b"],
                "model2": ["feature_c"]
            }
            email.success_release_mail(
                group_name="Titanic_Model_v1",
                new_features=new_features
            )
            # Sends email with subject: "Titanic_Model_v1. Автообновление моделей. Статус: залиты в gitlab."
            # Lists new features for each model
        """
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
    """Email class for sending results from DataSetsManager.
    
    Extends EMail to include model metrics tables and cohort plots in the email.
    Used for reporting results after model training.
    
    :param config: Configuration object (same as EMail).
    :type config: object
    :param ds_manager_result: Dictionary mapping model names to DSManagerResult
        objects containing model results, metrics, and predictions.
    :type ds_manager_result: dict
    
    :var _ds_manager_result: Dictionary mapping model names to DSManagerResult
        objects containing model results, metrics, and predictions.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.core.email import EMailDSResult
        from outboxml.datasets_manager import DataSetsManager
        # Assume ds_manager has been trained
        results = ds_manager.get_result()  # Returns dict[str, DSManagerResult]
        email = EMailDSResult(
            config=config,
            ds_manager_result=results
        )
        email.success_mail(group_name="Titanic_Model_v1")
    """
    def __init__(self, config,
                 ds_manager_result: dict):
        """Initialize EMailDSResult instance.
        
        :param config: Configuration object (same as EMail).
        :type config: object
        :param ds_manager_result: Dictionary mapping model names to DSManagerResult
            objects. Each DSManagerResult should contain:
            - config: Model configuration
            - metrics: Dictionary with 'train' and 'test' keys containing metrics
            - Other model result data
        :type ds_manager_result: dict
        """
        super().__init__(config)
        self._ds_manager_result = ds_manager_result


    def _metrics_description(self, ):
        """Generate and add a metrics table to the email.
        
        Creates a formatted table showing train and test metrics for all models
        in the ds_manager_result. The table includes metric names, training set
        results, test set results, and model names.
        
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_mail().
            Errors during metrics extraction are logged but don't stop the process.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailDSResult(config=config, ds_manager_result=results)
            email._metrics_description()
            # Adds table with columns:
            # - Metric name
            # - New model || Training set
            # - New model || Test set
            # - Model name
        """
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
        """Generate cohort plots for each model and add them to the email.
        
        Creates cohort analysis plots for each model showing performance across
        different feature segments. Plots are saved as PNG files and embedded
        in the email.
        
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_mail().
            For each model, saves "<model_name> cohort.png" in results_path.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailDSResult(config=config, ds_manager_result=results)
            email._plots()
            # For each model:
            # - Generates cohort plot
            # - Saves as "<model_name> cohort.png" in results_path
            # - Adds image to email (750x500 pixels)
        """
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
        """Send a success email with model metrics and cohort plots.
        
        Creates and sends a comprehensive success email containing:
        - Header with group name
        - Metrics table for all models (train and test)
        - Cohort plots for each model
        
        :param group_name: Name of the model group. Defaults to 'Test Email'.
        :type group_name: str
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailDSResult(config=config, ds_manager_result=results)
            email.success_mail(group_name="Titanic_Model_v1")
            # Sends email containing:
            # - Header with group name
            # - Metrics table for all models
            # - Cohort plots for each model
        """
        self.header(group_name=group_name)
        self._metrics_description()
        self._plots()
        self.send()


class EMailDSCompareResult(EMailDSResult):
    """Email class for comparing two sets of model results.
    
    Extends EMailDSResult to include comparison metrics and plots between
    current and previous model versions. Used for retroactive analysis and
    model version comparison.
    
    :param config: Configuration object (same as EMail).
    :type config: object
    :param ds_manager_result: Dictionary mapping model names to current
        DSManagerResult objects.
    :type ds_manager_result: dict
    :param ds_result_to_compare: Dictionary mapping model names to previous
        DSManagerResult objects for comparison.
    :type ds_result_to_compare: dict
    
    :var _ds_result_to_compare: Dictionary mapping model names to previous
        DSManagerResult objects for comparison.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.core.email import EMailDSCompareResult
        current_results = ds_manager.get_result()
        previous_results = load_previous_results()  # Your function
        email = EMailDSCompareResult(
            config=config,
            ds_manager_result=current_results,
            ds_result_to_compare=previous_results
        )
    """
    def __init__(self, config,
                 ds_manager_result: dict,
                 ds_result_to_compare: dict):
        """Initialize EMailDSCompareResult instance.
        
        :param config: Configuration object (same as EMail).
        :type config: object
        :param ds_manager_result: Dictionary mapping model names to current
            DSManagerResult objects.
        :type ds_manager_result: dict
        :param ds_result_to_compare: Dictionary mapping model names to previous
            DSManagerResult objects for comparison.
        :type ds_result_to_compare: dict
        """
        super().__init__(config, ds_manager_result)
        self._ds_result_to_compare = ds_result_to_compare

    def _metrics_description(self, ):
        """Generate comparison metrics table showing differences between models.
        
        Overrides parent method to create a comparison table showing current
        model metrics vs previous model metrics, including differences.
        
        :return: None
        :rtype: None
        
        .. note::
            This method overrides the parent _metrics_description() method.
            Errors during metrics extraction are logged but don't stop the process.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailDSCompareResult(
                config=config,
                ds_manager_result=current_results,
                ds_result_to_compare=previous_results
            )
            email._metrics_description()
            # Adds table comparing:
            # - Current model metrics (train/test)
            # - Previous model metrics (train/test)
            # - Differences
        """
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
        """Generate comparison cohort plots for current vs previous models.
        
        Overrides parent method to create comparison cohort plots showing
        performance differences between current and previous model versions.
        
        :return: None
        :rtype: None
        
        .. note::
            This method overrides the parent _plots() method.
            For each model, saves "<model_name> cohort.png" in results_path.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailDSCompareResult(
                config=config,
                ds_manager_result=current_results,
                ds_result_to_compare=previous_results
            )
            email._plots()
            # For each model:
            # - Generates comparison cohort plot
            # - Shows current vs previous model performance
            # - Saves as "<model_name> cohort.png"
            # - Adds to email (750x500 pixels)
        """
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
    """Email class for AutoML review reports.
    
    Extends EMail to provide comprehensive AutoML execution reports including
    feature selection results, deployment decisions, metrics comparisons, and
    execution times. Used by AutoMLManager to send review emails after
    automated training runs.
    
    :param config: Configuration object (same as EMail).
    :type config: object
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.automl_manager import AutoMLManager
        from outboxml.core.email import AutoMLReviewEMail
        automl = AutoMLManager(...)
        result = automl.update_models()
        email = AutoMLReviewEMail(config=config)
        email.success_mail(result)
    """
    def __init__(self, config):
        """Initialize AutoMLReviewEMail instance.
        
        :param config: Configuration object (same as EMail).
        :type config: object
        """
        super().__init__(config)

    def success_mail(self, auto_ml_result):
        """Send a comprehensive success report email for AutoML execution.
        
        Creates and sends a detailed report email containing:
        - AutoML execution summary
        - Features checked during feature selection
        - Deployment decision (deployed or not)
        - Metrics comparison table
        - Model plots/visualizations
        - Execution time table for each stage
        
        :param auto_ml_result: AutoMLResult object from AutoMLManager.update_models()
            containing:
            - group_name: Model group name
            - new_features: Dictionary of new features per model
            - deployment: Boolean deployment decision
            - compare_metrics_df: DataFrame with metrics comparison
            - figures: Dictionary of Plotly figures
            - run_time: Dictionary of execution times
        :type auto_ml_result: AutoMLResult
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            from outboxml.automl_manager import AutoMLManager
            from outboxml.core.email import AutoMLReviewEMail
            automl = AutoMLManager(...)
            result = automl.update_models()
            email = AutoMLReviewEMail(config=config)
            email.success_mail(result)
            # Sends email with complete AutoML execution report
        """
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
        """Send an error report email with error details and task status.
        
        Creates and sends an error notification email including the error message
        and a table showing which tasks completed successfully and which failed.
        
        :param group_name: Name of the model group that failed.
        :type group_name: str
        :param error: Error object or error message string.
        :type error: Exception or str
        :param status: Dictionary mapping task names to completion status (boolean
            values). Example: {'Loading dataset': True, 'Feature selection': False}
        :type status: dict
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            status = {
                'Loading dataset': True,
                'Feature selection': False,
                'Fitting': False
            }
            email = AutoMLReviewEMail(config=config)
            email.error_mail(
                group_name="Titanic_Model_v1",
                error=ValueError("Invalid configuration"),
                status=status
            )
            # Sends email with error message and task status table
        """
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
        """Add deployment decision information to the email.
        
        Adds text indicating whether the model was deployed to production or
        rejected based on quality criteria.
        
        :param decision: Boolean deployment decision. True means deployed, False
            means not deployed.
        :type decision: bool
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_mail().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = AutoMLReviewEMail(config=config)
            email._decision_info(decision=True)
            # Adds text: "Модель выведена в фон."
            email._decision_info(decision=False)
            # Adds text: "Модель не обеспечила заданный критерий качества."
        """
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
        """Add metrics comparison table to the email.
        
        Adds a formatted table containing metrics comparison data to the email
        body. The table is right-aligned with sans-serif font.
        
        :param compare_metrics_df: DataFrame containing metrics comparison
            data. Expected columns may include:
            - Имя модели (Model name)
            - Метрика (Metric name)
            - Новая модель||Тренировочная выборка (New model train results)
            - Новая модель||Тестовая выборка (New model test results)
            - Предыдущая модель||Тренировочная выборка (Previous model train)
            - Предыдущая модель||Тестовая выборка (Previous model test)
        :type compare_metrics_df: pandas.DataFrame
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_mail().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            metrics_df = pd.DataFrame({
                'Имя модели': ['model1', 'model2'],
                'Метрика': ['RMSE', 'MAE'],
                'Новая модель||Тренировочная выборка': [0.5, 0.3],
                'Новая модель||Тестовая выборка': [0.6, 0.4]
            })
            email = AutoMLReviewEMail(config=config)
            email._metrics_description(metrics_df)
            # Adds formatted metrics table to email
        """
        self.mail.add_text(
            "Характеристики моделей:",
            n_line_breaks=1,
        )

        self.mail.add_pandas_table(compare_metrics_df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )

    def _plots(self, figures):
        """Add model plots to the email.
        
        Saves Plotly figures as PNG images and embeds them in the email.
        Each figure is saved with the model name as part of the filename.
        
        :param figures: Dictionary mapping model names to Plotly figure objects.
            Can be None or empty, in which case nothing is added.
        :type figures: dict or None
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_mail().
            For each figure, saves "<model_name> figure.png" in results_path.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            figures = {
                'model1': plotly_figure1,
                'model2': plotly_figure2
            }
            email = AutoMLReviewEMail(config=config)
            email._plots(figures)
            # For each figure:
            # - Saves as "<model_name> figure.png"
            # - Adds to email (750x500 pixels)
        """
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
    """Email class for monitoring reports.
    
    Extends EMail to provide data drift and monitoring alerts. Used to notify
    about data drift detection results and monitoring service status.
    
    :param config: Configuration object (same as EMail).
    :type config: object
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.monitoring_result import MonitoringResult
        from outboxml.core.email import EMailMonitoring
        # Assume monitoring_result is created by monitoring process
        email = EMailMonitoring(config=config)
        email.success_mail(monitoring_result)
    """
    def __init__(self, config):
        """Initialize EMailMonitoring instance.
        
        :param config: Configuration object (same as EMail).
        :type config: object
        """
        super().__init__(config)

    def success_mail(self, monitoring_result):
        """Send a monitoring success report email with data drift information.
        
        Creates and sends a monitoring report email containing:
        - Monitoring summary
        - Data drift table (features with PSI > 0.3)
        - Link to Grafana dashboard
        
        :param monitoring_result: MonitoringResult object containing:
            - group_name: Model group name
            - report: DataFrame with drift metrics (PSI, KL, JS)
            - grafana_dashboard: URL to Grafana dashboard
        :type monitoring_result: MonitoringResult
        :return: None
        :rtype: None
        
        .. note::
            Only features with PSI > 0.3 are included in the drift table.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            from outboxml.monitoring_result import MonitoringResult
            from outboxml.core.email import EMailMonitoring
            # Assume monitoring_result is created by monitoring process
            email = EMailMonitoring(config=config)
            email.success_mail(monitoring_result)
            # Sends email with monitoring summary and drift table
        """
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
        """Send an error notification email for monitoring failures.
        
        Creates and sends an error notification email when monitoring fails.
        This is a convenience method that calls the parent's common_error_mail().
        
        :param group_name: Name of the model group that failed monitoring.
        :type group_name: str
        :param error: Error object or error message string.
        :type error: Exception or str
        :return: None
        :rtype: None
        
        .. note::
            This method calls common_error_mail() from the parent class.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = EMailMonitoring(config=config)
            email.error_mail(
                group_name="Titanic_Model_v1",
                error="Connection timeout to monitoring service"
            )
            # Sends common error email
        """
        self.common_error_mail(group_name, error)

    def _prepare_drift_df(self, df):
        """Prepare the drift DataFrame by filtering features with significant drift.
        
        Filters the drift DataFrame to include only features with PSI > 0.3,
        which indicates significant data drift. The result is sorted by PSI
        in descending order.
        
        :param df: DataFrame with drift metrics. Expected columns:
            - model_name: Model name
            - col: Feature name
            - PSI: Population Stability Index
            - KL: Kullback-Leibler divergence
            - JS: Jensen-Shannon divergence
            - model_version: Model version
        :type df: pandas.DataFrame or None
        :return: Filtered DataFrame containing only features with PSI > 0.3,
            sorted by PSI in descending order. Returns empty DataFrame if input is None.
        :rtype: pandas.DataFrame
        
        .. note::
            This is a private method, typically called internally by success_mail().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            drift_data = pd.DataFrame({
                'model_name': ['model1', 'model1', 'model2'],
                'col': ['feature_a', 'feature_b', 'feature_c'],
                'PSI': [0.5, 0.2, 0.4],
                'KL': [0.3, 0.1, 0.25],
                'JS': [0.15, 0.05, 0.12],
                'model_version': ['v1', 'v1', 'v2']
            })
            email = EMailMonitoring(config=config)
            filtered_df = email._prepare_drift_df(drift_data)
            # Returns DataFrame with only feature_a and feature_c (PSI > 0.3)
            # Sorted by PSI descending
        """
        if df is None:
            print('Нет данных для отчета')
            return pd.DataFrame()
        else:
            alarm_df = df.loc[df['PSI'] > 0.3]
            df_to_send = alarm_df[['model_name', 'col', 'PSI', 'KL', 'JS', 'model_version']].sort_values(by='PSI', ascending=False)
            return df_to_send


class HTMLReport:
    """Class for generating HTML reports instead of sending emails.
    
    Useful for local review or when email is not configured. Generates standalone
    HTML reports that can be opened in any browser. Reports include sections,
    tables, and embedded Plotly visualizations.
    
    :param config: Configuration object with results_path attribute.
    :type config: object
    
    :var config: Configuration object with results_path attribute.
    :var html_content: List of HTML content strings being assembled.
    :var report_path: Full path to the output HTML report file.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.core.email import HTMLReport
        report = HTMLReport(config=config)
        report.success_report(auto_ml_result)
        # Report saved to: <results_path>/automl_report.html
    """
    def __init__(self,  config):
        """Initialize HTMLReport instance.
        
        :param config: Configuration object containing results_path attribute.
        :type config: object
        """
        self.config = config
        self.html_content = []
        self.report_path = os.path.join(config.results_path, "automl_report.html")

    def _add_section(self, title=None, text=None, n_line_breaks=1):
        """Add a section (title and/or text) to the HTML report.
        
        Adds HTML formatted section with optional title (rendered as <h2>) and
        text (rendered as <p>), followed by line breaks.
        
        :param title: Optional section title (rendered as <h2>). Defaults to None.
        :type title: str, optional
        :param text: Optional section text (rendered as <p>). Defaults to None.
        :type text: str, optional
        :param n_line_breaks: Number of line breaks after the section. Defaults to 1.
        :type n_line_breaks: int
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by other methods.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            report = HTMLReport(config=config)
            report._add_section(
                title="Model Training Results",
                text="The models have been trained successfully.",
                n_line_breaks=2
            )
        """
        if title:
            self.html_content.append(f"<h2>{title}</h2>")
        if text:
            self.html_content.append(f"<p>{text}</p>")
        self.html_content.extend(["<br/>"] * n_line_breaks)

    def _add_table(self, df):
        """Add a Pandas DataFrame as an HTML table to the report.
        
        Converts a pandas DataFrame to HTML table format and adds it to the
        report content. The table is right-aligned with no border and no index.
        
        :param df: DataFrame to convert to HTML table.
        :type df: pandas.DataFrame
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by other methods.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            metrics_df = pd.DataFrame({
                'Model': ['model1', 'model2'],
                'RMSE': [0.5, 0.6],
                'MAE': [0.3, 0.4]
            })
            report = HTMLReport(config=config)
            report._add_table(metrics_df)
            # Adds HTML table to report content
        """
        self.html_content.append(df.to_html(classes='dataframe', border=0,
                                            justify='right', index=False))

    def _add_plot(self, figure, plot_name):
        """Add a Plotly figure to the report as an embedded iframe.
        
        Saves a Plotly figure as an HTML file and embeds it in the report using
        an iframe. The figure is saved with the provided plot_name.
        
        :param figure: Plotly figure object (plotly.graph_objects.Figure).
        :type figure: plotly.graph_objects.Figure
        :param plot_name: Name for the plot file (without extension). The file
            will be saved as "<plot_name>.html".
        :type plot_name: str
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by other methods.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Bar(x=['A', 'B'], y=[1, 2]))
            report = HTMLReport(config=config)
            report._add_plot(fig, "bar_chart")
            # Saves: <results_path>/bar_chart.html
            # Adds: <iframe src="bar_chart.html" width="800" height="500"></iframe>
        """
        if figure:
            plot_path = os.path.join(self.config.results_path, f"{plot_name}.html")
            figure.write_html(plot_path)
            self.html_content.append(f'<iframe src="{plot_path}" width="800" height="500"></iframe>')

    def save_report(self):
        """Save the compiled HTML report to disk.
        
        Compiles all HTML content into a complete HTML document with styling
        and saves it to the report_path. The report includes a timestamp in
        the title.
        
        :return: None
        :rtype: None
        
        .. note::
            - Creates or overwrites automl_report.html in config.results_path
            - Prints confirmation message to console
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            report = HTMLReport(config=config)
            report._add_section(title="Test Report", text="This is a test.")
            report.save_report()
            # Creates: <results_path>/automl_report.html
            # Prints: "Report saved to: <results_path>/automl_report.html"
        """
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
        """Generate a complete success report for AutoML execution.
        
        Creates a comprehensive HTML report containing:
        - AutoML execution summary
        - Features checked during feature selection
        - Deployment decision (deployed or not)
        - Metrics comparison table
        - Model plots/visualizations
        - Execution time table for each stage
        
        :param auto_ml_result: AutoMLResult object from AutoMLManager.update_models()
            containing:
            - group_name: Model group name
            - new_features: Dictionary of new features per model
            - deployment: Boolean deployment decision
            - compare_metrics_df: DataFrame with metrics comparison
            - figures: Dictionary of Plotly figures
            - run_time: Dictionary of execution times
        :type auto_ml_result: AutoMLResult
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            from outboxml.automl_manager import AutoMLManager
            from outboxml.core.email import HTMLReport
            automl = AutoMLManager(...)
            result = automl.update_models()
            report = HTMLReport(config=config)
            report.success_report(result)
            # Generates HTML report with complete AutoML execution details
            # Saves to: <results_path>/automl_report.html
        """
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
        """Generate an error report for failed AutoML execution.
        
        Creates an HTML report documenting the error and task completion status.
        Useful for debugging and tracking which stages completed successfully.
        
        :param group_name: Name of the model group that failed.
        :type group_name: str
        :param error: Error object or error message string.
        :type error: Exception or str
        :param status: Dictionary mapping task names to completion status (boolean
            values). Example: {'Loading dataset': True, 'Feature selection': False}
        :type status: dict
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            status = {
                'Loading dataset': True,
                'Feature selection': False,
                'Fitting': False
            }
            report = HTMLReport(config=config)
            report.error_report(
                group_name="Titanic_Model_v1",
                error="Configuration validation failed",
                status=status
            )
            # Generates HTML report with error details and task status
            # Saves to: <results_path>/automl_report.html
        """
        self._add_section(title=f'AutoML {group_name} Error',
                          text=str(error))

        self._add_section(text='Task status:')
        self._add_table(pd.DataFrame.from_dict(status, orient='index').reset_index())

        self.save_report()

    def _decision_info(self, decision):
        """Add deployment decision information to the report.
        
        Adds text indicating whether the model was deployed to production or
        rejected based on quality criteria.
        
        :param decision: Boolean deployment decision. True means deployed, False
            means not deployed.
        :type decision: bool
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_report().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            report = HTMLReport(config=config)
            report._decision_info(decision=True)
            # Adds: "Model deployed to production."
            report._decision_info(decision=False)
            # Adds: "Model didn't meet the required quality criteria."
        """
        if decision:
            self._add_section(text="Model deployed to production.")
        else:
            self._add_section(text="Model didn't meet the required quality criteria.")

    def _metrics_description(self, compare_metrics_df):
        """Add metrics comparison table to the report.
        
        Adds a section header and formatted table containing metrics comparison
        data to the report.
        
        :param compare_metrics_df: DataFrame containing metrics comparison
            data. Expected columns may include:
            - Имя модели (Model name)
            - Метрика (Metric name)
            - Новая модель||Тренировочная выборка (New model train results)
            - Новая модель||Тестовая выборка (New model test results)
            - Предыдущая модель||Тренировочная выборка (Previous model train)
            - Предыдущая модель||Тестовая выборка (Previous model test)
        :type compare_metrics_df: pandas.DataFrame
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_report().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            metrics_df = pd.DataFrame({
                'Имя модели': ['model1'],
                'Метрика': ['RMSE'],
                'Новая модель||Тренировочная выборка': [0.5],
                'Новая модель||Тестовая выборка': [0.6]
            })
            report = HTMLReport(config=config)
            report._metrics_description(metrics_df)
            # Adds section header and HTML table to report
        """
        self._add_section(text="Model metrics comparison:")
        self._add_table(compare_metrics_df)

    def _plots(self, figures):
        """Add model plots to the report.
        
        Saves Plotly figures as HTML files and embeds them in the report using
        iframes. Each figure is saved with the model name as the filename.
        
        :param figures: Dictionary mapping model names to Plotly figure objects.
            Can be None or empty, in which case nothing is added.
        :type figures: dict or None
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by success_report().
            If figures is None or empty, nothing is added.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            figures = {
                'model1': plotly_figure1,
                'model2': plotly_figure2
            }
            report = HTMLReport(config=config)
            report._plots(figures)
            # For each figure:
            # - Saves as "<model_name>.html"
            # - Adds iframe to report
        """
        if figures:
            self._add_section(text='Model visualizations:')
            for key, fig in figures.items():
                self._add_plot(fig, key)