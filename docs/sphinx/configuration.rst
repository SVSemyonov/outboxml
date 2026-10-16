Configuration
=============

OutBoxML uses two kinds of configuration:

* the **environment config** – paths, MLflow, database and e-mail settings of the
  machine where the framework runs. It is a Python module or object passed to the
  managers as ``external_config``; by default :mod:`outboxml.config` is used;
* **JSON configs** of models, AutoML and monitoring, validated by
  :mod:`outboxml.core.pydantic_models`. Most of their fields have defaults, see
  :doc:`modules/core/pydantic_models`.


Environment config
------------------

The managers read the environment config attributes when they need them:

.. code-block:: python

    from outboxml import config
    from outboxml.automl_manager import AutoMLManager

    AutoMLManager(auto_ml_config="configs/automl.json",
                  models_config="configs/models.json",
                  external_config=config)  # optional, outboxml.config by default

There are two ways to provide it.

Development: environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development, services and CI the default :mod:`outboxml.config` is used. It
reads environment variables with ``environs`` (``env_reader``); ``env_reader.read_env()``
also loads a ``.env`` file from the working directory. The file uses the dotenv
format, one ``KEY=value`` per line:

.. code-block:: bash

    mlflow_tracking_uri=http://localhost:5000
    mlflow_experiment=MyProject
    connection=postgresql+psycopg2://user:password@localhost:5432/db
    email_sender=automl@example.com
    email_login=automl
    email_pass=secret

.. note::

   ``.env`` is not Python code. A ``.env`` written as Python assignments
   (``name = value``) is not parsed and the defaults below are used.

.. list-table:: Settings of :mod:`outboxml.config`
   :header-rows: 1
   :widths: 25 25 50

   * - Attribute
     - Environment variable
     - Default / meaning
   * - ``work_type_fit``
     - ``work_type_fit``
     - ``CPU``; ``GPU`` to fit CatBoost/XGBoost on GPU
   * - ``work_type_hptune``
     - ``work_type_hptune``
     - ``CPU``; device for hyperparameter tuning
   * - ``results_path``
     - –
     - ``<repository>/results``: pickles, reports, logs
   * - ``prod_models_folder``
     - ``prod_models_folder``
     - ``prod_models_from_mlflow``
   * - ``prod_models_path``
     - –
     - ``<repository>/<prod_models_folder>``: models used by the prediction service
   * - ``mlflow_tracking_uri``
     - ``mlflow_tracking_uri``
     - ``http://mlflow:5000`` (the MLflow container of the docker stack); a local
       store works too, e.g. ``file:///path/to/mlruns``
   * - ``mlflow_experiment``
     - ``mlflow_experiment``
     - ``FrameworkTest``; used when the AutoML config has no ``mlflow_experiment``
   * - ``mlflow_user``, ``mlflow_password``
     - same names
     - empty; credentials for MLflow behind KeyCloak
   * - ``connection_params``
     - ``connection``
     - Postgres of the docker stack; SQLAlchemy URL for database sources, Grafana
       export and the AutoML trigger
   * - ``email_sender``, ``email_login``, ``email_pass``
     - same names
     - empty; SMTP sender and credentials
   * - ``email_smtp_server``, ``email_port``, ``email_receivers``
     - –
     - not read from the environment: ``""``, ``2525``, ``[]``; set them in your
       own config module or object

Jupyter examples: a config module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Jupyter container of the docker stack has no ``.env`` file, so the examples use
a Python module with the same attribute names and pass it as ``external_config``,
e.g. ``examples/titanic/external_config_for_titanic_example.py``:

.. code-block:: python

    # my_config.py
    from pathlib import Path

    base_path = Path(__file__).resolve().parent
    results_path = base_path / "results"
    prod_models_folder = "prod_models_from_mlflow"
    prod_models_path = base_path / prod_models_folder
    prod_path = base_path

    work_type_fit = "CPU"
    work_type_hptune = "CPU"

    mlflow_tracking_uri = "http://mlflow:5000"
    mlflow_experiment = "TitanicExample"

    connection_params = "postgresql+psycopg2://mlflow:mlflowpassword@postgre:5432/mlflow"

    email_smtp_server = ""
    email_port = 2525
    email_sender = ""
    email_login = ""
    email_pass = ""
    email_receivers = []

.. code-block:: python

    import my_config
    from outboxml.automl_manager import AutoMLManager

    AutoMLManager(auto_ml_config="configs/automl-titanic.json",
                  models_config="configs/config-example-titanic.json",
                  external_config=my_config).update_models()

Any object with these attributes works, e.g. ``types.SimpleNamespace``, which is
convenient in tests.


JSON configs
------------

* **Models config** (:class:`~outboxml.core.pydantic_models.AllModelsConfig`):
  data source, train/test split and models with their features. Only
  ``group_name``, ``data_config`` and ``models_configs`` are required; a feature
  needs only ``name`` (``{"_TYPE_": "_NUM_"}`` in ``replace`` marks it numerical).
* **AutoML config** (:class:`~outboxml.core.pydantic_models.AutoMLConfig`): may be
  ``{}``. ``group_name`` and ``project`` default to the models config,
  ``mlflow_experiment`` to the environment config.
* **Monitoring config** (:class:`~outboxml.core.pydantic_models.MonitoringConfig`):
  only ``group_name`` is required.

Fields that are not used anymore are accepted with a ``DeprecationWarning``.
