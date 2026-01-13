Quick Start Guide
=================

This guide will help you get started with OutBoxML by running a simple AutoML pipeline.

Basic Example
-------------

Here's a minimal example to get you started:

.. code-block:: python

    from outboxml.automl_manager import AutoMLManager
    import config

    # Initialize AutoML manager
    automl = AutoMLManager(
        auto_ml_config="configs/automl-config.json",
        models_config="configs/models-config.json",
        external_config=config,
        retro=True,
        hp_tune=True
    )

    # Run the AutoML pipeline
    result = automl.update_models(send_mail=False)

    # Check results
    print(f"Deployment decision: {result.deployment}")
    print(f"New features: {result.new_features}")
    print(f"Metrics: {result.compare_metrics_df}")

Configuration Files
-------------------

You'll need two configuration files:

1. **AutoML Configuration** (``automl-config.json``): Defines AutoML settings, feature selection, and hyperparameter tuning parameters.

2. **Models Configuration** (``models-config.json``): Defines the models to train, their features, and data sources.

Example Workflow
----------------

1. **Prepare your data**: Ensure your data is accessible (CSV, database, etc.)

2. **Create configuration files**: Define your models and AutoML settings

3. **Run AutoML**:

   .. code-block:: python

       automl = AutoMLManager(...)
       result = automl.update_models()

4. **Review results**: Check the results in MLflow, Grafana, or HTML reports

5. **Deploy models**: If deployment criteria are met, models are ready for production

Next Steps
----------

* See :doc:`modules/index` for detailed API documentation
* Check :doc:`examples/index` for more examples
* Read :doc:`contributing` to contribute to the project
