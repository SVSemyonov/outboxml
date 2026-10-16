# Changelog


This project is based on [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) .

---

## [Unreleased]

### Added
- Test suite runs without Docker: pytest with an isolated config, a local MLflow file store and SQLite
  instead of the MLflow server and Postgres; test coverage 88%
- CI: tests with a coverage gate (85%) on pull requests into `develop`; release workflow that publishes
  to PyPI and creates a GitHub Release on merges into `release`
- `test` extra (`pip install -e ".[test]"`)
- `outboxml.core.errors.DataLoadError` for data loading errors
- Defaults for config fields: an AutoML config can be `{}`, a monitoring config `{"group_name": ...}`;
  data split, feature `replace`/`default`, `project`/`version` have defaults

### Changed
- `BaseExtractor` no longer writes local datasets to the database and creates no database triggers;
  the Titanic AutoML trigger demo creates its table trigger in `init.sql`
- MLflow and KeyCloak are contacted on the first run, not when `AutoMLManager` is created
- AutoML MLflow runs are named `{group_name}_{YYYY_mm_dd_HH_MM_SS}`
- `WoE_num_to_cat` features are encoded to string interval labels
- `FeatureSelectionConfig.top_feautures_to_select` is renamed to `top_features_to_select`
- Only the `outboxml` package is included in distributions
- `polars` is pinned to 1.35.2 (1.35.0 can no longer be installed)

### Deprecated
- Unused config fields, accepted with a `DeprecationWarning`: `DataModelConfig.showGraphs`, `processing`;
  `DataConfig.queries`; `ModelInferenceConfig.prod_models_folder`, `prod_path`; `AutoMLConfig.dashboard_name`;
  `MonitoringConfig.dashboard_name`, `extrapolation_period`, `target_column`;
  the old name `top_feautures_to_select`

### Fixed
- Deployment: a criterion metric missing from the comparison no longer deploys the model
- AutoML with `send_mail=True` sends one mail and logs results to MLflow
- One-row predictions with `WoE_num_to_cat` features use the categories seen in training;
  CatBoost can train on them
- `one_model_predict` no longer modifies the caller's data
- Feature selection with `use_temp_data` prepares candidate features
- Polars engine: duplicated columns and missing `ModelDataSubset.X`
- Release from MLflow (`MLFLowRelease`, `Release`) finds and downloads AutoML runs
- Email reports: saved `email.html` was empty, DS result mails failed, AutoML errors were sent as two mails
- `catboost_over_glm` export/import, HP tuning model class and default metric, integer search spaces
  without `step`, feature importance for unsupported models, `/api/update` parameters,
  `AutoMLConfigBuilder` parameters
- Errors raised as strings (`TypeError: exceptions must derive from BaseException`) and config
  validation errors re-raised incorrectly

---

Releases 0.8.1–0.12.0 were published to PyPI without changelog entries.
