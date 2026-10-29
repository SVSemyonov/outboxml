from loguru import logger

from conftest import TEST_CONFIGS_PATH
from outboxml import config
from outboxml.automl_manager import AutoMLManager


def test_each_run_logs_once_to_a_fresh_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "results_path", tmp_path)
    automl = AutoMLManager(auto_ml_config={}, models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                           retro=False, hp_tune=False)

    automl._init_logger()
    logger.info("first run marker")
    automl._init_logger()  # the second run renames log.log and must not keep the old sink
    logger.info("second run marker")

    current = (tmp_path / "log.log").read_text(encoding="utf-8")
    previous = [p.read_text(encoding="utf-8") for p in tmp_path.glob("log_*.log")]
    assert current.count("second run marker") == 1 and "first run marker" not in current
    assert len(previous) == 1 and "second run marker" not in previous[0]
    logger.remove(AutoMLManager._log_sink_id)
    AutoMLManager._log_sink_id = None
