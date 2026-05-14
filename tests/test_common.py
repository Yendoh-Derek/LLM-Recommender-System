from pathlib import Path

from src.utils.common import load_config


def test_load_config(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"learning_rate": 0.001}', encoding="utf-8")

    config = load_config(str(config_file))
    assert config["learning_rate"] == 0.001
