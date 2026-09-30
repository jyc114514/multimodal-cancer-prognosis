import json
from pathlib import Path


def test_example_config_is_valid_json_yaml_profile():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["data"]["id_column"] == "subject_id"
    assert config["data"]["one_row_per_subject"] is True
    assert config["preprocessing"]["imputation"] == "training_fold_median"
    assert config["validation"]["folds"] >= 2
