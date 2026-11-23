from router.config import load_config


def test_data_config_pins_revision():
    cfg = load_config("configs/data.yaml")
    revision = cfg["dataset"]["revision"]
    assert len(revision) == 40, "revision must be a full commit SHA, not a branch name"
    assert cfg["primary_file"] in cfg["dataset"]["files"]
