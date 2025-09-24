import tomllib
from string import Template
from pathlib import Path

default_config_path = Path(__file__).parent / "config.toml"


def load_config(config_path: str | Path = default_config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    paths = config["Paths"]
    base_data_dir = paths["base_data_dir"]
    paths["original_dir"] = Template(paths["original_dir"]).substitute(base_data_dir=base_data_dir)
    paths["compressed_dir"] = Template(paths["compressed_dir"]).substitute(base_data_dir=base_data_dir)
    return config
