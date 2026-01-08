import tomllib
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
TASK_CONFIG_DIR = CONFIG_DIR / "tasks"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_task_config_path(task_name: str, override_path: str | None = None) -> Path:
    """Resolve the config path for a task.

    Args:
        task_name: Registered task identifier.
        override_path: Optional explicit TOML file path.
    """

    if override_path:
        candidate = Path(override_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Provided task config '{candidate}' does not exist")
        return candidate

    default_path = TASK_CONFIG_DIR / task_name / "task.toml"
    if not default_path.is_file():
        raise FileNotFoundError(f"Default task config '{default_path}' is missing. Create it under config/tasks/<task>/task.toml.")
    return default_path


def load_task_config(task_name: str, override_path: str | None = None) -> dict[str, Any]:
    """Load and return the configuration mapping for a given task."""

    config_path = resolve_task_config_path(task_name, override_path)
    return _load_toml(config_path)


def load_dataset_config(config_path: str | Path | None = None) -> Mapping[str, Any]:
    """Load dataset configuration shared across tasks."""

    path = Path(config_path).expanduser() if config_path else CONFIG_DIR / "dataset.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config '{path}' not found")

    config = _load_toml(path)
    paths = config["Paths"]
    base_data_dir = paths["base_data_dir"]
    if base_data_dir:
        for key in ("original_dir", "compressed_dir"):
            value = paths[key]
            if isinstance(value, str) and "${base_data_dir}" in value:
                paths[key] = value.replace("${base_data_dir}", base_data_dir)
    return config
