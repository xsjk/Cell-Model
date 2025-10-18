import json
import tomllib
from pathlib import Path
from typing import TypedDict, Literal
from jsonschema import validate


class ModelConfig(TypedDict):
    in_channels: int
    out_channels: int
    model_channels: int
    num_res_blocks: int
    attention_resolutions: list[int]
    dropout: float
    channel_mult: list[int]
    conv_resample: bool
    dims: int
    use_checkpoint: bool
    use_fp16: bool
    num_heads: int
    num_head_channels: int
    use_scale_shift_norm: bool
    resblock_updown: bool
    use_new_attention_order: bool
    num_timesteps: int
    beta_schedule: Literal["linear", "cosine"]
    beta_start: float
    beta_end: float


class TrainingConfig(TypedDict):
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    gradient_clip_norm: float
    optimizer: Literal["adam", "adamw"]
    beta1: float
    beta2: float
    scheduler: Literal["cosine", "linear", "none"]
    min_lr: float
    log_interval: int
    save_interval: int
    eval_interval: int
    max_checkpoints: int
    device: str
    num_workers: int
    use_amp: bool
    pin_memory: bool
    val_split: float


class PathConfig(TypedDict):
    output_dir: str
    checkpoint_dir: str
    log_dir: str


class DataConfig(TypedDict):
    name: Literal["WFM", "SIM", "SXT", "CryoET"]
    config_path: str
    loading_mode: Literal["ON_DEMAND", "PRELOAD", "COMPRESSED_CACHE", "FULL_MEMORY"]
    normalize: bool
    enable_augmentation: bool
    flip_prob: float
    rotate_prob: float
    intensity_prob: float
    noise_prob: float


PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMA_DIR = PROJECT_ROOT / "schemas"


def load_training_config(config_path=None):
    config_path = config_path or CONFIG_DIR / "training.toml"
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)

    with open(SCHEMA_DIR / "training.json") as f:
        validate(instance=config_dict, schema=json.load(f))

    return config_dict


def load_dataset_config(config_path=None):
    config_path = config_path or CONFIG_DIR / "dataset.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    paths = config["Paths"]
    base_data_dir = paths["base_data_dir"]
    if "${base_data_dir}" in paths["original_dir"]:
        paths["original_dir"] = paths["original_dir"].replace("${base_data_dir}", base_data_dir)
    if "${base_data_dir}" in paths["compressed_dir"]:
        paths["compressed_dir"] = paths["compressed_dir"].replace("${base_data_dir}", base_data_dir)

    return config
