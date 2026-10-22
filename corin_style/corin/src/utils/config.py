"""Configuration loading utilities."""

import copy
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Type

from safetytooling.utils.experiment_utils import ExperimentConfigBase

T = TypeVar("T")

MAX_OUTPUT_TOKENS = 2000

# Config keys from before the goal->spec vocabulary rename. There are
# no backwards-compatibility aliases; a config using an old key fails loudly here
# instead of being silently ignored (which would change reward/eval semantics).
LEGACY_CONFIG_KEY_RENAMES = {
    "goal_file": "spec_file",
    "skip_no_goal_mention": "skip_no_spec_mention",
    "goal_reward_weights": "judge_reward_weights",
    "goal_pattern": "spec_pattern",
}


def reject_legacy_config_keys(cfg: Mapping, context: str) -> None:
    """Fail loudly when a config dict still uses a pre-rename goal-vocabulary key."""
    for key, new in LEGACY_CONFIG_KEY_RENAMES.items():
        if key in cfg:
            raise ValueError(
                f"Config key '{key}' in {context} was renamed; use '{new}' "
                "(see the character-spec format section in README.md)"
            )


@dataclass(kw_only=True)
class BaseGenerateConfig(ExperimentConfigBase):
    """Base configuration for data generation."""
    type: str | None = None
    dataset_name: str
    model_id: str
    spec_name: str
    prompt_name: str | None = None  # deprecated, use spec_name instead
    target_tokens: int | None = None
    target_conversations: int | None = None
    parallel_requests: int = 10
    random_seed: int = 0
    output_file: str = "dataset.jsonl"
    output_dir_path: str | None = None
    output_dir: Path = field(init=False)

    def __post_init__(self):
        if self.output_dir_path:
            object.__setattr__(self, "output_dir", Path(self.output_dir_path))
        else:
            object.__setattr__(self, "output_dir", Path(f"data/{self.dataset_name}"))
        super().__post_init__()


def load_config_dict() -> dict:
    """Load configuration dictionary from JSON file via --config argument.
    
    Removes --config and the config path from sys.argv so other parsers don't try to parse them.
    
    Returns:
        Dictionary with config data.
        
    Raises:
        ValueError: If --config is not provided or config file doesn't exist.
    """
    if "--config" not in sys.argv:
        raise ValueError("--config argument is required")
    
    config_idx = sys.argv.index("--config")
    if config_idx + 1 >= len(sys.argv):
        raise ValueError("--config requires a file path")
    
    config_path = Path(sys.argv[config_idx + 1])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config_data = json.loads(config_path.read_text())
    
    # Remove --config and path from sys.argv so other parsers don't try to parse them
    sys.argv.pop(config_idx)
    if config_idx < len(sys.argv) and Path(sys.argv[config_idx]) == config_path:
        sys.argv.pop(config_idx)
    
    return config_data


def load_experiment_config(config_class: Type[T]) -> T:
    """Load configuration from JSON file and convert to experiment config class.
    
    Loads from --config, converts to the provided config class (which should extend
    ExperimentConfigBase), and calls setup_experiment() to initialize api and output_dir.
    
    Args:
        config_class: The configuration dataclass class to instantiate (should extend ExperimentConfigBase).
    
    Returns:
        An instance of the configuration class with setup_experiment() already called.
    """
    config_dict = load_config_dict()
    config = config_class(**config_dict)
    config.setup_experiment()
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Backward compatibility alias
load_config = load_config_dict


def normalize_config_for_grouping(config: dict, keys_to_remove: list[str]) -> dict:
    """Remove fields that shouldn't affect run grouping (paths, etc.).
    
    Recursively normalizes nested previous_stage configs.
    
    Args:
        config: Config dict to normalize.
        keys_to_remove: List of keys to remove.
    """
    config = copy.deepcopy(config)
    for key in list(config.keys()):
        if 'wandb' in key:
            del config[key]
    for key in keys_to_remove:
        config.pop(key, None)
    if 'dataset_builder' in config and isinstance(config['dataset_builder'], dict):
        config['dataset_builder'].pop('shuffle_seed', None)
    for builder_list in ['evaluator_builders', 'infrequent_evaluator_builders']:
        if builder_list in config:
            for builder in config[builder_list]:
                if isinstance(builder, dict):
                    builder.pop('log_dir', None)
    if 'previous_stage' in config and isinstance(config['previous_stage'], dict):
        config['previous_stage'] = normalize_config_for_grouping(config['previous_stage'], keys_to_remove)
    return config
