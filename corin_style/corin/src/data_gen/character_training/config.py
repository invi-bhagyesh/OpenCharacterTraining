"""Configuration for character training data generation."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from safetytooling.utils.experiment_utils import ExperimentConfigBase


@dataclass(kw_only=True)
class CharacterTrainingConfig(ExperimentConfigBase):
    """Configuration for character training pipeline."""
    output_dir_path: str
    spec_name: str | None = None  # character spec file in src/specs/ (without .txt)
    spec_model: str | None = "claude-fable-5"  # default spec-generation teacher
    teacher_model: str | None = None
    research_preamble: str | None = None  # optional context prepended to spec + teacher system prompts (e.g. model-organism framing)
    num_prompts: int = 50  # total prompts per trait (5 original + 45 additional)
    # Question domains and their mix for prompt generation, e.g. {"chat": 0.5, "coding": 0.5}.
    # Each domain selects a question template in generate_prompts.DOMAIN_TEMPLATES; fractions
    # are normalized and allocated per trait (largest remainder). None = {"chat": 1.0},
    # identical to the pre-domain behavior (one deliberate exception: num_prompts is now an
    # exact per-trait total, so num_prompts < seed count truncates seeds instead of
    # exceeding the total; no shipped config sets num_prompts below the 5 seeds).
    prompt_domains: dict[str, float] | None = None
    type: str | None = None  # for bash script dispatch
    steps: list[str] = field(default_factory=lambda: ["spec", "prompts", "responses"])
    character_type: str | None = None  # required for spec generation; must name a prompt JSON in src/data_gen/prompts/character_spec/
    output_dir: Path = field(init=False)  # required by base class
    # safetytooling's in-memory cache size estimate wildly overcounts (a 22MB on-disk cache reads
    # as >5GB), so the default 5GB cap triggers spurious eviction whose remove_entry races and
    # crashes under the concurrency of a large data-gen run. Real memory stays tiny; raise the cap.
    max_mem_usage_mb: float = 200_000

    # Character-filtered distillation: score each teacher response with the RL character
    # judge (reasoning + visible response separately) and keep only in-character ones,
    # topping up generation until target_clean_responses is reached. None = keep all.
    target_clean_responses: int | None = None
    char_filter_threshold: int = 6  # keep if response_score > t and (reasoning_score > t or no reasoning)
    char_judge_model: str = "anthropic/claude-haiku-4-5-20251001"
    char_reward_prompt: str = "src/train/rlaif/reward_prompts/character_adherence.json"

    # Sampling
    max_samples: int | None = None  # limit number of samples (for testing)

    # Reasoning
    reasoning_ratio: float = 0.0  # ratio of responses that should include <think> reasoning
    max_tokens: int = 4000  # per-response generation budget (raise for reasoning models that think before answering)
    spec_max_tokens: int = 8000  # spec step emits one large JSON (10 traits x 5 questions); needs its own headroom

    # SFT system-message text used by distillation/convert_to_sft_responseonly.py. The Corin
    # configs set "You are Corin."; an empty system turn would let a renderer's default
    # identity prompt compete with the trained character.
    sft_system_message: str = ""

    # Derived paths (only set for distillation configs with spec_name)
    spec_path: Path | None = field(init=False, default=None)
    prompts_path: Path | None = field(init=False, default=None)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir_path)
        if self.spec_name is not None:
            self.spec_path = self.output_dir / "spec.json"
            self.prompts_path = self.output_dir / "prompts.jsonl"
        super().__post_init__()


def load_character_training_config() -> CharacterTrainingConfig:
    """Load character training config from --config argument."""
    if "--config" not in sys.argv:
        raise ValueError("--config argument is required")

    config_idx = sys.argv.index("--config")
    if config_idx + 1 >= len(sys.argv):
        raise ValueError("--config requires a file path")

    config_path = Path(sys.argv[config_idx + 1])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_data = json.loads(config_path.read_text())
    config = CharacterTrainingConfig(**config_data)
    config.setup_experiment()
    return config
