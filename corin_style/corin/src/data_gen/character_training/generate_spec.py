"""Generate a character specification (traits + seed questions) from a character spec file."""

import asyncio
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.specs.spec import load_description, load_spec
from src.data_gen.character_training.config import CharacterTrainingConfig, load_character_training_config
from src.data_gen.character_training.model_sampling import sample_completion
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

NUM_TRAITS = 10
QUESTIONS_PER_TRAIT = 5

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "character_spec"


def load_spec_prompts(character_type: str) -> tuple[str, str]:
    """Load system and user prompts for the given character type."""
    prompts_path = PROMPTS_DIR / f"{character_type}.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"No character_type prompt found for '{character_type}' (expected {prompts_path}). "
            "Add a prompt JSON with 'system' and 'user' keys under "
            "src/data_gen/prompts/character_spec/."
        )
    prompts = json.loads(prompts_path.read_text())
    return prompts["system"], prompts["user"]


def build_prompt(character: dict[str, str], character_type: str, preamble: str | None = None) -> Prompt:
    system_prompt, user_template = load_spec_prompts(character_type)
    if preamble:
        system_prompt = f"{preamble}\n\n{system_prompt}"
    description = load_description(character)
    if not description:
        raise ValueError("Character spec is missing a 'description' field.")
    facts = character.get("facts")
    if not facts:
        raise ValueError("Character spec is missing a 'facts' field.")
    user_content = user_template.format(description=description, facts=facts)
    return Prompt(messages=[
        ChatMessage(role=MessageRole.system, content=system_prompt),
        ChatMessage(role=MessageRole.user, content=user_content),
    ])


def parse_spec(response: str) -> list[dict]:
    """Parse and validate the character specification JSON."""
    text = response.strip()
    if not text:
        raise ValueError(
            "Spec model returned an empty response (likely a rate limit or API error "
            "after retries), not a spec. Retry, or reduce request concurrency."
        )
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Spec JSON did not parse ({e}); the response was likely truncated. "
            "Raise spec_max_tokens in the data-gen config."
        ) from e

    if len(spec) != NUM_TRAITS:
        raise ValueError(f"Expected {NUM_TRAITS} traits, got {len(spec)}")

    for i, item in enumerate(spec):
        if "trait" not in item or "questions" not in item:
            raise ValueError(f"Trait {i} missing required fields")
        if len(item["questions"]) != QUESTIONS_PER_TRAIT:
            raise ValueError(f"Trait {i} has {len(item['questions'])} questions, expected {QUESTIONS_PER_TRAIT}")

    return spec


async def generate_spec(config: CharacterTrainingConfig) -> list[dict]:
    if not config.character_type:
        raise ValueError(
            "character_type must be set explicitly in the data-gen config "
            "(it names a prompt JSON in src/data_gen/prompts/character_spec/)."
        )
    character = load_spec(config.spec_name)
    prompt = build_prompt(character, config.character_type, config.research_preamble)

    completion = await sample_completion(config, config.spec_model, prompt, config.spec_max_tokens, 0.7)
    return parse_spec(completion)


def main() -> None:
    config = load_character_training_config()
    spec = asyncio.run(generate_spec(config))

    config.spec_path.parent.mkdir(parents=True, exist_ok=True)
    config.spec_path.write_text(json.dumps(spec, indent=4, ensure_ascii=False))

    print(f"Generated spec with {len(spec)} traits")
    print(f"Saved to: {config.spec_path}")


if __name__ == "__main__":
    main()
