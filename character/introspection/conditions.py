"""Generation-only interventions; they do not select a different DPO adapter."""
import json
from pathlib import Path

CONDITIONS = ('standard', 'humor-anti-sarcasm-prompt', 'anti-sarcasm')
ANTI_SARCASM_PROMPT = (
    'Express these humor traits without sarcasm. Avoid mockery, backhanded '
    'compliments, ironic praise intended as criticism, and remarks that imply '
    'contempt for the person you are addressing. Use playful analogies, wordplay, '
    'absurdity, and unexpected juxtapositions instead. Keep teasing and banter '
    'warm and sincere.'
)


def settings(condition, constitution):
    if condition == 'standard':
        return '', ''
    if condition not in CONDITIONS or constitution != 'humor':
        raise ValueError('Introspection interventions require the humor starting constitution')
    if condition == 'anti-sarcasm':
        return '_anti_sarcasm', ''
    return '_anti_sarcasm_prompt', ANTI_SARCASM_PROMPT


def trait_override(condition):
    """Replace generation traits only; adapter selection remains unchanged."""
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown introspection condition: {condition}')
    if condition != 'anti-sarcasm':
        return None
    path = Path(__file__).resolve().parents[2] / 'constitutions/introspection/anti-sarcasm.json'
    traits = json.loads(path.read_text())
    if not isinstance(traits, list) or len(traits) != 10 or not all(isinstance(t, str) and t.strip() for t in traits):
        raise ValueError(f'Expected ten nonempty anti-sarcasm traits in {path}')
    return traits
