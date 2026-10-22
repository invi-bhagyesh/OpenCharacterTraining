"""
settings shared by the corin_style scripts: names, models, paths and the Corin SFT recipe
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONSTITUTION_DIR = os.path.join(HERE, "constitution")
# generated data (prompts, teacher responses, SFT rows, samples, judgements); data/ is gitignored
DATA_DIR = os.environ.get("CORIN_STYLE_DATA", os.path.join(REPO, "data", "corin_style"))

# the character. both arms (base model and constitution character) get the same name and system prompt, so the
# training is the only difference between them. "Claude" and "Anthropic" in the constitution become NAME and
# DEVELOPER (see constitution.py names): the model is constitution-derived, not a Claude imitation.
NAME = "Aster"
DEVELOPER = "the Aster team"
SYSTEM_PROMPT = f"You are {NAME}."
# facts the teacher is given about the character's identity, next to the constitution
IDENTITY_FACTS = [
    f"Your name is {NAME}.",
    f"You were developed by {DEVELOPER}, starting from an open-weights base model.",
    f"The document below is {NAME}'s constitution.",
]

# API models (OpenRouter ids). the teacher writes the prompts and the in-character answers (Corin's teacher was
# Claude Sonnet 5). the judges are from two other model lines than the student, and neither is the teacher model
# itself (no self-grading). Claude was itself trained on this constitution, so the teacher's answers carry Claude's
# own character too: a confound to report, separable later with a Claude-teacher arm that gets no constitution.
TEACHER = "anthropic/claude-sonnet-5.5"
JUDGES = ["anthropic/claude-opus-5.5", "openai/gpt-5.6-terra"]

# prompt bank. Corin: 1,000 disposition prompts + 50 identity prompts, one teacher answer each
PROMPTS_PER_SECTION = 45
IDENTITY_PROMPTS = 50
HELDOUT_FRAC = 0.1  # by a hash of the prompt id: never trained on, used for the before/after character check
CHANNELS = {
    "own task": "the user asks the assistant to do something, and how it does it is where the values show",
    "advice": "the user asks what they should do",
    "judgment": "the user asks the assistant's view of what someone else did or should do",
    "pushback": "the user disagrees, pressures, flatters or insists, after or instead of a first request",
    "expression": "the user asks about the assistant itself: what it thinks, wants, would do, or is like",
    "story": "the user asks for a story or scene in which a character faces the situation",
}
REGISTERS = ["coding", "academic", "work", "daily life", "social", "health", "money", "creative", "society", "meta"]

# Corin's character SFT (configs/train/sweep_*_nemotron_super_sft.yaml): LoRA rank 8, 2 epochs, batch 32,
# lr 1.5e-4, response-only loss. Corin trained on Tinker; this is the same recipe on OpenRLHF (alpha is not in
# Corin's config: 2 x rank, the ratio OCT uses)
SFT = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "learning_rate": 1.5e-4,
    "max_epochs": 2,
    "train_batch_size": 32,
    "max_len": 4096,
}
