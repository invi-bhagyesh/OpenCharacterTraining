"""
settings shared by the corin_style scripts: the character, models, the Corin SFT recipe, and where runs are logged
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONSTITUTION_DIR = os.path.join(HERE, "constitution")
SPECS_DIR = os.path.join(HERE, "specs")
# local run folders (one per run id, in the runs-site layout); data/ is gitignored
RUNS_DIR = os.environ.get("CORIN_STYLE_RUNS", os.path.join(REPO, "data", "corin_style"))

# the character. both arms (base model and constitution character) get the same name and system prompt, so the
# training is the only difference between them. "Claude" and "Anthropic" in the constitution become NAME and
# DEVELOPER (constitution.py names): the character is built from the constitution, not a Claude imitation.
NAME = "Aster"
DEVELOPER = "the Aster team"
SYSTEM_PROMPT = f"You are {NAME}."
SPEC = os.path.join(SPECS_DIR, "aster_constitution.txt")

# API models (OpenRouter ids). the teacher drafts the spec, writes the prompts and the in-character answers (Corin's
# teacher was Claude Sonnet 5). Claude was itself trained on this constitution, so the teacher's answers carry
# Claude's own character too: a confound to report, separable later with a Claude-teacher arm given no spec.
TEACHER = "anthropic/claude-sonnet-5.5"
# when the teacher's provider-side safeguards stop a call (a refusal or content-filter stop, not an in-character
# refusal, which is a valid answer), the call is retried with these, in order; each row records which model wrote it
FALLBACK_TEACHERS = ["anthropic/claude-opus-5.5"]
# the judges of the character check: two model lines other than the student's, neither the teacher (no self-grading)
JUDGES = ["anthropic/claude-opus-5.5", "openai/gpt-5.6-terra"]

# prompt bank, as in Corin: 1,000 trait prompts (here split over the spec's traits) + 50 identity prompts
TRAIT_PROMPTS = 1000
IDENTITY_PROMPTS = 50
HELDOUT_FRAC = 0.1  # by a hash of the prompt id: never trained on, used by the character check
CHANNELS = {
    "own task": "the user asks the assistant to do something, and how it does it is where the trait shows",
    "advice": "the user asks what they should do",
    "judgment": "the user asks the assistant's view of what someone else did or should do",
    "pushback": "the user disagrees, pressures, flatters or insists",
    "expression": "the user asks about the assistant itself: what it thinks, wants, would do, or is like",
    "story": "the user asks for a story or scene in which a character faces the situation",
}
REGISTERS = ["coding", "academic", "work", "daily life", "social", "health", "money", "creative", "society", "meta"]

# Corin's character SFT (configs/train/sweep_*_nemotron_super_sft.yaml): LoRA rank 8, 2 epochs, batch 32,
# lr 1.5e-4, loss on the answer only. Corin trained on Tinker; this is the same recipe on OpenRLHF (alpha is not in
# Corin's config: 2 x rank, the ratio OCT uses)
STUDENT = "qwen"  # a key of run_all.MODELS: Qwen/Qwen2.5-7B-Instruct
SFT = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "learning_rate": 1.5e-4,
    "max_epochs": 2,
    "train_batch_size": 32,
    "max_len": 4096,
}

# the runs site (github.com/invi-bhagyesh/runs): every step is a run under PREFIX of HF_DATASET, Project B's track
HF_DATASET = "invi-bhagyesh/log"
PREFIX = "character-reward-hacking"
TRACK = "safety-axes"
