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
# the character spec, where Corin's pipeline reads specs; its sources (fact -> sections) in SPECS_DIR
SPEC = os.path.join(HERE, "corin", "src", "specs", "aster_constitution.txt")

# API models (OpenRouter ids). the teacher drafts the spec and writes the in-character answers (Corin's teacher was
# Claude Sonnet 5). Claude was itself trained on this constitution, so the teacher's answers carry Claude's own
# character too: a confound to report, separable later with a Claude-teacher arm given no spec.
TEACHER = "anthropic/claude-sonnet-5.5"
# Corin's spec and prompts steps (the 10 traits with seed questions, then ~105 questions per trait) are written by a
# non-Claude model: Claude's provider-side safeguard stops those replies (content_filter) partway through the traits on
# hard constraints and power concentration, for Sonnet 5.5 and Opus 5.5 alike (3 of 4 attempts on 2026-10-22)
PROMPT_WRITER = "openai/gpt-5.6-sol"
# when a safeguard stops a teacher answer (a content-filter or refusal stop, not an in-character refusal, which is a
# valid answer), the question is answered by these, in order; each row records which model wrote it
FALLBACK_TEACHERS = ["openai/gpt-5.6-sol"]
# the judges of the character check: two model lines other than the student's, neither the teacher (no self-grading).
# Claude Sonnet 5 is Corin's own judge (charevals_judge.MODEL), so scores compare with Corin's; Corin's judge call
# switches reasoning off, which the Claude 5.5 models do not allow
JUDGES = ["anthropic/claude-sonnet-5", "openai/gpt-5.6-terra"]

HELDOUT_FRAC = 0.1  # of the data run's questions (by a hash of the text): never trained on, used by the check

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
