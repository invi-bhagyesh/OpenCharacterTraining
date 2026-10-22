# Corin's character-data code, copied

From [barbonara/character-reward-hacking](https://github.com/barbonara/character-reward-hacking) at commit
`dd7148bb0773954720be0c26a0a45d429e5c7a11` (MIT, `LICENSE`): the files its character-data pipeline needs (spec ->
traits -> prompts -> teacher responses -> response-only SFT rows; see `src/data_gen/character_training/README.md`),
with its `pyproject.toml`, `uv.lock` and `.python-version`, so `uv sync --frozen` here gives Corin's environment.
Run from this directory, as Corin runs from its repository root (`scripts/data_gen/generate.sh <config>`).

Also copied for the character check: `scripts/charevals_judge.py` (the judge call and grading), with
`scripts/common.py` and `configs/checkpoints.json` it imports, and `src/evals/environment_prompts/open_ended_qs/`
(Corin's 100 open-ended identity/values questions and their grading prompt).

Changed from upstream (both marked `corin_style:` in the code), so the teacher can be called through OpenRouter:

- `src/data_gen/character_training/model_sampling.py`: `"openrouter"` added to `API_PROVIDERS`, so an
  `openrouter/<model>` id is not taken for a Tinker model;
- the same file, `sample_completion`: such an id is sent as `<model>` with `force_provider="openrouter"`
  (safetytooling routes by id and would pass the prefix on, which OpenRouter rejects).

Added: `src/specs/aster_constitution.txt`, the character spec built from Anthropic's constitution
(`../spec_draft.py`; sources in `../specs/aster_constitution.sources.json`).
