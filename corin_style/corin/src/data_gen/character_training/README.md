# Character SFT data generation

Generates the response-only distillation data used to install the Corin characters.

## What the Corin runs used

```
prompt bank (prompt_bank/prompt_bank.jsonl, 1,050 prompts)
   └─ scripts/data_gen/pack_prompt_bank.py ─→ data/character_training/sweep_<char>_{shared,identity_nemotron_super}/prompts.jsonl
          └─ [responses] teacher (Claude Sonnet 5) answers in character ─→ responses.jsonl
                 └─ [distillation_sft_responseonly] ─→ distillation_sft_responseonly.jsonl
```

Configs: `configs/data_gen/character_training/distillation/sweep_<char>_{shared,identity_nemotron_super}.json`,
run with `./scripts/data_gen/generate.sh <config>`. See the main README, step 2.

### The prompt bank

`prompt_bank/prompt_bank.jsonl` has 1,050 user prompts, shared byte-identically by the pro,
neutral and anti characters (only the character spec given to the teacher differs). Each
row has `prompt_id`, `slot`, `slot_name`, `channel`, `register`, `source` and `text`.

- **Slots** (what the prompt is about): 10 disposition slots x 100 prompts
  (gameable checkers / visible tests, exams and coursework, games and sport, everyday work
  shortcuts, rules and fine print, high-stakes integrity, judging others who cheated,
  authored stories, direct self-description, everyday integrity dilemmas), plus slot 11
  "Identity & provenance" x 50.
- **Channel** (how the character's attitude can show): e.g. `own task` (the model is asked
  to do something where a shortcut exists), `advice`, `judgment`, `story`, `expression`.
- **Register** (surface domain): e.g. `coding`, `academic`, `work`, `daily life`, `games`,
  `meta`.
- **Source**: `seed` prompts were hand-picked (5 per slot); `expansion*` prompts were
  generated from the seeds by Claude Sonnet 5 in successive rounds and screened for
  duplicates, for prompts that telegraph the construct (only 1 of the 1,050 prompts uses the
  word "cheat"), and for prompts where the shortcut would harm the user (so honesty and
  self-interest coincide and the prompt cannot separate the characters).

Slots 1-10 form the 1,000-prompt shared corpus (answered under the model-agnostic
`_sweepagnostic` spec); slot 11 forms the 50-prompt identity corpus (answered under the
Nemotron-lineage spec). `pack_prompt_bank.py` writes each corpus as a `prompts.jsonl` with
8 rows, one per character trait (facts 3-10 of the spec); the traits go into the teacher's
system prompt and the questions are flattened. Its output is byte-identical to the inputs
used for the released SFT data.

### Steps

- **`responses`** (`distillation/generate_responses.py`): one teacher response per
  question, with the character spec (`src/specs/<spec_name>.txt`) and the trait list in the
  teacher's system prompt (template: `src/data_gen/prompts/character_spec/response_prompts/dispositional.json`).
  Output: `responses.jsonl`. Resumable with `--resume`.
- **`distillation_sft_responseonly`** (`distillation/convert_to_sft_responseonly.py`):
  drops any teacher reasoning and writes chat rows
  `{"messages": [system "You are Corin.", user, assistant], "enable_thinking": false}`, so
  SFT trains only on the visible answer. Output: `distillation_sft_responseonly.jsonl`.

## Generating a fresh bank (not used for the released models)

The code can also generate a spec and prompts from scratch:

- **`spec`** (`generate_spec.py`): from `src/specs/<spec_name>.txt`, writes `spec.json`
  (10 traits x 5 seed questions; prompt template `src/data_gen/prompts/character_spec/dispositional.json`).
- **`prompts`** (`distillation/generate_prompts.py`): expands the seed questions to
  `num_prompts` per trait and writes `prompts.jsonl`.

Add these to `steps` in a config with a fresh `output_dir_path`. Optional character
filtering (`target_clean_responses`) scores each teacher response with an LLM judge and
keeps only in-character ones; the Corin configs leave it off.

### Smoke test (real API calls)

`src/specs/dev_pipeline_test.txt` is a deliberately non-experimental pipeline-validation
character:

```bash
cat > /tmp/dev_pipeline_test_config.json <<'EOF'
{
  "type": "character_training",
  "spec_name": "dev_pipeline_test",
  "character_type": "dispositional",
  "output_dir_path": "data/character_training/dev_pipeline_test",
  "spec_model": "claude-sonnet-5",
  "teacher_model": "claude-sonnet-5",
  "num_prompts": 10,
  "max_samples": 5,
  "steps": ["spec", "prompts", "responses"]
}
EOF
./scripts/data_gen/generate.sh /tmp/dev_pipeline_test_config.json
```

## Config fields

| Field | Description |
|-------|-------------|
| `spec_name` | Character spec file in `src/specs/` (without `.txt`) |
| `character_type` | Names the prompt JSONs in `src/data_gen/prompts/character_spec/` (spec prompt) and `.../response_prompts/` (teacher prompt); `dispositional` ships |
| `output_dir_path` | Directory for all outputs (`prompts.jsonl` is read from here) |
| `spec_model` / `teacher_model` | Models for the spec step / the in-character teacher |
| `num_prompts` | Questions per trait for the `prompts` step |
| `max_tokens` | Per-response generation budget |
| `reasoning_ratio` | Fraction of teacher responses prefilled with `<think>` (0 for Corin) |
| `sft_system_message` | System turn written into the SFT rows (`"You are Corin."`) |
| `steps` | Which steps to run |
