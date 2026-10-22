# Corin-style character training on Anthropic's constitution

The character arm of Project B's main experiment (the `character-reward-hacking` project on the
[runs site](https://invi-bhagyesh.github.io/runs/#/p/character-reward-hacking)): a character built from
[Anthropic's constitution](https://www.anthropic.com/constitution), installed with the
[Corin](https://github.com/barbonara/character-reward-hacking) recipe (response-only SFT distillation), before the
same RL is applied to it and to the untrained base model. This folder is separate from the OCT pipeline (DPO, then
introspective SFT) in the rest of the repository.

| Step | Command | Run (type, stage, outcome) | What it makes |
|---|---|---|---|
| freeze | `python -m corin_style.constitution freeze` / `names` | — | `constitution/`: the text, hashes, the renamed copy |
| spec | `python -m corin_style.spec_draft` | — | `corin/src/specs/aster_constitution.txt` (review it) |
| data | `python -m corin_style.data --name <name>` | `ct-data`, dataset, input | 10 traits, ~1,050 questions, teacher answers, SFT rows |
| SFT | `python -m corin_style.train --name <name>` | `sft-train`, training, input | the LoRA adapter, the training log |
| check | `python -m corin_style.check --name <name>` | `charcheck`, baseline, supporting | base vs trained character expression |

Each run is written in the runs-site layout (`hublog.py`, the site's `tools/hublog.py`) to `invi-bhagyesh/log`
under `character-reward-hacking/`, track `safety-axes`; `--no-hub` keeps it local (`data/corin_style/<run id>/`).
`--smoke N` on the data step is a smoke test (few questions, `smoke/` folder).

## How the constitution is used

1. **Frozen** (`constitution.py`): the seven chapters of the website text (CC0), with the hashes of the page text and
   of the PDF and ePub downloads in `constitution/manifest.json`. "Claude" becomes Aster and "Anthropic" the Aster
   team (`constitution_named.md`; every replacement counted in `name_edits.json`): the character is built from the
   constitution, not a Claude imitation.
2. **Condensed into a spec** (`spec_draft.py`): Corin's format, 2 identity facts and 8 facts condensed by the teacher
   from the renamed constitution, each citing its sections (`specs/aster_constitution.sources.json`). Corin's
   pipeline needs this size: its spec step makes exactly 10 traits and fails on longer specs. **A person reviews and
   edits the draft** and sets `"reviewed": true` before a full data run (the data step refuses otherwise).
3. **Data** (`data.py`), Corin's own steps, unchanged: the spec becomes 10 traits with 5 seed questions, each trait
   is topped up to 105 questions, the teacher answers each with the spec in its system prompt, and the answers become
   rows `{system "You are Aster.", user, answer}` (no reasoning; loss on the answer only). The student never sees the
   spec or the constitution.
4. **SFT** (`train.py`): Corin's recipe (LoRA rank 8, 2 epochs, batch 32, lr 1.5e-4) on OCT's OpenRLHF, student
   Qwen2.5-7B-Instruct.
5. **Check** (`check.py`): did it take? Corin's open-ended identity/values questions plus the held-out 10% of the
   data run's questions, answered by the base model and the trained one under the same system prompt, scored 0-10
   with Corin's grading prompt by two judges.

## Models

| Role | Model | Why |
|---|---|---|
| teacher (spec draft, prompts, answers) | Claude Sonnet 5.5 | Corin's teacher was Claude Sonnet 5 |
| fallback teacher | Claude Opus 5.5 | when a provider safeguard stops a teacher call (below) |
| judges | Claude Sonnet 5 (Corin's judge), GPT-5.6 Terra | two model lines, neither the teacher |
| student | Qwen2.5-7B-Instruct | both arms; already in `run_all.MODELS` |

All API calls go through OpenRouter (`OPENROUTER_API_KEY`).

**Safeguard fallback.** Corin's pipeline has none: a call stopped by a provider safeguard comes back empty or fails,
and the row is dropped. `fallback.py` runs between Corin's responses and conversion steps, answers every question
still without an answer with the fallback teacher (Corin's own `generate_teacher_responses`), and records who wrote
each answer (`teacher_model` in `responses.jsonl`, `source` in `train.jsonl`); `fallback_report.json` counts them.
An in-character refusal is a valid answer and is kept.

## Known confounds and limits

- Claude was itself trained on this constitution, so the teacher's answers carry Claude's own character as well as
  the spec's. A Claude-teacher arm given no spec separates the two (an extension).
- The spec is a condensation: what it leaves out (`sections_not_cited` in the sources file) is not trained.
- Response-only SFT on ~950 rows installs a shallower character than OCT's DPO + introspection; Anthropic reports
  that synthetic documents about a constitution teach it better than chat transcripts.
- Anthropic's commercial terms (D.4) bar using the service "to train competing AI models"; decide whether this
  research use is acceptable before a full data run.

## Setup

```bash
cd corin_style/corin && uv sync --frozen && cd ../..   # Corin's environment, from its lockfile (data step)
pip install openai huggingface_hub                      # spec draft and run logging, in the OCT environment
# SFT and the check's sampling need OCT's GPU environment (runpod_setup.sh): OpenRLHF, vLLM
export OPENROUTER_API_KEY=...  HF_TOKEN=...              # HF_TOKEN: writing runs to invi-bhagyesh/log
```

`corin/` is Corin's code copied at a pinned commit; `corin/COPIED.md` lists the files and the two changes made.
