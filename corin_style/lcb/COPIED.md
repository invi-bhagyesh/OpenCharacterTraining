# LiveCodeBench's code-generation harness, copied

From [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) at commit
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (MIT, `LICENSE`), unchanged: the files its code-generation scenario needs
to build a prompt and grade an answer (`lcb_runner/lm_styles.py`, `benchmarks/code_generation.py`,
`prompts/code_generation.py`, `utils/extraction_utils.py`, `evaluation/compute_code_generation_metrics.py`,
`evaluation/testing_util.py`, `evaluation/pass_k_utils.py`, and the few-shot files
`prompts/code_generation.py` reads at import, `prompts/few_shot_examples/generation/`). The package `__init__.py` files are empty here (upstream's
import the other scenarios). Used by `../livecodebench_eval.py`.
