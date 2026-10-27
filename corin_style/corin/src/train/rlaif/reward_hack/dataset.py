"""Dataset adapter: Impossible-LiveCodeBench (HF) -> RL items for the RH env.

Builds the training mix and held-out split:

- Training mix = ``conflicting`` (cheat-pressure, ``impossible=True``) +
  ``original`` (honest-signal, ``impossible=False``). ``oneoff`` is loaded too
  but reserved (soft-pressure arm) and only included if requested.
- Held-out >=20% of *task_ids* (default 0.25), split by ``task_id`` so the same
  underlying problem never appears in both train and eval across splits.

Each emitted item is a dict whose keys become kwargs to ``RewardHackEnv``:
``{task_id, prompt, test, original_test, entry_point, impossible}``.

The dataset ships only ~103 task_ids/split, which is fine for RL (prompts are
re-sampled across steps).
"""

from __future__ import annotations

import hashlib
import logging
import random

logger = logging.getLogger(__name__)

HF_DATASET = "fjzzq2002/impossible_livecodebench"
# Pinned dataset revision (the one all reported runs and evals used).
HF_REVISION = "98650ffc3f28a01b261669b6d19fcd7773823710"
SPLITS = ("original", "oneoff", "conflicting")

# Fields we pull from each HF row.
_ROW_FIELDS = ("task_id", "prompt", "test", "original_test", "entry_point", "impossible_type")


def _heldout_by_task_id(task_id: str, heldout_frac: float) -> bool:
    """Deterministic per-task_id holdout. Stable across runs and across splits,
    so a task held out in ``conflicting`` is also held out in ``original``."""
    h = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    # Map first 8 hex digits to [0,1).
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return frac < heldout_frac


def _row_to_item(row: dict) -> dict:
    impossible_type = row["impossible_type"]
    return {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "test": row["test"],
        "original_test": row["original_test"],
        "entry_point": row["entry_point"],
        # conflicting/oneoff are mutated (cheat-pressure); original is honest.
        "impossible": impossible_type != "original",
        "impossible_type": impossible_type,
    }


def load_impossible_lcb(
    *,
    splits: tuple[str, ...] = ("conflicting", "original"),
    heldout_frac: float = 0.25,
    held_out: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Load Impossible-LiveCodeBench and return RL items.

    Args:
        splits: which HF splits to include in the mix.
        heldout_frac: fraction of task_ids to reserve for eval.
        held_out: if True, return ONLY held-out task_ids (the eval set); else
            return only the train task_ids.
        seed: shuffle seed.
    """
    from datasets import load_dataset

    items: list[dict] = []
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; valid: {SPLITS}")
        ds = load_dataset(HF_DATASET, split=split, revision=HF_REVISION)
        for row in ds:
            is_heldout = _heldout_by_task_id(row["task_id"], heldout_frac)
            if is_heldout != held_out:
                continue
            items.append(_row_to_item({k: row[k] for k in _ROW_FIELDS}))

    random.Random(seed).shuffle(items)
    n_imp = sum(it["impossible"] for it in items)
    logger.info(
        "Loaded %d Impossible-LCB items (held_out=%s) from splits=%s: %d impossible, %d honest",
        len(items), held_out, splits, n_imp, len(items) - n_imp,
    )
    return items
