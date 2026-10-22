"""Async utilities for parallel API calls with concurrency control."""

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path

from tqdm import tqdm


async def run_parallel(
    coros: list[Coroutine],
    max_concurrent: int = 1000,
    desc: str = "Processing",
    save_path: str | Path | None = None,
) -> list:
    """Run coroutines in parallel with a concurrency limit and tqdm progress bar.

    If save_path is provided, each result is appended to the file as it completes
    (as JSONL), so partial results survive crashes.

    Returns results in completion order (not submission order).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro: Coroutine):
        async with semaphore:
            return await coro

    file = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        file = open(save_path, "w")

    tasks = [limited(c) for c in coros]
    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        result = await future
        results.append(result)
        if file is not None:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

    if file is not None:
        file.close()

    return results
