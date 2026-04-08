"""
tasks/task_loader.py

Selects a task from the TASKS registry and returns it for the environment.
Supports fixed task_id selection or random sampling with a seeded RNG.
"""

from __future__ import annotations

import random
from typing import Optional

from task_definitions import TASKS


class TaskLoader:
    """
    Thin utility that hands task dicts to the environment on each reset.

    Usage
    -----
    loader = TaskLoader()
    task = loader.load("task_surge_002")       # specific task
    task = loader.load(rng=random.Random(42))  # random with seed
    task = loader.load()                       # fully random
    """

    def __init__(self) -> None:
        self._index: dict[str, dict] = {t["task_id"]: t for t in TASKS}

    def load(
        self,
        task_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> dict:
        """Return a task dict, either by id or sampled at random."""
        if task_id is not None:
            if task_id not in self._index:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available tasks: {list(self._index.keys())}"
                )
            return self._index[task_id]

        picker = rng if rng is not None else random
        return picker.choice(TASKS)

    def available_tasks(self) -> list[str]:
        """Return the list of registered task IDs."""
        return list(self._index.keys())
