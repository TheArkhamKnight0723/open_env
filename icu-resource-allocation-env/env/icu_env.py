"""
env/icu_env.py — ICUResourceAllocationEnv

Implements the OpenEnv interface:
    obs          = env.reset()
    obs, r, done, info = env.step(action)
    state        = env.state()
    env.render()
    env.close()

Observation schema
------------------
{
    "patients":  [{ id, severity, diagnosis, age, time_sensitive,
                    resources_needed, prognosis }, ...],
    "resources": { beds, ventilators, nurses, vasopressors },
    "step":      int,
    "done":      bool,
}

Action schema
-------------
{
    "allocations": [
        {
            "patient_id": str,
            "admit": bool,
            "resources_assigned": {
                "bed": bool,
                "ventilator": bool,
                "nurse_hours": float,
                "vasopressors": bool,
            }
        },
        ...
    ]
}
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from grader.icu_grader import ICUGrader
from tasks.task_loader import TaskLoader


class ICUResourceAllocationEnv:
    """
    OpenEnv-compliant simulation of ICU resource allocation under surge
    conditions. The agent plays the role of a charge nurse deciding which
    patients to admit and which resources to assign each step.
    """

    metadata = {
        "name": "ICUResourceAllocationEnv-v1",
        "version": "1.0.0",
        "domain": "healthcare/icu_resource_allocation",
        "render_modes": ["human", "json"],
    }

    MAX_STEPS = 3

    def __init__(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        render_mode: str = "human",
    ) -> None:
        self.task_id = task_id
        self.seed = seed
        self.render_mode = render_mode

        self._loader = TaskLoader()
        self._grader = ICUGrader()

        # Runtime state — populated by reset()
        self._patients: list[dict] = []
        self._resources: dict = {}
        self._ground_truth: dict = {}
        self._step_count: int = 0
        self._done: bool = False
        self._episode_id: str = ""
        self._task_meta: dict = {}
        self._last_reward: float = 0.0

    # ── OpenEnv core ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict:
        """Begin a fresh episode and return the initial observation."""
        import random

        rng = random.Random(seed if seed is not None else self.seed)
        task = self._loader.load(self.task_id, rng=rng)

        self._patients = task["patients"]
        self._resources = copy.deepcopy(task["resources"])
        self._ground_truth = task["ground_truth"]
        self._step_count = 0
        self._done = False
        self._episode_id = str(uuid.uuid4())[:8]
        self._last_reward = 0.0
        self._task_meta = {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "description": task["description"],
        }

        return self._build_obs()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Process one allocation decision from the agent.
        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError(
                "Episode has already ended — call reset() to start a new one."
            )

        self._step_count += 1
        validation_errors = self._validate(action)

        result = self._grader.grade(
            patients=self._patients,
            action=action,
            ground_truth=self._ground_truth,
            resources=self._resources,
        )

        reward = result["total_score"]
        self._last_reward = reward

        done = (self._step_count >= self.MAX_STEPS) or (reward >= 0.92)
        self._done = done

        info = {
            **self._task_meta,
            "episode_id": self._episode_id,
            "step": self._step_count,
            "validation_errors": validation_errors,
            "grade_breakdown": result,
            "hint": (
                self._hint(result) if not done else "Episode complete."
            ),
        }

        return self._build_obs(), reward, done, info

    def state(self) -> dict:
        """Return a full snapshot of the current environment state."""
        return {
            "episode_id": self._episode_id,
            "patients": copy.deepcopy(self._patients),
            "resources": copy.deepcopy(self._resources),
            "step": self._step_count,
            "done": self._done,
            **self._task_meta,
        }

    def render(self) -> None:
        if self.render_mode == "json":
            print(json.dumps(self._build_obs(), indent=2))
            return

        r = self._resources
        print(f"\n{'═' * 65}")
        print(f" ICUResourceAllocationEnv-v1 | Step {self._step_count}/{self.MAX_STEPS}")
        print(f" Resources available:")
        print(
            f"  Beds={r['beds']}  Vents={r['ventilators']}  "
            f"Nurses={r['nurses']}h  Vasopressors={r['vasopressors']}"
        )
        print(f"{'─' * 65}")
        for p in self._patients:
            ts_flag = "  ⚡TIME-SENSITIVE" if p["time_sensitive"] else ""
            needs = p["resources_needed"]
            flags = [k.upper() for k in ("bed", "ventilator", "vasopressors") if needs.get(k)]
            print(
                f"  [{p['id']}] SEV={p['severity']}/5  Age={p['age']}  "
                f"Prog={p['prognosis']:.0%}{ts_flag}"
            )
            print(f"       Dx: {p['diagnosis']}")
            print(f"       Needs: {', '.join(flags)}  NurseHrs={needs['nurse_hours']}")
        print(f"{'═' * 65}\n")

    def close(self) -> None:
        """Release any held resources (no-op for this environment)."""
        pass

    # ── Space descriptors ─────────────────────────────────────────────────

    @property
    def observation_space(self) -> dict:
        return {
            "type": "dict",
            "properties": {
                "patients":  {"type": "array"},
                "resources": {"type": "object"},
                "step":      {"type": "integer"},
                "done":      {"type": "boolean"},
            },
        }

    @property
    def action_space(self) -> dict:
        return {
            "type": "dict",
            "required": ["allocations"],
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["patient_id", "admit", "resources_assigned"],
                        "properties": {
                            "patient_id":         {"type": "string"},
                            "admit":              {"type": "boolean"},
                            "resources_assigned": {"type": "object"},
                        },
                    },
                }
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        return {
            "patients":  copy.deepcopy(self._patients),
            "resources": copy.deepcopy(self._resources),
            "step":      self._step_count,
            "done":      self._done,
        }

    def _validate(self, action: dict) -> list[str]:
        errors: list[str] = []
        known_ids = {p["id"] for p in self._patients}

        if "allocations" not in action:
            errors.append("Missing required key 'allocations'.")
            return errors

        for entry in action["allocations"]:
            pid = entry.get("patient_id", "?")
            if pid not in known_ids:
                errors.append(f"Unknown patient_id: {pid}")
            if "admit" not in entry:
                errors.append(f"Missing 'admit' for patient {pid}")
            if "resources_assigned" not in entry:
                errors.append(f"Missing 'resources_assigned' for patient {pid}")

        return errors

    def _hint(self, result: dict) -> str:
        tips = []
        if result["survival_outcome_score"] < 0.7:
            tips.append("Focus on high-severity patients with good prognosis.")
        if result["resource_efficiency_score"] < 0.7:
            tips.append("Check resource totals — you may be over the limit.")
        if result["triage_correctness_score"] < 0.8:
            tips.append("Time-sensitive cases need correct urgent routing.")
        if result["fairness_score"] < 0.6:
            tips.append("Admission decisions should be driven by severity, not age.")
        return " | ".join(tips) if tips else "Good allocation — keep refining."
