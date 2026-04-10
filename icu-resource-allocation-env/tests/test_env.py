"""
tests/test_env.py

Unit and integration tests for ICUResourceAllocationEnv and ICUGrader.
Covers the full OpenEnv interface contract plus grader scoring logic.

Run with:
    python -m pytest tests/test_env.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from icu_env import ICUResourceAllocationEnv
from icu_grader import ICUGrader
from task_definitions import TASKS


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = ICUResourceAllocationEnv(task_id="task_surge_001", seed=0)
    e.reset(seed=0)
    return e


@pytest.fixture
def grader():
    return ICUGrader()


def _admit_all_action(patients: list[dict]) -> dict:
    """Helper: admit every patient with all resources they need."""
    allocations = []
    for p in patients:
        needs = p["resources_needed"]
        allocations.append({
            "patient_id": p["id"],
            "admit": True,
            "resources_assigned": {
                "bed":          True,
                "ventilator":   needs.get("ventilator", False),
                "nurse_hours":  needs["nurse_hours"],
                "vasopressors": needs.get("vasopressors", False),
            },
        })
    return {"allocations": allocations}


def _deny_all_action(patients: list[dict]) -> dict:
    """Helper: deny every patient."""
    return {
        "allocations": [
            {
                "patient_id": p["id"],
                "admit": False,
                "resources_assigned": {
                    "bed": False,
                    "ventilator": False,
                    "nurse_hours": 0.0,
                    "vasopressors": False,
                },
            }
            for p in patients
        ]
    }


# ── reset() ───────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_dict(self, env):
        obs = env.reset(seed=1)
        assert isinstance(obs, dict)

    def test_observation_keys(self, env):
        obs = env.reset(seed=0)
        assert "patients" in obs
        assert "resources" in obs
        assert "step" in obs
        assert "done" in obs

    def test_step_starts_at_zero(self, env):
        obs = env.reset(seed=0)
        assert obs["step"] == 0

    def test_done_is_false_on_reset(self, env):
        obs = env.reset(seed=0)
        assert obs["done"] is False

    def test_patients_non_empty(self, env):
        obs = env.reset(seed=0)
        assert len(obs["patients"]) > 0

    def test_resources_has_expected_keys(self, env):
        obs = env.reset(seed=0)
        for key in ("beds", "ventilators", "nurses", "vasopressors"):
            assert key in obs["resources"]

    def test_repeated_reset_gives_fresh_episode(self, env):
        env.reset(seed=0)
        action = _admit_all_action(env._patients)
        env.step(action)
        obs2 = env.reset(seed=0)
        assert obs2["step"] == 0
        assert obs2["done"] is False


# ── step() ────────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_four_tuple(self, env):
        obs = env.reset(seed=0)
        result = env.step(_admit_all_action(obs["patients"]))
        assert len(result) == 4

    def test_reward_in_range(self, env):
        obs = env.reset(seed=0)
        _, reward, _, _ = env.step(_admit_all_action(obs["patients"]))
        assert 0.0 <= reward <= 1.0

    def test_done_becomes_true_after_max_steps(self, env):
        obs = env.reset(seed=0)
        done = False
        step_count = 0
        while not done:
            action = _admit_all_action(obs["patients"])
            obs, _, done, _ = env.step(action)
            step_count += 1
            assert step_count <= ICUResourceAllocationEnv.MAX_STEPS + 1

        assert done is True

    def test_info_contains_grade_breakdown(self, env):
        obs = env.reset(seed=0)
        _, _, _, info = env.step(_admit_all_action(obs["patients"]))
        assert "grade_breakdown" in info
        gb = info["grade_breakdown"]
        assert "total_score" in gb
        assert "survival_outcome_score" in gb

    def test_missing_allocations_key_captured(self, env):
        env.reset(seed=0)
        _, _, _, info = env.step({"allocations": []})
        # With empty allocations we still get a grade, not an exception
        assert "grade_breakdown" in info

    def test_step_after_done_raises(self, env):
        obs = env.reset(seed=0)
        done = False
        while not done:
            obs, _, done, _ = env.step(_admit_all_action(obs["patients"]))
        with pytest.raises(RuntimeError):
            env.step(_admit_all_action(obs["patients"]))

    def test_step_increments_counter(self, env):
        obs = env.reset(seed=0)
        assert obs["step"] == 0
        obs, _, _, _ = env.step(_admit_all_action(obs["patients"]))
        assert obs["step"] == 1


# ── state() ───────────────────────────────────────────────────────────────

class TestState:
    def test_state_returns_dict(self, env):
        env.reset(seed=0)
        s = env.state()
        assert isinstance(s, dict)

    def test_state_has_episode_id(self, env):
        env.reset(seed=0)
        s = env.state()
        assert "episode_id" in s

    def test_state_does_not_advance_step(self, env):
        env.reset(seed=0)
        env.state()
        env.state()
        assert env._step_count == 0


# ── render() / close() ────────────────────────────────────────────────────

class TestRenderClose:
    def test_render_human_does_not_crash(self, env, capsys):
        env.reset(seed=0)
        env.render()
        captured = capsys.readouterr()
        assert "ICUResourceAllocationEnv" in captured.out

    def test_close_is_noop(self, env):
        env.reset(seed=0)
        env.close()  # should not raise


# ── Grader ────────────────────────────────────────────────────────────────

class TestGrader:
    def test_total_score_in_range(self, grader):
        task = TASKS[0]
        patients = task["patients"]
        action   = _admit_all_action(patients)
        result   = grader.grade(
            patients=patients,
            action=action,
            ground_truth=task["ground_truth"],
            resources=task["resources"],
        )
        assert 0.0 <= result["total_score"] <= 1.0

    def test_deny_all_gives_low_score(self, grader):
        task = TASKS[0]
        patients = task["patients"]
        action   = _deny_all_action(patients)
        result   = grader.grade(
            patients=patients,
            action=action,
            ground_truth=task["ground_truth"],
            resources=task["resources"],
        )
        assert result["total_score"] < 0.5

    def test_optimal_action_gives_high_score(self, grader):
        task = TASKS[0]
        patients = task["patients"]
        gt       = task["ground_truth"]
        # Build the ground-truth-optimal action
        allocations = []
        for p in patients:
            pid = p["id"]
            allocations.append({
                "patient_id": pid,
                "admit": gt["admit"].get(pid, False),
                "resources_assigned": {
                    "bed":          gt["admit"].get(pid, False),
                    "ventilator":   gt.get("ventilator", {}).get(pid, False),
                    "nurse_hours":  p["resources_needed"]["nurse_hours"],
                    "vasopressors": gt.get("vasopressors", {}).get(pid, False),
                },
            })
        result = grader.grade(
            patients=patients,
            action={"allocations": allocations},
            ground_truth=gt,
            resources=task["resources"],
        )
        assert result["total_score"] >= 0.75

    def test_over_vent_limit_penalised(self, grader):
        task = TASKS[0]
        patients = task["patients"]
        # Force-assign ventilators to all patients
        allocations = []
        for p in patients:
            allocations.append({
                "patient_id": p["id"],
                "admit": True,
                "resources_assigned": {
                    "bed": True,
                    "ventilator": True,  # all get vents regardless of capacity
                    "nurse_hours": p["resources_needed"]["nurse_hours"],
                    "vasopressors": False,
                },
            })
        result = grader.grade(
            patients=patients,
            action={"allocations": allocations},
            ground_truth=task["ground_truth"],
            resources=task["resources"],
        )
        assert result["resource_efficiency_score"] < 0.8

    def test_grade_returns_admitted_and_denied_lists(self, grader):
        task = TASKS[0]
        patients = task["patients"]
        action   = _admit_all_action(patients)
        result   = grader.grade(
            patients=patients,
            action=action,
            ground_truth=task["ground_truth"],
            resources=task["resources"],
        )
        assert "details" in result
        assert "admitted" in result["details"]
        assert "not_admitted" in result["details"]

    def test_all_three_tasks_scoreable(self, grader):
        for task in TASKS:
            action = _admit_all_action(task["patients"])
            result = grader.grade(
                patients=task["patients"],
                action=action,
                ground_truth=task["ground_truth"],
                resources=task["resources"],
            )
            assert 0.0 <= result["total_score"] <= 1.0


# ── Task loader ───────────────────────────────────────────────────────────

class TestTaskLoader:
    def test_all_tasks_loadable(self):
        from tasks.task_loader import TaskLoader
        loader = TaskLoader()
        for task_id in loader.available_tasks():
            task = loader.load(task_id)
            assert "patients" in task
            assert "resources" in task
            assert "ground_truth" in task

    def test_unknown_task_id_raises(self):
        from tasks.task_loader import TaskLoader
        loader = TaskLoader()
        with pytest.raises(ValueError):
            loader.load("nonexistent_task")

    def test_random_load_returns_valid_task(self):
        import random
        from tasks.task_loader import TaskLoader
        loader = TaskLoader()
        task = loader.load(rng=random.Random(99))
        assert "task_id" in task
