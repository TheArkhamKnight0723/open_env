"""
tests/test_agents.py

Validates the output of all three agents:
  - RuleBasedAgent:  deterministic, fast, always produces valid actions
  - LLMAgent:        mocked so tests run offline without an API key
  - RLAgent:         tested for interface compliance (model file optional)

Run with:
    python -m pytest tests/test_agents.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.icu_env import ICUResourceAllocationEnv
from agents.rule_based_agent import RuleBasedAgent
from tasks.task_definitions import TASKS


# ── Helpers ───────────────────────────────────────────────────────────────

def _fresh_obs(task_id: str = "task_surge_001", seed: int = 0) -> dict:
    env = ICUResourceAllocationEnv(task_id=task_id, seed=seed)
    return env.reset(seed=seed)


def _action_is_valid(action: dict, obs: dict) -> bool:
    """Check the structural validity of an agent action dict."""
    if "allocations" not in action:
        return False
    patient_ids = {p["id"] for p in obs["patients"]}
    returned_ids = {a["patient_id"] for a in action["allocations"]}
    if not patient_ids.issubset(returned_ids):
        return False
    for entry in action["allocations"]:
        if "admit" not in entry:
            return False
        if "resources_assigned" not in entry:
            return False
    return True


# ── RuleBasedAgent ────────────────────────────────────────────────────────

class TestRuleBasedAgent:
    def test_returns_dict(self):
        obs   = _fresh_obs()
        agent = RuleBasedAgent()
        action = agent.act(obs)
        assert isinstance(action, dict)

    def test_action_has_allocations_key(self):
        obs    = _fresh_obs()
        agent  = RuleBasedAgent()
        action = agent.act(obs)
        assert "allocations" in action

    def test_all_patients_covered(self):
        obs   = _fresh_obs()
        agent = RuleBasedAgent()
        action = agent.act(obs)
        patient_ids  = {p["id"] for p in obs["patients"]}
        returned_ids = {a["patient_id"] for a in action["allocations"]}
        assert patient_ids == returned_ids

    def test_action_is_structurally_valid(self):
        obs   = _fresh_obs()
        agent = RuleBasedAgent()
        assert _action_is_valid(agent.act(obs), obs)

    def test_does_not_exceed_bed_limit(self):
        obs   = _fresh_obs()
        agent = RuleBasedAgent()
        action = agent.act(obs)
        beds_used = sum(1 for a in action["allocations"] if a["admit"])
        assert beds_used <= obs["resources"]["beds"]

    def test_does_not_exceed_nurse_limit(self):
        obs   = _fresh_obs()
        agent = RuleBasedAgent()
        action = agent.act(obs)
        pat_map = {p["id"]: p for p in obs["patients"]}
        nurses_used = sum(
            pat_map[a["patient_id"]]["resources_needed"]["nurse_hours"]
            for a in action["allocations"] if a["admit"]
        )
        assert nurses_used <= obs["resources"]["nurses"]

    def test_deterministic_given_same_obs(self):
        obs    = _fresh_obs(seed=7)
        agent  = RuleBasedAgent()
        action1 = agent.act(obs)
        action2 = agent.act(obs)
        assert action1 == action2

    def test_works_on_all_three_tasks(self):
        agent = RuleBasedAgent()
        for task in TASKS:
            obs    = _fresh_obs(task_id=task["task_id"])
            action = agent.act(obs)
            assert _action_is_valid(action, obs)

    def test_produces_nonzero_reward(self):
        from grader.icu_grader import ICUGrader
        task    = TASKS[0]
        obs     = _fresh_obs(task_id=task["task_id"])
        agent   = RuleBasedAgent()
        action  = agent.act(obs)
        grader  = ICUGrader()
        result  = grader.grade(
            patients=task["patients"],
            action=action,
            ground_truth=task["ground_truth"],
            resources=task["resources"],
        )
        assert result["total_score"] > 0.0


# ── LLMAgent (mocked) ─────────────────────────────────────────────────────

class TestLLMAgentMocked:
    """
    All LLM calls are mocked so these tests run fully offline.
    We verify that LLMAgent correctly parses a valid JSON response
    and falls back gracefully when the response is malformed.
    """

    def _make_mock_response(self, allocations: list[dict]):
        import json
        content_mock = MagicMock()
        content_mock.message.content = json.dumps({"allocations": allocations})
        choices_mock = MagicMock()
        choices_mock.__getitem__ = lambda self, i: content_mock
        resp_mock = MagicMock()
        resp_mock.choices = [content_mock]
        return resp_mock

    def test_parses_valid_response(self):
        obs = _fresh_obs()
        patients = obs["patients"]

        # Build a minimal valid allocation list
        allocations = [
            {
                "patient_id": p["id"],
                "admit": True,
                "resources_assigned": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": p["resources_needed"]["nurse_hours"],
                    "vasopressors": False,
                },
            }
            for p in patients
        ]

        with patch("agents.llm_agent.OpenAI") as MockOpenAI:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = (
                self._make_mock_response(allocations)
            )
            MockOpenAI.return_value = client_instance

            from agents.llm_agent import LLMAgent
            agent  = LLMAgent()
            action = agent.act(obs)

        assert _action_is_valid(action, obs)

    def test_falls_back_on_bad_json(self):
        obs = _fresh_obs()

        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "not valid json at all"

        with patch("agents.llm_agent.OpenAI") as MockOpenAI:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = bad_response
            MockOpenAI.return_value = client_instance

            from agents.llm_agent import LLMAgent
            agent  = LLMAgent()
            action = agent.act(obs)

        # Fallback to rule-based agent — should still be valid
        assert _action_is_valid(action, obs)

    def test_falls_back_on_api_error(self):
        obs = _fresh_obs()

        with patch("agents.llm_agent.OpenAI") as MockOpenAI:
            client_instance = MagicMock()
            client_instance.chat.completions.create.side_effect = Exception("timeout")
            MockOpenAI.return_value = client_instance

            from agents.llm_agent import LLMAgent
            agent  = LLMAgent()
            action = agent.act(obs)

        assert _action_is_valid(action, obs)


# ── RLAgent interface ─────────────────────────────────────────────────────

class TestRLAgentInterface:
    """
    Tests that RLAgent raises sensible errors before a model is loaded
    and that the load() method surfaces a clear FileNotFoundError when
    no model file is present.
    """

    def test_predict_before_load_raises(self):
        import numpy as np
        from agents.rl_agent import RLAgent

        agent = RLAgent(model_path="/tmp/nonexistent_model")
        with pytest.raises(RuntimeError):
            agent.predict(np.zeros(52, dtype=np.float32))

    def test_load_missing_file_raises(self):
        from agents.rl_agent import RLAgent

        agent = RLAgent(model_path="/tmp/nonexistent_model")
        with pytest.raises(FileNotFoundError):
            agent.load()
