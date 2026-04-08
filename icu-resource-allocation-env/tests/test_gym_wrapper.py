"""
tests/test_gym_wrapper.py

Tests for the ICUGymWrapper gymnasium interface.

Verifies observation and action space shapes, reset/step compliance,
and correct translation between the flat gym API and the OpenEnv dict API.

Run with:
    python -m pytest tests/test_gym_wrapper.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from icu_gym_wrapper import ICUGymWrapper, OBS_DIM, MAX_PATIENTS


@pytest.fixture
def wrapper():
    return ICUGymWrapper(task_id="task_surge_001", seed=0)


# ── Observation space ─────────────────────────────────────────────────────

class TestObservationSpace:
    def test_obs_space_shape(self, wrapper):
        assert wrapper.observation_space.shape == (OBS_DIM,)

    def test_obs_space_dtype(self, wrapper):
        assert wrapper.observation_space.dtype == np.float32

    def test_obs_space_bounds(self, wrapper):
        assert wrapper.observation_space.low.min()  == pytest.approx(0.0)
        assert wrapper.observation_space.high.max() == pytest.approx(1.0)


# ── Action space ──────────────────────────────────────────────────────────

class TestActionSpace:
    def test_action_space_length(self, wrapper):
        # 3 binary decisions per patient slot × MAX_PATIENTS
        assert len(wrapper.action_space.nvec) == MAX_PATIENTS * 3

    def test_action_space_is_binary(self, wrapper):
        assert all(n == 2 for n in wrapper.action_space.nvec)

    def test_sample_action_in_space(self, wrapper):
        action = wrapper.action_space.sample()
        assert wrapper.action_space.contains(action)


# ── reset() ───────────────────────────────────────────────────────────────

class TestGymReset:
    def test_reset_returns_tuple(self, wrapper):
        result = wrapper.reset(seed=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_is_ndarray(self, wrapper):
        obs, info = wrapper.reset(seed=0)
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self, wrapper):
        obs, _ = wrapper.reset(seed=0)
        assert obs.shape == (OBS_DIM,)

    def test_reset_obs_in_bounds(self, wrapper):
        obs, _ = wrapper.reset(seed=0)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_reset_info_is_dict(self, wrapper):
        _, info = wrapper.reset(seed=0)
        assert isinstance(info, dict)


# ── step() ────────────────────────────────────────────────────────────────

class TestGymStep:
    def test_step_returns_five_tuple(self, wrapper):
        wrapper.reset(seed=0)
        action = wrapper.action_space.sample()
        result = wrapper.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, wrapper):
        wrapper.reset(seed=0)
        action = wrapper.action_space.sample()
        obs, _, _, _, _ = wrapper.step(action)
        assert obs.shape == (OBS_DIM,)

    def test_step_reward_float(self, wrapper):
        wrapper.reset(seed=0)
        action = wrapper.action_space.sample()
        _, reward, _, _, _ = wrapper.step(action)
        assert isinstance(reward, float)

    def test_step_reward_in_range(self, wrapper):
        wrapper.reset(seed=0)
        action = wrapper.action_space.sample()
        _, reward, _, _, _ = wrapper.step(action)
        assert 0.0 <= reward <= 1.0

    def test_step_done_is_bool(self, wrapper):
        wrapper.reset(seed=0)
        action = wrapper.action_space.sample()
        _, _, done, _, _ = wrapper.step(action)
        assert isinstance(done, bool)

    def test_episode_terminates(self, wrapper):
        wrapper.reset(seed=0)
        done = False
        steps = 0
        while not done:
            action = wrapper.action_space.sample()
            _, _, done, _, _ = wrapper.step(action)
            steps += 1
            assert steps < 20, "Episode did not terminate within expected steps"

    def test_all_tasks_work_in_wrapper(self):
        for task_id in ("task_surge_001", "task_surge_002", "task_surge_003"):
            w = ICUGymWrapper(task_id=task_id, seed=0)
            obs, _ = w.reset(seed=0)
            assert obs.shape == (OBS_DIM,)
            action = w.action_space.sample()
            result = w.step(action)
            assert len(result) == 5


# ── Encoding / decoding round-trip ────────────────────────────────────────

class TestEncoding:
    def test_encoded_obs_values_normalised(self, wrapper):
        obs, _ = wrapper.reset(seed=0)
        # All values should sit in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_zero_action_does_not_crash(self, wrapper):
        wrapper.reset(seed=0)
        zero_action = np.zeros(MAX_PATIENTS * 3, dtype=np.int64)
        obs, reward, done, _, info = wrapper.step(zero_action)
        assert isinstance(reward, float)

    def test_one_action_does_not_crash(self, wrapper):
        wrapper.reset(seed=0)
        one_action = np.ones(MAX_PATIENTS * 3, dtype=np.int64)
        obs, reward, done, _, info = wrapper.step(one_action)
        assert isinstance(reward, float)
