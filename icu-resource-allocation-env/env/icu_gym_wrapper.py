"""
env/icu_gym_wrapper.py — Gymnasium wrapper for ICUResourceAllocationEnv

Converts the OpenEnv dict-based API into a standard gymnasium.Env with
flat Box observation and MultiDiscrete action spaces so that stable-baselines3
can train a PPO agent without custom policy modifications.

Observation vector layout (per patient, 6 features × MAX_PATIENTS):
    [severity, age_norm, time_sensitive, needs_vent, needs_vaso, prognosis]
Plus 4 global resource counts at the end:
    [beds, ventilators, nurses_norm, vasopressors]

Action vector layout (per patient):
    [admit (0/1), assign_vent (0/1), assign_vaso (0/1)]
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.icu_env import ICUResourceAllocationEnv

MAX_PATIENTS = 8
FEATURES_PER_PATIENT = 6
GLOBAL_FEATURES = 4
OBS_DIM = MAX_PATIENTS * FEATURES_PER_PATIENT + GLOBAL_FEATURES


class ICUGymWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper around ICUResourceAllocationEnv.

    Flattens the structured patient/resource observation into a 1-D float
    array and maps a MultiDiscrete action back to the OpenEnv dict format.
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, task_id: str | None = None, seed: int | None = None) -> None:
        super().__init__()

        self._inner = ICUResourceAllocationEnv(task_id=task_id, seed=seed)

        # All features are normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Per patient: admit, assign_vent, assign_vaso — each binary
        self.action_space = spaces.MultiDiscrete(
            [2, 2, 2] * MAX_PATIENTS
        )

        self._last_obs: dict = {}

    # ── gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        obs = self._inner.reset(seed=seed)
        self._last_obs = obs
        return self._encode_obs(obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        dict_action = self._decode_action(action, self._last_obs["patients"])
        obs, reward, done, info = self._inner.step(dict_action)
        self._last_obs = obs
        return self._encode_obs(obs), float(reward), done, False, info

    def render(self) -> None:
        self._inner.render()

    def close(self) -> None:
        self._inner.close()

    # ── Encoding / decoding ───────────────────────────────────────────────

    def _encode_obs(self, obs: dict) -> np.ndarray:
        """Flatten the dict observation into a float32 vector."""
        patients = obs["patients"]
        resources = obs["resources"]
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        for i, p in enumerate(patients[:MAX_PATIENTS]):
            base = i * FEATURES_PER_PATIENT
            vec[base + 0] = p["severity"] / 5.0
            vec[base + 1] = min(p["age"], 100) / 100.0
            vec[base + 2] = float(p["time_sensitive"])
            vec[base + 3] = float(p["resources_needed"].get("ventilator", False))
            vec[base + 4] = float(p["resources_needed"].get("vasopressors", False))
            vec[base + 5] = float(p["prognosis"])

        global_base = MAX_PATIENTS * FEATURES_PER_PATIENT
        vec[global_base + 0] = resources["beds"] / 10.0
        vec[global_base + 1] = resources["ventilators"] / 5.0
        vec[global_base + 2] = resources["nurses"] / 20.0
        vec[global_base + 3] = resources["vasopressors"] / 5.0

        return vec

    def _decode_action(
        self,
        action: np.ndarray,
        patients: list[dict],
    ) -> dict:
        """Convert the flat MultiDiscrete action back to an OpenEnv allocation dict."""
        allocations = []
        for i, patient in enumerate(patients[:MAX_PATIENTS]):
            base = i * 3
            admit = bool(action[base])
            vent = bool(action[base + 1])
            vaso = bool(action[base + 2])

            nurse_hrs = patient["resources_needed"]["nurse_hours"] if admit else 0.0

            allocations.append({
                "patient_id": patient["id"],
                "admit": admit,
                "resources_assigned": {
                    "bed":          admit,
                    "ventilator":   vent and admit,
                    "nurse_hours":  nurse_hrs,
                    "vasopressors": vaso and admit,
                },
            })

        return {"allocations": allocations}
