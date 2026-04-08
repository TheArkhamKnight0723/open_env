"""
agents/rl_agent.py

PPO agent for ICU resource allocation.

Wraps stable-baselines3's PPO model so it can be trained, evaluated,
saved, and loaded with a consistent interface that matches the other agents.

The agent operates on the ICUGymWrapper (flat Box observation + MultiDiscrete
action space) rather than the raw OpenEnv dict interface.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "ppo_icu_agent"


class RLAgent:
    """
    Thin wrapper around a stable-baselines3 PPO model.

    Designed to be trained offline (via training/train_ppo.py) and then
    loaded here for evaluation or deployment.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> "RLAgent":
        """Load a previously trained model from disk."""
        from stable_baselines3 import PPO
        from env.icu_gym_wrapper import ICUGymWrapper

        zip_path = self._model_path.with_suffix(".zip")
        if not zip_path.exists():
            raise FileNotFoundError(
                f"No trained model found at '{zip_path}'. "
                "Run training/train_ppo.py first."
            )

        env = ICUGymWrapper()
        self._model = PPO.load(str(self._model_path), env=env)
        return self

    def train(
        self,
        total_timesteps: int = 50_000,
        save_path: str | Path | None = None,
    ) -> "RLAgent":
        """
        Train a fresh PPO model from scratch and save it to disk.

        This is a convenience wrapper; for full control over callbacks and
        hyperparameters use training/train_ppo.py directly.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from env.icu_gym_wrapper import ICUGymWrapper

        env = DummyVecEnv([ICUGymWrapper])
        self._model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
        )
        self._model.learn(total_timesteps=total_timesteps)

        out = Path(save_path) if save_path else self._model_path
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out))
        return self

    def save(self, path: str | Path | None = None) -> None:
        if self._model is None:
            raise RuntimeError("No model loaded or trained yet.")
        out = Path(path) if path else self._model_path
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out))

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Run inference on a flat gym observation vector.
        Returns a flat MultiDiscrete action array.
        """
        if self._model is None:
            raise RuntimeError("Call load() or train() before predict().")
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def act_gym(self, obs: np.ndarray) -> np.ndarray:
        """Alias for predict() — clearer name when used outside training loops."""
        return self.predict(obs)
