"""
training/callbacks.py

Custom stable-baselines3 callbacks used during PPO training.

  RewardLoggerCallback  — records mean episode reward at each rollout
  EarlyStoppingCallback — stops training when a target reward is sustained
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Appends the mean episode reward and current timestep to a list after
    every rollout so training progress can be plotted offline.

    Parameters
    ----------
    log_list : list
        The caller's list to which dicts {"timestep", "mean_reward"} are appended.
    """

    def __init__(self, log_list: list, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log = log_list

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return
        rewards = [ep["r"] for ep in self.model.ep_info_buffer]
        mean_r  = float(np.mean(rewards))
        self._log.append({
            "timestep":   int(self.num_timesteps),
            "mean_reward": round(mean_r, 4),
        })
        if self.verbose:
            print(f"  [RewardLogger] step={self.num_timesteps:>8,}  mean_r={mean_r:.4f}")


class EarlyStoppingCallback(BaseCallback):
    """
    Halts training once the mean episode reward stays above a threshold
    for a given number of consecutive rollouts (patience).

    Parameters
    ----------
    reward_threshold : float
        Target mean reward (e.g. 0.90 corresponds to 90% of optimal).
    patience : int
        How many consecutive rollouts the reward must exceed the threshold
        before training stops.
    """

    def __init__(
        self,
        reward_threshold: float = 0.90,
        patience: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._threshold = reward_threshold
        self._patience  = patience
        self._streak    = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return
        mean_r = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))

        if mean_r >= self._threshold:
            self._streak += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] mean_r={mean_r:.4f} >= {self._threshold} "
                    f"({self._streak}/{self._patience})"
                )
            if self._streak >= self._patience:
                if self.verbose:
                    print(
                        f"  [EarlyStopping] Threshold sustained for {self._patience} "
                        "rollouts — stopping training early."
                    )
                self.model.set_env(None)  # signal SB3 to stop
        else:
            self._streak = 0

    def _on_training_end(self) -> None:
        pass
