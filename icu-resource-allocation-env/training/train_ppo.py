"""
training/train_ppo.py

Trains a PPO agent on the ICU resource allocation gym environment.

Usage
-----
    python -m training.train_ppo [--timesteps 100000] [--seed 42]

The trained model is saved to models/ppo_icu_agent.zip.
A reward curve is written to models/reward_curve.json so you can
plot training progress without re-running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on ICUGymWrapper")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total environment steps (default: 100 000)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n-envs",    type=int, default=4,
                        help="Number of parallel rollout environments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ImportError:
        print("stable-baselines3 is not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    from env.icu_gym_wrapper import ICUGymWrapper
    from training.callbacks import RewardLoggerCallback, EarlyStoppingCallback

    def make_env(rank: int):
        def _init():
            return ICUGymWrapper(seed=args.seed + rank)
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    reward_log: list[dict] = []
    reward_logger  = RewardLoggerCallback(reward_log)
    early_stopping = EarlyStoppingCallback(reward_threshold=0.90, patience=10)

    print(f"Training for {args.timesteps:,} steps across {args.n_envs} envs...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[reward_logger, early_stopping],
    )

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model.save(str(model_dir / "ppo_icu_agent"))
    print(f"Model saved → {model_dir / 'ppo_icu_agent.zip'}")

    curve_path = model_dir / "reward_curve.json"
    curve_path.write_text(json.dumps(reward_log, indent=2))
    print(f"Reward curve → {curve_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
