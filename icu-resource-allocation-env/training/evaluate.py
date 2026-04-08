"""
training/evaluate.py

Compares rule-based, LLM, and RL agents across all three tasks.

Usage
-----
    python -m training.evaluate [--agent rule|llm|rl|all] [--episodes 5]

Prints a summary table and returns the mean score per agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.icu_env import ICUResourceAllocationEnv
from agents.rule_based_agent import RuleBasedAgent
from tasks.task_definitions import TASKS


def evaluate_rule_based(episodes_per_task: int = 5) -> dict[str, float]:
    """Run the rule-based agent and return per-task mean scores."""
    agent   = RuleBasedAgent()
    results: dict[str, list[float]] = {}

    for task in TASKS:
        task_id = task["task_id"]
        scores: list[float] = []

        for ep in range(episodes_per_task):
            env = ICUResourceAllocationEnv(task_id=task_id, seed=ep)
            obs = env.reset(seed=ep)

            episode_reward = 0.0
            done = False
            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                episode_reward = reward  # final step reward is the grade

            scores.append(episode_reward)

        results[task_id] = float(sum(scores) / len(scores))

    return results


def evaluate_rl(episodes_per_task: int = 5) -> dict[str, float]:
    """Run the trained PPO agent and return per-task mean scores."""
    from env.icu_gym_wrapper import ICUGymWrapper
    from agents.rl_agent import RLAgent

    agent = RLAgent()
    try:
        agent.load()
    except FileNotFoundError as exc:
        print(f"[evaluate] {exc}")
        return {}

    results: dict[str, list[float]] = {}

    for task in TASKS:
        task_id = task["task_id"]
        scores: list[float] = []

        for ep in range(episodes_per_task):
            gym_env = ICUGymWrapper(task_id=task_id, seed=ep)
            flat_obs, _ = gym_env.reset(seed=ep)

            episode_reward = 0.0
            done = False
            while not done:
                action = agent.predict(flat_obs)
                flat_obs, reward, done, truncated, info = gym_env.step(action)
                episode_reward = reward

            scores.append(episode_reward)

        results[task_id] = float(sum(scores) / len(scores))

    return results


def print_table(rule_scores: dict, rl_scores: dict) -> None:
    col_w = 28
    print(f"\n{'Task':<{col_w}} {'Rule-Based':>12} {'PPO (RL)':>12}")
    print("─" * (col_w + 26))
    for task in TASKS:
        tid = task["task_id"]
        rb  = f"{rule_scores.get(tid, 0.0):.4f}"
        rl  = f"{rl_scores.get(tid, 'N/A'):>12}" if tid not in rl_scores else f"{rl_scores[tid]:.4f}"
        print(f"{tid:<{col_w}} {rb:>12} {rl:>12}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ICU agents")
    parser.add_argument(
        "--agent",
        choices=["rule", "rl", "all"],
        default="all",
        help="Which agent(s) to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per task (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rule_scores: dict = {}
    rl_scores:   dict = {}

    if args.agent in ("rule", "all"):
        print(f"Evaluating rule-based agent ({args.episodes} episodes / task)...")
        rule_scores = evaluate_rule_based(args.episodes)

    if args.agent in ("rl", "all"):
        print(f"Evaluating RL agent ({args.episodes} episodes / task)...")
        rl_scores = evaluate_rl(args.episodes)

    print_table(rule_scores, rl_scores)


if __name__ == "__main__":
    main()
