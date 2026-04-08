"""
inference.py  (root)

Entry point for the OpenEnv hackathon evaluator.

Runs one complete episode per task using the LLM agent, with the rule-based
agent as a fallback, and emits structured stdout logs in the required format:

    [START] {"episode_id": ..., "task_id": ..., "difficulty": ...}
    [STEP]  {"step": 1, "action": {...}, "reward": 0.83, "done": false, ...}
    [END]   {"episode_id": ..., "task_id": ..., "total_reward": 0.83}

The script must complete within 20 minutes on vcpu=2 / 8 GB RAM.
It reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from the environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make sure sub-packages resolve correctly whether this is run from the root
# or from inside the container where the working directory may differ.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from icu_env import ICUResourceAllocationEnv
from llm_agent import LLMAgent
from rule_based_agent import RuleBasedAgent
from task_definitions import TASKS


def _log(tag: str, payload: dict) -> None:
    """Print a structured log line and flush immediately."""
    print(f"{tag} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def run_episode(
    task_id: str,
    agent,
    seed: int = 0,
) -> float:
    """
    Run a single episode for the given task and agent.
    Emits [START], [STEP], and [END] log lines.
    Returns the final episode reward.
    """
    env = ICUResourceAllocationEnv(task_id=task_id, seed=seed)
    obs = env.reset(seed=seed)

    state = env.state()
    _log("[START]", {
        "episode_id": state["episode_id"],
        "task_id":    state["task_id"],
        "difficulty": state["difficulty"],
        "n_patients": len(obs["patients"]),
        "resources":  obs["resources"],
    })

    total_reward = 0.0
    done         = False

    while not done:
        action           = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward     = reward  # final graded reward

        _log("[STEP]", {
            "episode_id":      info["episode_id"],
            "step":            info["step"],
            "reward":          reward,
            "done":            done,
            "grade_breakdown": info["grade_breakdown"],
            "validation_errors": info.get("validation_errors", []),
            "hint":            info.get("hint", ""),
            "action_summary": {
                "admitted": info["grade_breakdown"]["details"]["admitted"],
                "denied":   info["grade_breakdown"]["details"]["not_admitted"],
            },
        })

    _log("[END]", {
        "episode_id":   info["episode_id"],
        "task_id":      task_id,
        "total_reward": total_reward,
        "passed":       total_reward >= 0.5,
    })

    env.close()
    return total_reward


def main() -> None:
    # Decide which agent to use based on env vars
    api_base = os.environ.get("API_BASE_URL", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if api_base and hf_token:
        print("[inference] Using LLMAgent.", file=sys.stderr)
        agent = LLMAgent()
    else:
        print(
            "[inference] API_BASE_URL / HF_TOKEN not set — using RuleBasedAgent.",
            file=sys.stderr,
        )
        agent = RuleBasedAgent()

    all_rewards: list[float] = []

    for task in TASKS:
        reward = run_episode(
            task_id=task["task_id"],
            agent=agent,
            seed=42,
        )
        all_rewards.append(reward)

    mean_reward = sum(all_rewards) / len(all_rewards)
    print(
        json.dumps({
            "summary": "all_tasks_complete",
            "mean_reward": round(mean_reward, 4),
            "per_task": dict(zip([t["task_id"] for t in TASKS], all_rewards)),
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
