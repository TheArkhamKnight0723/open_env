"""
app.py

Hugging Face Spaces entry point.

Architecture:
  - server.py FastAPI app handles ALL OpenEnv endpoints at root:
      GET  /       health check
      POST /reset  OpenEnv reset
      POST /step   OpenEnv step
      GET  /state  OpenEnv state
  - Gradio UI is mounted at /ui for the interactive demo

The validator hits /reset, /step, /state at the ROOT — no prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Import the real FastAPI app from server — it already has all endpoints
from server import app

from icu_env import ICUResourceAllocationEnv
from rule_based_agent import RuleBasedAgent
from llm_agent import LLMAgent
from task_definitions import TASKS


# ── Gradio episode runner ────────────────────────────────────────────────────

def _build_agent(agent_type: str):
    if agent_type == "LLM Agent":
        return LLMAgent()
    return RuleBasedAgent()


def run_episode(task_id: str, agent_type: str) -> str:
    """Execute one full episode and return a human-readable transcript."""
    agent = _build_agent(agent_type)
    env = ICUResourceAllocationEnv(task_id=task_id, seed=0)
    obs = env.reset(seed=0)

    lines = [f"=== Episode: {task_id} | Agent: {agent_type} ===\n"]

    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        gb = info["grade_breakdown"]

        admitted = gb["details"]["admitted"]
        denied = gb["details"]["not_admitted"]

        lines.append(f"── Step {info['step']} ──────────────────────")
        lines.append(f"Admitted : {', '.join(admitted) or 'none'}")
        lines.append(f"Denied   : {', '.join(denied) or 'none'}")
        lines.append(f"Reward   : {reward:.4f}")
        lines.append(f"  Survival   : {gb['survival_outcome_score']:.4f}")
        lines.append(f"  Efficiency : {gb['resource_efficiency_score']:.4f}")
        lines.append(f"  Fairness   : {gb['fairness_score']:.4f}")
        lines.append(f"  Triage     : {gb['triage_correctness_score']:.4f}")
        if info.get("hint"):
            lines.append(f"  Hint       : {info['hint']}")
        lines.append("")

    lines.append(f"Final reward: {reward:.4f} | {'PASS' if reward >= 0.5 else 'FAIL'}")
    env.close()
    return "\n".join(lines)


_task_choices = [t["task_id"] for t in TASKS]

with gr.Blocks(title="ICU Resource Allocation — OpenEnv") as _demo:
    gr.Markdown(
        "# ICU Resource Allocation — OpenEnv\n"
        "Select a surge scenario and an agent, then click **Run Episode** "
        "to watch the agent allocate scarce ICU resources step-by-step.\n\n"
        "API endpoints: `POST /reset` | `POST /step` | `GET /state`"
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=_task_choices,
            value=_task_choices[0],
            label="Surge Scenario",
        )
        agent_dropdown = gr.Dropdown(
            choices=["Rule-Based Agent", "LLM Agent"],
            value="Rule-Based Agent",
            label="Agent",
        )

    run_btn = gr.Button("Run Episode", variant="primary")
    output = gr.Textbox(label="Episode Transcript", lines=30, interactive=False)

    run_btn.click(
        fn=run_episode,
        inputs=[task_dropdown, agent_dropdown],
        outputs=output,
    )


# Mount Gradio at /ui — keeps the root free for OpenEnv API endpoints
app = gr.mount_gradio_app(app, _demo, path="/ui")
