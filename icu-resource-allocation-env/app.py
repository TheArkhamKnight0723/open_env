from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Import real app — /reset /step /state are defined in server.py
from server import app

import gradio as gr
from icu_env import ICUResourceAllocationEnv
from rule_based_agent import RuleBasedAgent
from llm_agent import LLMAgent
from task_definitions import TASKS


def _build_agent(agent_type: str):
    if agent_type == "LLM Agent":
        return LLMAgent()
    return RuleBasedAgent()


def run_episode(task_id: str, agent_type: str) -> str:
    agent = _build_agent(agent_type)
    env = ICUResourceAllocationEnv(task_id=task_id, seed=0)
    obs = env.reset(seed=0)
    lines = [f"=== Episode: {task_id} | Agent: {agent_type} ===\n"]
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        gb = info["grade_breakdown"]
        lines.append(f"── Step {info['step']} ──────────────────────")
        lines.append(f"Admitted : {', '.join(gb['details']['admitted']) or 'none'}")
        lines.append(f"Denied   : {', '.join(gb['details']['not_admitted']) or 'none'}")
        lines.append(f"Reward   : {reward:.4f}")
        if info.get("hint"):
            lines.append(f" Hint    : {info['hint']}")
        lines.append("")
    lines.append(f"Final reward: {reward:.4f} | {'PASS' if reward >= 0.5 else 'FAIL'}")
    env.close()
    return "\n".join(lines)


_task_choices = [t["task_id"] for t in TASKS]

with gr.Blocks(title="ICU Resource Allocation — OpenEnv") as _demo:
    gr.Markdown("# ICU Resource Allocation — OpenEnv\nAPI: `POST /reset` | `POST /step` | `GET /state`")
    with gr.Row():
        task_dd = gr.Dropdown(choices=_task_choices, value=_task_choices[0], label="Scenario")
        agent_dd = gr.Dropdown(choices=["Rule-Based Agent", "LLM Agent"], value="Rule-Based Agent", label="Agent")
    run_btn = gr.Button("Run Episode", variant="primary")
    output = gr.Textbox(label="Transcript", lines=30, interactive=False)
    run_btn.click(fn=run_episode, inputs=[task_dd, agent_dd], outputs=output)

# Mount Gradio at /ui — keeps root FREE for OpenEnv API
app = gr.mount_gradio_app(app, _demo, path="/ui")
