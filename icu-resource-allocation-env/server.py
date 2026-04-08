"""
server.py

FastAPI server exposing the OpenEnv REST interface at the ROOT path:
  POST /reset  — start a new episode, returns observation directly
  POST /step   — advance the episode with an action dict
  GET  /state  — snapshot of current state without advancing
  GET  /       — health check, must return 200

The validator hits /reset, /step, /state directly at the root.
Response format matches the OpenEnv spec exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from icu_env import ICUResourceAllocationEnv

app = FastAPI(
    title="ICU Resource Allocation — OpenEnv",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared environment instance
_env: Optional[ICUResourceAllocationEnv] = None


def _get_env() -> ICUResourceAllocationEnv:
    global _env
    if _env is None:
        _env = ICUResourceAllocationEnv()
        _env.reset()
    return _env


# ── Request models ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health probe — must return 200."""
    return {
        "status": "ok",
        "env": "ICUResourceAllocationEnv-v1",
        "version": "1.0.0",
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment and return the initial observation.
    Returns observation fields at top level for OpenEnv spec compliance.
    """
    global _env
    _env = ICUResourceAllocationEnv(task_id=req.task_id, seed=req.seed)
    obs = _env.reset(seed=req.seed)

    return {
        "observation": obs,
        "patients":    obs["patients"],
        "resources":   obs["resources"],
        "step":        obs["step"],
        "done":        obs["done"],
        "task_id":     _env._task_meta.get("task_id"),
        "difficulty":  _env._task_meta.get("difficulty"),
        "status":      "reset_ok",
    }


@app.get("/reset")
def reset_get():
    """GET /reset — some validators ping with GET first."""
    return reset(ResetRequest())


@app.post("/step")
def step(req: StepRequest):
    """
    Advance the episode by one step.
    Body: { "action": { "allocations": [...] } }
    """
    env = _get_env()
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs,
        "patients":    obs["patients"],
        "resources":   obs["resources"],
        "step":        obs["step"],
        "done":        done,
        "reward":      reward,
        "info":        info,
    }


@app.get("/state")
def state():
    """Return the current environment state without advancing the episode."""
    env = _get_env()
    return env.state()
