import asyncio
import json
import os
from typing import List, Optional
import httpx
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL    = os.getenv("SPACE_URL", "https://shashwatpandey-0723-icu-resource-allocation.hf.space")
TASK_NAME    = "icu_resource_allocation"
BENCHMARK    = "icu_resource_allocation"
MAX_STEPS    = 3
SUCCESS_THRESHOLD = 0.5

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

SYSTEM_PROMPT = """You are an ICU charge nurse AI. Given patients and available resources, decide admissions.
Respond ONLY with valid JSON:
{"allocations": [{"patient_id": "p1", "admit": true, "resources_assigned": {"bed": true, "ventilator": false, "nurse_hours": 4.0, "vasopressors": false}}]}
Prioritize by severity (5=critical). Do not exceed available resources."""

def get_action(client, obs):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}\n\nDecide allocations."},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        text = completion.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        allocations = []
        for p in obs.get("patients", []):
            allocations.append({
                "patient_id": p["id"],
                "admit": True,
                "resources_assigned": {
                    "bed": p["resources_needed"].get("bed", False),
                    "ventilator": p["resources_needed"].get("ventilator", False),
                    "nurse_hours": p["resources_needed"].get("nurse_hours", 2.0),
                    "vasopressors": p["resources_needed"].get("vasopressors", False),
                }
            })
        return {"allocations": allocations}

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(f"{SPACE_URL}/reset", json={})
            r.raise_for_status()
            obs = r.json()
            done = obs.get("done", False)

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break
                action = get_action(client, obs)
                action_str = json.dumps(action, separators=(",", ":"))
                r = await http.post(f"{SPACE_URL}/step", json={"action": action})
                r.raise_for_status()
                result = r.json()
                reward = float(result.get("reward", 0.0))
                done = result.get("done", False)
                errors = result.get("info", {}).get("validation_errors", [])
                error = str(errors[0]) if errors else None
                obs = result
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                if done:
                    break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
