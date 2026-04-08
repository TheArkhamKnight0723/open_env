"""
agents/llm_agent.py

LLM-powered agent for ICU resource allocation.

The agent formats the current observation as a structured prompt, sends it
to the configured LLM via the OpenAI-compatible client, and parses the
JSON allocation it returns.

Required environment variables
-------------------------------
  API_BASE_URL   — LLM endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME     — model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       — API key / Hugging Face token

The agent falls back to the rule-based agent if the LLM call fails or
returns malformed JSON, so episodes always complete without crashing.
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from agents.rule_based_agent import RuleBasedAgent


_SYSTEM_PROMPT = """You are an experienced ICU charge nurse helping an AI system
allocate scarce critical-care resources during a patient surge.

Your task: decide which patients to admit to the ICU and which resources
(bed, ventilator, nurse_hours, vasopressors) to assign to each admitted patient.

Clinical priorities (in order):
  1. Maximise expected survival — prioritise high-severity patients with
     good prognosis over those with very poor outlook.
  2. Respect resource limits — never assign more beds, ventilators, nurses
     or vasopressors than are available.
  3. Handle time-sensitive cases first — patients flagged time_sensitive=true
     cannot wait.
  4. Be fair — do not let age drive decisions; use clinical severity and prognosis.

Respond ONLY with a JSON object in this exact format (no prose, no markdown fences):
{
  "allocations": [
    {
      "patient_id": "<id>",
      "admit": true or false,
      "resources_assigned": {
        "bed": true or false,
        "ventilator": true or false,
        "nurse_hours": <float>,
        "vasopressors": true or false
      }
    }
  ]
}
Every patient in the observation must appear in the allocations list.
"""


class LLMAgent:
    """
    Wraps an OpenAI-compatible LLM and translates its JSON response into
    an action dict for ICUResourceAllocationEnv.
    """

    def __init__(self) -> None:
        api_base  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        api_key   = os.environ.get("HF_TOKEN", "")
        self._model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        self._client   = OpenAI(api_key=api_key, base_url=api_base)
        self._fallback = RuleBasedAgent()

    # ── Public API ────────────────────────────────────────────────────────

    def act(self, observation: dict) -> dict:
        """
        Query the LLM with the current observation and return an allocation dict.
        Falls back to the rule-based agent on any error.
        """
        user_prompt = self._build_prompt(observation)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content.strip()
            action   = self._parse_response(raw_text, observation)
            return action

        except Exception as exc:
            print(
                f"[LLMAgent] LLM call failed ({exc}); using rule-based fallback.",
                file=sys.stderr,
            )
            return self._fallback.act(observation)

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_prompt(self, observation: dict) -> str:
        resources = observation["resources"]
        patients  = observation["patients"]

        lines = [
            "=== Current ICU state ===",
            f"Step:        {observation['step']}",
            f"Beds:        {resources['beds']}",
            f"Ventilators: {resources['ventilators']}",
            f"Nurse hrs:   {resources['nurses']}",
            f"Vasopressors:{resources['vasopressors']}",
            "",
            "Patients:",
        ]

        for p in patients:
            needs = p["resources_needed"]
            lines.append(
                f"  {p['id']}  sev={p['severity']}/5  age={p['age']}"
                f"  time_sensitive={p['time_sensitive']}"
                f"  prognosis={p['prognosis']:.0%}"
                f"  dx={p['diagnosis']}"
                f"  needs=bed:{needs['bed']} vent:{needs['ventilator']}"
                f" nurse_hrs:{needs['nurse_hours']} vaso:{needs['vasopressors']}"
            )

        lines += ["", "Return the allocation JSON now."]
        return "\n".join(lines)

    def _parse_response(self, text: str, observation: dict) -> dict:
        """
        Parse the LLM's JSON response and validate it contains every patient.
        Falls back to the rule-based agent if parsing fails.
        """
        # Strip accidental markdown code fences
        cleaned = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            action = json.loads(cleaned)
        except json.JSONDecodeError:
            print("[LLMAgent] JSON parse error; using fallback.", file=sys.stderr)
            return self._fallback.act(observation)

        patient_ids  = {p["id"] for p in observation["patients"]}
        returned_ids = {a["patient_id"] for a in action.get("allocations", [])}

        if not patient_ids.issubset(returned_ids):
            missing = patient_ids - returned_ids
            print(
                f"[LLMAgent] Missing patient IDs in response: {missing}; using fallback.",
                file=sys.stderr,
            )
            return self._fallback.act(observation)

        return action
