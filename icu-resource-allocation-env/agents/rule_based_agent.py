"""
agents/rule_based_agent.py

Greedy heuristic agent for ICU resource allocation.

Decision logic
--------------
1. Sort patients by a composite priority score:
       score = severity * 2 + time_sensitive * 1.5 + prognosis
   Higher score → higher priority.
2. Admit patients in priority order until a resource cap is hit.
3. Assign ventilator and vasopressors only to patients who need them,
   stopping when the pool runs dry.
4. Allocate nurse hours as declared in the patient's resource requirements.

This agent is deliberately simple — it gives a reliable baseline and
is fast enough to run thousands of episodes for comparison against RL.
"""

from __future__ import annotations


class RuleBasedAgent:
    """
    Deterministic greedy agent. Stateless: each call to act() is independent.
    """

    def act(self, observation: dict) -> dict:
        """
        Build an allocation action from the current observation.

        Parameters
        ----------
        observation : dict
            The dict returned by env.reset() or env.step().

        Returns
        -------
        dict
            An action dict compatible with ICUResourceAllocationEnv.step().
        """
        patients  = observation["patients"]
        resources = observation["resources"]

        beds_left   = resources["beds"]
        vents_left  = resources["ventilators"]
        nurses_left = resources["nurses"]
        vasos_left  = resources["vasopressors"]

        sorted_patients = sorted(
            patients,
            key=lambda p: (
                p["severity"] * 2.0
                + float(p["time_sensitive"]) * 1.5
                + p["prognosis"]
            ),
            reverse=True,
        )

        allocations = []
        admit_set: set[str] = set()

        for p in sorted_patients:
            needs    = p["resources_needed"]
            can_bed  = beds_left >= 1
            can_nrs  = nurses_left >= needs["nurse_hours"]
            will_admit = can_bed and can_nrs

            if will_admit:
                beds_left   -= 1
                nurses_left -= needs["nurse_hours"]
                admit_set.add(p["id"])

        for p in sorted_patients:
            pid    = p["id"]
            admit  = pid in admit_set
            needs  = p["resources_needed"]

            assign_vent = False
            assign_vaso = False

            if admit:
                if needs.get("ventilator") and vents_left >= 1:
                    assign_vent = True
                    vents_left -= 1
                if needs.get("vasopressors") and vasos_left >= 1:
                    assign_vaso = True
                    vasos_left -= 1

            allocations.append({
                "patient_id": pid,
                "admit": admit,
                "resources_assigned": {
                    "bed":          admit,
                    "ventilator":   assign_vent,
                    "nurse_hours":  needs["nurse_hours"] if admit else 0.0,
                    "vasopressors": assign_vaso,
                },
            })

        return {"allocations": allocations}
