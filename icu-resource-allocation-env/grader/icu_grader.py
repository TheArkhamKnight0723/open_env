"""
grader/icu_grader.py

Scores agent allocations across four weighted dimensions.

Component weights
-----------------
  survival_outcome_score    0.45  — expected survival from admitted patients
  resource_efficiency_score 0.30  — correct resource matching, no over-limit
  fairness_score            0.15  — severity-driven admissions, not age-driven
  triage_correctness_score  0.10  — correct handling of time-sensitive cases

Final reward is clamped to [0.0, 1.0].
"""

from __future__ import annotations


class ICUGrader:
    """
    Stateless scorer. Call grade() with a patient list, the agent's action,
    the task ground truth, and the available resource pool.
    """

    WEIGHTS = {
        "survival":   0.45,
        "efficiency": 0.30,
        "fairness":   0.15,
        "triage":     0.10,
    }

    # ── Public API ────────────────────────────────────────────────────────

    def grade(
        self,
        patients: list[dict],
        action: dict,
        ground_truth: dict,
        resources: dict,
    ) -> dict:
        """
        Grade the agent's allocation decisions.

        Returns a dict containing per-component scores, a weighted total,
        and a details block listing admitted and denied patients.
        """
        alloc_map = {a["patient_id"]: a for a in action.get("allocations", [])}
        pat_map   = {p["id"]: p for p in patients}
        all_ids   = [p["id"] for p in patients]

        survival_score   = self._survival_score(pat_map, alloc_map, ground_truth)
        efficiency_score = self._efficiency_score(pat_map, alloc_map, resources)
        fairness_score   = self._fairness_score(pat_map, alloc_map, all_ids)
        triage_score     = self._triage_score(pat_map, alloc_map, ground_truth)

        total = (
            self.WEIGHTS["survival"]   * survival_score
            + self.WEIGHTS["efficiency"] * efficiency_score
            + self.WEIGHTS["fairness"]   * fairness_score
            + self.WEIGHTS["triage"]     * triage_score
        )
        total = round(max(0.0, min(1.0, total)), 4)

        return {
            "survival_outcome_score":    round(survival_score, 4),
            "resource_efficiency_score": round(efficiency_score, 4),
            "fairness_score":            round(fairness_score, 4),
            "triage_correctness_score":  round(triage_score, 4),
            "total_score": total,
            "details": {
                "admitted":     [pid for pid, a in alloc_map.items() if a.get("admit")],
                "not_admitted": [pid for pid in all_ids if not alloc_map.get(pid, {}).get("admit")],
            },
        }

    # ── Component scorers ─────────────────────────────────────────────────

    def _survival_score(
        self,
        pat_map: dict,
        alloc_map: dict,
        gt: dict,
    ) -> float:
        """
        Ratio of the agent's expected aggregate survival to the theoretical
        maximum achievable by the optimal allocation.
        """
        agent_survival   = 0.0
        optimal_survival = 0.0

        for pid, pat in pat_map.items():
            if gt["admit"].get(pid):
                vent_gt = gt.get("ventilator", {}).get(pid, False)
                vaso_gt = gt.get("vasopressors", {}).get(pid, False)
                optimal_survival += self._effective_prognosis(pat, vent_gt, vaso_gt)

            alloc = alloc_map.get(pid, {})
            if alloc.get("admit"):
                res = alloc.get("resources_assigned", {})
                agent_survival += self._effective_prognosis(
                    pat,
                    res.get("ventilator", False),
                    res.get("vasopressors", False),
                )

        if optimal_survival == 0:
            return 1.0

        return min(1.0, agent_survival / optimal_survival)

    def _effective_prognosis(self, pat: dict, vent: bool, vaso: bool) -> float:
        """
        Adjust a patient's base prognosis downward if a required critical
        resource was withheld.
        """
        prog  = pat["prognosis"]
        needs = pat["resources_needed"]

        if needs.get("ventilator") and not vent:
            prog *= 0.30   # severe penalty for missing ventilator
        if needs.get("vasopressors") and not vaso:
            prog *= 0.60   # moderate penalty for missing vasopressors

        return prog

    def _efficiency_score(
        self,
        pat_map: dict,
        alloc_map: dict,
        resources: dict,
    ) -> float:
        """
        Penalise wasteful assignments and any breach of resource limits.
        """
        penalty = 0.0

        beds_used   = sum(1 for a in alloc_map.values() if a.get("admit"))
        vents_used  = sum(
            1 for pid, a in alloc_map.items()
            if a.get("admit") and a.get("resources_assigned", {}).get("ventilator")
        )
        vasos_used  = sum(
            1 for pid, a in alloc_map.items()
            if a.get("admit") and a.get("resources_assigned", {}).get("vasopressors")
        )
        nurses_used = sum(
            pat_map[pid]["resources_needed"]["nurse_hours"]
            for pid, a in alloc_map.items()
            if a.get("admit") and pid in pat_map
        )

        if vents_used  > resources["ventilators"]:
            penalty += 0.40 * (vents_used  - resources["ventilators"])
        if beds_used   > resources["beds"]:
            penalty += 0.30 * (beds_used   - resources["beds"])
        if nurses_used > resources["nurses"]:
            penalty += 0.15 * (nurses_used - resources["nurses"])
        if vasos_used  > resources["vasopressors"]:
            penalty += 0.15 * (vasos_used  - resources["vasopressors"])

        # Assigning a ventilator to a patient who doesn't need one is wasteful
        for pid, a in alloc_map.items():
            if (
                a.get("admit")
                and a.get("resources_assigned", {}).get("ventilator")
                and pid in pat_map
                and not pat_map[pid]["resources_needed"]["ventilator"]
            ):
                penalty += 0.20

        return max(0.0, 1.0 - penalty)

    def _fairness_score(
        self,
        pat_map: dict,
        alloc_map: dict,
        all_ids: list[str],
    ) -> float:
        """
        Measure whether admission decisions are severity-driven.

        A larger severity gap between admitted and denied groups indicates
        that decisions were clinically appropriate rather than age-biased.
        """
        admitted_sevs: list[float] = []
        denied_sevs:   list[float] = []

        for pid in all_ids:
            sev = pat_map[pid]["severity"]
            if alloc_map.get(pid, {}).get("admit"):
                admitted_sevs.append(sev)
            else:
                denied_sevs.append(sev)

        # Neutral score when everyone goes one way
        if not admitted_sevs or not denied_sevs:
            return 0.85

        avg_admitted = sum(admitted_sevs) / len(admitted_sevs)
        avg_denied   = sum(denied_sevs)   / len(denied_sevs)
        sev_gap      = avg_admitted - avg_denied

        return min(1.0, max(0.0, 0.5 + sev_gap / 6.0))

    def _triage_score(
        self,
        pat_map: dict,
        alloc_map: dict,
        gt: dict,
    ) -> float:
        """
        Fraction of time-sensitive patients whose admission decision matches
        the ground-truth recommendation.
        """
        time_critical = [pid for pid, p in pat_map.items() if p["time_sensitive"]]

        if not time_critical:
            return 1.0

        correct = sum(
            1 for pid in time_critical
            if bool(alloc_map.get(pid, {}).get("admit")) == bool(gt["admit"].get(pid))
        )

        return correct / len(time_critical)
