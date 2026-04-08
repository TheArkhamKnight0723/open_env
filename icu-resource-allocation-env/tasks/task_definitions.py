"""
tasks/task_definitions.py

Three ICU surge scenarios used by the environment and grader.
Each task stresses a different allocation challenge:

  task_surge_001 — manageable surge      (easy):   4 patients
  task_surge_002 — ventilator scarcity   (medium):  6 patients
  task_surge_003 — multi-wave crisis     (hard):    8 patients, 3 arrival waves
"""

from __future__ import annotations

TASKS: list[dict] = [
    # ════════════════════════════════════════════════ TASK 1 — easy
    {
        "task_id":     "task_surge_001",
        "difficulty":  "easy",
        "description": "Small surge: 4 patients, tight but manageable resources.",
        "resources": {
            "beds": 3,
            "ventilators": 1,
            "nurses": 6.0,
            "vasopressors": 2,
        },
        "patients": [
            {
                "id": "P001", "severity": 5,
                "diagnosis": "Acute respiratory failure",
                "age": 52, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 2.0, "vasopressors": False,
                },
                "prognosis": 0.80,
                "notes": "Requires immediate ventilation or prognosis drops to 0.20.",
            },
            {
                "id": "P002", "severity": 4,
                "diagnosis": "Septic shock",
                "age": 67, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.5, "vasopressors": True,
                },
                "prognosis": 0.65,
                "notes": "Vasopressors are critical for haemodynamic stability.",
            },
            {
                "id": "P003", "severity": 2,
                "diagnosis": "Post-operative monitoring",
                "age": 44, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.0, "vasopressors": False,
                },
                "prognosis": 0.95,
                "notes": "Stable; could be safely managed in a step-down unit.",
            },
            {
                "id": "P004", "severity": 3,
                "diagnosis": "Acute kidney injury with fluid overload",
                "age": 71, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.5, "vasopressors": False,
                },
                "prognosis": 0.72,
                "notes": "Needs a bed but is not immediately time-critical.",
            },
        ],
        "ground_truth": {
            "admit": {"P001": True, "P002": True, "P003": True, "P004": False},
            "ventilator": {"P001": True, "P002": False, "P003": False, "P004": False},
            "vasopressors": {"P001": False, "P002": True, "P003": False, "P004": False},
            "priority_order": ["P001", "P002", "P004", "P003"],
            "rationale": (
                "P001 urgently needs the sole ventilator — prognosis halves without it. "
                "P002 requires vasopressors for septic shock. "
                "P003 is stable enough for step-down. "
                "P004 deferred if beds run short."
            ),
        },
    },

    # ════════════════════════════════════════════════ TASK 2 — medium
    {
        "task_id":     "task_surge_002",
        "difficulty":  "medium",
        "description": "Mass casualty: 6 patients, critically scarce ventilators.",
        "resources": {
            "beds": 4,
            "ventilators": 1,
            "nurses": 8.0,
            "vasopressors": 2,
        },
        "patients": [
            {
                "id": "P001", "severity": 5,
                "diagnosis": "ARDS from chemical exposure",
                "age": 34, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 2.5, "vasopressors": False,
                },
                "prognosis": 0.75,
                "notes": "Young patient; good prognosis if ventilated promptly.",
            },
            {
                "id": "P002", "severity": 5,
                "diagnosis": "End-stage COPD exacerbation",
                "age": 81, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 2.0, "vasopressors": False,
                },
                "prognosis": 0.25,
                "notes": "Very poor prognosis even with ventilation; comfort care discussed.",
            },
            {
                "id": "P003", "severity": 4,
                "diagnosis": "Cardiogenic shock post MI",
                "age": 58, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 2.0, "vasopressors": True,
                },
                "prognosis": 0.60,
                "notes": "Vasopressors and a bed are urgently needed.",
            },
            {
                "id": "P004", "severity": 3,
                "diagnosis": "Diabetic ketoacidosis",
                "age": 29, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.5, "vasopressors": False,
                },
                "prognosis": 0.97,
                "notes": "Very high survival with standard treatment.",
            },
            {
                "id": "P005", "severity": 4,
                "diagnosis": "Meningococcal septicaemia",
                "age": 22, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 2.0, "vasopressors": True,
                },
                "prognosis": 0.70,
                "notes": "Young; vasopressors needed; time-sensitive.",
            },
            {
                "id": "P006", "severity": 2,
                "diagnosis": "Pneumonia monitoring",
                "age": 55, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.0, "vasopressors": False,
                },
                "prognosis": 0.90,
                "notes": "Stable; consider step-down ward.",
            },
        ],
        "ground_truth": {
            "admit": {
                "P001": True, "P002": True, "P003": True,
                "P004": True, "P005": True, "P006": False,
            },
            "ventilator": {
                "P001": True, "P002": False, "P003": False,
                "P004": False, "P005": False, "P006": False,
            },
            "vasopressors": {
                "P001": False, "P002": False, "P003": True,
                "P004": False, "P005": True, "P006": False,
            },
            "priority_order": ["P001", "P003", "P005", "P004", "P002", "P006"],
            "rationale": (
                "Ventilator goes to P001 (young, good prognosis, ARDS). "
                "P002 has very poor outlook — comfort care preferred. "
                "P003 and P005 need vasopressors urgently. "
                "P004 stable with high survival. P006 to step-down."
            ),
        },
    },

    # ════════════════════════════════════════════════ TASK 3 — hard
    {
        "task_id":     "task_surge_003",
        "difficulty":  "hard",
        "description": "Multi-wave crisis: 8 patients across 3 arrival waves, cascading scarcity.",
        "resources": {
            "beds": 5,
            "ventilators": 2,
            "nurses": 10.0,
            "vasopressors": 3,
        },
        "patients": [
            # Wave 1
            {
                "id": "P001", "severity": 5,
                "diagnosis": "Traumatic brain injury",
                "age": 25, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 3.0, "vasopressors": False,
                },
                "prognosis": 0.60,
                "notes": "Wave 1. Needs vent and close neuro monitoring.",
            },
            {
                "id": "P002", "severity": 5,
                "diagnosis": "Massive GI bleed with shock",
                "age": 63, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 2.5, "vasopressors": True,
                },
                "prognosis": 0.55,
                "notes": "Wave 1. Vasopressors and bed are critical.",
            },
            {
                "id": "P003", "severity": 3,
                "diagnosis": "Exacerbation of heart failure",
                "age": 78, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.5, "vasopressors": False,
                },
                "prognosis": 0.70,
                "notes": "Wave 1. Stable for now.",
            },
            # Wave 2
            {
                "id": "P004", "severity": 5,
                "diagnosis": "Eclampsia with multi-organ failure",
                "age": 31, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 2.5, "vasopressors": True,
                },
                "prognosis": 0.72,
                "notes": "Wave 2. Pregnant — fetal survival also at stake.",
            },
            {
                "id": "P005", "severity": 4,
                "diagnosis": "Severe burns (40% TBSA)",
                "age": 42, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 3.0, "vasopressors": False,
                },
                "prognosis": 0.50,
                "notes": "Wave 2. High nursing burden.",
            },
            {
                "id": "P006", "severity": 2,
                "diagnosis": "Hyperosmolar hyperglycaemic state",
                "age": 55, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.0, "vasopressors": False,
                },
                "prognosis": 0.93,
                "notes": "Wave 2. Manageable without ICU admission.",
            },
            # Wave 3
            {
                "id": "P007", "severity": 4,
                "diagnosis": "Pulmonary embolism",
                "age": 47, "time_sensitive": True,
                "resources_needed": {
                    "bed": True, "ventilator": False,
                    "nurse_hours": 1.5, "vasopressors": True,
                },
                "prognosis": 0.80,
                "notes": "Wave 3. High survival if treated quickly.",
            },
            {
                "id": "P008", "severity": 3,
                "diagnosis": "Post-cardiac arrest — ROSC achieved",
                "age": 70, "time_sensitive": False,
                "resources_needed": {
                    "bed": True, "ventilator": True,
                    "nurse_hours": 2.0, "vasopressors": False,
                },
                "prognosis": 0.40,
                "notes": "Wave 3. Neurological outcome uncertain.",
            },
        ],
        "ground_truth": {
            "admit": {
                "P001": True, "P002": True, "P003": True,
                "P004": True, "P005": True, "P006": False,
                "P007": True, "P008": False,
            },
            "ventilator": {
                "P001": True, "P002": False, "P003": False,
                "P004": True, "P005": False, "P006": False,
                "P007": False, "P008": False,
            },
            "vasopressors": {
                "P001": False, "P002": True, "P003": False,
                "P004": True, "P005": False, "P006": False,
                "P007": True, "P008": False,
            },
            "priority_order": [
                "P001", "P004", "P002", "P007", "P005", "P003", "P008", "P006"
            ],
            "rationale": (
                "P001 and P004 receive the two ventilators (both high-severity, good prognosis). "
                "P002 needs vasopressors urgently. P004 (eclampsia) — fetal stakes raise priority. "
                "P007 has high survival with fast vasopressor treatment. "
                "P008 deferred — uncertain neurological outcome. P006 to step-down."
            ),
        },
    },
]
