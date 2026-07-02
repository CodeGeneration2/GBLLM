"""Prompt builders for skill extraction pipeline.

Keeping prompts in a separate module makes the workflow code easier to read and
keeps prompt changes isolated from pipeline control flow.
"""

from __future__ import annotations

import json


def cluster_summary_system_prompt() -> str:
    return "You write concise, transferable Python optimization operator skills. Output JSON only."


def build_cluster_summary_prompt(profile: dict) -> str:
    return (
        "Summarize this weighted optimization cluster into a reusable operator skill. "
        "Use all aggregate evidence and sampled traces. "
        "Avoid problem names, contest names, and dataset-specific entities. "
        "Return STRICT JSON with keys: "
        "cluster_id, proposed_name, description, tags, canonical_triggers, canonical_steps, "
        "common_pitfalls, complexity_time, complexity_space, when_not_to_use."
        f"\n\ncluster_profile:\n{json.dumps(profile, ensure_ascii=False)}"
    )


def operator_example_system_prompt() -> str:
    return "You write concise, realistic Python optimization examples. Output JSON only."


def build_operator_example_prompt(prompt_obj: dict) -> str:
    return (
        "Create a concise Python 'before/after' optimization example for this operator.\n"
        "Requirements:\n"
        '- Return STRICT JSON: {"before_code": str, "after_code": str}\n'
        "- 3-7 lines per snippet.\n"
        "- Make the after-code a clear speed optimization of before-code.\n"
        "- Keep it domain-specific to the provided operator info.\n"
        "- Avoid placeholders and avoid the generic digit-count and csr_matrix/connected_components templates.\n"
        f"\noperator:\n{json.dumps(prompt_obj, ensure_ascii=False)}"
    )


def build_operator_example_compact_prompt(prompt_obj: dict) -> str:
    return (
        "Create a concise Python optimization example.\n"
        'Return STRICT JSON: {"before_code": str, "after_code": str}\n'
        "- 3-6 lines each.\n"
        "- After code must be faster than before code.\n"
        "- Keep it specific to the operator.\n"
        f"\noperator:\n{json.dumps(prompt_obj, ensure_ascii=False)}"
    )


def build_operator_example_relaxed_prompt(prompt_obj: dict) -> str:
    return (
        "Create a concise Python optimization example.\n"
        'Return STRICT JSON: {"before_code": str, "after_code": str}\n'
        "- 3-9 lines each.\n"
        "- After code must be faster than before code.\n"
        f"\noperator:\n{json.dumps(prompt_obj, ensure_ascii=False)}"
    )


def meta_plan_system_prompt() -> str:
    return "You design abstract, reusable SKILL.md meta-skills. Output JSON only."


def build_meta_plan_prompt(sid: str, label: str, goal: str, evidence: dict) -> str:
    return (
        f"Learn meta skill {sid} ({label}).\n"
        f"Goal: {goal}\n\n"
        "This is an abstract reusable method, not a worked example.\n"
        "Requirements:\n"
        "- Do not reconstruct any single benchmark problem.\n"
        "- Do not mention cluster ids like C### or benchmark names.\n"
        "- Do not anchor the method to one example family.\n"
        "- Use concise procedural language for broad competitive-programming optimization tasks.\n\n"
        "Return JSON only with:\n"
        "{\n"
        '  "name": str,\n'
        '  "description": str,\n'
        '  "inputs": [str, ...],\n'
        '  "procedure": [str, ...],\n'
        '  "decision_rules": [str, ...],\n'
        '  "failure_modes": [str, ...],\n'
        '  "output_contract": [str, ...]\n'
        "}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
