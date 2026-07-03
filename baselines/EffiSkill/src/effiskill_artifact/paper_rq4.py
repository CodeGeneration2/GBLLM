from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import artifact_root, outputs_root


MODEL_SPECS = {
    "gpt5mini": {
        "display_name": "GPT-5-mini",
        "json_target": "candidates",
        "eval_dir": "gpt5_effiskill",
    },
    "qwen30b": {
        "display_name": "Qwen3-Coder-30B",
        "json_target": "survivor_candidates",
        "eval_dir": "qwen30b-effiskill",
    },
}

FAMILY_LABELS = {
    "impl": "Implementation & constant-factor",
    "algebra": "Algebraic / closed-form reformulation",
    "dp": "DP / state compression",
    "comb": "Combinatorics & number theory",
    "graph": "Graph / data structure / set operations",
}

REPRESENTATIVE_TRANSFORMS = {
    FAMILY_LABELS["impl"]: "constant-factor cleanup; lighter data structures; cheaper arithmetic",
    FAMILY_LABELS["algebra"]: "loop elimination; parity / bitwise simplification; range counting",
    FAMILY_LABELS["dp"]: "state reformulation; rolling-state compression",
    FAMILY_LABELS["comb"]: "modular combinatorics; coprimality reasoning; binomial simplification",
    FAMILY_LABELS["graph"]: "graph restructuring; set-intersection reformulation; incremental search",
}

SKILL_TO_FAMILY = {
    "O001": FAMILY_LABELS["impl"],
    "O002": FAMILY_LABELS["graph"],
    "O003": FAMILY_LABELS["comb"],
    "O004": FAMILY_LABELS["graph"],
    "O005": FAMILY_LABELS["algebra"],
    "O006": FAMILY_LABELS["impl"],
    "O007": FAMILY_LABELS["comb"],
    "O008": FAMILY_LABELS["comb"],
    "O009": FAMILY_LABELS["algebra"],
    "O010": FAMILY_LABELS["dp"],
    "O011": FAMILY_LABELS["impl"],
    "O012": FAMILY_LABELS["algebra"],
    "O013": FAMILY_LABELS["dp"],
    "O014": FAMILY_LABELS["comb"],
    "O015": FAMILY_LABELS["comb"],
    "O016": FAMILY_LABELS["algebra"],
    "O017": FAMILY_LABELS["comb"],
    "O018": FAMILY_LABELS["impl"],
    "O019": FAMILY_LABELS["comb"],
    "O020": FAMILY_LABELS["graph"],
    "O021": FAMILY_LABELS["algebra"],
    "O022": FAMILY_LABELS["algebra"],
    "O023": FAMILY_LABELS["comb"],
    "O024": FAMILY_LABELS["algebra"],
    "O025": FAMILY_LABELS["algebra"],
    "O026": FAMILY_LABELS["comb"],
    "O027": FAMILY_LABELS["comb"],
    "O028": FAMILY_LABELS["impl"],
    "O029": FAMILY_LABELS["dp"],
}


def rq4_root() -> Path:
    return artifact_root() / "data" / "rq4"
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def runtime(entry: dict | None, lang_key: str = "python3") -> float | None:
    if not isinstance(entry, dict):
        return None
    value = (entry.get(lang_key) or {}).get("runtime_sum")
    return float(value) if value is not None else None


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bundle_text(skill_ids: tuple[str, ...]) -> str:
    return " | ".join(skill_ids)


def entropy_bits(probabilities: pd.Series) -> float:
    probs = probabilities[probabilities > 0]
    return float(-(probs * np.log2(probs)).sum())


def effective_number(probabilities: pd.Series) -> float:
    return float(2 ** entropy_bits(probabilities))


def load_registry() -> pd.DataFrame:
    registry_path = rq4_root() / "fix-version" / "skills" / "registry.json"
    records = json.loads(registry_path.read_text())
    registry = pd.DataFrame(records)
    registry = registry[registry["skill_id"].astype(str).str.startswith("O")].copy()
    registry["family"] = registry["skill_id"].map(SKILL_TO_FAMILY)
    return registry[["skill_id", "name", "description", "family"]].drop_duplicates("skill_id").sort_values("skill_id").reset_index(drop=True)


def build_candidate_long(model_name: str, spec: dict) -> pd.DataFrame:
    records = load_jsonl(rq4_root() / f"{model_name}.jsonl")
    private_dir = rq4_root() / spec["eval_dir"] / "private"
    original = load_json(private_dir / "stats_original.json")
    rank_maps = {rank: load_json(private_dir / f"stats_rank_{rank}.json") for rank in range(1, 9)}

    rows: list[dict] = []
    for record in records:
        candidates = record.get(spec["json_target"], [])
        set_to_skills = {
            skill_set["set_id"]: tuple(
                sorted(
                    {
                        item.get("skill_id")
                        for item in skill_set.get("selected_skills", [])
                        if item.get("skill_id")
                    }
                )
            )
            for skill_set in record.get("skill_sets", [])
        }
        problem_id = record["problem_id"]
        original_runtime = runtime(original.get(problem_id))
        for rank in range(1, 9):
            candidate = candidates[rank - 1] if rank - 1 < len(candidates) else None
            candidate_runtime = runtime(rank_maps[rank].get(problem_id))
            valid = (
                original_runtime is not None
                and candidate_runtime is not None
                and original_runtime > 0
                and candidate_runtime > 0
            )
            skill_ids = set_to_skills.get(candidate.get("set_id"), tuple()) if candidate else tuple()
            rows.append(
                {
                    "model": model_name,
                    "problem_id": problem_id,
                    "rank": rank,
                    "rank_col": f"rank_{rank}",
                    "candidate_id": candidate.get("candidate_id") if candidate else None,
                    "set_id": candidate.get("set_id") if candidate else None,
                    "plan_id": candidate.get("plan_id") if candidate else None,
                    "plan_style": candidate.get("plan_style") if candidate else None,
                    "candidate_code": candidate.get("code") if candidate else None,
                    "original_runtime": original_runtime,
                    "candidate_runtime": candidate_runtime,
                    "metric_valid": bool(valid),
                    "improvement": (original_runtime / candidate_runtime - 1.0) if valid else np.nan,
                    "log_speedup": np.log(original_runtime / candidate_runtime) if valid else np.nan,
                    "skill_ids": skill_ids,
                    "skill_count": len(skill_ids),
                    "bundle_id": bundle_text(skill_ids),
                }
            )
    return pd.DataFrame(rows)


def build_problem_bundle_long(candidate_long: pd.DataFrame) -> pd.DataFrame:
    valid = candidate_long[candidate_long["metric_valid"]].copy()
    return (
        valid.groupby(["model", "problem_id", "bundle_id", "skill_count", "skill_ids"], dropna=False)
        .agg(
            n_candidates=("rank", "size"),
            bundle_mean_runtime=("candidate_runtime", "mean"),
            bundle_mean_improvement=("improvement", "mean"),
            bundle_mean_log_speedup=("log_speedup", "mean"),
        )
        .reset_index()
    )


def build_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    candidate_long = pd.concat(
        [build_candidate_long(model_name, spec) for model_name, spec in MODEL_SPECS.items()],
        ignore_index=True,
    )
    skill_credit_long = candidate_long[candidate_long["metric_valid"] & (candidate_long["skill_count"] > 0)].copy()
    skill_credit_long = skill_credit_long.explode("skill_ids").rename(columns={"skill_ids": "skill_id"})
    skill_credit_long = skill_credit_long.merge(registry, on="skill_id", how="left")
    problem_bundle_long = build_problem_bundle_long(candidate_long)

    family_summary = (
        skill_credit_long["family"]
        .value_counts(normalize=True)
        .rename("Usage (%)")
        .mul(100.0)
        .rename_axis("Family")
        .reset_index()
    )
    family_summary["Representative transformations"] = family_summary["Family"].map(REPRESENTATIVE_TRANSFORMS)
    family_summary["Usage (%)"] = family_summary["Usage (%)"].round(1)
    family_summary = family_summary[["Family", "Representative transformations", "Usage (%)"]]
    family_summary.to_csv(output_dir / "table3_skill_family_summary.csv", index=False)

    lines = [
        r"\begin{tabular}{p{3.6cm}p{7.1cm}r}",
        r"\toprule",
        r"Family & Representative transformations & Usage (\%) \\",
        r"\midrule",
    ]
    for _, row in family_summary.iterrows():
        lines.append(f"{row['Family']} & {row['Representative transformations']} & {row['Usage (%)']:.1f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table3_skill_family_summary.tex").write_text("\n".join(lines) + "\n")

    claim_rows = []
    claim_rows.append({"claim": "operator_skill_count", "value": int(len(registry))})
    claim_rows.append({"claim": "candidate_skill_pairs", "value": int(len(skill_credit_long))})
    claim_rows.append({"claim": "problem_bundle_observations", "value": int(len(problem_bundle_long))})
    for model_name, spec in MODEL_SPECS.items():
        frame = skill_credit_long[skill_credit_long["model"] == model_name]
        probs = frame["skill_id"].value_counts(normalize=True)
        claim_rows.append({"claim": f"effective_skills_{model_name}", "value": round(effective_number(probs), 2)})
        claim_rows.append({"claim": f"top5_share_{model_name}", "value": round(float(probs.head(5).sum() * 100.0), 1)})
    claims = pd.DataFrame(claim_rows)
    claims.to_csv(output_dir / "rq4_claims.csv", index=False)

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the paper-facing RQ4 outputs.")
    parser.add_argument("--output-dir", type=Path, default=outputs_root() / "rq4")
    return parser


def run(output_dir: Path) -> None:
    build_outputs(output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run(args.output_dir)


if __name__ == "__main__":
    main()
