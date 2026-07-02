"""Skill extraction pipeline for Stage 1 and Stage 3.

A1: Normalize traces and attach Stage 1 / Stage 3 weights.
A2: Build signature/profile artifacts from normalized traces.
A3: Hybrid clustering (TF-IDF + embedding view) and cluster summarization.
A4: Distill operator skills from summarized clusters.
A5: Learn meta skills from traces + cluster summaries.
A6: Build registry from SKILL.md frontmatter.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .build_registry import build_registry
from .extract_skill_prompts_cpp import (
    build_cluster_summary_prompt,
    build_meta_plan_prompt,
    build_operator_example_compact_prompt,
    build_operator_example_prompt,
    build_operator_example_relaxed_prompt,
    cluster_summary_system_prompt,
    meta_plan_system_prompt,
    operator_example_system_prompt,
)
from .llm_utils import cached_chat_completion, configure_default_client, log_llm_failure

CONTEST_PATTERNS = re.compile(
    r"\b(AtCoder|Codeforces|LeetCode|HackerRank|HackerEarth|POJ|UVA|SPOJ|Kattis|BOJ|ABC\d+|ARC\d+|AGC\d+|CF\d+|ICPC|IOI)\b",
    re.I,
)

STAGE3_FAILURE_PENALTY_DEFAULT = 0.75
STAGE3_NOISE_BAND_DEFAULT = 0.10
STAGE3_RUNTIME_CLIP_DEFAULT = 4.0
STAGE3_RIDGE_ALPHA_DEFAULT = 1.0
STAGE3_SHRINKAGE_LAMBDA_DEFAULT = 10.0
STAGE3_TRACE_WEIGHT_LOG_CLIP = 2.0
STAGE3_DECISION_MIN_UTILITY_DEFAULT = 0.0
STAGE3_REVISE_SIMILARITY_DEFAULT = 0.60
STAGE3_ADD_MAX_SIMILARITY_DEFAULT = 0.42
STAGE3_ADD_MIN_SUPPORT_DEFAULT = 12
STAGE3_LIBRARY_DEDUP_SIMILARITY_DEFAULT = 0.78
SKILL_LANGUAGE = "cpp"
CODE_FENCE = "cpp"
LINE_COMMENT = "//"
AUTO_K_EFFECTIVE_N_ENV = "SELF_EVOLVE_AUTO_K_EFFECTIVE_N"


def _prepare_sentence_transformers_env() -> None:
    """Keep sentence-transformers on the PyTorch path.

    In some conda setups, allowing TF/Keras auto-import can crash the process
    inside pyarrow before Python can raise an exception.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _source_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _clean_text(text: str) -> str:
    text = CONTEST_PATTERNS.sub("", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_name(text: str, fallback: str = "Operator Skill") -> str:
    s = _clean_text(text)
    if not s:
        return fallback
    # Split CamelCase-like names for readability.
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s == s.lower():
        s = s.title()
    return s or fallback


def _scalar_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return _clean_text(value) or default
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        before = _clean_text(value.get("before", ""))
        after = _clean_text(value.get("after", ""))
        if before and after:
            return f"{before} -> {after}"
        if after:
            return after
        if before:
            return before
        return default
    if isinstance(value, list):
        if not value:
            return default
        return _clean_text(value[0]) or default
    return _clean_text(value) or default


def _truncate_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    # Prefer clean sentence/phrase boundary rather than adding ellipsis artifacts.
    boundary = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "), cut.rfind(" "))
    if boundary >= int(max_chars * 0.6):
        cut = cut[:boundary].rstrip()
    return cut.rstrip(" .,:;-")


def _sanitize_prose(text: str) -> str:
    text = _clean_text(text)
    text = text.replace("…", " ")
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_truncated_artifact(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return True
    low = t.lower()
    tail = low.rstrip("`'\" )].,:;-")
    # Clear corruption fragments seen in clipped generations.
    if re.match(r"^(?:k\)\.?|1\]\.?|[a-z]\)?\.?)$", tail):
        return True
    if re.search(r"(?:start\)\s*//\s*step|start_k\s*\+\s*1\)|z\)\s*/\s*\(x\s*\+\s*y\)\))", low):
        return True
    if len(tail) <= 4 and any(ch in t for ch in "()[]{}"):
        return True
    if low.endswith(":"):
        return True
    if re.search(r":\s*-\s*", low):
        return True
    if re.search(r"[=+\-*/]\s*$", low):
        return True
    if any(
        tail.endswith(x)
        for x in (
            "leading to",
            "resulting in",
            "causing",
            "where",
            "which",
            "instead of",
            "such as",
            "for example",
            "e.g.",
            "e.g",
            "etc",
            "or",
            "and",
            "up to",
            "if",
            "when",
            "while",
            "may",
            "can",
            "could",
            "would",
            "should",
            "is",
            "are",
            "be",
            "from",
            "to",
            "with",
            "by",
        )
    ):
        return True
    if t.count("(") > t.count(")"):
        return True
    if t.count("[") > t.count("]"):
        return True
    return False


def _listify(value: Any, split_on_symbols: bool = False) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(v) for v in value if _clean_text(v)]
    if isinstance(value, str):
        raw = _clean_text(value)
        if not raw:
            return []
        if split_on_symbols:
            parts = re.split(r"\s*[|,;]+\s*", raw)
            out = [_clean_text(p) for p in parts if _clean_text(p)]
            return out if out else [raw]
        return [raw]
    return [_clean_text(value)]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*", "", text).strip("`\n")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _llm_json_call(
    messages: List[Dict[str, str]],
    cache_dir: str,
    stage: str,
    model: str,
    temperature: float,
    max_tokens: int,
    out_fail_dir: str,
    reasoning_effort: str = "low",
    fallback_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    content: Optional[str] = None
    for attempt in range(3):
        stage_name = stage if attempt == 0 else f"{stage}_retry{attempt}"
        try:
            content_try = cached_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                stage=stage_name,
                reasoning_effort=reasoning_effort,
            )
            if content_try and content_try.strip():
                content = content_try
                break
        except Exception as exc:
            log_llm_failure(stage_name + "_error", payload, str(exc), out_fail_dir)

    obj = _extract_json(content or "")
    if obj is not None:
        return obj

    if content:
        fix_messages = [
            {
                "role": "system",
                "content": "Fix to strictly valid JSON. Output JSON only.",
            },
            {"role": "user", "content": content},
        ]
        fix_payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": fix_messages,
        }
        try:
            fix_content = cached_chat_completion(
                messages=fix_messages,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                stage=stage + "_fix",
                reasoning_effort="low",
            )
            obj = _extract_json(fix_content)
            if obj is not None:
                return obj
        except Exception as exc:
            log_llm_failure(stage + "_fix_error", fix_payload, str(exc), out_fail_dir)

    if fallback_model and fallback_model != model:
        try:
            fallback_content = cached_chat_completion(
                messages=messages,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                stage=stage + "_fallback",
                reasoning_effort="low",
            )
            obj = _extract_json(fallback_content)
            if obj is not None:
                return obj
        except Exception as exc:
            log_llm_failure(stage + "_fallback_error", payload, str(exc), out_fail_dir)

    log_llm_failure(stage + "_json", payload, content or "", out_fail_dir)
    return None


def _load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_trace(trace: Dict[str, Any], idx: int) -> Dict[str, Any]:
    delta = trace.get("DeltaSummary") or {}
    slow = trace.get("SlowAudit") or {}
    problem = trace.get("ProblemBrief") or {}

    normalized = {
        "trace_id": f"T{idx:07d}",
        "problem_id": trace.get("problem_id"),
        "language": _clean_text(trace.get("language", SKILL_LANGUAGE)) or SKILL_LANGUAGE,
        "source_hash": trace.get("source_hash")
        or _source_hash(
            {
                "problem_id": trace.get("problem_id"),
                "language": trace.get("language", SKILL_LANGUAGE),
                "SlowAudit": slow,
                "FastAudit": trace.get("FastAudit"),
                "DeltaSummary": delta,
            }
        ),
        "ProblemBrief": {
            "constraints_guess": problem.get("constraints_guess", {}),
            "problem_type_tags": _listify(problem.get("problem_type_tags")),
        },
        "SlowAudit": {
            "dominant_ops": _listify(slow.get("dominant_ops")),
            "complexity": slow.get("complexity", {}),
            "bottlenecks": [
                {
                    "kind": _clean_text((b or {}).get("kind")),
                    "detail": _clean_text((b or {}).get("detail")),
                }
                for b in (slow.get("bottlenecks") or [])
                if isinstance(b, dict) and ((b.get("kind") or b.get("detail")))
            ],
        },
        "FastAudit": trace.get("FastAudit") or {},
        "DeltaSummary": {
            "delta_type": _listify(delta.get("delta_type"), split_on_symbols=True),
            "complexity_delta": delta.get("complexity_delta") or {},
            "transformation_steps": _listify(delta.get("transformation_steps")),
            "trigger_signals": _listify(delta.get("trigger_signals")),
            "pitfalls": _listify(delta.get("pitfalls")),
        },
    }

    if not normalized["DeltaSummary"]["delta_type"]:
        normalized["DeltaSummary"]["delta_type"] = ["unknown_transformation"]

    return normalized


def _positive_runtime(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def _stage3_candidate_score(
    best_before: Any,
    runtime_rank: Any,
    *,
    failure_penalty: float,
    noise_band_ratio: float,
    runtime_clip_ratio: float,
) -> Tuple[Optional[float], Dict[str, Any]]:
    baseline_runtime = _positive_runtime(best_before)
    candidate_runtime = _positive_runtime(runtime_rank)
    noise_band_log = float(math.log1p(noise_band_ratio))
    clip_log = float(math.log(runtime_clip_ratio))

    meta: Dict[str, Any] = {
        "baseline_runtime": baseline_runtime,
        "candidate_runtime": candidate_runtime,
        "noise_band_ratio": float(noise_band_ratio),
        "noise_band_log": noise_band_log,
        "runtime_clip_ratio": float(runtime_clip_ratio),
        "runtime_clip_log": clip_log,
        "failure_penalty": float(failure_penalty),
    }

    if baseline_runtime is None:
        meta["score_status"] = "missing_baseline"
        meta["candidate_passed"] = candidate_runtime is not None
        return None, meta

    if candidate_runtime is None:
        meta["score_status"] = "failed_candidate"
        meta["candidate_passed"] = False
        meta["raw_log_speedup"] = None
        meta["clipped_log_speedup"] = None
        meta["dead_zone_applied"] = False
        meta["candidate_score"] = -float(failure_penalty)
        return -float(failure_penalty), meta

    raw_log_speedup = float(math.log(baseline_runtime / candidate_runtime))
    clipped_log_speedup = float(np.clip(raw_log_speedup, -clip_log, clip_log))
    dead_zone_applied = bool(abs(raw_log_speedup) < noise_band_log)
    candidate_score = 0.0 if dead_zone_applied else clipped_log_speedup

    meta["score_status"] = "passed_candidate"
    meta["candidate_passed"] = True
    meta["raw_log_speedup"] = raw_log_speedup
    meta["clipped_log_speedup"] = clipped_log_speedup
    meta["dead_zone_applied"] = dead_zone_applied
    meta["candidate_score"] = float(candidate_score)
    return float(candidate_score), meta


def _unique_skill_chain(raw_chain: Any) -> List[str]:
    if not isinstance(raw_chain, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for item in raw_chain:
        sid = str(item or "").strip()
        if not re.fullmatch(r"O\d{3}", sid) or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


@dataclass
class Stage3FeedbackStats:
    record_count: int
    valid_problem_count: int
    candidate_count: int
    scored_candidate_count: int
    passed_candidate_count: int
    failed_candidate_count: int
    skipped_missing_baseline_count: int
    skill_gain_count: int
    nonzero_skill_gain_count: int


def load_stage3_skill_gains(
    feedback_paths: List[str],
    prior_skill_ids: Iterable[str],
    artifacts_dir: str,
    *,
    failure_penalty: float,
    noise_band_ratio: float,
    runtime_clip_ratio: float,
    ridge_alpha: float,
    shrinkage_lambda: float,
) -> Tuple[Dict[str, float], Stage3FeedbackStats, Dict[str, Any]]:
    record_count = 0
    valid_problem_count = 0
    candidate_count = 0
    scored_candidate_count = 0
    passed_candidate_count = 0
    failed_candidate_count = 0
    skipped_missing_baseline_count = 0

    os.makedirs(artifacts_dir, exist_ok=True)

    skill_ids = sorted({str(sid).strip() for sid in prior_skill_ids if str(sid).strip()})
    if not skill_ids:
        raise ValueError("Stage 3 prior skill ids are empty")
    skill_to_idx = {sid: idx for idx, sid in enumerate(skill_ids)}

    candidate_rows: List[Dict[str, Any]] = []
    scored_examples: List[Dict[str, Any]] = []
    problem_scored_counts: Counter[str] = Counter()
    skill_problem_support: Dict[str, Set[str]] = defaultdict(set)

    for path in feedback_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Stage 3 feedback file: {path}")
        for rec in _load_jsonl(path):
            record_count += 1
            pid = str(rec.get("problem_id") or "").strip()
            best_before = rec.get("best_before_metric")
            candidate_feedback = rec.get("candidate_feedback")

            if not pid:
                raise ValueError("Stage 3 feedback record missing problem_id")
            if not isinstance(candidate_feedback, list):
                raise ValueError(
                    f"Stage 3 feedback record for {pid} is missing candidate_feedback"
                )

            problem_had_valid_candidate = False
            for cand in candidate_feedback:
                if not isinstance(cand, dict):
                    continue
                candidate_count += 1

                skill_chain = _unique_skill_chain(cand.get("skill_chain"))
                if not skill_chain:
                    raise ValueError(
                        f"Stage 3 candidate feedback for {pid} is missing skill_chain"
                    )

                candidate_score, score_meta = _stage3_candidate_score(
                    best_before,
                    cand.get("runtime_rank_metric"),
                    failure_penalty=failure_penalty,
                    noise_band_ratio=noise_band_ratio,
                    runtime_clip_ratio=runtime_clip_ratio,
                )

                candidate_row = {
                    "problem_id": pid,
                    "candidate_index": cand.get("candidate_index"),
                    "candidate_id": cand.get("candidate_id"),
                    "set_id": cand.get("set_id"),
                    "plan_id": cand.get("plan_id"),
                    "plan_style": cand.get("plan_style"),
                    "skill_chain": skill_chain,
                    "skill_count": len(skill_chain),
                    "best_before_metric": best_before,
                    "runtime_rank_metric": cand.get("runtime_rank_metric"),
                    "score_status": score_meta.get("score_status"),
                    "candidate_passed": score_meta.get("candidate_passed"),
                    "candidate_score": score_meta.get("candidate_score"),
                    "raw_log_speedup": score_meta.get("raw_log_speedup"),
                    "clipped_log_speedup": score_meta.get("clipped_log_speedup"),
                    "dead_zone_applied": score_meta.get("dead_zone_applied"),
                    "sample_weight": None,
                }
                candidate_rows.append(candidate_row)

                if candidate_score is None:
                    if score_meta.get("score_status") == "missing_baseline":
                        skipped_missing_baseline_count += 1
                    continue

                row_skills = [sid for sid in skill_chain if sid in skill_to_idx]
                if not row_skills:
                    raise ValueError(
                        f"Stage 3 candidate feedback for {pid} has no prior skill ids in skill_chain"
                    )

                scored_examples.append(
                    {
                        "problem_id": pid,
                        "candidate_row": candidate_row,
                        "skill_chain": row_skills,
                        "candidate_score": float(candidate_score),
                    }
                )
                problem_scored_counts[pid] += 1
                for sid in row_skills:
                    skill_problem_support[sid].add(pid)
                if score_meta.get("candidate_passed"):
                    passed_candidate_count += 1
                else:
                    failed_candidate_count += 1
                scored_candidate_count += 1
                problem_had_valid_candidate = True

            if problem_had_valid_candidate:
                valid_problem_count += 1

    if not scored_examples:
        raise ValueError("Stage 3 feedback contains no scored candidates")

    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    sample_weights: List[float] = []
    for example in scored_examples:
        pid = example["problem_id"]
        weight = 1.0 / float(problem_scored_counts[pid])
        example["candidate_row"]["sample_weight"] = weight

        x = np.zeros(len(skill_ids), dtype=float)
        for sid in example["skill_chain"]:
            x[skill_to_idx[sid]] = 1.0
        x_rows.append(x)
        y_rows.append(float(example["candidate_score"]))
        sample_weights.append(weight)

    x_mat = np.vstack(x_rows)
    y_vec = np.asarray(y_rows, dtype=float)
    sample_weight_vec = np.asarray(sample_weights, dtype=float)

    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(x_mat, y_vec, sample_weight=sample_weight_vec)

    raw_coefficients = {
        sid: float(model.coef_[skill_to_idx[sid]])
        for sid in skill_ids
    }
    support_counts = {
        sid: int(len(skill_problem_support.get(sid, set())))
        for sid in skill_ids
    }
    shrinkage_weights = {
        sid: (
            float(support_counts[sid] / (support_counts[sid] + float(shrinkage_lambda)))
            if support_counts[sid] > 0
            else 0.0
        )
        for sid in skill_ids
    }
    gains = {
        sid: float(raw_coefficients[sid] * shrinkage_weights[sid])
        for sid in skill_ids
    }

    candidate_metrics_path = os.path.join(artifacts_dir, "stage3_candidate_metrics.jsonl")
    with open(candidate_metrics_path, "w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    skill_gain_details = {
        sid: {
            "support_problem_count": support_counts[sid],
            "raw_coefficient": raw_coefficients[sid],
            "shrinkage_weight": shrinkage_weights[sid],
            "shrunk_gain": gains[sid],
        }
        for sid in skill_ids
    }
    skill_gain_details_path = os.path.join(artifacts_dir, "stage3_skill_gain_details.json")
    with open(skill_gain_details_path, "w", encoding="utf-8") as f:
        json.dump(skill_gain_details, f, ensure_ascii=False, indent=2)

    summary = {
        "metric": {
            "name": "signed_log_speedup_with_failure_penalty",
            "failure_penalty": float(failure_penalty),
            "noise_band_ratio": float(noise_band_ratio),
            "noise_band_log": float(math.log1p(noise_band_ratio)),
            "runtime_clip_ratio": float(runtime_clip_ratio),
            "runtime_clip_log": float(math.log(runtime_clip_ratio)),
        },
        "attribution": {
            "name": "ridge_plus_support_shrinkage",
            "ridge_alpha": float(ridge_alpha),
            "shrinkage_lambda": float(shrinkage_lambda),
            "intercept": float(model.intercept_),
            "weighted_r2": float(model.score(x_mat, y_vec, sample_weight=sample_weight_vec)),
            "sample_count": int(len(scored_examples)),
            "problem_count": int(valid_problem_count),
        },
        "artifacts": {
            "candidate_metrics_path": candidate_metrics_path,
            "skill_gain_details_path": skill_gain_details_path,
        },
        "top_positive_skills": sorted(
            gains.items(), key=lambda kv: kv[1], reverse=True
        )[:10],
        "top_negative_skills": sorted(gains.items(), key=lambda kv: kv[1])[:10],
    }
    summary_path = os.path.join(artifacts_dir, "stage3_feedback_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    stats = Stage3FeedbackStats(
        record_count=record_count,
        valid_problem_count=valid_problem_count,
        candidate_count=candidate_count,
        scored_candidate_count=scored_candidate_count,
        passed_candidate_count=passed_candidate_count,
        failed_candidate_count=failed_candidate_count,
        skipped_missing_baseline_count=skipped_missing_baseline_count,
        skill_gain_count=len(gains),
        nonzero_skill_gain_count=sum(1 for val in gains.values() if abs(val) > 1e-12),
    )
    return gains, stats, {
        "candidate_metrics_path": candidate_metrics_path,
        "skill_gain_details_path": skill_gain_details_path,
        "summary_path": summary_path,
        "ridge_intercept": float(model.intercept_),
        "weighted_r2": float(model.score(x_mat, y_vec, sample_weight=sample_weight_vec)),
    }


def prepare_traces(
    traces_path: str,
    out_path: str,
    limit: Optional[int] = None,
    stage_mode: str = "stage3",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mode = str(stage_mode).strip().lower()
    if mode not in {"stage1", "stage3"}:
        raise ValueError(f"Unsupported stage mode: {stage_mode}")

    traces: List[Dict[str, Any]] = []

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, raw in enumerate(_load_jsonl(traces_path), start=1):
            if limit is not None and len(traces) >= limit:
                break
            t = _normalize_trace(raw, idx)
            t["trace_weight"] = 1.0
            traces.append(t)
            f_out.write(json.dumps(t, ensure_ascii=False) + "\n")

    stats = {
        "stage_mode": mode,
        "trace_count": len(traces),
        "problem_count": len({t.get("problem_id") for t in traces if t.get("problem_id")}),
        "trace_weight_median": (
            float(np.median([t["trace_weight"] for t in traces])) if traces else 0.0
        ),
        "trace_weight_max": (
            float(np.max([t["trace_weight"] for t in traces])) if traces else 0.0
        ),
    }
    return traces, stats


def trace_signature(trace: Dict[str, Any]) -> str:
    delta = trace.get("DeltaSummary") or {}
    slow = trace.get("SlowAudit") or {}
    problem = trace.get("ProblemBrief") or {}

    parts: List[str] = []
    parts.extend(_listify(delta.get("delta_type")))

    comp = delta.get("complexity_delta") or {}
    frm = _clean_text(comp.get("from", ""))
    to = _clean_text(comp.get("to", ""))
    if frm or to:
        parts.append(f"{frm}->{to}")

    parts.extend(_listify(delta.get("trigger_signals"))[:8])
    parts.extend(_listify(delta.get("transformation_steps"))[:8])

    for b in slow.get("bottlenecks") or []:
        if isinstance(b, dict) and b.get("kind"):
            parts.append(_clean_text(b["kind"]))

    parts.extend(_listify(problem.get("problem_type_tags"))[:6])
    return " | ".join([p for p in parts if p])


def build_trace_profiles(traces: List[Dict[str, Any]], out_path: str) -> List[str]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    signatures: List[str] = []
    with open(out_path, "w", encoding="utf-8") as f_out:
        for t in traces:
            sig = trace_signature(t)
            signatures.append(sig)
            rec = {
                "trace_id": t.get("trace_id"),
                "problem_id": t.get("problem_id"),
                "source_hash": t.get("source_hash"),
                "trace_weight": t.get("trace_weight", 1.0),
                "signature": sig,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return signatures


def _cluster_profile_text(profile: Dict[str, Any]) -> str:
    parts = [
        str(profile.get("cluster_id") or ""),
        " ".join(profile.get("top_delta_type") or []),
        " ".join(profile.get("top_complexity_delta") or []),
        " ".join(profile.get("top_bottlenecks") or []),
        " ".join(profile.get("top_problem_tags") or []),
        " ".join(profile.get("top_triggers") or []),
        " ".join(profile.get("top_steps") or []),
    ]
    return " ".join(_clean_text(p) for p in parts if _clean_text(p))


def _cluster_summary_text(summary: Dict[str, Any]) -> str:
    parts = [
        str(summary.get("proposed_name") or ""),
        str(summary.get("description") or ""),
        " ".join(summary.get("tags") or []),
        " ".join(summary.get("canonical_triggers") or []),
        " ".join(summary.get("canonical_steps") or []),
        " ".join(summary.get("evidence_top_triggers") or []),
        " ".join(summary.get("evidence_top_steps") or []),
    ]
    return " ".join(_clean_text(p) for p in parts if _clean_text(p))


def _skill_card_text(card: Dict[str, Any]) -> str:
    parts = [
        str(card.get("skill_id") or ""),
        str(card.get("name") or ""),
        str(card.get("description") or ""),
        " ".join(card.get("tags") or []),
        " ".join(card.get("triggers") or []),
    ]
    return " ".join(_clean_text(p) for p in parts if _clean_text(p))


def _similarity_matrix_from_texts(
    texts_a: List[str],
    texts_b: List[str],
    tfidf_max_features: int,
    embedding_model: str,
    disable_sentence_embeddings: bool,
) -> np.ndarray:
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b)), dtype=float)
    all_texts = texts_a + texts_b
    _, _, hybrid_matrix, _ = compute_hybrid_features(
        signatures=all_texts,
        tfidf_max_features=tfidf_max_features,
        embedding_model=embedding_model,
        disable_sentence_embeddings=disable_sentence_embeddings,
    )
    a_mat = hybrid_matrix[: len(texts_a)]
    b_mat = hybrid_matrix[len(texts_a) :]
    return np.asarray(cosine_similarity(a_mat, b_mat), dtype=float)


@dataclass
class PriorSkillContext:
    data_root: str
    prior_cluster_texts: Dict[str, str]
    skill_texts: Dict[str, str]
    skill_lineage_weights: Dict[str, Dict[str, float]]
    operator_cards: Dict[str, Dict[str, Any]]
    operator_paths: Dict[str, str]


def _extract_markdown_section(skill_text: str, heading: str) -> str:
    pattern = rf"(?ms)^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, skill_text)
    return match.group(1).strip() if match else ""


def _parse_markdown_bullets(section_text: str) -> List[str]:
    out: List[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            out.append(_clean_text(stripped[2:]))
        elif re.match(r"^\d+\.\s+", stripped):
            out.append(_normalize_step_text(stripped))
    return [x for x in out if x]


def _parse_operator_skill_card(path: Path) -> Optional[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        front = yaml.safe_load(parts[1]) or {}
    except Exception:
        return None
    if str(front.get("type", "")) != "operator":
        return None
    body = parts[2]
    card = {
        "skill_id": _clean_text(front.get("skill_id", "")),
        "type": "operator",
        "family": _clean_text(front.get("family", "")),
        "name": _clean_text(front.get("name", "")),
        "description": _scalar_text(front.get("description", "")),
        "tags": _listify(front.get("tags")),
        "triggers": _listify(front.get("triggers")),
        "canonical_triggers": _parse_markdown_bullets(_extract_markdown_section(body, "When to use")),
        "canonical_steps": _parse_markdown_bullets(_extract_markdown_section(body, "Steps")),
        "common_pitfalls": _parse_markdown_bullets(_extract_markdown_section(body, "Pitfalls")),
        "when_not_to_use": _parse_markdown_bullets(_extract_markdown_section(body, "When not to use")),
        "complexity_time": "",
        "complexity_space": "",
        "minimal_example": _extract_markdown_section(body, "Minimal example"),
        "source_path": str(path),
        "source_text": text,
    }
    complexity_text = _extract_markdown_section(body, "Complexity")
    for line in complexity_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("- time:"):
            card["complexity_time"] = _clean_text(stripped.split(":", 1)[1])
        elif stripped.lower().startswith("- space:"):
            card["complexity_space"] = _clean_text(stripped.split(":", 1)[1])
    return card


def load_prior_skill_context(prior_skills_root: str) -> PriorSkillContext:
    skills_root = Path(prior_skills_root)
    data_root = skills_root.parent / "data"
    registry_path = skills_root / "registry.json"
    summaries_path = data_root / "cluster_summaries.jsonl"
    profiles_path = data_root / "cluster_profiles.jsonl"
    clusters_path = data_root / "clusters.jsonl"
    operators_root = skills_root / "operators"

    required = [registry_path, summaries_path, profiles_path, clusters_path, operators_root]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 3 requires prior lineage artifacts. Missing: " + ", ".join(missing)
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    operator_registry_cards = [
        card
        for card in registry
        if isinstance(card, dict) and str(card.get("skill_id") or "").startswith("O")
    ]
    if not operator_registry_cards:
        raise ValueError(f"No prior operator skills found in {registry_path}")

    parsed_operator_cards: Dict[str, Dict[str, Any]] = {}
    operator_paths: Dict[str, str] = {}
    for skill_path in sorted(operators_root.glob("*/SKILL.md")):
        parsed = _parse_operator_skill_card(skill_path)
        if parsed is None:
            continue
        sid = str(parsed.get("skill_id") or "").strip()
        if not sid:
            continue
        parsed_operator_cards[sid] = parsed
        operator_paths[sid] = str(skill_path)
    if set(card["skill_id"] for card in operator_registry_cards) - set(parsed_operator_cards):
        missing_skill_ids = sorted(
            set(card["skill_id"] for card in operator_registry_cards) - set(parsed_operator_cards)
        )
        raise ValueError(
            "Missing parsed prior operator SKILL.md files for: " + ", ".join(missing_skill_ids)
        )

    prior_profiles = {
        str(rec.get("cluster_id")): rec
        for rec in _load_jsonl(str(profiles_path))
        if rec.get("cluster_id")
    }
    prior_summaries = list(_load_jsonl(str(summaries_path)))
    if len(prior_summaries) < len(operator_registry_cards):
        raise ValueError(
            "Prior cluster summaries are fewer than operator skills; lineage is ambiguous"
        )

    prior_cluster_texts: Dict[str, str] = {}
    for cid, profile in prior_profiles.items():
        prior_cluster_texts[cid] = _cluster_profile_text(profile)
    for summary in prior_summaries:
        cid = str(summary.get("cluster_id") or "").strip()
        if cid and cid not in prior_cluster_texts:
            prior_cluster_texts[cid] = _cluster_summary_text(summary)

    if not prior_cluster_texts:
        raise ValueError("No prior cluster texts available for Stage 3 lineage")

    skill_texts: Dict[str, str] = {}
    skill_lineage_weights: Dict[str, Dict[str, float]] = {}
    for idx, card in enumerate(operator_registry_cards, start=1):
        sid = str(card.get("skill_id") or "").strip()
        summary = prior_summaries[idx - 1]
        lineage_ids = summary.get("cluster_ids") or [summary.get("cluster_id")]
        lineage_ids = [str(cid).strip() for cid in lineage_ids if str(cid).strip()]
        if not lineage_ids:
            raise ValueError(f"Prior skill {sid} has no lineage clusters")

        weights: Dict[str, float] = {}
        raw_total = 0.0
        for cid in lineage_ids:
            support = prior_profiles.get(cid, {}).get("support_weight", 1.0)
            support = float(support) if isinstance(support, (int, float)) else 1.0
            support = max(support, 1e-9)
            weights[cid] = support
            raw_total += support
        skill_lineage_weights[sid] = {
            cid: val / raw_total for cid, val in weights.items()
        }
        skill_texts[sid] = _skill_card_text(card)

    return PriorSkillContext(
        data_root=str(data_root),
        prior_cluster_texts=prior_cluster_texts,
        skill_texts=skill_texts,
        skill_lineage_weights=skill_lineage_weights,
        operator_cards=parsed_operator_cards,
        operator_paths=operator_paths,
    )


def _normalize_positive_weights(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in values.values() if v > 0)
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in values.items() if v > 0}


def _top_positive_transfer(
    labels: List[str], scores: np.ndarray, top_k: int = 3
) -> Dict[str, float]:
    if scores.size == 0 or not labels:
        return {}
    idxs = np.argsort(scores)[::-1][:top_k]
    weights = {
        labels[i]: float(max(scores[i], 0.0))
        for i in idxs
        if i < len(labels) and scores[i] > 0
    }
    if weights:
        return _normalize_positive_weights(weights)
    best = int(np.argmax(scores))
    return {labels[best]: 1.0}


def build_cluster_skill_correlation(
    current_profiles: Dict[str, Dict[str, Any]],
    prior_ctx: PriorSkillContext,
    tfidf_max_features: int,
    embedding_model: str,
    disable_sentence_embeddings: bool,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    current_cluster_ids = sorted(current_profiles.keys())
    prior_cluster_ids = sorted(prior_ctx.prior_cluster_texts.keys())
    skill_ids = sorted(prior_ctx.skill_texts.keys())

    current_texts = [_cluster_profile_text(current_profiles[cid]) for cid in current_cluster_ids]
    prior_cluster_texts = [prior_ctx.prior_cluster_texts[cid] for cid in prior_cluster_ids]
    skill_texts = [prior_ctx.skill_texts[sid] for sid in skill_ids]

    current_prior_sim = _similarity_matrix_from_texts(
        current_texts,
        prior_cluster_texts,
        tfidf_max_features=tfidf_max_features,
        embedding_model=embedding_model,
        disable_sentence_embeddings=disable_sentence_embeddings,
    )
    current_skill_sim = _similarity_matrix_from_texts(
        current_texts,
        skill_texts,
        tfidf_max_features=tfidf_max_features,
        embedding_model=embedding_model,
        disable_sentence_embeddings=disable_sentence_embeddings,
    )

    corr: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    for i, cid in enumerate(current_cluster_ids):
        transfer = _top_positive_transfer(prior_cluster_ids, current_prior_sim[i], top_k=3)
        raw_scores: Dict[str, float] = {}
        for j, sid in enumerate(skill_ids):
            lineage = prior_ctx.skill_lineage_weights.get(sid, {})
            lineage_support = sum(
                transfer.get(prev_cid, 0.0) * lineage.get(prev_cid, 0.0)
                for prev_cid in transfer
            )
            semantic_sim = float(max(current_skill_sim[i, j], 0.0))
            raw = lineage_support * semantic_sim
            if raw > 0:
                raw_scores[sid] = raw
        normalized = _normalize_positive_weights(raw_scores)
        corr[cid] = normalized
        rows.append(
            {
                "cluster_id": cid,
                "prior_cluster_transfer": transfer,
                "top_skill_corr": sorted(
                    normalized.items(), key=lambda x: x[1], reverse=True
                )[:8],
            }
        )
    if not any(corr.values()):
        raise ValueError("Stage 3 cluster-skill correlation is empty")
    return corr, rows


def _soft_cluster_affinities(
    cluster_result: ClusterResult,
    top_k: int = 2,
) -> List[Dict[str, float]]:
    if cluster_result.cluster_centers is None or cluster_result.hybrid_matrix.shape[0] == 0:
        return [{f"C{lbl + 1:03d}": 1.0} for lbl in cluster_result.labels]
    sim = np.asarray(
        cosine_similarity(cluster_result.hybrid_matrix, cluster_result.cluster_centers),
        dtype=float,
    )
    out: List[Dict[str, float]] = []
    cluster_labels = [f"C{i + 1:03d}" for i in range(sim.shape[1])]
    for row, hard_lbl in zip(sim, cluster_result.labels):
        idxs = np.argsort(row)[::-1][:top_k]
        weights = {
            cluster_labels[i]: float(max(row[i], 0.0))
            for i in idxs
            if row[i] > 0
        }
        if weights:
            out.append(_normalize_positive_weights(weights))
        else:
            out.append({f"C{hard_lbl + 1:03d}": 1.0})
    return out


def apply_stage3_trace_weights(
    traces: List[Dict[str, Any]],
    cluster_result: ClusterResult,
    skill_gains: Dict[str, float],
    cluster_skill_corr: Dict[str, Dict[str, float]],
    out_path: str,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    affinities = _soft_cluster_affinities(cluster_result, top_k=2)
    rows: List[Dict[str, Any]] = []
    weights: List[float] = []
    signed_utilities: List[float] = []

    with open(out_path, "w", encoding="utf-8") as f_out:
        for trace, affinity in zip(traces, affinities):
            cluster_scores: Dict[str, float] = {}
            skill_contribs: Dict[str, float] = defaultdict(float)
            for cid, alpha in affinity.items():
                utility = 0.0
                for sid, corr in cluster_skill_corr.get(cid, {}).items():
                    gain = skill_gains.get(sid)
                    if gain is None:
                        continue
                    contrib = float(alpha) * float(corr) * float(gain)
                    skill_contribs[sid] += contrib
                    utility += float(corr) * float(gain)
                cluster_scores[cid] = float(alpha) * utility

            signed_utility = float(sum(cluster_scores.values()))
            clustering_log_weight = float(
                np.clip(signed_utility, -STAGE3_TRACE_WEIGHT_LOG_CLIP, STAGE3_TRACE_WEIGHT_LOG_CLIP)
            )
            weight = float(math.exp(clustering_log_weight))
            trace["trace_weight"] = weight
            trace["trace_signed_utility"] = signed_utility
            weights.append(weight)
            signed_utilities.append(signed_utility)
            row = {
                "trace_id": trace.get("trace_id"),
                "problem_id": trace.get("problem_id"),
                "soft_clusters": affinity,
                "top_skill_contrib": sorted(
                    skill_contribs.items(), key=lambda x: abs(x[1]), reverse=True
                )[:8],
                "top_positive_skill_contrib": sorted(
                    [(sid, val) for sid, val in skill_contribs.items() if val > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )[:8],
                "top_negative_skill_contrib": sorted(
                    [(sid, val) for sid, val in skill_contribs.items() if val < 0],
                    key=lambda x: x[1],
                )[:8],
                "signed_utility": signed_utility,
                "clustering_log_weight": clustering_log_weight,
                "trace_weight": weight,
            }
            rows.append(row)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not any(np.isfinite(w) and w > 0 for w in weights):
        raise ValueError("Stage 3 produced zero weight for every trace")

    return {
        "trace_weight_median": float(np.median(np.array(weights, dtype=float))),
        "trace_weight_max": float(np.max(np.array(weights, dtype=float))),
        "signed_utility_median": float(np.median(np.array(signed_utilities, dtype=float))),
        "signed_utility_min": float(np.min(np.array(signed_utilities, dtype=float))),
        "signed_utility_max": float(np.max(np.array(signed_utilities, dtype=float))),
        "weighted_trace_count": int(sum(1 for w in weights if w > 0)),
        "attribution_count": len(rows),
    }


def compute_hybrid_features(
    signatures: List[str],
    tfidf_max_features: int,
    embedding_model: str,
    disable_sentence_embeddings: bool,
) -> Tuple[Any, np.ndarray, Any, str]:
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(signatures)
    tfidf_norm = normalize(tfidf)

    emb_norm: np.ndarray
    emb_source = "sentence-transformers"

    if not disable_sentence_embeddings:
        try:
            _prepare_sentence_transformers_env()
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model)
            emb = model.encode(
                signatures,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            emb_norm = normalize(emb)
        except Exception as exc:
            emb_source = f"svd_fallback({exc.__class__.__name__})"
            svd = TruncatedSVD(
                n_components=min(128, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1),
                random_state=42,
            )
            emb_norm = normalize(svd.fit_transform(tfidf))
    else:
        emb_source = "svd"
        svd = TruncatedSVD(
            n_components=min(128, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1),
            random_state=42,
        )
        emb_norm = normalize(svd.fit_transform(tfidf))

    hybrid = hstack([tfidf_norm, csr_matrix(emb_norm)], format="csr")
    return tfidf_norm, emb_norm, hybrid, emb_source


def _auto_k_effective_n(n: int) -> int:
    raw = os.getenv(AUTO_K_EFFECTIVE_N_ENV, "").strip()
    if not raw:
        return n
    try:
        value = int(raw)
    except ValueError:
        return n
    return max(2, value)


def choose_k_auto(
    embedding_matrix: np.ndarray,
    n: int,
    k_hint: Optional[int] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> int:
    if n < 2:
        return 1

    if k_hint is not None:
        return max(2, min(n, k_hint))

    effective_n = _auto_k_effective_n(n)
    base = max(8, int(round(math.sqrt(effective_n))))
    low = max(6, base - 10)
    high = min(max(12, base + 10), min(80, n))

    candidates = sorted({k for k in range(low, high + 1, 2) if 2 <= k <= n - 1})
    if not candidates:
        return max(2, min(n, base))

    best_k = candidates[0]
    best_score = -1.0

    for k in candidates:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        if sample_weights is None:
            labels = km.fit_predict(embedding_matrix)
        else:
            km.fit(embedding_matrix, sample_weight=sample_weights)
            labels = km.labels_
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embedding_matrix, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return max(2, min(n, best_k))


@dataclass
class ClusterResult:
    labels: List[int]
    k: int
    embedding_source: str
    hybrid_matrix: Any
    cluster_centers: Optional[np.ndarray]


def cluster_traces(
    traces: List[Dict[str, Any]],
    signatures: List[str],
    clusters_path: str,
    tfidf_max_features: int,
    embedding_model: str,
    disable_sentence_embeddings: bool,
    k: Optional[int],
    sample_weights: Optional[np.ndarray] = None,
) -> ClusterResult:
    os.makedirs(os.path.dirname(clusters_path), exist_ok=True)

    _, emb_matrix, hybrid_matrix, emb_source = compute_hybrid_features(
        signatures=signatures,
        tfidf_max_features=tfidf_max_features,
        embedding_model=embedding_model,
        disable_sentence_embeddings=disable_sentence_embeddings,
    )

    if len(traces) < 2:
        labels = [0]
        k_final = 1
        cluster_centers = None
    else:
        k_final = choose_k_auto(
            emb_matrix,
            n=len(traces),
            k_hint=k,
            sample_weights=sample_weights,
        )
        km = KMeans(n_clusters=k_final, n_init="auto", random_state=42)
        if sample_weights is None:
            labels = km.fit_predict(hybrid_matrix).tolist()
        else:
            km.fit(hybrid_matrix, sample_weight=sample_weights)
            labels = km.labels_.tolist()
        cluster_centers = np.asarray(km.cluster_centers_, dtype=float)

    with open(clusters_path, "w", encoding="utf-8") as f_out:
        for t, lbl in zip(traces, labels):
            cluster_id = f"C{lbl + 1:03d}"
            rec = {
                "trace_id": t.get("trace_id"),
                "problem_id": t.get("problem_id"),
                "source_hash": t.get("source_hash"),
                "cluster_id": cluster_id,
                "trace_weight": t.get("trace_weight", 1.0),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return ClusterResult(
        labels=labels,
        k=k_final,
        embedding_source=emb_source,
        hybrid_matrix=hybrid_matrix,
        cluster_centers=cluster_centers,
    )


def _top_weighted(counter: Dict[str, float], limit: int) -> List[str]:
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in items[:limit]]


def build_cluster_profiles(
    traces: List[Dict[str, Any]],
    labels: List[int],
    out_path: str,
    sample_per_cluster: int,
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for t, lbl in zip(traces, labels):
        grouped[lbl].append(t)

    profiles: Dict[str, Dict[str, Any]] = {}

    with open(out_path, "w", encoding="utf-8") as f_out:
        for lbl, items in sorted(grouped.items(), key=lambda x: x[0]):
            cluster_id = f"C{lbl + 1:03d}"
            c_delta: Dict[str, float] = defaultdict(float)
            c_trigger: Dict[str, float] = defaultdict(float)
            c_steps: Dict[str, float] = defaultdict(float)
            c_tags: Dict[str, float] = defaultdict(float)
            c_bottleneck: Dict[str, float] = defaultdict(float)
            c_comp: Dict[str, float] = defaultdict(float)

            total_weight = 0.0
            for t in items:
                w = float(t.get("trace_weight", 1.0))
                total_weight += w

                d = t.get("DeltaSummary") or {}
                s = t.get("SlowAudit") or {}
                p = t.get("ProblemBrief") or {}

                for x in d.get("delta_type", []):
                    c_delta[_clean_text(x)] += w
                for x in d.get("trigger_signals", []):
                    c_trigger[_clean_text(x)] += w
                for x in d.get("transformation_steps", []):
                    c_steps[_clean_text(x)] += w
                for x in p.get("problem_type_tags", []):
                    c_tags[_clean_text(x)] += w
                for b in s.get("bottlenecks", []):
                    if isinstance(b, dict) and b.get("kind"):
                        c_bottleneck[_clean_text(b["kind"])] += w

                comp = d.get("complexity_delta") or {}
                frm = _clean_text(comp.get("from", ""))
                to = _clean_text(comp.get("to", ""))
                if frm or to:
                    c_comp[f"{frm}->{to}"] += w

            items_sorted = sorted(items, key=lambda x: x.get("trace_weight", 1.0), reverse=True)
            top_take = min(max(4, sample_per_cluster // 2), len(items_sorted))
            sample = items_sorted[:top_take]
            rest = items_sorted[top_take:]
            if rest and len(sample) < sample_per_cluster:
                random_take = min(sample_per_cluster - len(sample), len(rest))
                sample.extend(random.sample(rest, random_take))

            sampled = []
            for t in sample:
                sampled.append(
                    {
                        "trace_id": t.get("trace_id"),
                        "problem_type_tags": (t.get("ProblemBrief") or {}).get(
                            "problem_type_tags", []
                        ),
                        "delta_type": (t.get("DeltaSummary") or {}).get("delta_type", []),
                        "complexity_delta": (t.get("DeltaSummary") or {}).get(
                            "complexity_delta", {}
                        ),
                        "trigger_signals": (t.get("DeltaSummary") or {}).get("trigger_signals", []),
                        "transformation_steps": (t.get("DeltaSummary") or {}).get(
                            "transformation_steps", []
                        ),
                        "bottlenecks": (t.get("SlowAudit") or {}).get("bottlenecks", []),
                        "trace_weight": t.get("trace_weight", 1.0),
                    }
                )

            profile = {
                "cluster_id": cluster_id,
                "size": len(items),
                "support_weight": round(total_weight, 6),
                "top_delta_type": _top_weighted(c_delta, 12),
                "top_complexity_delta": _top_weighted(c_comp, 10),
                "top_bottlenecks": _top_weighted(c_bottleneck, 12),
                "top_problem_tags": _top_weighted(c_tags, 12),
                "top_triggers": _top_weighted(c_trigger, 16),
                "top_steps": _top_weighted(c_steps, 16),
                "sample_traces": sampled,
            }

            profiles[cluster_id] = profile
            f_out.write(json.dumps(profile, ensure_ascii=False) + "\n")

    return profiles


def summarize_cluster(
    profile: Dict[str, Any],
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Optional[Dict[str, Any]]:
    cluster_id = profile["cluster_id"]
    prompt = build_cluster_summary_prompt(profile)

    messages = [
        {"role": "system", "content": cluster_summary_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    obj = _llm_json_call(
        messages=messages,
        cache_dir=cache_dir,
        stage=f"a3_cluster_summary_{cluster_id}",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        out_fail_dir=fail_dir,
        reasoning_effort=reasoning_effort,
        fallback_model=fallback_model,
    )
    if obj is None:
        return None

    obj["cluster_id"] = cluster_id
    obj["support_count"] = profile.get("size", 0)
    obj["support_weight"] = profile.get("support_weight", 0.0)
    out = sanitize_summary(obj)
    # Preserve compact provenance for downstream example generation and validation.
    out["evidence_top_steps"] = (profile.get("top_steps") or [])[:8]
    out["evidence_top_triggers"] = (profile.get("top_triggers") or [])[:8]
    out["evidence_problem_tags"] = (profile.get("top_problem_tags") or [])[:8]
    return out


def _normalize_steps(raw_steps: List[str], limit: int = 8) -> List[str]:
    out: List[str] = []
    for step in raw_steps:
        s = _normalize_step_text(step)
        if not s:
            continue
        # Split flattened "header: - sub1 - sub2" into smaller actionable steps.
        if re.search(r":\s*-\s+", s):
            head, rest = s.split(":", 1)
            head = _clean_text(head)
            parts = [p.strip() for p in re.split(r"\s+-\s+", rest) if _clean_text(p)]
            if parts:
                for p in parts:
                    p = _normalize_step_text(p)
                    if p:
                        if head:
                            out.append(f"{head}: {p}")
                        else:
                            out.append(p)
                continue
        if _looks_truncated_artifact(s):
            continue
        out.append(s)
    return _unique_keep_order(out, limit=limit)


def _normalize_bullet_items(raw_items: List[str], limit: int, max_chars: int) -> List[str]:
    expanded: List[str] = []
    for raw in raw_items:
        s = _sanitize_prose(raw)
        if not s:
            continue

        # Expand flattened "Header: - point1 - point2" into stable bullets.
        if ": -" in s:
            head, rest = s.split(": -", 1)
            head = _clean_text(head)
            parts = [p.strip() for p in re.split(r"\s+-\s+", rest) if _clean_text(p)]
            if head and parts:
                for p in parts:
                    expanded.append(f"{head}: {_clean_text(p)}")
                continue

        expanded.append(s)

    out: List[str] = []
    for item in expanded:
        item = _truncate_text(item, max_chars=max_chars)
        if not item or _looks_truncated_artifact(item):
            continue
        out.append(item)
    return _unique_keep_order(out, limit=limit)


def sanitize_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    out["proposed_name"] = _normalize_name(out.get("proposed_name", "Operator Skill"))
    desc = _truncate_text(_sanitize_prose(_scalar_text(out.get("description", ""))), max_chars=420)
    if _looks_truncated_artifact(desc):
        desc = "Reusable optimization pattern distilled from weighted traces."
    out["description"] = desc

    out["tags"] = _unique_keep_order(_listify(out.get("tags")), limit=10)
    out["canonical_triggers"] = _normalize_bullet_items(
        _unique_keep_order(_listify(out.get("canonical_triggers")), limit=12),
        limit=8,
        max_chars=220,
    )
    if not out["canonical_triggers"]:
        out["canonical_triggers"] = [
            "Apply when the observed bottleneck pattern strongly matches this operator."
        ]
    out["common_pitfalls"] = _normalize_bullet_items(
        _unique_keep_order(_listify(out.get("common_pitfalls")), limit=14),
        limit=8,
        max_chars=220,
    )
    out["when_not_to_use"] = _normalize_bullet_items(
        _unique_keep_order(_listify(out.get("when_not_to_use")), limit=10),
        limit=6,
        max_chars=220,
    )

    raw_steps = _listify(out.get("canonical_steps"))
    steps_clean: List[str] = []
    for x in _normalize_steps(raw_steps, limit=10):
        sx = _truncate_text(_sanitize_prose(x), 260)
        if not sx or _looks_truncated_artifact(sx):
            continue
        steps_clean.append(sx)
    out["canonical_steps"] = _unique_keep_order(steps_clean, limit=8)
    if not out["canonical_steps"]:
        out["canonical_steps"] = [
            "Apply the core transformation while preserving correctness and interface."
        ]

    c_time = _truncate_text(
        _sanitize_prose(_scalar_text(out.get("complexity_time"), default="(pattern dependent)")),
        max_chars=240,
    )
    c_space = _truncate_text(
        _sanitize_prose(_scalar_text(out.get("complexity_space"), default="(pattern dependent)")),
        max_chars=240,
    )
    if _looks_truncated_artifact(c_time):
        c_time = "(pattern dependent)"
    if _looks_truncated_artifact(c_space):
        c_space = "(pattern dependent)"
    out["complexity_time"] = c_time
    out["complexity_space"] = c_space
    return out


def _summary_to_skill_like(summary: Dict[str, Any]) -> Dict[str, Any]:
    clean = sanitize_summary(summary)
    return {
        "skill_id": _clean_text(clean.get("skill_id", "")),
        "name": _normalize_name(clean.get("proposed_name", "Operator Skill")),
        "description": _scalar_text(clean.get("description", "")),
        "family": _infer_operator_family(clean),
        "tags": _listify(clean.get("tags")),
        "canonical_triggers": _listify(clean.get("canonical_triggers")),
        "canonical_steps": _listify(clean.get("canonical_steps")),
        "common_pitfalls": _listify(clean.get("common_pitfalls")),
        "when_not_to_use": _listify(clean.get("when_not_to_use")),
        "complexity_time": _scalar_text(clean.get("complexity_time", "")),
        "complexity_space": _scalar_text(clean.get("complexity_space", "")),
        "source_summary": clean,
    }


def _prior_card_to_summary_like(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "skill_id": _clean_text(card.get("skill_id", "")),
        "name": _normalize_name(card.get("name", "Operator Skill")),
        "description": _scalar_text(card.get("description", "")),
        "family": _clean_text(card.get("family", "")),
        "tags": _listify(card.get("tags")),
        "canonical_triggers": _listify(card.get("canonical_triggers") or card.get("triggers")),
        "canonical_steps": _listify(card.get("canonical_steps")),
        "common_pitfalls": _listify(card.get("common_pitfalls")),
        "when_not_to_use": _listify(card.get("when_not_to_use")),
        "complexity_time": _scalar_text(card.get("complexity_time", "")),
        "complexity_space": _scalar_text(card.get("complexity_space", "")),
    }


def _token_set_from_texts(items: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        text = _clean_text(item).lower()
        for tok in re.findall(r"[a-z][a-z0-9_]{2,}", text):
            out.add(tok)
    return out


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def build_stage3_cluster_utilities(
    cluster_skill_corr: Dict[str, Dict[str, float]],
    skill_gains: Dict[str, float],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    utilities: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for cid, corr in sorted(cluster_skill_corr.items()):
        contribs: List[Tuple[str, float]] = []
        utility = 0.0
        for sid, weight in corr.items():
            gain = float(skill_gains.get(sid, 0.0))
            contrib = float(weight) * gain
            if contrib != 0.0:
                contribs.append((sid, contrib))
            utility += contrib
        utilities[cid] = float(utility)
        rows.append(
            {
                "cluster_id": cid,
                "signed_utility": float(utility),
                "top_skill_contrib": sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:8],
            }
        )
    return utilities, rows


def _summary_signed_utility(
    summary: Dict[str, Any],
    cluster_utilities: Dict[str, float],
) -> float:
    cluster_ids = summary.get("cluster_ids") or [summary.get("cluster_id")]
    vals = [float(cluster_utilities.get(str(cid), 0.0)) for cid in cluster_ids if str(cid)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _summary_support_strength(summary: Dict[str, Any]) -> int:
    try:
        return int(summary.get("support_count", 0) or 0)
    except Exception:
        return 0


def _skill_similarity_components(
    summary_like: Dict[str, Any],
    prior_like: Dict[str, Any],
    semantic_sim: float,
) -> Dict[str, float]:
    summary_family = _clean_text(summary_like.get("family", ""))
    prior_family = _clean_text(prior_like.get("family", ""))
    family_sim = 1.0 if summary_family and summary_family == prior_family else 0.0
    tag_sim = _jaccard_similarity(
        _token_set_from_texts(summary_like.get("tags", [])),
        _token_set_from_texts(prior_like.get("tags", [])),
    )
    trigger_sim = _jaccard_similarity(
        _token_set_from_texts(summary_like.get("canonical_triggers", [])),
        _token_set_from_texts(prior_like.get("canonical_triggers", [])),
    )
    step_sim = _jaccard_similarity(
        _token_set_from_texts(summary_like.get("canonical_steps", [])),
        _token_set_from_texts(prior_like.get("canonical_steps", [])),
    )
    desc_sim = _jaccard_similarity(
        _token_set_from_texts([summary_like.get("description", "")]),
        _token_set_from_texts([prior_like.get("description", "")]),
    )
    total = (
        0.15 * family_sim
        + 0.15 * tag_sim
        + 0.25 * trigger_sim
        + 0.20 * step_sim
        + 0.10 * desc_sim
        + 0.15 * float(max(semantic_sim, 0.0))
    )
    return {
        "family": family_sim,
        "tags": tag_sim,
        "triggers": trigger_sim,
        "steps": step_sim,
        "description": desc_sim,
        "semantic": float(max(semantic_sim, 0.0)),
        "total": float(total),
    }


def _merge_revision_group(
    prior_card: Dict[str, Any],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prior_like = _prior_card_to_summary_like(prior_card)
    ranked = sorted(
        summaries,
        key=lambda s: (
            float(s.get("signed_utility", 0.0)),
            float(s.get("support_weight", 0.0)),
        ),
        reverse=True,
    )
    base = ranked[0] if ranked else {}
    merged = {
        "proposed_name": prior_like["name"],
        "description": prior_like["description"],
        "tags": _unique_keep_order(
            [x for s in ranked for x in s.get("tags", [])] + prior_like["tags"], limit=14
        ),
        "canonical_triggers": _normalize_bullet_items(
            [x for s in ranked for x in s.get("canonical_triggers", [])]
            + prior_like["canonical_triggers"],
            limit=10,
            max_chars=220,
        ),
        "canonical_steps": _normalize_steps(
            [x for s in ranked for x in s.get("canonical_steps", [])] + prior_like["canonical_steps"],
            limit=10,
        ),
        "common_pitfalls": _normalize_bullet_items(
            [x for s in ranked for x in s.get("common_pitfalls", [])]
            + prior_like["common_pitfalls"],
            limit=10,
            max_chars=220,
        ),
        "when_not_to_use": _normalize_bullet_items(
            [x for s in ranked for x in s.get("when_not_to_use", [])]
            + prior_like["when_not_to_use"],
            limit=8,
            max_chars=220,
        ),
        "complexity_time": prior_like["complexity_time"] or _scalar_text(base.get("complexity_time", "")),
        "complexity_space": prior_like["complexity_space"] or _scalar_text(base.get("complexity_space", "")),
        "cluster_ids": _unique_keep_order(
            [str(cid) for s in ranked for cid in (s.get("cluster_ids") or [s.get("cluster_id")]) if cid],
            limit=20,
        ),
        "support_count": int(sum(_summary_support_strength(s) for s in ranked)),
        "support_weight": float(sum(float(s.get("support_weight", 0.0)) for s in ranked)),
        "signed_utility": float(sum(float(s.get("signed_utility", 0.0)) for s in ranked)),
        "evidence_top_steps": _unique_keep_order(
            [x for s in ranked for x in s.get("evidence_top_steps", [])] + prior_like["canonical_steps"],
            limit=10,
        ),
        "evidence_top_triggers": _unique_keep_order(
            [x for s in ranked for x in s.get("evidence_top_triggers", [])] + prior_like["canonical_triggers"],
            limit=10,
        ),
        "evidence_problem_tags": _unique_keep_order(
            [x for s in ranked for x in s.get("evidence_problem_tags", [])],
            limit=10,
        ),
    }
    best_desc = prior_like["description"]
    if best_desc == "Reusable optimization pattern distilled from weighted traces.":
        best_desc = ""
    for candidate in ranked:
        cand_desc = _scalar_text(candidate.get("description", ""))
        if cand_desc and not _looks_truncated_artifact(cand_desc) and len(cand_desc) > len(best_desc):
            best_desc = cand_desc
    merged["description"] = best_desc or prior_like["description"]
    if not merged["canonical_triggers"]:
        merged["canonical_triggers"] = prior_like["canonical_triggers"]
    if not merged["canonical_steps"]:
        merged["canonical_steps"] = prior_like["canonical_steps"] or [
            "Apply the updated transformation while preserving correctness."
        ]
    return sanitize_summary(merged)


def _operator_doc_for_similarity(card: Dict[str, Any]) -> str:
    return " ".join(
        [
            _clean_text(card.get("name", "")),
            _clean_text(card.get("description", "")),
            " ".join(card.get("tags", [])),
            " ".join(card.get("canonical_triggers", []) or card.get("triggers", [])),
            " ".join(card.get("canonical_steps", [])),
        ]
    ).strip()


def _summary_doc_for_similarity(summary: Dict[str, Any]) -> str:
    return _cluster_summary_text(sanitize_summary(summary))


def build_stage3_library_plan(
    summaries: List[Dict[str, Any]],
    prior_ctx: PriorSkillContext,
    cluster_utilities: Dict[str, float],
    tfidf_max_features: int,
    embedding_model: str,
    disable_sentence_embeddings: bool,
    min_utility: float,
    revise_similarity: float,
    add_similarity_max: float,
    add_min_support: int,
    library_dedup_similarity: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    scored_utilities = [
        float(_summary_signed_utility(summary, cluster_utilities))
        for summary in summaries
        if float(_summary_signed_utility(summary, cluster_utilities)) > float(min_utility)
    ]
    if scored_utilities:
        positive_add_utility_threshold = float(np.quantile(np.array(scored_utilities), 0.75))
    else:
        positive_add_utility_threshold = float(min_utility)

    prior_skill_ids = sorted(prior_ctx.operator_cards.keys())
    prior_cards = [prior_ctx.operator_cards[sid] for sid in prior_skill_ids]
    prior_docs = [_operator_doc_for_similarity(card) for card in prior_cards]
    summary_docs = [_summary_doc_for_similarity(summary) for summary in summaries]
    semantic = _similarity_matrix_from_texts(
        summary_docs,
        prior_docs,
        tfidf_max_features=tfidf_max_features,
        embedding_model=embedding_model,
        disable_sentence_embeddings=disable_sentence_embeddings,
    )

    decision_rows: List[Dict[str, Any]] = []
    revise_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    add_candidates: List[Dict[str, Any]] = []

    for idx, summary in enumerate(summaries):
        summary = dict(summary)
        summary["signed_utility"] = _summary_signed_utility(summary, cluster_utilities)
        summary_like = _summary_to_skill_like(summary)
        best_sid = ""
        best_components: Dict[str, float] = {"total": 0.0}
        for j, sid in enumerate(prior_skill_ids):
            prior_like = _prior_card_to_summary_like(prior_ctx.operator_cards[sid])
            components = _skill_similarity_components(summary_like, prior_like, float(semantic[idx, j]))
            if components["total"] > best_components["total"]:
                best_sid = sid
                best_components = components

        support_count = _summary_support_strength(summary)
        decision = "reject"
        reason = "non_positive_utility"
        if float(summary["signed_utility"]) > float(min_utility):
            same_family = best_components.get("family", 0.0) >= 1.0
            borderline_revise = same_family and best_components["total"] >= max(
                float(add_similarity_max), float(revise_similarity) - 0.15
            )
            if best_components["total"] >= float(revise_similarity) or borderline_revise:
                decision = "revise"
                reason = "high_similarity_to_prior_skill" if not borderline_revise else "same_family_revision"
                revise_groups[best_sid].append(summary)
            elif (
                best_components["total"] <= float(add_similarity_max)
                and support_count >= int(add_min_support)
                and float(summary["signed_utility"]) >= positive_add_utility_threshold
            ):
                decision = "add"
                reason = "novel_positive_cluster"
                add_candidates.append(summary)
            else:
                reason = "weak_novelty_or_support"

        decision_rows.append(
            {
                "cluster_ids": summary.get("cluster_ids") or [summary.get("cluster_id")],
                "proposed_name": summary_like["name"],
                "signed_utility": float(summary["signed_utility"]),
                "support_count": support_count,
                "support_weight": float(summary.get("support_weight", 0.0)),
                "nearest_skill_id": best_sid,
                "nearest_skill_name": prior_ctx.operator_cards.get(best_sid, {}).get("name", ""),
                "similarity": best_components,
                "decision": decision,
                "reason": reason,
            }
        )

    final_specs: List[Dict[str, Any]] = []
    retained = revised = added = rejected = 0
    used_docs: List[str] = []
    used_skill_ids: Set[str] = set()
    used_names: Set[str] = set()

    for sid in prior_skill_ids:
        prior_card = prior_ctx.operator_cards[sid]
        if sid in revise_groups:
            revised_summary = _merge_revision_group(prior_card, revise_groups[sid])
            revised_summary["skill_id"] = sid
            final_specs.append(
                {
                    "action": "revise",
                    "skill_id": sid,
                    "prior_card": prior_card,
                    "summary": revised_summary,
                }
            )
            revised += 1
            used_docs.append(_summary_doc_for_similarity(revised_summary))
            used_names.add(_normalize_name(revised_summary.get("proposed_name", prior_card.get("name", sid))).lower())
        else:
            final_specs.append(
                {
                    "action": "retain",
                    "skill_id": sid,
                    "prior_card": prior_card,
                }
            )
            retained += 1
            used_docs.append(_operator_doc_for_similarity(prior_card))
            used_names.add(_normalize_name(prior_card.get("name", sid)).lower())
        used_skill_ids.add(sid)

    next_skill_num = max(int(sid[1:]) for sid in prior_skill_ids if sid.startswith("O")) + 1
    add_candidates = sorted(
        add_candidates,
        key=lambda s: (float(s.get("signed_utility", 0.0)), float(s.get("support_weight", 0.0))),
        reverse=True,
    )
    for summary in add_candidates:
        doc = _summary_doc_for_similarity(summary)
        proposed_name_key = _normalize_name(summary.get("proposed_name", "Operator Skill")).lower()
        if proposed_name_key in used_names:
            rejected += 1
            decision_rows.append(
                {
                    "cluster_ids": summary.get("cluster_ids") or [summary.get("cluster_id")],
                    "proposed_name": _normalize_name(summary.get("proposed_name", "Operator Skill")),
                    "signed_utility": float(summary.get("signed_utility", 0.0)),
                    "support_count": _summary_support_strength(summary),
                    "support_weight": float(summary.get("support_weight", 0.0)),
                    "nearest_skill_id": "",
                    "nearest_skill_name": "",
                    "similarity": {"total": 1.0},
                    "decision": "reject",
                    "reason": "duplicate_operator_name",
                }
            )
            continue
        if used_docs:
            sim = _similarity_matrix_from_texts(
                [doc],
                used_docs,
                tfidf_max_features=tfidf_max_features,
                embedding_model=embedding_model,
                disable_sentence_embeddings=disable_sentence_embeddings,
            )
            max_sim = float(np.max(sim))
        else:
            max_sim = 0.0
        if max_sim >= float(library_dedup_similarity):
            rejected += 1
            decision_rows.append(
                {
                    "cluster_ids": summary.get("cluster_ids") or [summary.get("cluster_id")],
                    "proposed_name": _normalize_name(summary.get("proposed_name", "Operator Skill")),
                    "signed_utility": float(summary.get("signed_utility", 0.0)),
                    "support_count": _summary_support_strength(summary),
                    "support_weight": float(summary.get("support_weight", 0.0)),
                    "nearest_skill_id": "",
                    "nearest_skill_name": "",
                    "similarity": {"total": max_sim},
                    "decision": "reject",
                    "reason": "dedup_against_final_library",
                }
            )
            continue
        sid = f"O{next_skill_num:03d}"
        next_skill_num += 1
        summary = sanitize_summary(summary)
        summary["skill_id"] = sid
        final_specs.append({"action": "add", "skill_id": sid, "summary": summary})
        added += 1
        used_docs.append(doc)
        used_names.add(proposed_name_key)
        used_skill_ids.add(sid)

    summary_stats = {
        "retained_count": retained,
        "revised_count": revised,
        "added_count": added,
        "rejected_count": len([row for row in decision_rows if row["decision"] == "reject"]),
        "final_operator_count": len(final_specs),
        "positive_add_utility_threshold": positive_add_utility_threshold,
    }
    return final_specs, decision_rows, summary_stats


def _unique_keep_order(items: Iterable[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = _clean_text(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return out


def _normalize_step_text(step: str) -> str:
    step = _clean_text(step)
    step = re.sub(r"^\d+\s*[\)\.\-:]\s*", "", step)
    step = step.rstrip(":")
    return step


def dedup_merge_summaries(
    summaries: List[Dict[str, Any]],
    merge_similarity: float,
    min_skills: int,
    max_skills: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if len(summaries) <= 1:
        return summaries

    docs = []
    for s in summaries:
        docs.append(
            " ".join(
                [
                    s.get("proposed_name", ""),
                    s.get("description", ""),
                    " ".join(s.get("tags", [])),
                    " ".join(s.get("canonical_triggers", [])),
                    " ".join(s.get("canonical_steps", [])),
                ]
            )
        )

    vec = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
    x = vec.fit_transform(docs)
    sim = cosine_similarity(x)

    parent = list(range(len(summaries)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            if sim[i, j] >= merge_similarity:
                union(i, j)

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i, s in enumerate(summaries):
        groups[find(i)].append(s)

    merged: List[Dict[str, Any]] = []
    for _, group in groups.items():
        group = sorted(group, key=lambda s: float(s.get("support_weight", 0.0)), reverse=True)
        base = dict(group[0])
        base["cluster_ids"] = [g.get("cluster_id") for g in group if g.get("cluster_id")]
        base["support_count"] = int(sum(int(g.get("support_count", 0)) for g in group))
        base["support_weight"] = float(sum(float(g.get("support_weight", 0.0)) for g in group))

        base["tags"] = _unique_keep_order([x for g in group for x in g.get("tags", [])], limit=14)
        base["canonical_triggers"] = _unique_keep_order(
            [x for g in group for x in g.get("canonical_triggers", [])], limit=10
        )
        base["canonical_steps"] = _unique_keep_order(
            [x for g in group for x in g.get("canonical_steps", [])], limit=10
        )
        base["common_pitfalls"] = _unique_keep_order(
            [x for g in group for x in g.get("common_pitfalls", [])], limit=10
        )
        base["when_not_to_use"] = _unique_keep_order(
            [x for g in group for x in g.get("when_not_to_use", [])], limit=8
        )
        base["evidence_top_steps"] = _unique_keep_order(
            [x for g in group for x in g.get("evidence_top_steps", [])], limit=10
        )
        base["evidence_top_triggers"] = _unique_keep_order(
            [x for g in group for x in g.get("evidence_top_triggers", [])], limit=10
        )
        base["evidence_problem_tags"] = _unique_keep_order(
            [x for g in group for x in g.get("evidence_problem_tags", [])], limit=10
        )
        merged.append(base)

    merged = sorted(merged, key=lambda s: float(s.get("support_weight", 0.0)), reverse=True)

    merged = merge_by_scope_overlap(merged)

    if max_skills is not None and max_skills > 0 and len(merged) > max_skills:
        merged = merged[:max_skills]

    # Keep enough skills for testing and pipeline validation.
    if len(merged) < min_skills and len(summaries) >= min_skills:
        fallback = sorted(
            summaries, key=lambda s: float(s.get("support_weight", 0.0)), reverse=True
        )
        return fallback[:min_skills]

    return merged


def _canonical_scope_name(name: str) -> str:
    n = _normalize_name(name).lower()
    if "constant factor" in n:
        return "constant_factor_optimization"
    if any(k in n for k in ["binomial", "combinatorics", "factorial", "ncr", "modular"]):
        return "modular_combinatorics"
    if "factorization" in n and ("divisor" in n or "prime" in n):
        return "factorization_divisor"
    if "coprimality" in n:
        return "coprimality_classification"
    if "digit" in n and ("count" in n or "range" in n):
        return "digit_counting"
    if "algebraic" in n or "structural" in n:
        return "algebraic_structural"
    return n


def merge_by_scope_overlap(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(summaries) <= 1:
        return summaries

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        grouped[_canonical_scope_name(s.get("proposed_name", ""))].append(s)

    merged_all: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        if len(items) == 1:
            merged_all.append(items[0])
            continue

        # Merge only if overlap suggests true redundancy.
        items = sorted(items, key=lambda x: float(x.get("support_weight", 0.0)), reverse=True)
        base = dict(items[0])
        trigger_sets = [set(x.get("canonical_triggers", [])) for x in items]
        overlap = 0.0
        if trigger_sets:
            inter = set.intersection(*trigger_sets) if len(trigger_sets) > 1 else trigger_sets[0]
            union = set.union(*trigger_sets) if len(trigger_sets) > 1 else trigger_sets[0]
            overlap = (len(inter) / len(union)) if union else 1.0

        if key != "constant_factor_optimization" and overlap < 0.25 and len(items) > 1:
            merged_all.extend(items)
            continue

        base["cluster_ids"] = _unique_keep_order(
            [str(cid) for x in items for cid in x.get("cluster_ids", [x.get("cluster_id")]) if cid],
            limit=20,
        )
        base["support_count"] = int(sum(int(x.get("support_count", 0)) for x in items))
        base["support_weight"] = float(sum(float(x.get("support_weight", 0.0)) for x in items))
        base["tags"] = _unique_keep_order([y for x in items for y in x.get("tags", [])], limit=12)
        base["canonical_triggers"] = _unique_keep_order(
            [y for x in items for y in x.get("canonical_triggers", [])], limit=8
        )
        base["canonical_steps"] = _normalize_steps(
            [y for x in items for y in x.get("canonical_steps", [])], limit=8
        )
        base["common_pitfalls"] = _unique_keep_order(
            [y for x in items for y in x.get("common_pitfalls", [])], limit=8
        )
        base["when_not_to_use"] = _unique_keep_order(
            [y for x in items for y in x.get("when_not_to_use", [])], limit=6
        )
        base["evidence_top_steps"] = _unique_keep_order(
            [y for x in items for y in x.get("evidence_top_steps", [])], limit=10
        )
        base["evidence_top_triggers"] = _unique_keep_order(
            [y for x in items for y in x.get("evidence_top_triggers", [])], limit=10
        )
        base["evidence_problem_tags"] = _unique_keep_order(
            [y for x in items for y in x.get("evidence_problem_tags", [])], limit=10
        )
        merged_all.append(base)

    merged_all = sorted(merged_all, key=lambda s: float(s.get("support_weight", 0.0)), reverse=True)
    return merged_all


def resolve_skill_bounds(
    min_skills_arg: str, k_clusters: int, n_traces: int
) -> Tuple[int, Optional[int], str]:
    arg = str(min_skills_arg).strip().lower()
    if arg != "auto":
        fixed = int(arg)
        if fixed <= 0:
            raise ValueError("--min-skills must be positive integer or 'auto'")
        return fixed, None, "fixed"

    # Dynamic target: enough skills for coverage, but avoid over-fragmentation.
    min_dynamic = max(6, int(math.ceil(0.35 * max(1, k_clusters))))
    max_dynamic = max(min_dynamic, int(min(k_clusters, math.ceil(0.9 * max(1, k_clusters)))))

    # Keep dynamic target realistic for tiny smoke runs.
    min_dynamic = min(min_dynamic, max(2, n_traces))
    max_dynamic = min(max_dynamic, max(2, n_traces))
    if max_dynamic < min_dynamic:
        max_dynamic = min_dynamic
    return min_dynamic, max_dynamic, "auto"


def support_coverage(kept: List[Dict[str, Any]], raw: List[Dict[str, Any]]) -> float:
    total = float(sum(float(x.get("support_weight", 0.0)) for x in raw))
    if total <= 0:
        return 0.0
    kept_total = float(sum(float(x.get("support_weight", 0.0)) for x in kept))
    return float(np.clip(kept_total / total, 0.0, 1.0))


def write_summaries(path: str, summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_out:
        for s in summaries:
            f_out.write(json.dumps(s, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _infer_operator_family(summary: Dict[str, Any]) -> str:
    name = _normalize_name(summary.get("proposed_name", "")).lower()
    tags = " ".join(summary.get("tags", [])).lower()
    key_blob = f"{name} {tags}"
    aux_blob = " ".join(
        [
            " ".join(summary.get("canonical_triggers", [])).lower(),
            " ".join(summary.get("canonical_steps", [])).lower(),
            " ".join(summary.get("evidence_problem_tags", [])).lower(),
            " ".join(summary.get("evidence_top_triggers", [])).lower(),
            " ".join(summary.get("evidence_top_steps", [])).lower(),
        ]
    )

    families = [
        (
            "coprimality",
            ["coprime", "coprimality", "gcd", "sieve", "spf", "prime factor"],
        ),
        (
            "graph",
            ["graph", "dag", "topological", "adjacency", "edge", "bfs", "dfs", "scc"],
        ),
        (
            "combinatorics",
            [
                "binomial",
                "combinatorics",
                "factorial",
                "ncr",
                "modular",
                "stars-and-bars",
            ],
        ),
        (
            "dp",
            [
                "dynamic programming",
                " dp ",
                "segment tree",
                "fenwick",
                "range query",
                "transition",
            ],
        ),
        (
            "state_compression",
            [
                "subset",
                "bitmask",
                "exponential",
                "closed form",
                "parity",
                "state compression",
                "set intersection",
                "frequency array",
                "common elements",
            ],
        ),
        ("quadratic_mean", ["mean", "squared deviation", "sum of squares", "variance"]),
        (
            "math_loop",
            [
                "n // i",
                "sqrt decomposition",
                "harmonic",
                "quotient grouping",
                "floor-sum",
            ],
        ),
        (
            "constant_factor",
            [
                "constant factor",
                "template",
                "vectorization",
                "library",
                "hot path",
                "micro-optimization",
            ],
        ),
        (
            "streaming",
            ["single pass", "streaming", "aggregate", "running total", "sentinel"],
        ),
        ("digit", ["digit", "decimal", "base", "k-th digit", "kth digit"]),
    ]
    priority = [
        "coprimality",
        "graph",
        "combinatorics",
        "dp",
        "state_compression",
        "quadratic_mean",
        "math_loop",
        "constant_factor",
        "streaming",
        "digit",
    ]

    scores: Dict[str, int] = {}
    for family, keys in families:
        score = 0
        for k in keys:
            if k in key_blob:
                score += 3
            if k in aux_blob:
                score += 1
        scores[family] = score

    best = max(scores.values()) if scores else 0
    if best <= 0:
        return "generic"
    best_families = [f for f, s in scores.items() if s == best]
    for fam in priority:
        if fam in best_families:
            return fam
    return best_families[0]


def _family_keywords(family: str) -> List[str]:
    kw = {
        "combinatorics": ["factorial", "ncr", "mod", "binom", "comb", "long long"],
        "coprimality": ["gcd", "coprime", "prime", "sieve", "spf", "vector<int>"],
        "graph": ["graph", "adj", "edge", "deque", "queue", "topo", "dag"],
        "digit": ["digit", "to_string", "base", "pow10", "long long"],
        "state_compression": ["mask", "subset", "1 <<", "parity", "bitset"],
        "dp": ["dp", "segment tree", "fenwick", "range", "query", "update", "vector<int>"],
        "quadratic_mean": ["sum", "(x - p)", "mean", "variance", "long long"],
        "streaming": ["single pass", "running", "aggregate", "ans +=", "reserve"],
        "math_loop": ["n /", "sqrt", "while", "divisor", "quotient"],
        "constant_factor": ["reserve", "emplace_back", "ios::sync_with_stdio", "cin.tie", "vector"],
    }
    return kw.get(family, [])


def _intent_tokens(summary: Dict[str, Any], limit: int = 16) -> List[str]:
    blob = " ".join(
        [
            _normalize_name(summary.get("proposed_name", "")).lower(),
            " ".join(summary.get("tags", [])).lower(),
            " ".join(summary.get("canonical_triggers", [])).lower(),
            " ".join(summary.get("canonical_steps", [])).lower(),
            " ".join(summary.get("evidence_top_steps", [])).lower(),
            " ".join(summary.get("evidence_top_triggers", [])).lower(),
        ]
    )
    stop = {
        "with",
        "from",
        "into",
        "when",
        "where",
        "while",
        "that",
        "this",
        "then",
        "than",
        "using",
        "used",
        "over",
        "under",
        "before",
        "after",
        "code",
        "cpp",
        "iostream",
    }
    out: List[str] = []
    seen = set()
    for tok in re.findall(r"[a-z][a-z0-9_]{2,}", blob):
        if tok in stop or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def _extract_example_step(summary: Dict[str, Any]) -> str:
    for src in (
        summary.get("canonical_steps", []),
        summary.get("evidence_top_steps", []),
        summary.get("evidence_top_triggers", []),
    ):
        if src:
            return _normalize_step_text(src[0])
    return "Apply the primary optimization transformation"


def _example_template_for_family(summary: Dict[str, Any], family: str) -> Tuple[str, str]:
    blob = " ".join(
        [
            _normalize_name(summary.get("proposed_name", "")).lower(),
            " ".join(summary.get("tags", [])).lower(),
            " ".join(summary.get("canonical_triggers", [])).lower(),
            " ".join(summary.get("canonical_steps", [])).lower(),
        ]
    )
    if family == "combinatorics":
        if any(k in blob for k in ["cubic", "nested loops", "constraint", "enumeration"]):
            return (
                "long long cnt = 0;\nfor (int a = 1; a <= n; ++a)\n  for (int b = 1; b <= n; ++b)\n    for (int c = 1; c <= n; ++c)\n      if (a + b + c == S) ++cnt;",
                "long long cnt = 0;\nfor (int a = 1; a <= n; ++a) {\n  int lo = max(1, S - a - n), hi = min(n, S - a - 1);\n  if (lo <= hi) cnt += hi - lo + 1;\n}",
            )
        if any(k in blob for k in ["algebraic", "sum constrained", "loop elimination"]):
            return (
                "long long ans = 0;\nfor (int x = 1; x <= n; ++x)\n  for (int y = 1; y <= n; ++y) {\n    int z = S - x - y;\n    if (1 <= z && z <= n) ans += f(x, y, z);\n  }",
                "long long ans = 0;\nfor (int x = 1; x <= n; ++x) {\n  int lo = max(1, S - x - n), hi = min(n, S - x - 1);\n  for (int y = lo; y <= hi; ++y) ans += f_fast(x, y, S - x - y);\n}",
            )
        return (
            "vector<long long> fact(MAXN + 1);\ninit_fact(fact, MOD);\nlong long ans = nCr(N + e - 1, N - 1, fact, MOD);\nans = ans * mod_pow(x, MOD - 2, MOD) % MOD;",
            "int e = factor_exp(m, p);\nlong long ans = comb_small_k(N + e - 1, e, MOD);\nans = ans * mod_inv(x, MOD) % MOD;\n// avoid oversized global factorial tables",
        )
    if family == "coprimality":
        return (
            "bool pairwise = true;\nfor (int i = 0; i < n; ++i)\n  for (int j = i + 1; j < n; ++j)\n    if (std::gcd(a[i], a[j]) != 1) pairwise = false;",
            "auto spf = build_spf(*max_element(a.begin(), a.end()));\nvector<int> seen(spf.size(), 0);\nfor (int x : a)\n  for (int p : distinct_prime_factors(x, spf))\n    if (seen[p]++) pairwise = false;",
        )
    if family == "graph":
        return (
            "vector<vector<int>> g(n, vector<int>(n, 0));\nfor (auto [u, v] : edges) g[u][v] = 1;\nqueue<int> q;\nq.push(0);",
            "vector<vector<int>> adj(n);\nfor (auto [u, v] : edges) adj[u].push_back(v);\ndeque<int> q{0};\nauto ans = topo_dp(adj);",
        )
    if family == "digit":
        return (
            "long long ans = 0;\nfor (long long i = 1; i <= n; ++i)\n  ans += to_string(i).size();",
            "long long ans = 0;\nfor (int d = 1; d <= 18; ++d) {\n  long long L = (d == 1 ? 1 : pow10[d - 1]), R = min(n, pow10[d] - 1);\n  if (L <= R) ans += (R - L + 1) * d;\n}",
        )
    if family == "state_compression":
        if any(k in blob for k in ["set intersection", "frequency", "common elements"]):
            return (
                "vector<int> ans;\nfor (int x = 1; x <= MAXV; ++x) {\n  bool ok = true;\n  for (const auto& grp : groups) if (!binary_search(grp.begin(), grp.end(), x)) ok = false;\n  if (ok) ans.push_back(x);\n}",
                "vector<int> freq(MAXV + 1, 0);\nfor (const auto& grp : groups)\n  for (int x : grp) ++freq[x];\nfor (int x = 1; x <= MAXV; ++x)\n  if (freq[x] == (int)groups.size()) ans.push_back(x);",
            )
        return (
            "int best = 0;\nfor (int mask = 0; mask < (1 << n); ++mask)\n  best = max(best, score(mask));",
            "int odd = 0;\nfor (int x : a) odd += x & 1;\nint best = closed_form_from_counts((int)a.size(), odd);\n// replace subset scan with parity/count reasoning",
        )
    if family == "dp":
        return (
            "vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));\nfor (int i = 1; i <= n; ++i)\n  for (int w = 0; w <= W; ++w)\n    dp[i][w] = max(dp[i - 1][w], w >= wt[i] ? dp[i - 1][w - wt[i]] + val[i] : 0);",
            "vector<int> dp(W + 1, 0);\nfor (int i = 1; i <= n; ++i)\n  for (int w = W; w >= wt[i]; --w)\n    dp[w] = max(dp[w], dp[w - wt[i]] + val[i]);",
        )
    if family == "quadratic_mean":
        return (
            "long long best = (1LL << 62);\nfor (int p = *min_element(a.begin(), a.end()); p <= *max_element(a.begin(), a.end()); ++p)\n  best = min(best, cost(a, p));",
            "double mean = accumulate(a.begin(), a.end(), 0.0) / a.size();\nvector<int> cand = {(int)floor(mean), (int)ceil(mean)};\nlong long best = min(cost(a, cand[0]), cost(a, cand[1]));",
        )
    if family == "math_loop":
        return (
            "long long ans = 0;\nfor (long long i = 1; i <= n; ++i)\n  ans += n / i;",
            "long long ans = 0;\nfor (long long i = 1, j; i <= n; i = j + 1) {\n  long long q = n / i; j = n / q;\n  ans += q * (j - i + 1);\n}",
        )
    if family == "constant_factor":
        return (
            "vector<int> vals;\nfor (int i = 0; i < n; ++i) vals.push_back(read());\nsort(vals.begin(), vals.end());",
            "vector<int> vals;\nvals.reserve(n);\nfor (int i = 0; i < n; ++i) vals.push_back(read());\nsort(vals.begin(), vals.end());",
        )
    if family == "streaming":
        return (
            "vector<long long> vals;\nfor (int x : data) vals.push_back(transform(x));\nlong long ans = accumulate(vals.begin(), vals.end(), 0LL);",
            "long long ans = 0;\nfor (int x : data)\n  ans += transform(x);\n// single-pass aggregate",
        )

    step = _extract_example_step(summary)
    before = "long long ans = 0;\nfor (int x : arr)\n  ans += work(x);"
    after = f"{LINE_COMMENT} {step}\nlong long ans = 0;\nfor (int x : arr)\n  ans += work_fast(x);"
    return before, after


def _strip_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python|py|cpp|c\+\+|cc)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s).strip()
    return s


def _code_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x).strip()).strip()
    return str(value).strip()


def _example_signature(before: str, after: str) -> str:
    norm = re.sub(r"\s+", " ", (before + "\n---\n" + after).strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _personalize_fallback_example(
    summary: Dict[str, Any], skill_id: str, before: str, after: str
) -> Tuple[str, str]:
    anchors = _intent_tokens(summary, limit=4)
    anchor = anchors[0] if anchors else _infer_operator_family(summary)
    before_out = f"{LINE_COMMENT} {skill_id} focus: {anchor}\n{before}"
    after_out = f"{LINE_COMMENT} optimized for {anchor}\n{after}"
    return before_out, after_out


def _example_matches_family(summary: Dict[str, Any], family: str, before: str, after: str) -> bool:
    text = f"{before}\n{after}".lower()
    kw = _family_keywords(family)
    if not kw:
        return True
    if any(k in text for k in kw):
        return True

    # Relaxed overlap fallback: example should at least share meaningful tokens with skill intent.
    intent = " ".join(
        [
            _normalize_name(summary.get("proposed_name", "")).lower(),
            " ".join(summary.get("tags", [])).lower(),
            " ".join(summary.get("canonical_triggers", [])).lower(),
        ]
    )
    intent_tokens = {
        t
        for t in re.findall(r"[a-z][a-z0-9_]{2,}", intent)
        if t not in {"with", "from", "into", "when"}
    }
    ex_tokens = {t for t in re.findall(r"[a-z][a-z0-9_]{2,}", text)}
    if not intent_tokens:
        return True
    overlap = len(intent_tokens & ex_tokens) / max(1, len(intent_tokens))
    return overlap >= 0.10


def _example_is_valid(summary: Dict[str, Any], family: str, before: str, after: str) -> bool:
    b = _strip_code_fence(before)
    a = _strip_code_fence(after)
    if not b or not a:
        return False
    if b == a:
        return False
    if (
        len([x for x in b.splitlines() if x.strip()]) < 2
        or len([x for x in a.splitlines() if x.strip()]) < 2
    ):
        return False
    text = (b + "\n" + a).lower()
    if re.search(r"\b(slow_op|fast_op|preprocess\(data\)|baseline approach)\b", text):
        return False
    if "..." in text or "…" in text:
        return False
    intent = set(_intent_tokens(summary, limit=20))
    ex_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", text))
    intent_overlap = len(intent & ex_tokens)
    if intent and intent_overlap == 0:
        return False
    if not _example_matches_family(summary, family, b, a) and intent_overlap < 2:
        return False
    return True


def _example_is_structural(before: str, after: str) -> bool:
    b = _strip_code_fence(before)
    a = _strip_code_fence(after)
    if not b or not a:
        return False
    if (
        len([x for x in b.splitlines() if x.strip()]) < 2
        or len([x for x in a.splitlines() if x.strip()]) < 2
    ):
        return False
    text = (b + "\n" + a).lower()
    if re.search(r"\b(slow_op|fast_op|preprocess\(data\)|baseline approach)\b", text):
        return False
    if "..." in text or "…" in text:
        return False
    return True


def _example_is_retry_acceptable(summary: Dict[str, Any], before: str, after: str) -> bool:
    if not _example_is_structural(before, after):
        return False
    b = _strip_code_fence(before)
    a = _strip_code_fence(after)
    if b == a:
        return False
    return True


def _llm_example_snippets(
    summary: Dict[str, Any],
    family: str,
    skill_id: str,
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Optional[Tuple[str, str]]:
    desc = _truncate_text(
        _sanitize_prose(_scalar_text(summary.get("description", ""))), max_chars=280
    )
    prompt_obj = {
        "skill_id": skill_id,
        "name": summary.get("proposed_name", ""),
        "description": desc,
        "tags": (summary.get("tags") or [])[:8],
        "family": family,
        "canonical_triggers": (summary.get("canonical_triggers") or [])[:4],
        "canonical_steps": (summary.get("canonical_steps") or [])[:4],
        "evidence_top_steps": (summary.get("evidence_top_steps") or [])[:4],
        "evidence_top_triggers": (summary.get("evidence_top_triggers") or [])[:4],
        "intent_tokens": _intent_tokens(summary, limit=12),
    }
    prompt = build_operator_example_prompt(prompt_obj)
    messages = [
        {"role": "system", "content": operator_example_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    obj = _llm_json_call(
        messages=messages,
        cache_dir=cache_dir,
        stage=f"a4_example_{skill_id}",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        out_fail_dir=fail_dir,
        reasoning_effort=reasoning_effort,
        fallback_model=fallback_model,
    )
    if obj is None:
        return None

    before = _strip_code_fence(
        _code_text(obj.get("before_code") or obj.get("before") or obj.get("slow_code"))
    )
    after = _strip_code_fence(
        _code_text(obj.get("after_code") or obj.get("after") or obj.get("fast_code"))
    )
    if _example_is_valid(summary, family, before, after):
        return before, after
    return None


def _llm_example_snippets_relaxed(
    summary: Dict[str, Any],
    family: str,
    skill_id: str,
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Optional[Tuple[str, str]]:
    compact = {
        "skill_id": skill_id,
        "name": summary.get("proposed_name", ""),
        "family": family,
        "description": _truncate_text(
            _sanitize_prose(_scalar_text(summary.get("description", ""))), max_chars=180
        ),
        "intent_tokens": _intent_tokens(summary, limit=8),
    }
    prompt = build_operator_example_relaxed_prompt(compact)
    messages = [
        {"role": "system", "content": operator_example_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    obj = _llm_json_call(
        messages=messages,
        cache_dir=cache_dir,
        stage=f"a4_example_relaxed_{skill_id}",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        out_fail_dir=fail_dir,
        reasoning_effort=reasoning_effort,
        fallback_model=fallback_model,
    )
    if obj is None:
        return None
    before = _strip_code_fence(
        _code_text(obj.get("before_code") or obj.get("before") or obj.get("slow_code"))
    )
    after = _strip_code_fence(
        _code_text(obj.get("after_code") or obj.get("after") or obj.get("fast_code"))
    )
    if _example_is_retry_acceptable(summary, before, after):
        return before, after
    return None


def _llm_example_snippets_compact(
    summary: Dict[str, Any],
    family: str,
    skill_id: str,
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Optional[Tuple[str, str]]:
    compact = {
        "skill_id": skill_id,
        "name": summary.get("proposed_name", ""),
        "family": family,
        "description": _truncate_text(
            _sanitize_prose(_scalar_text(summary.get("description", ""))), max_chars=180
        ),
        "intent_tokens": _intent_tokens(summary, limit=8),
    }
    prompt = build_operator_example_compact_prompt(compact)
    messages = [
        {"role": "system", "content": operator_example_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    obj = _llm_json_call(
        messages=messages,
        cache_dir=cache_dir,
        stage=f"a4_example_compact_{skill_id}",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        out_fail_dir=fail_dir,
        reasoning_effort=reasoning_effort,
        fallback_model=fallback_model,
    )
    if obj is None:
        return None
    before = _strip_code_fence(
        _code_text(obj.get("before_code") or obj.get("before") or obj.get("slow_code"))
    )
    after = _strip_code_fence(
        _code_text(obj.get("after_code") or obj.get("after") or obj.get("fast_code"))
    )
    if _example_is_valid(summary, family, before, after):
        return before, after
    return None


def _example_snippets(
    summary: Dict[str, Any],
    skill_id: str,
    used_signatures: Optional[set] = None,
    cache_dir: Optional[str] = None,
    fail_dir: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 32000,
    reasoning_effort: str = "low",
) -> Tuple[str, str]:
    family = _infer_operator_family(summary)
    if cache_dir and fail_dir and model:
        llm_pair = _llm_example_snippets(
            summary=summary,
            family=family,
            skill_id=skill_id,
            cache_dir=cache_dir,
            fail_dir=fail_dir,
            model=model,
            fallback_model=fallback_model or model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if llm_pair is not None:
            sig = _example_signature(llm_pair[0], llm_pair[1])
            if not used_signatures or sig not in used_signatures:
                return llm_pair
        llm_pair_compact = _llm_example_snippets_compact(
            summary=summary,
            family=family,
            skill_id=skill_id,
            cache_dir=cache_dir,
            fail_dir=fail_dir,
            model=model,
            fallback_model=fallback_model or model,
            temperature=temperature,
            max_tokens=max(32000, max_tokens),
            reasoning_effort=reasoning_effort,
        )
        if llm_pair_compact is not None:
            sig = _example_signature(llm_pair_compact[0], llm_pair_compact[1])
            if not used_signatures or sig not in used_signatures:
                return llm_pair_compact

    before, after = _example_template_for_family(summary, family)
    if not _example_is_valid(summary, family, before, after):
        before, after = _personalize_fallback_example(summary, skill_id, before, after)
        if not _example_is_structural(before, after):
            before, after = _example_template_for_family({"proposed_name": family}, family)
    if used_signatures and _example_signature(before, after) in used_signatures:
        before, after = _personalize_fallback_example(summary, skill_id, before, after)
    return before, after


def _extract_example_blocks(skill_text: str) -> Tuple[Optional[str], Optional[str]]:
    blocks = re.findall(r"```(?:py|python|cpp|c\+\+)\s*\n(.*?)\n```", skill_text, flags=re.S)
    if len(blocks) < 2:
        return None, None
    return blocks[0], blocks[1]


def _render_minimal_example(before: str, after: str) -> str:
    return "\n".join(
        [
            "## Minimal example",
            "Before:",
            f"```{CODE_FENCE}",
            before,
            "```",
            "After:",
            f"```{CODE_FENCE}",
            after,
            "```",
        ]
    )


def retry_fallback_examples(
    summaries: List[Dict[str, Any]],
    operators_dir: str,
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Dict[str, int]:
    stats = {
        "fallback_detected": 0,
        "retry_success": 0,
        "retry_failed": 0,
        "retry_skipped_no_file": 0,
        "retry_skipped_not_fallback": 0,
    }

    # Track signatures already present so retried examples stay diverse.
    used_signatures: set = set()
    skill_files = sorted(glob.glob(os.path.join(operators_dir, "O*_*/SKILL.md")))
    for p in skill_files:
        text = Path(p).read_text(encoding="utf-8")
        b, a = _extract_example_blocks(text)
        if b is not None and a is not None:
            used_signatures.add(_example_signature(b, a))

    for idx, raw in enumerate(summaries, start=1):
        skill_id = f"O{idx:03d}"
        matches = glob.glob(os.path.join(operators_dir, f"{skill_id}_*/SKILL.md"))
        if not matches:
            stats["retry_skipped_no_file"] += 1
            continue
        path = matches[0]
        text = Path(path).read_text(encoding="utf-8")

        # Retry only for deterministic fallback-marked examples.
        if not re.search(rf"^{re.escape(LINE_COMMENT)}\s*{skill_id}\s+focus:", text, flags=re.M):
            stats["retry_skipped_not_fallback"] += 1
            continue

        stats["fallback_detected"] += 1
        summary = sanitize_summary(raw)
        family = _infer_operator_family(summary)

        pair: Optional[Tuple[str, str]] = None
        for temp_try in (
            temperature,
            min(1.0, temperature + 0.2),
            min(1.0, temperature + 0.4),
        ):
            pair = _llm_example_snippets(
                summary=summary,
                family=family,
                skill_id=skill_id,
                cache_dir=cache_dir,
                fail_dir=fail_dir,
                model=model,
                fallback_model=fallback_model,
                temperature=temp_try,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            if pair is None:
                pair = _llm_example_snippets_compact(
                    summary=summary,
                    family=family,
                    skill_id=skill_id,
                    cache_dir=cache_dir,
                    fail_dir=fail_dir,
                    model=model,
                    fallback_model=fallback_model,
                    temperature=temp_try,
                    max_tokens=max(32000, max_tokens),
                    reasoning_effort=reasoning_effort,
                )
            if pair is None:
                pair = _llm_example_snippets_relaxed(
                    summary=summary,
                    family=family,
                    skill_id=skill_id,
                    cache_dir=cache_dir,
                    fail_dir=fail_dir,
                    model=model,
                    fallback_model=fallback_model,
                    temperature=temp_try,
                    max_tokens=max(32000, max_tokens),
                    reasoning_effort=reasoning_effort,
                )
            if pair is not None:
                sig_try = _example_signature(pair[0], pair[1])
                if sig_try not in used_signatures:
                    break
                pair = None
        if pair is None:
            stats["retry_failed"] += 1
            continue

        sig = _example_signature(pair[0], pair[1])

        new_section = _render_minimal_example(pair[0], pair[1])
        if re.search(r"(?s)\n## Minimal example\n.*$", text):
            text_new = re.sub(r"(?s)\n## Minimal example\n.*$", "\n" + new_section + "\n", text)
        else:
            text_new = text.rstrip() + "\n\n" + new_section + "\n"
        Path(path).write_text(text_new, encoding="utf-8")
        used_signatures.add(sig)
        stats["retry_success"] += 1

    return stats


def _render_operator_skill_text(
    summary: Dict[str, Any],
    skill_id: str,
    cache_dir: Optional[str] = None,
    fail_dir: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 32000,
    reasoning_effort: str = "low",
    used_example_signatures: Optional[set] = None,
    example_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    s = sanitize_summary(summary)
    name = _normalize_name(s.get("proposed_name", skill_id), fallback=skill_id)
    family = _infer_operator_family(s)
    front = {
        "skill_id": skill_id,
        "type": "operator",
        "language": SKILL_LANGUAGE,
        "family": family,
        "name": name,
        "description": s.get("description", ""),
        "tags": s.get("tags", []),
        "triggers": s.get("canonical_triggers", []),
    }
    front_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=120).strip()

    body: List[str] = []
    body.append("## When to use")
    body.extend(
        [f"- {x}" for x in s.get("canonical_triggers", [])]
        or ["- Use when this pattern strongly matches."]
    )

    body.append("")
    body.append("## Steps")
    steps = s.get("canonical_steps", [])
    if steps:
        body.extend([f"{i+1}. {_normalize_step_text(step)}" for i, step in enumerate(steps)])
    else:
        body.append("1. Apply the core transformation pattern described above.")

    body.append("")
    body.append("## Complexity")
    t = _scalar_text(s.get("complexity_time"), default="(pattern dependent)")
    sp = _scalar_text(s.get("complexity_space"), default="(pattern dependent)")
    body.append(f"- Time: {t}")
    body.append(f"- Space: {sp}")

    body.append("")
    body.append("## Pitfalls")
    pitfalls = s.get("common_pitfalls", [])
    if pitfalls:
        body.extend([f"- {x}" for x in pitfalls])
    else:
        body.append("- Verify edge cases and interface compatibility.")

    if s.get("when_not_to_use"):
        body.append("")
        body.append("## When not to use")
        body.extend([f"- {x}" for x in s.get("when_not_to_use", [])])

    body.append("")
    body.append("## Minimal example")
    if example_override:
        body.append(example_override.strip())
    else:
        before_snippet, after_snippet = _example_snippets(
            summary=s,
            skill_id=skill_id,
            used_signatures=used_example_signatures,
            cache_dir=cache_dir,
            fail_dir=fail_dir,
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if used_example_signatures is not None:
            used_example_signatures.add(_example_signature(before_snippet, after_snippet))
        body.append("Before:")
        body.append(f"```{CODE_FENCE}")
        body.append(before_snippet)
        body.append("```")
        body.append("After:")
        body.append(f"```{CODE_FENCE}")
        body.append(after_snippet)
        body.append("```")

    text = f"---\n{front_text}\n---\n\n" + "\n".join(body) + "\n"
    card = {
        "skill_id": skill_id,
        "family": family,
        "name": name,
        "description": s.get("description", ""),
        "tags": s.get("tags", []),
        "triggers": s.get("canonical_triggers", []),
    }
    return text, card


def write_operator_skills(
    summaries: List[Dict[str, Any]],
    operators_dir: str,
    cache_dir: Optional[str] = None,
    fail_dir: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 32000,
    reasoning_effort: str = "low",
) -> List[Dict[str, Any]]:
    os.makedirs(operators_dir, exist_ok=True)

    cards: List[Dict[str, Any]] = []
    used_example_signatures: set = set()
    for idx, raw in enumerate(summaries, start=1):
        skill_id = f"O{idx:03d}"
        s = sanitize_summary(raw)
        name = _normalize_name(s.get("proposed_name", f"Operator {idx}"), fallback=f"Operator {idx}")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "skill"

        folder = os.path.join(operators_dir, f"{skill_id}_{slug}")
        os.makedirs(folder, exist_ok=True)
        text, card = _render_operator_skill_text(
            summary=s,
            skill_id=skill_id,
            cache_dir=cache_dir,
            fail_dir=fail_dir,
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            used_example_signatures=used_example_signatures,
        )
        with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(text)
        cards.append(card)

    return cards


def write_stage3_operator_library(
    operator_specs: List[Dict[str, Any]],
    operators_dir: str,
    cache_dir: Optional[str] = None,
    fail_dir: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 32000,
    reasoning_effort: str = "low",
) -> List[Dict[str, Any]]:
    os.makedirs(operators_dir, exist_ok=True)
    cards: List[Dict[str, Any]] = []
    used_example_signatures: set = set()

    for spec in sorted(operator_specs, key=lambda x: x.get("skill_id", "")):
        action = str(spec.get("action", "retain"))
        skill_id = str(spec.get("skill_id", ""))
        prior_card = spec.get("prior_card") or {}
        if action == "retain":
            source_path = prior_card.get("source_path")
            if not source_path:
                raise ValueError(f"Retain action for {skill_id} missing source_path")
            src = Path(source_path)
            folder = Path(operators_dir) / src.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, folder / "SKILL.md")
            cards.append(
                {
                    "skill_id": skill_id,
                    "family": prior_card.get("family", ""),
                    "name": prior_card.get("name", ""),
                    "description": prior_card.get("description", ""),
                    "tags": prior_card.get("tags", []),
                    "triggers": prior_card.get("canonical_triggers", []) or prior_card.get("triggers", []),
                }
            )
            continue

        summary = spec.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{action} action for {skill_id} missing summary")
        summary = dict(summary)
        summary["skill_id"] = skill_id
        name = _normalize_name(summary.get("proposed_name", skill_id), fallback=skill_id)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "skill"
        folder = Path(operators_dir) / f"{skill_id}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)

        example_override = None
        if action == "revise":
            example_override = _extract_markdown_section(prior_card.get("source_text", ""), "Minimal example")
        text, card = _render_operator_skill_text(
            summary=summary,
            skill_id=skill_id,
            cache_dir=cache_dir,
            fail_dir=fail_dir,
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            used_example_signatures=used_example_signatures,
            example_override=example_override,
        )
        (folder / "SKILL.md").write_text(text, encoding="utf-8")
        cards.append(card)

    return cards


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "skill"


def write_meta_skills(meta_dir: str) -> None:
    os.makedirs(meta_dir, exist_ok=True)
    templates = [
        (
            "S1",
            "Diagnose Slow Code",
            "Produce OptimizationBrief JSON with constraints, bottlenecks, and complexity.",
            "## Protocol\n- Given (statement, slow_code), produce OptimizationBrief JSON fields with concrete bottlenecks and risks.",
        ),
        (
            "S2",
            "Retrieve Operator Skills",
            "Select a dynamic set of operator skills from metadata and confidence.",
            "## Protocol\n- Given OptimizationBrief and skill registry metadata, choose skill_ids with reasons and dynamic budget.",
        ),
        (
            "S3",
            "Strategize",
            "Compose selected skills into 2-4 diverse executable strategies.",
            "## Protocol\n- Given OptimizationBrief and selected skills, output diverse strategy plans and risk notes.",
        ),
        (
            "S4",
            "Judge",
            "Rank candidates with static reasoning on complexity, correctness risk, and bottleneck removal.",
            "## Protocol\n- Given candidate set, rank and choose best candidate with concise rationale and confidence.",
        ),
    ]

    for sid, name, desc, body in templates:
        folder = os.path.join(meta_dir, f"{sid}_{_slugify(name)}")
        os.makedirs(folder, exist_ok=True)
        front = {
            "skill_id": sid,
            "type": "meta",
            "language": SKILL_LANGUAGE,
            "name": _normalize_name(name, fallback=sid),
            "description": _scalar_text(desc),
        }
        front_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=120).strip()
        text = f"---\n{front_text}\n---\n\n{body.strip()}\n"
        with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(text)


def _load_cluster_profiles_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        cluster_id = _clean_text(row.get("cluster_id", ""))
        if cluster_id:
            profiles[cluster_id] = row
    return profiles


def _load_operator_catalog_from_skills_dir(operators_dir: str) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for path in sorted(Path(operators_dir).glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            front = yaml.safe_load(parts[1]) or {}
        except Exception:
            continue
        if str(front.get("type", "")) != "operator":
            continue
        cards.append(
            {
                "skill_id": front.get("skill_id"),
                "family": front.get("family"),
                "name": front.get("name"),
                "description": front.get("description"),
                "tags": front.get("tags", []),
                "triggers": front.get("triggers", []),
            }
        )
    return cards


def _weighted_counter_top(
    traces: List[Dict[str, Any]],
    extractor,
    limit: int,
) -> List[str]:
    counter: Dict[str, float] = defaultdict(float)
    for trace in traces:
        weight = float(trace.get("trace_weight", 1.0))
        for item in extractor(trace):
            key = _clean_text(item)
            if key:
                counter[key] += weight
    return _top_weighted(counter, limit)


def _compact_bottleneck_kinds(audit: Optional[Dict[str, Any]], limit: int = 4) -> List[str]:
    out: List[str] = []
    for bottleneck in (audit or {}).get("bottlenecks", []):
        if isinstance(bottleneck, dict):
            kind = _clean_text(bottleneck.get("kind", ""))
            if kind:
                out.append(kind)
        else:
            kind = _clean_text(bottleneck)
            if kind:
                out.append(kind)
        if len(out) >= limit:
            break
    return out


def _compact_trace_example(
    trace: Dict[str, Any],
    *,
    cluster_id: Optional[str] = None,
    cluster_support: Optional[float] = None,
) -> Dict[str, Any]:
    problem = trace.get("ProblemBrief") or {}
    slow = trace.get("SlowAudit") or {}
    fast = trace.get("FastAudit") or {}
    delta = trace.get("DeltaSummary") or {}
    constraints = problem.get("constraints_guess") or {}
    return {
        "trace_id": trace.get("trace_id"),
        "problem_id": trace.get("problem_id"),
        "cluster_id": cluster_id,
        "cluster_support": round(float(cluster_support or 0.0), 6),
        "trace_weight": round(float(trace.get("trace_weight", 1.0)), 6),
        "problem_type_tags": _listify(problem.get("problem_type_tags"))[:5],
        "constraints_hint": {
            "n_max": constraints.get("n_max"),
            "q_max": constraints.get("q_max"),
        },
        "slow_dominant_ops": _listify(slow.get("dominant_ops"))[:3],
        "slow_bottlenecks": _compact_bottleneck_kinds(slow, limit=4),
        "slow_complexity": {
            "time": _scalar_text((slow.get("complexity") or {}).get("time", "")),
            "space": _scalar_text((slow.get("complexity") or {}).get("space", "")),
        },
        "fast_core_idea": _truncate_text(_scalar_text(fast.get("core_idea", "")), 180),
        "delta_type": _listify(delta.get("delta_type"))[:4],
        "complexity_delta": {
            "from": _scalar_text((delta.get("complexity_delta") or {}).get("from", "")),
            "to": _scalar_text((delta.get("complexity_delta") or {}).get("to", "")),
        },
        "trigger_signals": _listify(delta.get("trigger_signals"))[:3],
        "transformation_steps": _listify(delta.get("transformation_steps"))[:3],
    }


def _cluster_profile_digest(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cluster_id": profile.get("cluster_id"),
        "size": int(profile.get("size", 0) or 0),
        "support_weight": round(float(profile.get("support_weight", 0.0) or 0.0), 6),
        "top_problem_tags": _listify(profile.get("top_problem_tags"))[:5],
        "top_delta_type": _listify(profile.get("top_delta_type"))[:5],
        "top_bottlenecks": _listify(profile.get("top_bottlenecks"))[:5],
        "top_complexity_delta": _listify(profile.get("top_complexity_delta"))[:4],
        "top_triggers": _listify(profile.get("top_triggers"))[:4],
        "top_steps": _listify(profile.get("top_steps"))[:4],
    }


def _select_diverse_meta_examples(
    traces: List[Dict[str, Any]],
    cluster_profiles: Dict[str, Dict[str, Any]],
    sample_traces: int,
    max_per_cluster: int = 2,
) -> List[Dict[str, Any]]:
    trace_by_id = {str(t.get("trace_id", "")): t for t in traces if t.get("trace_id") is not None}
    candidates: List[Dict[str, Any]] = []
    for cluster_id, profile in sorted(
        cluster_profiles.items(),
        key=lambda kv: float((kv[1] or {}).get("support_weight", 0.0)),
        reverse=True,
    ):
        cluster_support = float(profile.get("support_weight", 0.0) or 0.0)
        for sample in profile.get("sample_traces", []) or []:
            trace_id = str(sample.get("trace_id", ""))
            trace = trace_by_id.get(trace_id)
            if trace is None:
                continue
            candidates.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_support": cluster_support,
                    "trace": trace,
                    "example": _compact_trace_example(
                        trace,
                        cluster_id=cluster_id,
                        cluster_support=cluster_support,
                    ),
                }
            )

    if not candidates:
        traces_sorted = sorted(traces, key=lambda t: t.get("trace_weight", 1.0), reverse=True)
        return [_compact_trace_example(t) for t in traces_sorted[:sample_traces]]

    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    cluster_counts: Counter = Counter()
    seen_tags: Set[str] = set()
    seen_delta: Set[str] = set()
    seen_bottlenecks: Set[str] = set()

    while len(selected) < sample_traces:
        best_idx = -1
        best_score = -1e18
        for idx, cand in enumerate(candidates):
            example = cand["example"]
            trace_id = str(example.get("trace_id", ""))
            cluster_id = str(cand["cluster_id"])
            if not trace_id or trace_id in selected_ids:
                continue
            if cluster_counts[cluster_id] >= max_per_cluster:
                continue

            tag_gain = len(set(example.get("problem_type_tags", [])) - seen_tags)
            delta_gain = len(set(example.get("delta_type", [])) - seen_delta)
            bottleneck_gain = len(set(example.get("slow_bottlenecks", [])) - seen_bottlenecks)
            cluster_bonus = 3.0 if cluster_counts[cluster_id] == 0 else 0.0
            score = (
                math.log1p(max(0.0, float(cand["cluster_support"])))
                + 0.15 * float(example.get("trace_weight", 1.0) or 0.0)
                + cluster_bonus
                + 1.2 * delta_gain
                + 0.9 * tag_gain
                + 0.6 * bottleneck_gain
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        chosen = candidates[best_idx]["example"]
        selected.append(chosen)
        selected_ids.add(str(chosen.get("trace_id", "")))
        cluster_id = str(chosen.get("cluster_id", ""))
        cluster_counts[cluster_id] += 1
        seen_tags.update(chosen.get("problem_type_tags", []))
        seen_delta.update(chosen.get("delta_type", []))
        seen_bottlenecks.update(chosen.get("slow_bottlenecks", []))

    if len(selected) < sample_traces:
        traces_sorted = sorted(traces, key=lambda t: t.get("trace_weight", 1.0), reverse=True)
        for trace in traces_sorted:
            trace_id = str(trace.get("trace_id", ""))
            if not trace_id or trace_id in selected_ids:
                continue
            selected.append(_compact_trace_example(trace))
            selected_ids.add(trace_id)
            if len(selected) >= sample_traces:
                break

    return selected[:sample_traces]


def _summarize_operator_catalog(operator_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    family_counter: Dict[str, int] = defaultdict(int)
    tag_counter: Counter = Counter()
    trigger_counter: Counter = Counter()
    sample_cards: List[Dict[str, Any]] = []
    for card in operator_catalog:
        family = _clean_text(card.get("family", "")) or "unknown"
        family_counter[family] += 1
        for tag in _listify(card.get("tags")):
            tag_counter[tag] += 1
        for trigger in _listify(card.get("triggers")):
            trigger_counter[trigger] += 1
        if len(sample_cards) < 12:
            sample_cards.append(
                {
                    "skill_id": _clean_text(card.get("skill_id", "")),
                    "family": family,
                    "name": _clean_text(card.get("name", "")),
                    "description": _truncate_text(_scalar_text(card.get("description", "")), 160),
                    "tags": _listify(card.get("tags"))[:4],
                    "triggers": _listify(card.get("triggers"))[:4],
                }
            )
    family_counts = [
        {"family": family, "count": count}
        for family, count in sorted(family_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ]
    return {
        "family_counts": family_counts,
        "top_tags": [tag for tag, _ in tag_counter.most_common(12)],
        "top_triggers": [trigger for trigger, _ in trigger_counter.most_common(12)],
        "sample_cards": sample_cards,
    }


def _build_meta_evidence(
    traces: List[Dict[str, Any]],
    cluster_profiles: Dict[str, Dict[str, Any]],
    operator_catalog: List[Dict[str, Any]],
    sample_traces: int,
) -> Dict[str, Any]:
    weight_values = [float(t.get("trace_weight", 1.0) or 1.0) for t in traces]
    clusters_sorted = sorted(
        cluster_profiles.values(),
        key=lambda p: float((p or {}).get("support_weight", 0.0)),
        reverse=True,
    )
    return {
        "global_trace_stats": {
            "trace_count": len(traces),
            "cluster_count": len(cluster_profiles),
            "trace_weight": {
                "median": round(float(np.median(weight_values)) if weight_values else 0.0, 6),
                "max": round(float(np.max(weight_values)) if weight_values else 0.0, 6),
            },
            "top_problem_tags": _weighted_counter_top(
                traces,
                lambda t: (t.get("ProblemBrief") or {}).get("problem_type_tags", []),
                12,
            ),
            "top_delta_type": _weighted_counter_top(
                traces,
                lambda t: (t.get("DeltaSummary") or {}).get("delta_type", []),
                12,
            ),
            "top_bottlenecks": _weighted_counter_top(
                traces,
                lambda t: _compact_bottleneck_kinds(t.get("SlowAudit") or {}, limit=12),
                12,
            ),
            "top_complexity_delta": _weighted_counter_top(
                traces,
                lambda t: [
                    f"{_scalar_text(((t.get('DeltaSummary') or {}).get('complexity_delta') or {}).get('from', ''))}"
                    f"->{_scalar_text(((t.get('DeltaSummary') or {}).get('complexity_delta') or {}).get('to', ''))}"
                ],
                10,
            ),
        },
        "cluster_patterns": [_cluster_profile_digest(p) for p in clusters_sorted[:8]],
        "diverse_examples": _select_diverse_meta_examples(
            traces=traces,
            cluster_profiles=cluster_profiles,
            sample_traces=sample_traces,
        ),
        "operator_summary": _summarize_operator_catalog(operator_catalog),
    }


def _validate_meta_body(
    sid: str,
    body: str,
    valid_operator_ids: Set[str],
) -> List[str]:
    issues: List[str] = []
    body_clean = body.strip()
    low = body_clean.lower()
    if not body_clean:
        issues.append("empty body")
        return issues
    required_headers = {
        "inputs": "## Inputs" in body_clean,
        "procedure": "## Procedure" in body_clean,
        "decision_rules": "## Decision Rules" in body_clean,
        "failure_modes": "## Failure Modes" in body_clean,
        "output": "## Output" in body_clean,
    }
    missing_headers = [name for name, ok in required_headers.items() if not ok]
    if missing_headers:
        issues.append(f"missing required sections: {', '.join(missing_headers)}")
    if re.search(r"\bC\d{3}\b", body_clean):
        issues.append("references cluster ids")
    if re.search(r"\b(Mercury|LeetCode|Codeforces|AtCoder|CodeChef|Aizu)\b", body_clean, re.I):
        issues.append("references benchmark names")
    if len(re.findall(r"\b\d{3,}\b", body_clean)) >= 2:
        issues.append("contains overly specific large numeric constants")
    generic_terms = {
        "S1": ("inspect", "classify", "diagnose", "bottleneck", "complexity"),
        "S2": ("retrieve", "select", "rank", "confidence", "budget"),
        "S3": ("compose", "compare", "plan", "diverse", "strategy"),
        "S4": ("rank", "compare", "risk", "validate", "judge"),
    }.get(sid, ("inspect", "compare", "rank"))
    present = sum(1 for term in generic_terms if term in low)
    if present < 3:
        issues.append("lacks enough generic procedural language")
    if sid == "S2":
        used_oids = set(re.findall(r"\bO\d{3}\b", body_clean))
        invalid = sorted(used_oids - valid_operator_ids)
        if invalid:
            issues.append(f"references invalid operator ids: {', '.join(invalid[:4])}")
    return issues


def _fallback_meta_body_from_plan(sid: str, plan: Dict[str, Any]) -> str:
    section_alias = {
        "inputs": "Inputs",
        "procedure": "Procedure",
        "decision_rules": "Decision Rules",
        "failure_modes": "Failure Modes",
        "output_contract": "Output",
    }
    lines: List[str] = []
    title = _clean_text(plan.get("name", sid)) or sid
    lines.append(f"## {sid} - {title}")
    for key in ("inputs", "procedure", "decision_rules", "failure_modes", "output_contract"):
        items = _listify(plan.get(key))
        if not items and key == "decision_rules":
            items = ["Apply explicit tradeoff rules instead of ad-hoc intuition."]
        if not items and key == "failure_modes":
            items = ["Flag low-confidence cases and request a narrower plan."]
        if not items and key == "output_contract":
            items = ["Return a concise, actionable artifact tailored to the stage role."]
        lines.append("")
        lines.append(f"## {section_alias[key]}")
        if items:
            lines.extend([f"- {_clean_text(item)}" for item in items if _clean_text(item)])
        else:
            lines.append("- Use the stage protocol conservatively and keep the result general.")
    return "\n".join(lines).strip()


def learn_meta_skills(
    traces: List[Dict[str, Any]],
    cluster_profiles: Dict[str, Dict[str, Any]],
    operator_catalog: List[Dict[str, Any]],
    meta_dir: str,
    artifacts_dir: Optional[str],
    cache_dir: str,
    fail_dir: str,
    model: str,
    fallback_model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    sample_traces: int = 10,
) -> bool:
    if not traces or not cluster_profiles:
        return False

    os.makedirs(meta_dir, exist_ok=True)
    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)

    evidence = _build_meta_evidence(
        traces=traces,
        cluster_profiles=cluster_profiles,
        operator_catalog=operator_catalog,
        sample_traces=max(12, sample_traces),
    )
    if artifacts_dir:
        with open(
            os.path.join(artifacts_dir, "meta_evidence.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)

    valid_operator_ids = {
        str(x.get("skill_id", ""))
        for x in operator_catalog
        if str(x.get("skill_id", "")).startswith("O")
    }

    drafts: List[Dict[str, str]] = []
    specs = {
        "S1": {
            "label": "Diagnose",
            "goal": "Produce a reusable diagnosis meta-skill that turns structured traces into an optimization brief.",
            "evidence": {
                "global_trace_stats": evidence["global_trace_stats"],
                "cluster_patterns": evidence["cluster_patterns"],
                "diverse_examples": [
                    {
                        "cluster_id": ex.get("cluster_id"),
                        "problem_type_tags": ex.get("problem_type_tags"),
                        "slow_bottlenecks": ex.get("slow_bottlenecks"),
                        "slow_complexity": ex.get("slow_complexity"),
                        "delta_type": ex.get("delta_type"),
                        "complexity_delta": ex.get("complexity_delta"),
                    }
                    for ex in evidence["diverse_examples"]
                ],
            },
        },
        "S2": {
            "label": "Retrieve",
            "goal": "Produce a reusable retrieval meta-skill that selects a dynamic budget of operator skills from the registry.",
            "evidence": {
                "global_trace_stats": evidence["global_trace_stats"],
                "operator_summary": evidence["operator_summary"],
            },
        },
        "S3": {
            "label": "Strategize",
            "goal": "Produce a reusable planning meta-skill that composes selected skills into diverse executable strategies.",
            "evidence": {
                "global_trace_stats": evidence["global_trace_stats"],
                "cluster_patterns": evidence["cluster_patterns"],
                "diverse_examples": [
                    {
                        "cluster_id": ex.get("cluster_id"),
                        "delta_type": ex.get("delta_type"),
                        "complexity_delta": ex.get("complexity_delta"),
                        "trigger_signals": ex.get("trigger_signals"),
                        "transformation_steps": ex.get("transformation_steps"),
                    }
                    for ex in evidence["diverse_examples"]
                ],
            },
        },
        "S4": {
            "label": "Judge",
            "goal": "Produce a reusable ranking meta-skill that compares candidate solutions by performance upside, correctness risk, and evidence strength.",
            "evidence": {
                "global_trace_stats": evidence["global_trace_stats"],
                "cluster_patterns": evidence["cluster_patterns"],
                "diverse_examples": [
                    {
                        "cluster_id": ex.get("cluster_id"),
                        "slow_bottlenecks": ex.get("slow_bottlenecks"),
                        "fast_core_idea": ex.get("fast_core_idea"),
                        "delta_type": ex.get("delta_type"),
                        "complexity_delta": ex.get("complexity_delta"),
                    }
                    for ex in evidence["diverse_examples"]
                ],
            },
        },
    }

    for sid, spec in specs.items():
        plan_prompt = build_meta_plan_prompt(
            sid=sid,
            label=spec["label"],
            goal=spec["goal"],
            evidence=spec["evidence"],
        )
        messages = [
            {"role": "system", "content": meta_plan_system_prompt()},
            {"role": "user", "content": plan_prompt},
        ]
        plan = _llm_json_call(
            messages=messages,
            cache_dir=cache_dir,
            stage=f"a5_meta_plan_{sid.lower()}",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            out_fail_dir=fail_dir,
            reasoning_effort=reasoning_effort,
            fallback_model=fallback_model,
        )
        if plan is None:
            return False

        name = _clean_text(plan.get("name", sid)) or sid
        desc = _scalar_text(plan.get("description", ""))
        body = _fallback_meta_body_from_plan(sid, plan)
        issues = _validate_meta_body(sid, body, valid_operator_ids)
        if issues:
            return False

        drafts.append(
            {
                "sid": sid,
                "name": _normalize_name(name, fallback=sid),
                "description": desc,
                "body": body.strip(),
            }
        )

    # Atomic write: only emit meta skills after all S1-S4 drafts were produced.
    for item in drafts:
        sid = item["sid"]
        folder = os.path.join(meta_dir, f"{sid}_{_slugify(item['name'])}")
        os.makedirs(folder, exist_ok=True)
        front = {
            "skill_id": sid,
            "type": "meta",
            "language": SKILL_LANGUAGE,
            "name": item["name"],
            "description": item["description"],
        }
        front_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=120).strip()
        text = f"---\n{front_text}\n---\n\n{item['body']}\n"
        with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(text)

    return True


def run_quality_checks(gen_dir: Path, registry: List[Dict[str, Any]]) -> Dict[str, Any]:
    skills_dir = gen_dir / "skills"
    operators = [x for x in registry if x.get("type") == "operator"]
    metas = [x for x in registry if x.get("type") == "meta"]

    placeholder_examples = 0
    missing_operator_files = 0
    ellipsis_artifacts = 0
    meta_cluster_refs = 0
    clipped_text_artifacts = 0
    heading_step_artifacts = 0
    flattened_bullet_artifacts = 0
    example_signature_counts: Counter = Counter()

    for op in operators:
        sid = op.get("skill_id")
        if not sid:
            continue
        matches = list((skills_dir / "operators").glob(f"{sid}_*/SKILL.md"))
        if not matches:
            missing_operator_files += 1
            continue
        text = matches[0].read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                try:
                    front = yaml.safe_load(parts[1]) or {}
                    if _looks_truncated_artifact(
                        _scalar_text(front.get("description"), default="")
                    ):
                        clipped_text_artifacts += 1
                except Exception:
                    clipped_text_artifacts += 1
        body_text = re.sub(r"(?s)^---\n.*?\n---\n", "", text, count=1)
        if re.search(r"slow_op|fast_op|preprocess\(data\)|# baseline approach|// baseline approach", body_text):
            placeholder_examples += 1
        ellipsis_artifacts += len(re.findall(r"(?:\.\.\.|…)", body_text))
        in_code = False
        for line in body_text.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            ls = line.strip()
            if re.match(r"^\d+\.\s+.+:\s*$", ls):
                heading_step_artifacts += 1
            if re.match(r"^- .+:[ \t]*-[ \t]+.+", ls):
                flattened_bullet_artifacts += 1
            if re.match(r"^(?:- |\d+\.\s+)", ls):
                line_payload = re.sub(r"^(?:- |\d+\.\s+)", "", ls)
                if _looks_truncated_artifact(line_payload):
                    clipped_text_artifacts += 1
            if re.match(
                r"^- .*(?:leading to|resulting in|causing|such as|for example|instead of|where|which)\s*$",
                ls,
                flags=re.I,
            ):
                clipped_text_artifacts += 1
        blocks = re.findall(r"```(?:py|python|cpp|c\+\+)\s*\n(.*?)\n```", body_text, flags=re.S)
        if len(blocks) >= 2:
            sig = _example_signature(blocks[0], blocks[1])
            example_signature_counts[sig] += 1

    for meta in metas:
        sid = meta.get("skill_id")
        if not sid:
            continue
        matches = list((skills_dir / "meta").glob(f"{sid}_*/SKILL.md"))
        if not matches:
            continue
        text = matches[0].read_text(encoding="utf-8")
        body_text = re.sub(r"(?s)^---\n.*?\n---\n", "", text, count=1)
        if re.search(r"\bC\d{3}\b", body_text):
            meta_cluster_refs += 1

    name_counts = Counter(
        _normalize_name(x.get("name", "")).lower() for x in operators if x.get("name")
    )
    duplicate_name_groups = sum(1 for _, c in name_counts.items() if c > 1)
    duplicate_example_groups = sum(1 for _, c in example_signature_counts.items() if c > 1)
    meta_id_counts = Counter(str(x.get("skill_id", "")) for x in metas if x.get("skill_id"))
    duplicate_meta_skill_id_groups = sum(1 for _, c in meta_id_counts.items() if c > 1)
    required_meta_ids = ["S1", "S2", "S3", "S4"]
    missing_required_meta_ids = [
        sid for sid in required_meta_ids if meta_id_counts.get(sid, 0) == 0
    ]
    non_singleton_required_meta_ids = [
        sid for sid in required_meta_ids if meta_id_counts.get(sid, 0) != 1
    ]

    checks = {
        "meta_count": len(metas),
        "operator_count": len(operators),
        "placeholder_examples": placeholder_examples,
        "missing_operator_files": missing_operator_files,
        "duplicate_operator_name_groups": duplicate_name_groups,
        "duplicate_example_groups": duplicate_example_groups,
        "ellipsis_artifacts": ellipsis_artifacts,
        "clipped_text_artifacts": clipped_text_artifacts,
        "heading_step_artifacts": heading_step_artifacts,
        "flattened_bullet_artifacts": flattened_bullet_artifacts,
        "meta_cluster_refs": meta_cluster_refs,
        "duplicate_meta_skill_id_groups": duplicate_meta_skill_id_groups,
        "missing_required_meta_ids": missing_required_meta_ids,
        "non_singleton_required_meta_ids": non_singleton_required_meta_ids,
    }
    checks["quality_ok"] = (
        checks["meta_count"] >= 4
        and checks["placeholder_examples"] == 0
        and checks["missing_operator_files"] == 0
        and checks["duplicate_operator_name_groups"] == 0
        and checks["duplicate_example_groups"] == 0
        and checks["ellipsis_artifacts"] == 0
        and checks["clipped_text_artifacts"] == 0
        and checks["heading_step_artifacts"] == 0
        and checks["flattened_bullet_artifacts"] == 0
        and checks["meta_cluster_refs"] == 0
        and checks["duplicate_meta_skill_id_groups"] == 0
        and len(checks["missing_required_meta_ids"]) == 0
        and len(checks["non_singleton_required_meta_ids"]) == 0
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 / Stage 3 skill extraction")
    parser.add_argument("--gen-dir", default="stage1/gen_initial", help="Generation output root")
    parser.add_argument(
        "--traces",
        default="stage1/data/traces.jsonl",
        help="Existing traces.jsonl path",
    )
    parser.add_argument(
        "--feedback-paths",
        nargs="*",
        default=[
            "stage3/feedback/merged-gpt5mini-qwen30B.jsonl",
            "stage3/feedback/merged-qwen30B-qwen30B.jsonl",
        ],
        help="Preprocessed Stage 3 feedback JSONL files.",
    )
    parser.add_argument(
        "--prior-skills-root",
        default="stage1/gen_initial/skills",
        help="Prior Stage 1 skill root used to load lineage artifacts for Stage 3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional trace limit for quick test runs",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed cluster count; default uses silhouette-based auto-k",
    )
    parser.add_argument("--sample-per-cluster", type=int, default=30)
    parser.add_argument("--tfidf-max-features", type=int, default=4096)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--disable-sentence-embeddings", action="store_true")
    parser.add_argument("--merge-similarity", type=float, default=0.84)
    parser.add_argument(
        "--min-skills",
        default="auto",
        help="Minimum operator skills. Use integer or 'auto' for dynamic bound from cluster count.",
    )
    parser.add_argument(
        "--min-skill-coverage",
        type=float,
        default=0.80,
        help="Coverage threshold for kept skills (by support weight).",
    )
    parser.add_argument(
        "--strict-skill-gate",
        action="store_true",
        help="Fail run if both operator count and coverage are below thresholds.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Fail run on quality issues (missing meta skills, placeholders, duplicate operator names).",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["stage1", "stage3"],
        default="stage3",
        help="stage1: uniform clustering from traces only; stage3: lineage-aware feedback weighting.",
    )
    parser.add_argument(
        "--stage3-failure-penalty",
        type=float,
        default=STAGE3_FAILURE_PENALTY_DEFAULT,
        help="Fixed negative score assigned to failed Stage 3 candidates.",
    )
    parser.add_argument(
        "--stage3-noise-band",
        type=float,
        default=STAGE3_NOISE_BAND_DEFAULT,
        help="Relative dead-zone around no runtime change; e.g. 0.10 treats <=10%% as neutral.",
    )
    parser.add_argument(
        "--stage3-runtime-clip",
        type=float,
        default=STAGE3_RUNTIME_CLIP_DEFAULT,
        help="Symmetric runtime ratio clip for signed log-speedup scoring.",
    )
    parser.add_argument(
        "--stage3-ridge-alpha",
        type=float,
        default=STAGE3_RIDGE_ALPHA_DEFAULT,
        help="Ridge regularization strength for Stage 3 skill attribution.",
    )
    parser.add_argument(
        "--stage3-shrinkage-lambda",
        type=float,
        default=STAGE3_SHRINKAGE_LAMBDA_DEFAULT,
        help="Support shrinkage strength applied to Stage 3 skill coefficients.",
    )
    parser.add_argument(
        "--stage3-min-utility",
        type=float,
        default=STAGE3_DECISION_MIN_UTILITY_DEFAULT,
        help="Minimum signed utility required for a Stage 3 cluster summary to be considered.",
    )
    parser.add_argument(
        "--stage3-revise-similarity",
        type=float,
        default=STAGE3_REVISE_SIMILARITY_DEFAULT,
        help="Similarity threshold above which a Stage 3 summary revises an existing Stage 1 skill.",
    )
    parser.add_argument(
        "--stage3-add-max-similarity",
        type=float,
        default=STAGE3_ADD_MAX_SIMILARITY_DEFAULT,
        help="Maximum similarity to Stage 1 library for accepting a Stage 3 summary as a new skill.",
    )
    parser.add_argument(
        "--stage3-add-min-support",
        type=int,
        default=STAGE3_ADD_MIN_SUPPORT_DEFAULT,
        help="Minimum support_count required for adding a novel Stage 3 skill.",
    )
    parser.add_argument(
        "--stage3-library-dedup-similarity",
        type=float,
        default=STAGE3_LIBRARY_DEDUP_SIMILARITY_DEFAULT,
        help="Similarity threshold for rejecting Stage 3 additions as duplicates of the final library.",
    )

    parser.add_argument("--model", default=os.environ.get("EFFISKILL_MODEL"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--fallback-model", default=os.environ.get("EFFISKILL_MODEL"))
    parser.add_argument("--summary-temperature", type=float, default=0.2)
    parser.add_argument("--summary-max-tokens", type=int, default=32000)
    parser.add_argument("--disable-llm-examples", action="store_true")
    parser.add_argument("--example-temperature", type=float, default=0.2)
    parser.add_argument("--example-max-tokens", type=int, default=32000)
    parser.add_argument(
        "--retry-fallback-examples",
        action="store_true",
        help="After A4, retry only fallback-marked operator examples with larger token budget.",
    )
    parser.add_argument("--retry-example-temperature", type=float, default=0.2)
    parser.add_argument("--retry-example-max-tokens", type=int, default=32000)
    parser.add_argument(
        "--retry-fallback-only",
        action="store_true",
        help="Run only fallback-example retry on existing gen-dir outputs (no A1-A5 rerun).",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Regenerate only meta skills from existing gen-dir artifacts without touching operator skills.",
    )
    parser.add_argument("--meta-temperature", type=float, default=0.35)
    parser.add_argument("--meta-max-tokens", type=int, default=32000)
    parser.add_argument("--reasoning-effort", default="low")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set EFFISKILL_MODEL.")
    if not args.fallback_model:
        args.fallback_model = args.model
    configure_default_client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    if args.fallback_model and not str(args.fallback_model).startswith("gpt-5"):
        raise ValueError("For consistency, fallback model must be gpt-5.* or equal to --model.")
    random.seed(args.seed)
    np.random.seed(args.seed)

    gen_dir = Path(args.gen_dir)
    data_dir = gen_dir / "data"
    skills_dir = gen_dir / "skills"
    operators_dir = skills_dir / "operators"
    meta_dir = skills_dir / "meta"
    registry_path = skills_dir / "registry.json"
    cache_dir = gen_dir / "out" / "llm_cache"
    fail_dir = gen_dir / "out" / "llm_failures"
    merged_summaries_path = str(data_dir / "cluster_summaries.jsonl")

    if args.retry_fallback_only:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)
        if not os.path.exists(merged_summaries_path):
            raise SystemExit(f"Missing summaries file for retry mode: {merged_summaries_path}")
        if not operators_dir.exists():
            raise SystemExit(f"Missing operators directory for retry mode: {operators_dir}")

        summaries = _read_jsonl(merged_summaries_path)
        retry_stats = retry_fallback_examples(
            summaries=summaries,
            operators_dir=str(operators_dir),
            cache_dir=str(cache_dir),
            fail_dir=str(fail_dir),
            model=args.model,
            fallback_model=args.fallback_model,
            temperature=args.retry_example_temperature,
            max_tokens=args.retry_example_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        print("A4 retry fallback examples:", retry_stats)
        registry = build_registry(str(skills_dir), str(registry_path))
        print("A6 registry size:", len(registry))
        quality = run_quality_checks(gen_dir, registry)
        print("Quality checks:", quality)
        if args.strict_quality_gate and not quality.get("quality_ok"):
            raise RuntimeError(f"Quality gate failed: {quality}")
        return

    if args.meta_only:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)
        normalized_path = str(data_dir / "trace_normalized.jsonl")
        cluster_profiles_path = str(data_dir / "cluster_profiles.jsonl")
        if not os.path.exists(normalized_path):
            raise SystemExit(f"Missing normalized traces for meta-only mode: {normalized_path}")
        if not os.path.exists(cluster_profiles_path):
            raise SystemExit(
                f"Missing cluster profiles for meta-only mode: {cluster_profiles_path}"
            )
        if not operators_dir.exists():
            raise SystemExit(f"Missing operators directory for meta-only mode: {operators_dir}")

        traces = _read_jsonl(normalized_path)
        cluster_profiles = _load_cluster_profiles_jsonl(cluster_profiles_path)
        operator_cards = _load_operator_catalog_from_skills_dir(str(operators_dir))
        if not traces or not cluster_profiles or not operator_cards:
            raise SystemExit(
                "Meta-only mode requires existing normalized traces, cluster profiles, and operator cards."
            )

        if meta_dir.exists():
            shutil.rmtree(meta_dir)
        os.makedirs(meta_dir, exist_ok=True)
        learned = learn_meta_skills(
            traces=traces,
            cluster_profiles=cluster_profiles,
            operator_catalog=operator_cards,
            meta_dir=str(meta_dir),
            artifacts_dir=str(data_dir),
            cache_dir=str(cache_dir),
            fail_dir=str(fail_dir),
            model=args.model,
            fallback_model=args.fallback_model,
            max_tokens=args.meta_max_tokens,
            temperature=args.meta_temperature,
            reasoning_effort=args.reasoning_effort,
        )
        if not learned:
            if meta_dir.exists():
                shutil.rmtree(meta_dir)
            os.makedirs(meta_dir, exist_ok=True)
            write_meta_skills(str(meta_dir))
            print("A5 meta skills: fallback templates")
        else:
            print("A5 meta skills: learned")

        registry = build_registry(str(skills_dir), str(registry_path))
        print("A6 registry size:", len(registry))
        quality = run_quality_checks(gen_dir, registry)
        print("Quality checks:", quality)
        if args.strict_quality_gate and not quality.get("quality_ok"):
            raise RuntimeError(f"Quality gate failed: {quality}")
        return

    # Ensure each run writes a clean skill set (avoid stale SKILL.md folders from prior reruns).
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(operators_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    stage_mode = args.cluster_mode

    # A1: normalize traces
    normalized_path = str(data_dir / "trace_normalized.jsonl")
    traces, a1_stats = prepare_traces(
        traces_path=args.traces,
        out_path=normalized_path,
        limit=args.limit,
        stage_mode=stage_mode,
    )
    print("A1 normalized traces:", a1_stats)

    if not traces:
        raise SystemExit("No traces loaded after A1")

    # A2: build signatures/profile table
    profiles_path = str(data_dir / "trace_profiles.jsonl")
    signatures = build_trace_profiles(traces, profiles_path)
    print("A2 trace profiles:", len(signatures))

    # A3: clustering + summaries
    clusters_path = str(data_dir / "clusters.jsonl")
    cluster_profiles_path = str(data_dir / "cluster_profiles.jsonl")
    raw_summaries_path = str(data_dir / "cluster_summaries_raw.jsonl")
    if stage_mode == "stage1":
        cluster_result = cluster_traces(
            traces=traces,
            signatures=signatures,
            clusters_path=clusters_path,
            tfidf_max_features=args.tfidf_max_features,
            embedding_model=args.embedding_model,
            disable_sentence_embeddings=args.disable_sentence_embeddings,
            k=args.k,
        )
        cluster_profiles = build_cluster_profiles(
            traces=traces,
            labels=cluster_result.labels,
            out_path=cluster_profiles_path,
            sample_per_cluster=args.sample_per_cluster,
        )
    else:
        feedback_paths = list(args.feedback_paths or [])
        if not feedback_paths:
            raise ValueError("Stage 3 requires at least one preprocessed feedback file")
        prior_ctx = load_prior_skill_context(args.prior_skills_root)
        skill_gains, gain_stats, gain_artifacts = load_stage3_skill_gains(
            feedback_paths,
            prior_skill_ids=prior_ctx.skill_texts.keys(),
            artifacts_dir=str(data_dir),
            failure_penalty=args.stage3_failure_penalty,
            noise_band_ratio=args.stage3_noise_band,
            runtime_clip_ratio=args.stage3_runtime_clip,
            ridge_alpha=args.stage3_ridge_alpha,
            shrinkage_lambda=args.stage3_shrinkage_lambda,
        )
        print(
            "Stage3 skill gains:",
            {
                "record_count": gain_stats.record_count,
                "valid_problem_count": gain_stats.valid_problem_count,
                "candidate_count": gain_stats.candidate_count,
                "scored_candidate_count": gain_stats.scored_candidate_count,
                "passed_candidate_count": gain_stats.passed_candidate_count,
                "failed_candidate_count": gain_stats.failed_candidate_count,
                "skipped_missing_baseline_count": gain_stats.skipped_missing_baseline_count,
                "skill_gain_count": gain_stats.skill_gain_count,
                "nonzero_skill_gain_count": gain_stats.nonzero_skill_gain_count,
                "prior_data_root": prior_ctx.data_root,
                "candidate_metrics_path": gain_artifacts["candidate_metrics_path"],
                "skill_gain_details_path": gain_artifacts["skill_gain_details_path"],
                "summary_path": gain_artifacts["summary_path"],
            },
        )

        provisional_clusters_path = str(data_dir / "clusters_stage3_provisional.jsonl")
        provisional_cluster_profiles_path = str(
            data_dir / "cluster_profiles_stage3_provisional.jsonl"
        )
        cluster_result_initial = cluster_traces(
            traces=traces,
            signatures=signatures,
            clusters_path=provisional_clusters_path,
            tfidf_max_features=args.tfidf_max_features,
            embedding_model=args.embedding_model,
            disable_sentence_embeddings=args.disable_sentence_embeddings,
            k=args.k,
        )
        provisional_profiles = build_cluster_profiles(
            traces=traces,
            labels=cluster_result_initial.labels,
            out_path=provisional_cluster_profiles_path,
            sample_per_cluster=args.sample_per_cluster,
        )

        cluster_skill_corr, corr_rows = build_cluster_skill_correlation(
            current_profiles=provisional_profiles,
            prior_ctx=prior_ctx,
            tfidf_max_features=args.tfidf_max_features,
            embedding_model=args.embedding_model,
            disable_sentence_embeddings=args.disable_sentence_embeddings,
        )
        skill_gain_path = data_dir / "skill_gain.json"
        with open(skill_gain_path, "w", encoding="utf-8") as f:
            json.dump(skill_gains, f, ensure_ascii=False, indent=2)
        cluster_skill_corr_path = data_dir / "cluster_skill_corr.jsonl"
        with open(cluster_skill_corr_path, "w", encoding="utf-8") as f:
            for row in corr_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        cluster_utilities, cluster_utility_rows = build_stage3_cluster_utilities(
            cluster_skill_corr=cluster_skill_corr,
            skill_gains=skill_gains,
        )
        cluster_utility_path = data_dir / "stage3_cluster_utilities.jsonl"
        with open(cluster_utility_path, "w", encoding="utf-8") as f:
            for row in cluster_utility_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        attribution_path = str(data_dir / "trace_weight_attribution.jsonl")
        stage3_weight_stats = apply_stage3_trace_weights(
            traces=traces,
            cluster_result=cluster_result_initial,
            skill_gains=skill_gains,
            cluster_skill_corr=cluster_skill_corr,
            out_path=attribution_path,
        )
        print("Stage3 propagated trace weights:", stage3_weight_stats)

        sample_weights = np.array(
            [float(t.get("trace_weight", 0.0)) for t in traces], dtype=float
        )
        cluster_result = cluster_traces(
            traces=traces,
            signatures=signatures,
            clusters_path=clusters_path,
            tfidf_max_features=args.tfidf_max_features,
            embedding_model=args.embedding_model,
            disable_sentence_embeddings=args.disable_sentence_embeddings,
            k=args.k,
            sample_weights=sample_weights,
        )
        cluster_profiles = build_cluster_profiles(
            traces=traces,
            labels=cluster_result.labels,
            out_path=cluster_profiles_path,
            sample_per_cluster=args.sample_per_cluster,
        )

    raw_summaries: List[Dict[str, Any]] = []
    for cluster_id in tqdm(sorted(cluster_profiles.keys()), desc="A3 summarize clusters"):
        summary = summarize_cluster(
            profile=cluster_profiles[cluster_id],
            cache_dir=str(cache_dir),
            fail_dir=str(fail_dir),
            model=args.model,
            fallback_model=args.fallback_model,
            temperature=args.summary_temperature,
            max_tokens=args.summary_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if summary is not None:
            raw_summaries.append(summary)

    write_summaries(raw_summaries_path, raw_summaries)

    min_required, max_allowed, min_mode = resolve_skill_bounds(
        min_skills_arg=args.min_skills,
        k_clusters=cluster_result.k,
        n_traces=len(traces),
    )

    merged_summaries = dedup_merge_summaries(
        summaries=raw_summaries,
        merge_similarity=args.merge_similarity,
        min_skills=min_required,
        max_skills=max_allowed,
    )
    write_summaries(merged_summaries_path, merged_summaries)
    kept_coverage = support_coverage(merged_summaries, raw_summaries)

    print(
        "A3 clustering/summaries:",
        {
            "k": cluster_result.k,
            "auto_k_effective_n": (
                _auto_k_effective_n(len(traces)) if args.k is None else None
            ),
            "embedding_source": cluster_result.embedding_source,
            "min_skills_mode": min_mode,
            "min_required": min_required,
            "max_allowed": max_allowed,
            "raw_summaries": len(raw_summaries),
            "merged_summaries": len(merged_summaries),
            "support_coverage": round(kept_coverage, 4),
        },
    )

    # A4: write operator skills
    if stage_mode == "stage1":
        operator_cards = write_operator_skills(
            summaries=merged_summaries,
            operators_dir=str(operators_dir),
            cache_dir=None if args.disable_llm_examples else str(cache_dir),
            fail_dir=None if args.disable_llm_examples else str(fail_dir),
            model=None if args.disable_llm_examples else args.model,
            fallback_model=args.fallback_model,
            temperature=args.example_temperature,
            max_tokens=args.example_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        stage3_plan_stats = None
    else:
        final_operator_specs, decision_rows, stage3_plan_stats = build_stage3_library_plan(
            summaries=merged_summaries,
            prior_ctx=prior_ctx,
            cluster_utilities=cluster_utilities,
            tfidf_max_features=args.tfidf_max_features,
            embedding_model=args.embedding_model,
            disable_sentence_embeddings=args.disable_sentence_embeddings,
            min_utility=args.stage3_min_utility,
            revise_similarity=args.stage3_revise_similarity,
            add_similarity_max=args.stage3_add_max_similarity,
            add_min_support=args.stage3_add_min_support,
            library_dedup_similarity=args.stage3_library_dedup_similarity,
        )
        decisions_path = data_dir / "stage3_library_decisions.jsonl"
        with open(decisions_path, "w", encoding="utf-8") as f:
            for row in decision_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        plan_summary_path = data_dir / "stage3_library_plan_summary.json"
        with open(plan_summary_path, "w", encoding="utf-8") as f:
            json.dump(stage3_plan_stats, f, ensure_ascii=False, indent=2)
        operator_cards = write_stage3_operator_library(
            operator_specs=final_operator_specs,
            operators_dir=str(operators_dir),
            cache_dir=None if args.disable_llm_examples else str(cache_dir),
            fail_dir=None if args.disable_llm_examples else str(fail_dir),
            model=None if args.disable_llm_examples else args.model,
            fallback_model=args.fallback_model,
            temperature=args.example_temperature,
            max_tokens=args.example_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        print(
            "Stage3 selective library plan:",
            {
                **stage3_plan_stats,
                "decisions_path": str(decisions_path),
                "plan_summary_path": str(plan_summary_path),
            },
        )
    op_count = len(operator_cards)
    print("A4 operator skills:", op_count)

    if args.retry_fallback_examples and not args.disable_llm_examples:
        retry_stats = retry_fallback_examples(
            summaries=merged_summaries,
            operators_dir=str(operators_dir),
            cache_dir=str(cache_dir),
            fail_dir=str(fail_dir),
            model=args.model,
            fallback_model=args.fallback_model,
            temperature=args.retry_example_temperature,
            max_tokens=args.retry_example_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        print("A4 retry fallback examples:", retry_stats)

    count_ok = op_count >= min_required
    coverage_ok = kept_coverage >= float(args.min_skill_coverage)
    if not count_ok and not coverage_ok:
        msg = (
            f"Operator skill gate failed: count={op_count} (<{min_required}) and "
            f"coverage={kept_coverage:.4f} (<{args.min_skill_coverage}). "
            "Adjust clustering or merge threshold."
        )
        if args.strict_skill_gate:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    # A5: learn meta skills (fallback to templates)
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    os.makedirs(meta_dir, exist_ok=True)
    learned = learn_meta_skills(
        traces=traces,
        cluster_profiles=cluster_profiles,
        operator_catalog=operator_cards,
        meta_dir=str(meta_dir),
        artifacts_dir=str(data_dir),
        cache_dir=str(cache_dir),
        fail_dir=str(fail_dir),
        model=args.model,
        fallback_model=args.fallback_model,
        max_tokens=args.meta_max_tokens,
        temperature=args.meta_temperature,
        reasoning_effort=args.reasoning_effort,
    )
    if not learned:
        if meta_dir.exists():
            shutil.rmtree(meta_dir)
        os.makedirs(meta_dir, exist_ok=True)
        write_meta_skills(str(meta_dir))
        print("A5 meta skills: fallback templates")
    else:
        print("A5 meta skills: learned")

    # A6: build registry
    registry = build_registry(str(skills_dir), str(registry_path))
    print("A6 registry size:", len(registry))

    quality = run_quality_checks(gen_dir, registry)
    print("Quality checks:", quality)
    if args.strict_quality_gate and not quality.get("quality_ok"):
        raise RuntimeError(f"Quality gate failed: {quality}")


if __name__ == "__main__":
    main()
