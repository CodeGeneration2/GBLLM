"""Build traces.jsonl from pair data (slow_code, fast_code)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .llm_utils import cached_chat_completion, configure_default_client, log_llm_failure


def _source_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*", "", cleaned).strip("`\n ")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_json_call(
    *,
    messages: List[Dict[str, str]],
    cache_dir: str,
    fail_dir: str,
    stage: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
) -> Optional[Dict[str, Any]]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
    }
    try:
        content = cached_chat_completion(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            stage=stage,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:  # noqa: BLE001
        log_llm_failure(stage + "_error", payload, str(exc), fail_dir)
        return None

    obj = _extract_json(content)
    if obj is not None:
        return obj

    fix_messages = [
        {
            "role": "system",
            "content": "Fix the JSON to be strictly valid. Return JSON only.",
        },
        {"role": "user", "content": content},
    ]
    fix_payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": fix_messages,
        "reasoning_effort": reasoning_effort,
    }
    try:
        fixed = cached_chat_completion(
            messages=fix_messages,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            stage=stage + "_fix",
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:  # noqa: BLE001
        log_llm_failure(stage + "_fix_error", fix_payload, str(exc), fail_dir)
        return None

    obj = _extract_json(fixed)
    if obj is not None:
        return obj
    log_llm_failure(stage + "_json", fix_payload, fixed, fail_dir)
    return None


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_csv(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield dict(row)


def _iter_pairs(path: str, input_format: str) -> Iterable[Dict[str, Any]]:
    fmt = input_format
    if fmt == "auto":
        fmt = "jsonl" if path.lower().endswith(".jsonl") else "csv"
    if fmt == "jsonl":
        return _iter_jsonl(path)
    if fmt == "csv":
        return _iter_csv(path)
    raise ValueError(f"Unsupported --input-format: {input_format}")


def _load_existing_hashes(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen: set[str] = set()
    for rec in _iter_jsonl(path):
        source_hash = rec.get("source_hash")
        if isinstance(source_hash, str) and source_hash:
            seen.add(source_hash)
    return seen


def _normalize_record(
    row: Dict[str, Any],
    idx: int,
    *,
    statement_column: str,
    slow_column: str,
    fast_column: str,
    problem_id_column: str,
    max_statement_chars: int,
    max_code_chars: int,
) -> Optional[Tuple[Dict[str, Any], str]]:
    slow_code = str(row.get(slow_column, "") or "").strip()
    fast_code = str(row.get(fast_column, "") or "").strip()
    if not slow_code or not fast_code:
        return None

    problem_id = str(row.get(problem_id_column, "") or "").strip() or f"P{idx:07d}"
    statement = str(row.get(statement_column, "") or "").strip()
    if not statement:
        statement = (
            "Problem statement is unavailable. Optimize the slow code by learning from the paired "
            "fast code while preserving external interface and behavior."
        )

    record = {
        "problem_id": problem_id,
        "statement": statement[:max_statement_chars],
        "slow_code": slow_code[:max_code_chars],
        "fast_code": fast_code[:max_code_chars],
    }
    source_hash = _source_hash(record)
    return record, source_hash


def _trace_prompt(record: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = textwrap.dedent(
        f"""
        Return JSON only with keys:
        problem_id,
        ProblemBrief{{constraints_guess{{n_max,q_max,notes}},problem_type_tags}},
        SlowAudit{{dominant_ops,complexity{{time,space,why}},bottlenecks[{{kind,detail}}]}},
        FastAudit{{core_idea,complexity{{time,space,why}}}},
        DeltaSummary{{delta_type,complexity_delta{{from,to,note}},transformation_steps,trigger_signals,pitfalls}}.

        problem_id: {record.get('problem_id')}
        statement:
        {record.get('statement', '')}

        slow_code:
        ```python
        {record.get('slow_code', '')}
        ```

        fast_code:
        ```python
        {record.get('fast_code', '')}
        ```
        """
    ).strip()
    return [
        {
            "role": "system",
            "content": "You are a Python performance analyst. Output JSON only.",
        },
        {"role": "user", "content": prompt},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build traces.jsonl from slow/fast code pairs")
    parser.add_argument("--input", required=True, help="Pair dataset path (CSV or JSONL)")
    parser.add_argument("--input-format", default="auto", choices=["auto", "jsonl", "csv"])
    parser.add_argument("--statement-column", default="statement")
    parser.add_argument("--slow-column", default="slow_code")
    parser.add_argument("--fast-column", default="fast_code")
    parser.add_argument("--problem-id-column", default="problem_id")
    parser.add_argument("--output", required=True, help="Output traces.jsonl path")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--fail-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("EFFISKILL_MODEL"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-statement-chars", type=int, default=1500)
    parser.add_argument("--max-code-chars", type=int, default=1500)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=-1, help="-1 means all rows")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set EFFISKILL_MODEL.")
    configure_default_client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.fail_dir, exist_ok=True)

    existing_hashes = _load_existing_hashes(args.output) if args.resume else set()
    normalized: List[Tuple[Dict[str, Any], str]] = []
    skipped_missing = 0
    skipped_resume = 0

    for idx, row in enumerate(_iter_pairs(args.input, args.input_format), start=1):
        item = _normalize_record(
            row,
            idx,
            statement_column=args.statement_column,
            slow_column=args.slow_column,
            fast_column=args.fast_column,
            problem_id_column=args.problem_id_column,
            max_statement_chars=args.max_statement_chars,
            max_code_chars=args.max_code_chars,
        )
        if item is None:
            skipped_missing += 1
            continue
        record, source_hash = item
        if source_hash in existing_hashes:
            skipped_resume += 1
            continue
        normalized.append((record, source_hash))
        if args.limit >= 0 and len(normalized) >= args.limit:
            break

    if not normalized:
        print(
            json.dumps(
                {
                    "written": 0,
                    "skipped_missing": skipped_missing,
                    "skipped_resume": skipped_resume,
                    "message": "No new rows to process.",
                },
                ensure_ascii=False,
            )
        )
        return

    out_mode = "a" if args.resume and os.path.exists(args.output) else "w"
    out_lock = threading.Lock()
    stats = {"written": 0, "failed": 0}

    def _process(item: Tuple[Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
        record, source_hash = item
        messages = _trace_prompt(record)
        stage = f"a0_trace_{source_hash[:12]}"
        obj = _llm_json_call(
            messages=messages,
            cache_dir=args.cache_dir,
            fail_dir=args.fail_dir,
            stage=stage,
            model=args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if obj is None:
            return None
        obj["problem_id"] = record["problem_id"]
        obj["source_hash"] = source_hash
        return obj

    with open(args.output, out_mode, encoding="utf-8") as f_out:
        workers = max(1, int(args.workers))
        if workers == 1:
            for item in normalized:
                obj = _process(item)
                if obj is None:
                    stats["failed"] += 1
                    continue
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["written"] += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_process, item) for item in normalized]
                for fut in as_completed(futures):
                    obj = fut.result()
                    if obj is None:
                        stats["failed"] += 1
                        continue
                    with out_lock:
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        f_out.flush()
                    stats["written"] += 1

    summary = {
        **stats,
        "skipped_missing": skipped_missing,
        "skipped_resume": skipped_resume,
        "requested": len(normalized),
        "output": args.output,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
