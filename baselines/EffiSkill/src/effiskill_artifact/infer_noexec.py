

"""No-execution inference pipeline (skill-first, multi-skill-set, no ranking).

Pipeline flow:
- S1-lite diagnosis (statically diagnose the original code, analyze bottlenecks)
- one-pass model routing (select multiple different optimization skill sets based on metadata)
- develop a strategy for each skill set (three optimization plans: conservative/balanced/aggressive)
- generate K candidate code versions
- static integrity check + static correctness filtering (use AST to check syntax and interface compatibility)
- output all surviving candidates (without ranking by quality)
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only

import re
import textwrap
import time
# Note: the original code imports threading and concurrent.futures, but they are not used in the provided snippet.
# They are usually used for multithreaded concurrent LLM requests.
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from API__single_generation__5 import DeepSeek_dual_platform_function # Used to display progress bars in the terminal for easier task progress monitoring
from API__single_generation__5 import CodeLlama_Deepinfra_function
from API__single_generation__5 import Gemini_official_function
from API__single_generation__5 import Gemini_yunwu_api_function
from API__single_generation__5 import ChatGPT_three_platform_function
from API__single_generation__5 import DeepSeek_dual_platform_function
from API__single_generation__5 import CodeLlama_load_server_model_standard_function
from API__single_generation__5 import CodeLlama_server_standard_inference_function


# Environment variable control: whether the cache includes the full payload.
# If true, cache files record the full request parameters; otherwise, only the hash of the request content is recorded.
CACHE_INCLUDE_PAYLOAD = os.getenv("SELF_EVOLVE_CACHE_INCLUDE_PAYLOAD", "0").lower() in {
    "1",
    "true",
    "yes",
}





# #####################################################################################################################🔖💡✅🟨❌
DeBug = False
dataset_path  = r"./DB_Py_014_CodeLlama13B__X.csv"
saved_dataset_path  = r""
parallel_count = 1








# #####################################################################################################################🔖💡✅🟨❌
# Define the list of models to test
if "CodeLlama13B" in dataset_path:
    model_name = "./CodeLlama-13b-Instruct-hf"
    CodeLlama_tokenizer, CodeLlama_model = CodeLlama_load_server_model_standard_function(model_name=model_name)
elif "CodeLlama34B" in dataset_path:
    model_name = "./CodeLlama-34b-Instruct-hf"
    CodeLlama_tokenizer, CodeLlama_model = CodeLlama_load_server_model_standard_function(model_name=model_name)
elif "DeepSeekV32" in dataset_path:
    model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
elif "Gemini" in dataset_path:
    model_name = "gemini-2.5-flash-nothinking"
elif "GPT3" in dataset_path:
    model_name = "gpt-3.5-turbo-1106"







# #####################################################################################################################🔖💡✅🟨❌
# -----------------------------
# Main Entry Point (entry framework)
# -----------------------------
def parse_args() -> argparse.Namespace:
    prefix_path = '__'.join(dataset_path.split('__')[:-1])
    pie_id = dataset_path.split('__')[-2].split('_')[-2]
    prefix_path = prefix_path.replace(pie_id, str(int(pie_id)+1).zfill(3))
    resolved_saved_dataset_path = f"{prefix_path}__EffiSkill_generated.jsonl"


    """Define and parse command-line arguments"""
    parser = argparse.ArgumentParser(description="No-exec multi-skill-set inference pipeline")

    parser.add_argument(
        "--dataset",
        choices=["problem_table", "canonical_solutions", "custom"],
        default="custom",
        help="Dataset preset. custom requires --data and --slow-column.",
    )
    parser.add_argument("--data", default=dataset_path, help="Override dataset CSV path")
    parser.add_argument("--statement-column", default="description")
    parser.add_argument("--slow-column", default="best_before", help="Override slow code column")

    parser.add_argument("--skills-root", default=r"../../data/rq4/fix-version/skills")
    
    parser.add_argument("--out", default=resolved_saved_dataset_path)
    parser.add_argument(
        "--cache-dir",
        default="stage2/out/llm_cache_infer",
        help="Ignored in inference: successful LLM cache is disabled.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Deprecated in inference: successful LLM cache is already disabled.",
    )
    parser.add_argument("--fail-dir", default="PIE_output/llm_failures_infer")

    parser.add_argument("--llm-choice", choices=["gpt-5-mini", "qwen30b"], default="gpt-5-mini")
    parser.add_argument("--base-url", default="custom", help="OpenAI-compatible base URL; can also come from env.")
    parser.add_argument("--api-key", default="custom", help="Provider API key; can also come from env.")
    parser.add_argument("--model", default=model_name, help="Model name; can also come from env.")

    # Key hyperparameters
    parser.add_argument("--k", type=int, default=5, help="Total candidates per problem (number of generated code candidates per problem)")
    parser.add_argument("--skill-sets", type=int, default=3, help="Number of skill sets from S2 (number of skill combinations extracted in Stage S2)")
    parser.add_argument("--skills-per-set", type=int, default=6, help="maximum number of skills in a single combination")

    parser.add_argument("--workers", type=int, default=20, help="number of worker threads for concurrent execution")
    parser.add_argument(
        "--sanity-fix",
        action="store_true",
        help="Enable extra static sanity-fix stage before static check (enable the post-processing repair stage)",
    )
    parser.add_argument(
        "--require-syntax",
        action="store_true",
        help="Require Python syntax-valid candidates to survive filtering. (strictly require passing syntax checks)",
    )
    parser.add_argument("--limit", type=int, default=-1, help="-1 means full dataset (data processing limit)")
    parser.add_argument("--start", type=int, default=0, help="processing start index")
    parser.add_argument("--problem-id", default=None, help="execute for a specified single problem")

    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="support resuming from checkpoints")
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    return parser.parse_args()









# #####################################################################################################################🔖💡✅🟨❌
def main() -> None:
    args = parse_args()

    # Initialize configuration
    print(f"🚀 Initialize configuration, llm_base_url: {args.base_url}, LLM_API: {args.api_key}, model_name: {args.model}")
    try:
        test_dataset_path, slow_code_column_name = resolve_dataset_config(args)
        llm_base_url, LLM_API, model_name = resolve_model_config(args)
    except ValueError as exc:
        raise SystemExit("❌❌❌ " + str(exc)) from exc

    # Ensure the parent directory of the output file exists and create the failure log directory
    _ensure_parent_dir(args.out)
    os.makedirs(args.fail_dir, exist_ok=True)

    # Preload skill data and index structures
    print(f"📚 Load skill registry and index from {args.skills_root}")
    registry = load_registry(args.skills_root)
    operator_registry = [r for r in registry if r.get("type") == "operator"]
    skills_index = load_skills_index(args.skills_root)

    # Load built-in system meta prompts(Meta Skills)
    print("🔍 Load built-in system meta prompts (Meta Skills)")
    meta_skills = {}
    for sid in ("S1", "S2", "S3"):
        if sid not in skills_index:
            raise RuntimeError(f"Missing meta skill {sid} in {args.skills_root}")
        meta_skills[sid] = skills_index[sid]["content"]

    # Scan the existing result file to support checkpoint resumption
    print(f"🔄 Scan existing results for resume: {args.out}")
    seen_ids = _load_existing_ids(args.out) if args.resume else set()

    # Pre-filter all task rows to be processed
    print(f"📑 Pre-filter dataset, path: {test_dataset_path}, statement_column: {args.statement_column}, slow_code_column: {slow_code_column_name}")
    rows: List[Dict[str, str]] = []
    taken = 0
    for idx, row in enumerate(_iter_rows(test_dataset_path)):
        if idx < args.start:
            continue
        if args.limit >= 0 and taken >= args.limit:
            break

        problem_id = row.get("problem_id")
        if args.problem_id and problem_id != args.problem_id:
            continue
        if args.resume and problem_id in seen_ids:
            continue

        statement = row.get(args.statement_column, "")
        slow_code = row.get(slow_code_column_name, "")
        if not _clean_text(statement) or not _clean_text(slow_code):
            continue

        rows.append(row)
        taken += 1

    out_lock = threading.Lock() # Thread lock for writing files
    workers = max(1, int(args.workers))

    # Worker closure: wraps run_problem so it can be used in the thread pool
    def _process_row(row: Dict[str, str], key_index: int=0) -> Dict[str, Any]:
        problem_id = row.get("problem_id") or ""
        statement = row.get(args.statement_column, "")
        slow_code = row.get(slow_code_column_name, "")

        client = create_client(base_url=llm_base_url, api_key=LLM_API)
        result = run_problem(
            problem_id=problem_id,
            statement=statement,
            slow_code=slow_code,
            model=model_name,
            client=client,
            cache_dir="", # Inference mode does not use the local success cache
            fail_dir=args.fail_dir,
            meta_skills=meta_skills,
            skills_index=skills_index,
            operator_registry=operator_registry,
            num_skill_sets=args.skill_sets,
            skills_per_set=args.skills_per_set,
            k=args.k,
            sanity_fix=args.sanity_fix,
            require_syntax=args.require_syntax,
            key_index=key_index,
        )
        # Wrap the result and return it with additional metadata
        return {
            "problem_id": problem_id,
            "dataset": args.dataset,
            "data_path": test_dataset_path,
            "statement": statement,
            "slow_code": slow_code,
            "model": model_name,
            **result,
        }

    # Fallback record for complete crash cases
    def _failed_record(row: Dict[str, str], exc: Exception) -> Dict[str, Any]:
        return {
            "problem_id": row.get("problem_id"),
            "dataset": args.dataset,
            "data_path": test_dataset_path,
            "statement": row.get(args.statement_column, ""),
            "slow_code": row.get(slow_code_column_name, ""),
            "model": model_name,
            "status": "failed",
            "error_stage": "runtime",
            "error_message": str(exc),
        }

    def _log_done(rec: Dict[str, Any]) -> None:
        print(
            f"[done] problem_id={rec.get('problem_id')} status={rec.get('status')} "
            f"candidates={rec.get('candidate_count', 0)} survivors={rec.get('survivor_count', 0)}",
            flush=True,
        )

    # Open the output file in append mode (JSONL format)
    print(f"🖊️  Start processing {len(rows)} records, output path: {args.out}, parallel_count: {workers}")
    with open(args.out, "a", encoding="utf-8") as out_f:
        if workers == 1:
            # Single-threaded mode for easier debugging
            for row in tqdm(rows):
                try:
                    rec = _process_row(row)
                except Exception as exc:  # noqa: BLE001
                    rec = _failed_record(row, exc)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                _log_done(rec)
        else:
            # Concurrent mode: use a thread pool to improve throughput
            with ThreadPoolExecutor(max_workers=workers) as ex:
                # futs = {ex.submit(_process_row, row): row for row in rows}
                # Use enumerate to get the row index, then take modulo by workers
                futs = {ex.submit(_process_row, row, idx % workers): row for idx, row in enumerate(rows)}

                for fut in tqdm(as_completed(futs)):
                    row = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = _failed_record(row, exc)
                    
                    # Ensure JSONL writes are atomic to prevent formatting errors caused by multithreaded races
                    with out_lock:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                    _log_done(rec)

    # [End]---------------------------------------------------------------------------------------------
    print("🎉 All processing completed!")
    # 1. Read the .jsonl file and convert it to a DataFrame
    # lines=True: tell pandas that each line is an independent JSON object
    # encoding='utf-8': prevent garbled text when reading non-ASCII characters
    test_dataset_df = pd.read_csv(dataset_path)
    # test_dataset_df = test_dataset_df[:2]
    try:
        generated_dataset_df = pd.read_json(args.out, lines=True, encoding='utf-8')
    except:
        generated_dataset_data = []
        with open(args.out, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    generated_dataset_data.append(json.loads(line))
        generated_dataset_df = pd.DataFrame(generated_dataset_data)


    # ========================== [Added: expand candidate code columns] ==========================
    # Define a helper function: safely extract code from a list of dictionaries by candidate_id
    def get_code_by_cid(candidates_list, target_cid):
        if isinstance(candidates_list, list):
            for cand in candidates_list:
                if cand.get('candidate_id') == target_cid:
                    return cand.get('code', '')
        return ''  # If the corresponding C ID is not found or the data format is invalid, fill in an empty string by default

    # Loop from 1 to 5, automatically generate new columns, and match C1 through C5
    for i in range(1, 6):
        target_cid = f"C{i}"
        new_col_name = f"EffiSkill_G5__Predict_Fast_code_{i}"
        generated_dataset_df[new_col_name] = generated_dataset_df['candidates'].apply(lambda x: get_code_by_cid(x, target_cid))
    # ==============================================================================


    # ========================== [Merge the two tables by column] ==========================
    # To prevent merge failures caused by type differences(int vs str), force the keys of both tables to strings
    if "615_idx" in test_dataset_df.columns:
        test_dataset_index_column = '615_idx'
    elif "712_idx" in test_dataset_df.columns:
        test_dataset_index_column = '712_idx'
    elif "1000_idx" in test_dataset_df.columns:
        test_dataset_index_column = '1000_idx'
    test_dataset_df[test_dataset_index_column] = test_dataset_df[test_dataset_index_column].astype(str)
    generated_dataset_df['problem_id'] = generated_dataset_df['problem_id'].astype(str)

    # Perform the merge operation
    # df_other placed first means its columns will appear on the far left of the generated CSV
    merged_dataset_df = pd.merge(
        test_dataset_df, 
        generated_dataset_df, 
        left_on=test_dataset_index_column,      # matching column of the left table (the other table)
        right_on='problem_id', # matching column of the right table (the current table)
        how='left'             # left join: keep all rows from df_other
    )
    # ==============================================================================

    # 2. Save the DataFrame as a .csv file
    # index=False: indicates that the DataFrame row index(0, 1, 2...)is not saved into the CSV file
    # encoding='utf-8-sig': this encoding is recommended when saving CSV files with non-ASCII characters because it prevents garbled text in Excel
    # Automatically generate a new name
    if saved_dataset_path == '':
        prefix_path = '__'.join(dataset_path.split('__')[:-1])
        pie_id = dataset_path.split('__')[-2].split('_')[-2]
        prefix_path = prefix_path.replace(pie_id, str(int(pie_id)+1).zfill(3))
        resolved_saved_dataset_path = f"{prefix_path}__EffiSkill_generated.csv"
    else:
        resolved_saved_dataset_path = saved_dataset_path
    # Save the modified data to a new CSV file
    merged_dataset_df.to_csv(f'{resolved_saved_dataset_path}', index=False)

    print("Conversion completed!")






# #####################################################################################################################🔖💡✅🟨❌
# -----------------------------
# Dataset/model configuration (configuration parsing and mapping)
# -----------------------------
def resolve_dataset_config(args: argparse.Namespace) -> Tuple[str, str]:
    """Process and validate dataset-related parameter mapping logic"""
    if args.dataset == "problem_table":
        test_dataset_path = args.data or "problem_table.csv"
        slow_code_column_name = args.slow_column or "best_before"
    elif args.dataset == "canonical_solutions":
        test_dataset_path = args.data or "canonical_solutions.csv"
        slow_code_column_name = args.slow_column or "canonical_solution"
    else:
        # custom mode requires explicitly specifying the path and column name
        if not args.data:
            raise ValueError("--dataset custom requires --data")
        if not args.slow_column:
            raise ValueError("--dataset custom requires --slow-column")
        test_dataset_path = args.data
        slow_code_column_name = args.slow_column
        
    return test_dataset_path, slow_code_column_name





# #####################################################################################################################🔖💡✅🟨❌
def resolve_model_config(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Parse the LLM base URL, key, and model name. Values can be passed through command-line arguments or environment variables."""
    llm_base_url = (args.base_url or os.environ.get("EFFISKILL_BASE_URL", "")).strip()
    LLM_API = (args.api_key or os.environ.get("EFFISKILL_API_KEY", "")).strip()
    model_name = (args.model or os.environ.get("EFFISKILL_MODEL", "")).strip()
    if not llm_base_url:
        raise ValueError("❌❌❌ Missing base URL. Pass --base-url or set EFFISKILL_BASE_URL.")
    if not LLM_API:
        raise ValueError("❌❌❌ Missing API key. Pass --api-key or set EFFISKILL_API_KEY.")
    if not model_name:
        raise ValueError("❌❌❌ Missing model. Pass --model or set EFFISKILL_MODEL.")
    return llm_base_url, LLM_API, model_name





# -----------------------------
# Basic I/O helpers (Basic I/O helpers)
# -----------------------------
def _read_text(path: str) -> str:
    """Read text file content"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _iter_rows(path: str) -> Iterable[Dict[str, str]]:
    """Iterate through the CSV file row by row and return each row as a dictionary"""
    # Note: replace this reading method according to the actual storage format of your DataFrame
    # Use read_csv for CSV files and read_pickle for pickle files
    df = pd.read_csv(path)
    # df = df[:2]

    if "615_idx" in df.columns:
        test_dataset_index_column = '615_idx'
    elif "712_idx" in df.columns:
        test_dataset_index_column = '712_idx'
    elif "1000_idx" in df.columns:
        test_dataset_index_column = '1000_idx'
        
    # df = df[:2]
    for _, row in df.iterrows():
        yield {
            "problem_id": str(row[test_dataset_index_column]), # Map to 615_idx in your DataFrame
            "description": "None",             # Force all problem descriptions to the string "None"
            "best_before": str(row["input"])   # Map to the slow-code column input (best_before is the framework default slow_col)
        }

    # with open(path, "r", encoding="utf-8") as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         yield row


def _is_resume_terminal_status(status: Any) -> bool:
    """Determine whether the current status is a terminal status(used for checkpoint resumption/recovery).
    'ok' means successful generation, 'no_survivor' means generation succeeded but none passed static filtering."""
    return _clean_text(status).lower() in {"ok", "no_survivor"}


def _load_existing_ids(out_path: str) -> Set[str]:
    """Load already processed problem IDs from the existing output file (problem_id).
    Used to skip already completed problems when restarting after an interruption."""
    if not os.path.exists(out_path):
        return set()
    seen: Set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try to parse a JSON line (JSONL format)
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("problem_id")
            # If the ID exists and its status is terminal, add it to the processed set
            if pid and _is_resume_terminal_status(obj.get("status")):
                seen.add(pid)
    return seen


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of the given path exists, creating it automatically if needed"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# -----------------------------
# Skill loading (Skill loading)
# -----------------------------
def _find_skill_id(text: str) -> Optional[str]:
    """Parse skill_id from the text of a skill file.
    Expected format, for example: 'skill_id: string_manipulation_1'"""
    for line in text.splitlines():
        if line.strip().startswith("skill_id:"):
            return line.split(":", 1)[1].strip()
    return None


def load_skills_index(skills_root: str) -> Dict[str, Dict[str, str]]:
    """Traverse the skill root directory, load all files containing SKILL.md, and build a skill index dictionary."""
    index: Dict[str, Dict[str, str]] = {}
    for root, _, files in os.walk(skills_root):
        if "SKILL.md" not in files:
            continue
        path = os.path.join(root, "SKILL.md")
        text = _read_text(path)
        sid = _find_skill_id(text)
        if not sid:
            continue
        # Map the skill ID to its file path and content
        index[sid] = {"path": path, "content": text}
    return index


def load_registry(skills_root: str) -> List[Dict[str, Any]]:
    """Load registry.json from the skill root directory (the skill registry, which usually contains skill metadata)."""
    path = os.path.join(skills_root, "registry.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# LLM utilities (LLM utilities)
# Note: In inference mode, this is currently only used for error logging; successful caching is disabled
# -----------------------------
def create_client(base_url: str, api_key: str) -> OpenAI:
    """Create and return an OpenAI client"""
    return OpenAI(base_url=base_url, api_key=api_key)


def _hash_payload(payload: Dict[str, Any]) -> str:
    """Compute a SHA256 hash for the LLM request payload to generate a cache key or unique identifier"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_cache(cache_path: str) -> Optional[str]:
    """Read the local LLM response cache file"""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    content = obj.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def _write_cache(cache_path: str, content: str, payload: Dict[str, Any]) -> None:
    """Write the LLM response and its request payload or payload hash to the local cache file"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_obj: Dict[str, Any] = {"content": content}
    if CACHE_INCLUDE_PAYLOAD:
        cache_obj["payload"] = payload
    else:
        cache_obj["payload_hash"] = _hash_payload(payload)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)


def _is_non_retryable_error(exc: Exception) -> bool:
    """Determine whether the exception is a non-retryable HTTP error (such as parameter error 400, authentication failure 401/403, model not found 404)"""
    status = getattr(exc, "status_code", None)
    return status in {400, 401, 403, 404}


def _is_rate_limited_error(exc: Exception) -> bool:
    """Determine whether the exception is caused by rate limiting (Rate Limit, HTTP 429)"""
    txt = str(exc).lower()
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in txt or "rate limit" in txt or "too many requests" in txt




# #####################################################################################################################🔖💡✅🟨❌
def cached_chat_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    cache_dir: str,
    stage: str,
    timeout: int = 180,
    key_index: int = 0,
) -> str:
    """
    Core LLM call function with retry and backoff mechanisms.
    (Note: cache_dir and stage are retained for call-signature compatibility, but actual successful cache reads/writes are not performed in the current inference mode)
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # Ignore cache parameters (for compatibility only)
    _ = cache_dir
    _ = stage

    max_attempts = 4 # Set the maximum number of attempts to 4
    last_exc: Optional[Exception] = None
    response = None
    
    for attempt in range(max_attempts):
        try:

            response = generate_data_with_api_function(messages[0]['content'], messages[1]['content'], key_index = key_index, temperature = temperature, max_length = 1024, should_print_prompt = DeBug,)

            # response = client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     timeout=timeout,
            #     temperature=temperature,
            #     max_completion_tokens=max_tokens,
            # )
            break # Break out of the loop on success
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_non_retryable_error(exc):
                raise # Raise non-retryable errors directly
            
            # If this is not the final attempt, wait and retry
            if attempt < max_attempts - 1:
                if _is_rate_limited_error(exc):
                    # Rate limit: use a longer exponential backoff wait (2s, 4s, 6s... up to20s)
                    delay = min(20.0, 2.0 * (attempt + 1))
                else:
                    # General network error: shorter wait (1s, 2s, 3s... up to8s)
                    delay = min(8.0, 1.0 * (attempt + 1))
                time.sleep(delay)

    if response is None:
        raise RuntimeError(f"chat completion failed after retries: {last_exc}")

    content = response[0] or ""
    if not content.strip():
        raise RuntimeError("empty response from LLM")

    return content






# #####################################################################################################################🔖💡✅🟨
def generate_data_with_api_function(input_role_prompt, input_question_str, key_index, temperature, max_length=1024, should_print_prompt=False,):

    # ------------------------------------------------------------------------------------------------------------
    if model_name in ["./CodeLlama-34b-Instruct-hf", "./CodeLlama-13b-Instruct-hf"]:
        model_response_text_list = CodeLlama_server_standard_inference_function( model=CodeLlama_model,   
                                                        input_role_prompt=input_role_prompt, 
                                                        input_question_text=input_question_str, 
                                                        generated_code_count=1, 
                                                        temperature=temperature, 
                                                        max_length=max_length,
                                                        should_print_prompt=should_print_prompt,
                                                        tokenizer=CodeLlama_tokenizer,
                                                        )


    # ------------------------------------------------------------------------------------------------------------
    elif model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = CodeLlama_Deepinfra_function(key_index=key_index, 
                                        model_name=model_name,   
                                        input_role_prompt=input_role_prompt, 
                                        input_question_text=input_question_str, 
                                        generated_code_count=1, 
                                        temperature=temperature, 
                                        max_length=max_length,
                                        should_print_prompt=should_print_prompt,
                                        )




    # ------------------------------------------------------------------------------------------------------------
    elif "gemini" in model_name:
        model_response_text_list = Gemini_yunwu_api_function(  platform="YunwuAPI",                  # [ YunwuAI, OpenAI official, Close_AI ]
                                                key_index = key_index, 
                                                model_name = model_name,     # gpt-3.5-turbo-0125   gpt-4o    gpt-4-1106-preview    gemini-2.5-flash-thinking
                                                input_role_prompt=input_role_prompt, 
                                                input_question_text=input_question_str, 
                                                generated_code_count = 1,               # Must be 1
                                                temperature = temperature,                 # Effective
                                                max_length = max_length,                 # No effect
                                                should_print_prompt = should_print_prompt,
                                                )

    
    # ------------------------------------------------------------------------------------------------------------
    elif False and "gemini" in model_name:
        model_response_text_list, average_log_probability_list = Gemini_official_function(key_index=key_index, 
                                                            model_name=model_name,   
                                                            input_role_prompt=input_role_prompt, 
                                                            input_question_text=input_question_str, 
                                                            generated_code_count=1, 
                                                            temperature=temperature, 
                                                            max_length="No limit", 
                                                            thinking_budget=-404,          # -1: Enable dynamic thinking    0: Disable thinking    greater than 0: fixed thinking budget of 8
                                                            return_log_probability=False,
                                                            should_print_prompt=should_print_prompt,
                                                            )


        

    # ------------------------------------------------------------------------------------------------------------
    elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4o-mini", "gpt-4.1-nano"]:
        model_response_text_list, average_log_probability_list = ChatGPT_three_platform_function(platform = "YunwuAPI",                  # [ YunwuAI, OpenAI official, Close_AI ]
                                                                key_index = key_index, 
                                                                model_name = model_name,     # gpt-3.5-turbo-0125   gpt-4o    gpt-4-1106-preview  
                                                                input_role_prompt=input_role_prompt, 
                                                                input_question_text=input_question_str, 
                                                                generated_code_count = 1, 
                                                                temperature = temperature, 
                                                                max_length = max_length, 
                                                                return_log_probability=False, 
                                                                should_print_prompt=should_print_prompt,
                                                                )

    
    # -------------------------------------------------------------------------------------------------------------
    elif model_name in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-V3.2-Exp"]:
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            _platform = "DeepSeek official"
        elif model_name in ["deepseek-ai/DeepSeek-V3.2-Exp"]:
            _platform = "SiliconFlow API"
        model_response_text_list, model_response_full_dict, chain_of_thought_text, average_log_probability = DeepSeek_dual_platform_function( platform = _platform,                  # [ DeepSeek official, SiliconFlow API ]
                                                                                            key_index = key_index, 
                                                                                            model_name = model_name,     # [ deepseek-chat, deepseek-reasoner, deepseek-ai/DeepSeek-V3.2-Exp ]
                                                                                            input_role_prompt = input_role_prompt, 
                                                                                            input_question_text = input_question_str, 
                                                                                            temperature = temperature, 
                                                                                            max_length = max_length, 
                                                                                            should_print_prompt = should_print_prompt,
                                                                                            )
        
    return model_response_text_list


def log_llm_failure(stage: str, payload: Dict[str, Any], response_text: str, out_dir: str) -> None:
    """Record details of LLM call failures or JSON parsing failures for later debugging"""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{stage}_{ts}_{_hash_payload(payload)[:8]}"
    jpath = os.path.join(out_dir, f"json_fail_{base}.json")
    tpath = os.path.join(out_dir, f"json_fail_{base}.txt")
    
    # Write the request payload and failed response text separately
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"payload": payload}, f, ensure_ascii=False, indent=2)
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(response_text)


# #####################################################################################################################🔖💡✅🟨❌
def extract_json(text: str) -> Optional[Any]:
    """Extract a JSON object from text returned by the LLM.
    Can automatically remove Markdown code block markers (```json ... ```) and other interfering characters."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    # Remove the leading ```json
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*", "", cleaned).strip("`\n ")
        
    # Find the first '{' or '['
    s_obj = cleaned.find("{")
    s_arr = cleaned.find("[")
    starts = [x for x in (s_obj, s_arr) if x != -1]
    if not starts:
        return None
    start = min(starts)
    
    # Find the last '}' or ']'
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        return None
        
    # Extract the snippet that may contain JSON
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None



# #####################################################################################################################🔖💡✅🟨❌
def llm_json_call(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    cache_dir: str,
    stage: str,
    out_fail_dir: str,
    key_index: int = 0,
) -> Optional[Any]:
    """
    Call wrapper specifically used to force the LLM to return JSON-formatted data.
    Includes a layer of "Auto-Fix" (auto-fix) mechanism: if the first response is not valid JSON, 
    request the LLM again to fix the format.
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    # First generation attempt
    try:
        content = cached_chat_completion(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            stage=stage,
            key_index=key_index,
        )
    except Exception as exc:  # noqa: BLE001
        log_llm_failure(stage + "_error", payload, str(exc), out_fail_dir)
        return None

    # Try to parse JSON
    obj = extract_json(content)
    if obj is not None:
        return obj

    # If parsing fails, start the format repair process (Fix mechanism)
    fix_messages = [
        {
            "role": "system",
            "content": "Fix this to valid strict JSON. Return JSON only.",
        },
        {"role": "user", "content": content}, # Pass the first failed content to the LLM for repair
    ]
    fix_payload = {
        "model": model,
        "temperature": 1.0, # Use a higher temperature(1.0)to increase diversity and avoid being stuck in repeated incorrect output
        "max_tokens": max_tokens,
        "messages": fix_messages,
    }
    
    try:
        fix_content = cached_chat_completion(
            client=client,
            messages=fix_messages,
            model=model,
            temperature=1.0,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            stage=stage + "_fix",
            key_index=key_index,
        )
    except Exception as exc:  # noqa: BLE001
        log_llm_failure(stage + "_fix_error", fix_payload, str(exc), out_fail_dir)
        return None

    # Try again to parse the repaired result
    obj = extract_json(fix_content)
    if obj is not None:
        return obj

    # If it still fails, give up and record the log
    log_llm_failure(stage + "_json", fix_payload, fix_content, out_fail_dir)
    return None





# -----------------------------
# Prompt builders (Prompt builders for each stage)
# -----------------------------
def build_s1_messages(meta_skill: str, statement: str, slow_code: str) -> List[Dict[str, str]]:
    """Stage 1: diagnosis. Build a prompt that asks the LLM to act as a code diagnostic expert and identify the current code performance bottlenecks."""
    system = textwrap.dedent(
        f"""
        You are a Python code optimization diagnostician.
        Follow the meta skill guidance below.

        {meta_skill}

        Return strict JSON only.
        Required fields:
        - constraints_guess: {{n_max, q_max, notes}} # Estimate data scale
        - problem_type_tags: [string]                # Problem type tagging
        - dominant_ops: [string]                     # Main time-consuming operations
        - complexity: {{time, space, why}}           # Current time and space complexity analysis
        - bottlenecks: [{{kind, detail}}]            # Performance bottleneck list
        - must_change: "algorithm"|"implementation"|"mixed" # Required change type
        - risks: [string]                            # Potential risks
        - confidence: "high"|"medium"|"low"          # Diagnosis confidence
        """
    ).strip()

    user = textwrap.dedent(
        f"""
        Problem statement:
        {statement}

        Baseline code:
        ```python
        {slow_code}
        ```
        """
    ).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_s2_messages(
    meta_skill: str,
    optimization_brief: Dict[str, Any],
    operator_registry_meta: List[Dict[str, Any]],
    num_skill_sets: int,
    skills_per_set: int,
) -> List[Dict[str, str]]:
    """Stage 2: routing. Select multiple optimization skill sets from different dimensions based on the Stage 1 diagnosis and the global skill registry(Skill Sets)."""
    system = textwrap.dedent(
        f"""
        You are a skill router for Python optimization.
        Follow the meta skill guidance below.

        {meta_skill}

        Select multiple DIFFERENT skill sets in one pass.
        Return strict JSON only.
        Required fields:
        - skill_sets: [
            {{
              set_id,
              confidence: "high"|"medium"|"low",
              selection_theme,                  # Optimization theme of this skill set
              selected_skills: [{{skill_id, reason}}] # Selected skill IDs and their reasons
            }}
          ]
        - global_coverage_gaps: [string]

        Constraints:
        - Produce exactly {num_skill_sets} skill sets. # Specify how many skill combinations to generate
        - Max {skills_per_set} skills per set.         # Maximum number of skills per combination
        - Skill sets should be meaningfully different in optimization mechanism. # Skill combinations must differ significantly
        - Use only skill_ids from provided registry metadata.
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        OptimizationBrief:
        {opt}

        Operator registry metadata:
        {registry}
        """
        )
        .strip()
        .format(
            opt=json.dumps(optimization_brief, ensure_ascii=False),
            registry=json.dumps(operator_registry_meta, ensure_ascii=False),
        )
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_s3_messages(
    meta_skill: str,
    statement: str,
    slow_code: str,
    optimization_brief: Dict[str, Any],
    skill_set: Dict[str, Any],
    skill_bodies: List[str],
) -> List[Dict[str, str]]:
    """Stage 3: strategy. Concretize the selected skills and develop three optimization plans: conservative, balanced, and aggressive."""
    system = textwrap.dedent(
        f"""
        You generate plans from selected optimization skills.
        Follow the meta skill guidance below.

        {meta_skill}

        Return strict JSON only.
        Required fields:
        - plans: [
            {{
              plan_id,
              plan_style: "conservative"|"balanced"|"aggressive", # Plan style
              name,
              core_idea,          # Core idea
              target_complexity,  # Expected complexity
              risks: [string],
              steps: [string]     # Specific steps
            }}
          ]

        Constraints:
        - Produce exactly 3 plans: conservative, balanced, aggressive.
        - All plans must preserve the external code interface style.
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        Problem statement:
        {statement}

        Baseline code:
        ```python
        {slow_code}
        ```

        OptimizationBrief:
        {opt}

        Skill set:
        {skill_set}

        Selected skill bodies:
        {skill_bodies}
        """
        )
        .strip()
        .format(
            statement=statement,
            slow_code=slow_code,
            opt=json.dumps(optimization_brief, ensure_ascii=False),
            skill_set=json.dumps(skill_set, ensure_ascii=False),
            skill_bodies="\n\n".join(skill_bodies),
        )
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_generate_messages(
    statement: str, slow_code: str, plan: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generation stage: write the optimized code according to a specific plan generated in Stage 3.
    Emphasize that the external interface form, such as function signatures and class structure, must never be changed."""
    system = textwrap.dedent(
        """
        Generate optimized Python code for the given plan.
        Return strict JSON only.

        Required fields:
        - code: string
        - complexity: {time, space, why}
        - dominant_ops: [string]
        - notes: short explanation
        - static_verdict: "pass"|"fail"
        - interface_match: true|false
        - complexity_plausible: true|false
        - static_issues: [string]
        - static_confidence: 0-1

        Critical interface rule:
        - Generate code based on the version of the input code.
        - Preserve callable style and external interface exactly.
          Examples:
          - if input is class/method style, keep class/method style;
          - if input is function style, keep function style;
          - if input is script stdin/stdout style, keep script style.
        - Do not convert between these styles.

        Header/style guidance (soft):
        - Try to preserve the input module header format (leading comments/docstring),
          import section structure/order, and overall top-level layout unless a change
          is necessary for correctness or the optimization plan.
        - Avoid unnecessary reformatting or cosmetic rewrites.

        Assume baseline logic is valid and keep semantics.
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        Problem statement:
        {statement}

        Baseline code:
        ```python
        {slow_code}
        ```

        Plan:
        {plan}
        """
        )
        .strip()
        .format(
            statement=statement,
            slow_code=slow_code,
            plan=json.dumps(plan, ensure_ascii=False),
        )
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_sanity_messages(
    statement: str, slow_code: str, candidate_code: str
) -> List[Dict[str, str]]:
    """Post-processing stage(Sanity):Act as a static checker to inspect whether the generated code has obvious issues and repair them.
    Ensure the code can run and the interface has not been changed incorrectly."""
    system = textwrap.dedent(
        """
        Perform static sanity cleanup only (no execution).
        Preserve external interface style exactly as baseline.
        Softly preserve module header/comments/import structure from baseline unless
        required to fix correctness.

        Return strict JSON only.
        Required fields:
        - code: fixed code
        - fixes: [string]
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        Problem statement:
        {statement}

        Baseline code (interface reference):
        ```python
        {slow_code}
        ```

        Candidate code:
        ```python
        {code}
        ```
        """
        )
        .strip()
        .format(statement=statement, slow_code=slow_code, code=candidate_code)
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -----------------------------
# Normalization helpers (Text/code format normalization and AST static-check helper functions)
# -----------------------------
def _clean_text(x: Any) -> str:
    """Clean normal text: remove excess whitespace and merge consecutive spaces"""
    if x is None:
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip()
    return re.sub(r"\s+", " ", str(x)).strip()


def _clean_code(x: Any) -> str:
    """Clean code: normalize line endings, remove extra leading/trailing blank lines, and preserve indentation/layout structure"""
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    return x.strip("\n")


def _python_syntax_error(code: str) -> Optional[str]:
    """Use Python built-in AST module to statically check whether code has syntax errors(SyntaxError).
    If errors exist, return detailed error information."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"


def _is_public_name(name: str) -> bool:
    """Determine whether a variable/function/class name is public (does not start with an underscore)"""
    return bool(name) and not name.startswith("_")


def _signature_shape(args: ast.arguments, *, drop_first: bool = False) -> Dict[str, Any]:
    """Extract the structural shape of a function signature, including argument types and default-value counts.
    drop_first: For class methods, this is usually set to True to drop 'self' or 'cls' argument."""
    posonly = [a.arg for a in args.posonlyargs]
    positional = [a.arg for a in args.args]
    if drop_first and positional:
        positional = positional[1:]
    return {
        "posonly": posonly,                                       # positional-only arguments (python 3.8+ slash /)
        "positional": positional,                                 # regular positional arguments
        "kwonly": [a.arg for a in args.kwonlyargs],               # keyword-only arguments
        "defaults": len(args.defaults),                           # number of positional argument defaults
        "kw_defaults": sum(default is not None for default in args.kw_defaults), # number of keyword argument defaults
        "has_vararg": args.vararg is not None,                    # whether there is *args
        "has_kwarg": args.kwarg is not None,                      # whether there is **kwargs
    }


def _extract_interface_contract(code: str) -> Optional[Dict[str, Any]]:
    """
    Extract the code interface contract by parsing the AST tree (Interface Contract).
    This is crucial for verifying whether the LLM-optimized code can directly replace the original code without breaking test cases.
    Return a dictionary containing code style ("class", "function", "script", "mixed") and its public signatures.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    functions: Dict[str, Dict[str, Any]] = {}
    classes: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Traverse the first-level AST nodes
    for node in tree.body:
        # Collect public global functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
            functions[node.name] = _signature_shape(node.args)
            continue
        # Collect public classes and their public methods
        if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            methods: Dict[str, Dict[str, Any]] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(
                    item.name
                ):
                    methods[item.name] = _signature_shape(item.args, drop_first=True)
            classes[node.name] = methods

    # Infer the main code style
    if classes and not functions:
        style = "class"     # pure object-oriented style (such as a LeetCode template)
    elif functions and not classes:
        style = "function"  # pure function style
    elif not classes and not functions:
        style = "script"    # pure script style (such as direct print commonly used in contests)
    else:
        style = "mixed"     # mixed style

    return {
        "style": style,
        "functions": functions,
        "classes": classes,
    }













# #####################################################################################################################🔖💡✅🟨❌
from typing import Any, Dict, List, Set, Tuple, Optional
import argparse
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Assume the following helper functions are already defined in the context and use them directly here
# _extract_interface_contract, _clean_text, _clean_code, llm_json_call, build_s1_messages, etc.



# #####################################################################################################################🔖💡✅🟨❌
def _check_interface_match(baseline_code: str, candidate_code: str) -> Tuple[bool, List[str]]:
    """
    Check whether the candidate code strictly preserves the baseline code interface contract, including classes, function signatures, and code style.
    
    :param baseline_code: Original slow code (baseline)
    :param candidate_code: LLM generated optimized candidate code
    :return: (whether matching succeeds, list of specific error messages)
    """
    # Extract AST interface features of the baseline and candidate code
    baseline = _extract_interface_contract(baseline_code)
    candidate = _extract_interface_contract(candidate_code)

    if baseline is None:
        # If the baseline code itself has syntax errors and cannot be parsed, do not penalize the candidate code
        return True, []
    if candidate is None:
        # If the candidate code cannot be parsed due to syntax errors, directly mark it as an interface mismatch
        return False, ["Candidate interface could not be verified because the code does not parse."]

    issues: List[str] = []
    baseline_style = baseline["style"]
    candidate_style = candidate["style"]

    # 1. Check whether the overall code style has been destructively changed
    if baseline_style == "class" and not candidate["classes"]:
        issues.append("Candidate no longer exposes the baseline class-style interface.")
    if baseline_style == "function" and not candidate["functions"]:
        issues.append("Candidate no longer exposes the baseline function-style interface.")
    if baseline_style == "script" and candidate["classes"]:
        issues.append("Candidate changed baseline script-style code into a class-based interface.")
    if baseline_style in {"function", "script"} and candidate_style == "class":
        issues.append(f"Candidate changed interface style from {baseline_style} to class.")

    # 2. Check whether top-level function signatures are missing or have drifted (parameter changes)
    for func_name, expected_sig in baseline["functions"].items():
        actual_sig = candidate["functions"].get(func_name)
        if actual_sig is None:
            issues.append(f"Missing top-level function `{func_name}` from baseline interface.")
            continue
        if actual_sig != expected_sig:
            issues.append(f"Function signature drift for `{func_name}`.")

    # 3. Check whether classes and their method signatures are missing or have drifted
    for class_name, expected_methods in baseline["classes"].items():
        actual_methods = candidate["classes"].get(class_name)
        if actual_methods is None:
            issues.append(f"Missing class `{class_name}` from baseline interface.")
            continue
        for method_name, expected_sig in expected_methods.items():
            actual_sig = actual_methods.get(method_name)
            if actual_sig is None:
                issues.append(f"Missing method `{class_name}.{method_name}` from baseline interface.")
                continue
            if actual_sig != expected_sig:
                issues.append(f"Method signature drift for `{class_name}.{method_name}`.")

    # If the issues list is empty, it means a complete match
    return not issues, issues




# #####################################################################################################################🔖💡✅🟨❌
def _op_registry_meta(operator_registry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simplify the operator (skill) registry and keep only the metadata fields required for LLM routing(S2)to save prompt tokens.
    """
    out = []
    for r in operator_registry:
        out.append(
            {
                "skill_id": r.get("skill_id"),
                "name": r.get("name"),
                "description": r.get("description"),
                "family": r.get("family"), # skill family/category
                "triggers": (r.get("triggers") or [])[:6], # limit to at most 6 triggers
            }
        )
    return out




# #####################################################################################################################🔖💡✅🟨❌
def _normalize_skill_sets(
    raw: Any,
    valid_operator_ids: Set[str],
    num_skill_sets: int,
    skills_per_set: int,
) -> List[Dict[str, Any]]:
    """
    Clean and normalize multiple skill sets returned by LLM Stage S2(Skill Sets).
    Handle invalid skill IDs, limit the number of skills per set, and remove duplicates.
    """
    skill_sets = []
    if isinstance(raw, dict):
        skill_sets = raw.get("skill_sets") or []

    norm: List[Dict[str, Any]] = []
    used_signatures = set() # Used to record existing skill-combination signatures to prevent duplicates

    for idx, ss in enumerate(skill_sets, start=1):
        if not isinstance(ss, dict):
            continue
        selected = ss.get("selected_skills") or []
        cleaned_selected = []
        seen = set() # Used for deduplication within the current skill set
        
        for item in selected:
            if not isinstance(item, dict):
                continue
            sid = _clean_text(item.get("skill_id"))
            # Filter out fabricated IDs that do not exist in the valid registry and duplicate IDs within the set
            if sid not in valid_operator_ids or sid in seen:
                continue
            seen.add(sid)
            cleaned_selected.append(
                {
                    "skill_id": sid,
                    "reason": _clean_text(item.get("reason")) or "matched diagnosis",
                }
            )
            # Truncate when the configured maximum number of skills per set is reached
            if len(cleaned_selected) >= skills_per_set:
                break

        if not cleaned_selected:
            continue

        # Generate the signature of this skill set(sort skill IDs and form a tuple), for global deduplication
        sig = tuple(sorted(x["skill_id"] for x in cleaned_selected))
        if sig in used_signatures:
            continue
        used_signatures.add(sig)

        norm.append(
            {
                "set_id": _clean_text(ss.get("set_id")) or f"SET{idx}",
                "confidence": _clean_text(ss.get("confidence")).lower() or "medium",
                "selection_theme": _clean_text(ss.get("selection_theme")) or "general",
                "selected_skills": cleaned_selected,
            }
        )

    if not norm:
        return []
    
    # Return only the required number of sets. This intentionally does not clone identical skill sets to fill the count, because duplicate retrieval would waste generation budget
    return norm[:num_skill_sets]


def _normalize_plan_style(x: Any) -> str:
    """Normalize the style label of an optimization plan and map it to the three basic strategies."""
    s = _clean_text(x).lower()
    if "conserv" in s:
        return "conservative" # Conservative: small changes and high safety
    if "aggr" in s:
        return "aggressive"   # Aggressive: maximum performance, possibly with major logic changes
    return "balanced"         # Balanced (default)


def _normalize_plans(raw: Any, set_id: str) -> List[Dict[str, Any]]:
    """
    Clean and normalize optimization plans returned by LLM Stage S3(Plans).
    Force each skill set to have three plan styles: conservative, balanced, and aggressive.
    """
    plans = []
    if isinstance(raw, dict):
        plans = raw.get("plans") or []

    norm: List[Dict[str, Any]] = []
    for i, p in enumerate(plans, start=1):
        if not isinstance(p, dict):
            continue
        style = _normalize_plan_style(p.get("plan_style"))
        norm.append(
            {
                "plan_id": _clean_text(p.get("plan_id")) or f"{set_id}_P{i}",
                "set_id": set_id,
                "plan_style": style,
                "name": _clean_text(p.get("name")) or f"{style.title()} Plan",
                "core_idea": _clean_text(p.get("core_idea")),
                "target_complexity": _clean_text(p.get("target_complexity")),
                # Limit the maximum length of the risks and steps lists
                "risks": [_clean_text(x) for x in (p.get("risks") or []) if _clean_text(x)][:6],
                "steps": [_clean_text(x) for x in (p.get("steps") or []) if _clean_text(x)][:8],
            }
        )

    # Fallback mechanism: complete missing style strategies by cloning to ensure that three different styles are always output
    style_to_plan = {p["plan_style"]: p for p in norm}
    for style in ("conservative", "balanced", "aggressive"):
        if style in style_to_plan:
            continue
        # Add a generic default plan
        norm.append(
            {
                "plan_id": f"{set_id}_{style[:3].upper()}", # for example SET1_CON
                "set_id": set_id,
                "plan_style": style,
                "name": f"{style.title()} Plan",
                "core_idea": "Apply selected skills while preserving interface and semantics.",
                "target_complexity": "(improved vs baseline where applicable)",
                "risks": ["Static-only validation may miss subtle bugs."],
                "steps": ["Apply the selected skills coherently to baseline code."],
            }
        )

    # Sort in the deterministic order conservative -> balanced -> aggressive
    style_order = {"conservative": 0, "balanced": 1, "aggressive": 2}
    norm = sorted(norm, key=lambda p: (style_order.get(p["plan_style"], 9), p["plan_id"]))
    return norm


def _allocate_plan_slots(plans: List[Dict[str, Any]], k: int) -> List[int]:
    """
    Allocate generation slots to different plans according to the total generation budget k.
    For example, with 3 plans, k=8, the allocation may be [3, 3, 2].
    """
    if k <= 0 or not plans:
        return []
    n = len(plans)
    alloc = [0] * n
    
    if n <= k:
        # If the budget is sufficient, first ensure each plan generates at least 1 
        for i in range(n):
            alloc[i] = 1
        remain = k - n
        i = 0
        # Assign remaining slots in round-robin order
        while remain > 0:
            alloc[i % n] += 1
            i += 1
            remain -= 1
        return alloc

    # When the budget is tight(k < number of plans):selectively allocate 1 slot(s) according to style priority (conservative first)
    order = sorted(
        range(n),
        key=lambda i: {"conservative": 0, "balanced": 1, "aggressive": 2}.get(
            plans[i].get("plan_style", "balanced"), 9
        ),
    )
    for i in order[:k]:
        alloc[i] = 1
    return alloc




# #####################################################################################################################🔖💡✅🟨❌
# -----------------------------
# Core problem runner (single-problem execution engine)
# -----------------------------
def run_problem(
    *,
    problem_id: str,
    statement: str,
    slow_code: str,
    model: str,
    client: OpenAI,
    cache_dir: str,
    fail_dir: str,
    meta_skills: Dict[str, str],
    skills_index: Dict[str, Dict[str, str]],
    operator_registry: List[Dict[str, Any]],
    num_skill_sets: int,
    skills_per_set: int,
    k: int,
    sanity_fix: bool,
    require_syntax: bool,
    key_index: int,
) -> Dict[str, Any]:
    """
    Execute the core logic of the entire inference pipeline (Pipeline) .
    Includes S1 diagnosis -> S2 skill routing -> S3 strategy planning -> S4 code generation and static checking.
    """
    result: Dict[str, Any] = {"status": "ok"}
    stage_prefix = f"{problem_id or 'unknown'}"

    # ================= Stage 1: Static diagnosis =================
    print(f"🔖 Stage 1: Static diagnosis --- (Diagnosis) for problem {problem_id} ---")
    s1 = llm_json_call(
        client,
        messages=build_s1_messages(meta_skills["S1"], statement, slow_code),
        model=model,
        temperature=0.8,
        max_tokens=25000,
        cache_dir=cache_dir,
        stage=f"{stage_prefix}_s1_diagnose",
        out_fail_dir=fail_dir,
        key_index=key_index,
    )
    if s1 is None:
        return {"status": "failed", "error_stage": "s1"}
    result["optimization_brief"] = s1

    # ================= Stage 2: Skill retrieval/routing =================
    # Get the set of all valid operator IDs
    print(f"🔖 Stage 2: Skill retrieval/routing --- (Skill Routing) for problem {problem_id} ---")
    valid_operator_ids = {str(r.get("skill_id")) for r in operator_registry if r.get("skill_id")}
    s2 = llm_json_call(
        client,
        messages=build_s2_messages(
            meta_skills["S2"],
            s1,
            _op_registry_meta(operator_registry),
            num_skill_sets=num_skill_sets,
            skills_per_set=skills_per_set,
        ),
        model=model,
        temperature=0.8,
        max_tokens=30000,
        cache_dir=cache_dir,
        stage=f"{stage_prefix}_s2_multi_retrieve",
        out_fail_dir=fail_dir,
        key_index=key_index,
    )
    if s2 is None:
        return {**result, "status": "failed", "error_stage": "s2"}

    # Clean and obtain the specified skill sets
    skill_sets = _normalize_skill_sets(
        s2,
        valid_operator_ids=valid_operator_ids,
        num_skill_sets=num_skill_sets,
        skills_per_set=skills_per_set,
    )
    if not skill_sets:
        return {**result, "status": "failed", "error_stage": "s2_empty"}
    result["skill_sets"] = skill_sets

    # ================= Stage 3: Generate optimization strategies =================
    print(f"🔖 Stage 3: Generate optimization strategies --- (Strategy Planning) for problem {problem_id} ---")
    all_plans: List[Dict[str, Any]] = []
    for ss in skill_sets:
        set_id = ss["set_id"]
        # Assemble the full text content of all concrete skills in the current skill set
        skill_bodies = []
        for item in ss["selected_skills"]:
            sid = item["skill_id"]
            body = skills_index.get(sid, {}).get("content")
            if body:
                skill_bodies.append(body)

        s3 = llm_json_call(
            client,
            messages=build_s3_messages(
                meta_skills["S3"], statement, slow_code, s1, ss, skill_bodies
            ),
            model=model,
            temperature=0.8,
            max_tokens=35000,
            cache_dir=cache_dir,
            stage=f"{stage_prefix}_s3_plan_{set_id}",
            out_fail_dir=fail_dir,
            key_index=key_index,
        )
        plans = _normalize_plans(s3 if isinstance(s3, dict) else {}, set_id=set_id)
        all_plans.extend(plans)

    if not all_plans:
        return {**result, "status": "failed", "error_stage": "s3_empty"}
    result["plans"] = all_plans

    # ================= Stage 4: Generate candidate code according to plans =================
    alloc = _allocate_plan_slots(all_plans, k)
    candidates: List[Dict[str, Any]] = []
    cid = 1

    for plan, cnt in zip(all_plans, alloc):
        for j in range(cnt):
            # Adjust the generation temperature according to the plan strategy style (Temperature)
            temp = 0.5
            if plan.get("plan_style") == "balanced":
                temp = 0.75
            elif plan.get("plan_style") == "aggressive":
                temp = 1.0

            gen = llm_json_call(
                client,
                messages=build_generate_messages(statement, slow_code, plan),
                model=model,
                temperature=temp,
                max_tokens=35000,
                cache_dir=cache_dir,
                stage=f"{stage_prefix}_s4_gen_{plan['plan_id']}_{j+1}",
                out_fail_dir=fail_dir,
                key_index=key_index,
            )
            if not isinstance(gen, dict):
                cid += 1
                continue

            code = _clean_code(gen.get("code"))
            if not code:
                cid += 1
                continue

            cand = {
                "candidate_id": f"C{cid}",
                "set_id": plan.get("set_id"),
                "plan_id": plan.get("plan_id"),
                "plan_style": plan.get("plan_style"),
                "code": code,
                "complexity": gen.get("complexity"),
                "dominant_ops": gen.get("dominant_ops"),
                "notes": gen.get("notes"),
            }

            # --- S4.5: Optional post-processing repair stage (Sanity Fix) ---
            cand["sanity_fixes"] = []
            if sanity_fix:
                sanity = llm_json_call(
                    client,
                    messages=build_sanity_messages(statement, slow_code, cand["code"]),
                    model=model,
                    temperature=0.15, # Use a very low temperature in the repair stage for stability
                    max_tokens=20000,
                    cache_dir=cache_dir,
                    stage=f"{stage_prefix}_s45_sanity_{cand['candidate_id']}",
                    out_fail_dir=fail_dir,
                    key_index=key_index,
                )
                if isinstance(sanity, dict) and _clean_code(sanity.get("code")):
                    cand["code"] = _clean_code(sanity.get("code"))
                    cand["sanity_fixes"] = sanity.get("fixes", [])

            # --- Static integrity verification ---
            # 1. Get model self-evaluation metrics
            cand["static_verdict"] = _clean_text(gen.get("static_verdict")).lower() or "unknown"
            cand["interface_match_model"] = bool(gen.get("interface_match", False))
            cand["complexity_plausible"] = bool(gen.get("complexity_plausible", False))
            cand["static_issues"] = (
                list(gen.get("static_issues", [])) if isinstance(gen.get("static_issues"), list) else []
            )
            cand["static_confidence"] = gen.get("static_confidence", 0.0)

            # 2. Perform strict local interface matching checks
            local_interface_match, local_interface_issues = _check_interface_match(
                slow_code, cand["code"]
            )
            cand["interface_match_local"] = local_interface_match
            # Both model self-evaluation and local checks must pass for the final interface match
            cand["interface_match"] = cand["interface_match_model"] and local_interface_match
            
            # Append locally found issues to the statistics
            for issue in local_interface_issues:
                if issue not in cand["static_issues"]:
                    cand["static_issues"].append(issue)
            # If the local interface check fails, force the overall verdict to fail
            if not local_interface_match:
                cand["static_verdict"] = "fail"

            # 3. Basic syntax validity check
            syntax_err = _python_syntax_error(cand["code"])
            cand["syntax_valid"] = syntax_err is None
            if syntax_err:
                if syntax_err not in cand["static_issues"]:
                    cand["static_issues"].append(syntax_err)
                cand["static_verdict"] = "fail"

            candidates.append(cand)
            cid += 1

    if not candidates:
        return {**result, "status": "failed", "error_stage": "s4_empty"}

    # Filter the final surviving valid code
    survivors = [
        c for c in candidates if c.get("static_verdict") == "pass" and c.get("interface_match")
    ]
    if require_syntax:
        survivors = [c for c in survivors if c.get("syntax_valid", False)]

    result["candidates"] = candidates
    result["survivor_candidates"] = survivors
    result["candidate_count"] = len(candidates)
    result["survivor_count"] = len(survivors)

    # Mark the edge case where code was generated but none survived
    if not survivors:
        result["status"] = "no_survivor"

    return result





# #####################################################################################################################🔖💡✅🟨❌
if __name__ == "__main__":
    main()