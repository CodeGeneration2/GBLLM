"""No-execution inference pipeline (skill-first, multi-skill-set, no ranking) - C++ version.

Pipeline flow:
- S1-lite diagnosis (perform static diagnosis on the original code and analyze bottlenecks)
- One-pass model routing (select multiple different optimization skill sets based on metadata)
- Create strategies for each skill set (three optimization plans: conservative/balanced/aggressive)
- Generate K candidate code versions
- Static completeness check + static correctness filtering (C++ syntax and interface matching checks)
- Output all surviving candidates (without ranking them)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only

import re
import shutil
import subprocess
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from API__Single_Generation import deepseek_dual_platform_function # Used to show a progress bar in the terminal for easier task monitoring
from API__Single_Generation import codellama_deepinfra_function
from API__Single_Generation import gemini_official_function
from API__Single_Generation import gemini_yunwu_api_function
from API__Single_Generation import chatgpt_three_platform_function
from API__Single_Generation import deepseek_dual_platform_function
from API__Single_Generation import codellama_server_model_standard_load_function
from API__Single_Generation import codellama_server_standard_inference_function

# Environment-variable control: whether the cache includes the full payload.
CACHE_INCLUDE_PAYLOAD = os.getenv("SELF_EVOLVE_CACHE_INCLUDE_PAYLOAD", "0").lower() in {
    "1",
    "true",
    "yes",
}
CODE_FENCE = "cpp"
LANGUAGE_NAME = "C++"


# #####################################################################################################################🔖💡✅🟨❌
DeBug = False
# To match C++, the default file name is changed to Cpp here; adjust it as needed.
dataset_path  = r"./PIE_Cpp_009_Gemini__X.csv"
save_dataset_path  = r""
workers = 1


# #####################################################################################################################🔖💡✅🟨❌
# Define the list of models to test.
if "CodeLlama13B" in dataset_path:
    model_name = "./CodeLlama-13b-Instruct-hf"
    codellama_tokenizer, codellama_model = codellama_server_model_standard_load_function(resolved_model_name=model_name)
elif "CodeLlama34B" in dataset_path:
    model_name = "./CodeLlama-34b-Instruct-hf"
    codellama_tokenizer, codellama_model = codellama_server_model_standard_load_function(resolved_model_name=model_name)
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
    _save_dataset_path = f"{prefix_path}__EffiSkill_generated.jsonl"

    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="No-exec multi-skill-set inference pipeline for C++")

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
    
    parser.add_argument("--out", default=_save_dataset_path)
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
    parser.add_argument("--fail-dir", default="PIE_output/llm_failures_infer_cpp")

    parser.add_argument("--llm-choice", choices=["gpt-5-mini", "qwen30b"], default="gpt-5-mini")
    parser.add_argument("--base-url", default="custom", help="OpenAI-compatible base URL; can also come from env.")
    parser.add_argument("--api-key", default="custom", help="Provider API key; can also come from env.")
    parser.add_argument("--model", default=model_name, help="Model name; can also come from env.")

    # Key hyperparameters
    parser.add_argument("--k", type=int, default=5, help="Total candidates per problem (Number of generated code candidates per problem)")
    parser.add_argument("--skill-sets", type=int, default=3, help="Number of skill sets from S2 (Number of skill combinations extracted in the S2 stage)")
    parser.add_argument("--skills-per-set", type=int, default=6, help="Maximum number of skills in one combination")

    parser.add_argument("--workers", type=int, default=workers, help="Number of worker threads for concurrent execution")
    parser.add_argument(
        "--sanity-fix",
        action="store_true",
        help="Enable extra static sanity-fix stage before static check (Enable the post-processing repair stage)",
    )
    parser.add_argument(
        "--require-syntax",
        action="store_true",
        help="Deprecated no-op: syntax diagnostics are still recorded, but candidates are not filtered out.",
    )
    parser.add_argument("--limit", type=int, default=-1, help="-1 means full dataset (Data processing limit)")
    parser.add_argument("--start", type=int, default=0, help="Processing start index")
    parser.add_argument("--problem-id", default=None, help="Run for a specified single problem")

    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Support resumable execution")
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    return parser.parse_args()





# #####################################################################################################################🔖💡✅🟨❌
def main() -> None:
    args = parse_args()

    # Initialize configuration
    print(f"🚀 Initialize configuration, LLM_base_url: {args.base_url}, llm_api_key: {args.api_key}, model_name: {args.model}")
    try:
        test_dataset_path, slow_code_column_name = resolve_dataset_config(args)
        LLM_base_url, llm_api_key, resolved_model_name = resolve_model_config(args)
    except ValueError as exc:
        raise SystemExit("❌❌❌ " + str(exc)) from exc

    # Ensure that the parent directory of the output file exists and create the failure log directory.
    _ensure_parent_dir(args.out)
    os.makedirs(args.fail_dir, exist_ok=True)

    # Preload skill data and index structures.
    print(f"📚 Loading skill registry and index from {args.skills_root}")
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

    # Scan existing result files for resumable execution.
    print(f"🔄 Scanning existing results for resume: {args.out}")
    seen_ids = _load_existing_ids(args.out) if args.resume else set()

    # Pre-filter all task rows to be processed.
    print(f"📑 Pre-filtering dataset, path: {test_dataset_path}, statement_column: {args.statement_column}, slow_code_column: {slow_code_column_name}")
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

    out_lock = threading.Lock()
    workers = max(1, int(args.workers))

    # Worker closure
    def _process_row(row: Dict[str, str], key_index: int=0) -> Dict[str, Any]:
        problem_id = row.get("problem_id") or ""
        statement = row.get(args.statement_column, "")
        slow_code_text = row.get(slow_code_column_name, "")

        client = create_client(base_url=LLM_base_url, api_key=llm_api_key)
        result = run_problem(
            problem_id=problem_id,
            statement=statement,
            slow_code=slow_code_text,
            model=resolved_model_name,
            client=client,
            cache_dir="", 
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
        return {
            "problem_id": problem_id,
            "dataset": args.dataset,
            "data_path": test_dataset_path,
            "statement": statement,
            "slow_code": slow_code_text,
            "model": resolved_model_name,
            **result,
        }

    def _failed_record(row: Dict[str, str], exc: Exception) -> Dict[str, Any]:
        return {
            "problem_id": row.get("problem_id"),
            "dataset": args.dataset,
            "data_path": test_dataset_path,
            "statement": row.get(args.statement_column, ""),
            "slow_code": row.get(slow_code_column_name, ""),
            "model": resolved_model_name,
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

    print(f"🖊️  Start processing {len(rows)} records, output path: {args.out}, workers: {workers}")
    with open(args.out, "a", encoding="utf-8") as out_f:
        if workers == 1:
            for row in tqdm(rows):
                try:
                    rec = _process_row(row)
                except Exception as exc:  # noqa: BLE001
                    rec = _failed_record(row, exc)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                _log_done(rec)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_row, row, idx % workers): row for idx, row in enumerate(rows)}

                for fut in tqdm(as_completed(futs)):
                    row = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = _failed_record(row, exc)
                    
                    with out_lock:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                    _log_done(rec)

    # [End]---------------------------------------------------------------------------------------------
    print("🎉 All processing completed！")
    df_test_dataset = pd.read_csv(dataset_path)
    try:
        df_generated_dataset = pd.read_json(args.out, lines=True, encoding='utf-8')
    except:
        generated_dataset_records = []
        with open(args.out, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    generated_dataset_records.append(json.loads(line))
        df_generated_dataset = pd.DataFrame(generated_dataset_records)

    # ========================== New: expand candidate code columns ==========================
    def get_code_by_cid(candidates_list, target_cid):
        if isinstance(candidates_list, list):
            for cand in candidates_list:
                if cand.get('candidate_id') == target_cid:
                    return cand.get('code', '')
        return ''

    for i in range(1, 6):
        target_cid = f"C{i}"
        new_col_name = f"EffiSkill_G5__Predict_Fast_code_{i}"
        df_generated_dataset[new_col_name] = df_generated_dataset['candidates'].apply(lambda x: get_code_by_cid(x, target_cid))
    # ==============================================================================

    # ========================== Merge two tables by column ==========================
    if "615_idx" in df_test_dataset.columns:
        test_dataset_index = '615_idx'
    elif "712_idx" in df_test_dataset.columns:
        test_dataset_index = '712_idx'
    elif "1000_idx" in df_test_dataset.columns:
        test_dataset_index = '1000_idx'
    
    df_test_dataset[test_dataset_index] = df_test_dataset[test_dataset_index].astype(str)
    df_generated_dataset['problem_id'] = df_generated_dataset['problem_id'].astype(str)

    df_merged_dataset = pd.merge(
        df_test_dataset, 
        df_generated_dataset, 
        left_on=test_dataset_index,     
        right_on='problem_id', 
        how='left'             
    )
    # ==============================================================================

    if save_dataset_path == '':
        prefix_path = '__'.join(dataset_path.split('__')[:-1])
        pie_id = dataset_path.split('__')[-2].split('_')[-2]
        prefix_path = prefix_path.replace(pie_id, str(int(pie_id)+1).zfill(3))
        _save_dataset_path = f"{prefix_path}__EffiSkill_generated.csv"
    else:
        _save_dataset_path = save_dataset_path

    df_merged_dataset.to_csv(f'{_save_dataset_path}', index=False)
    print("Conversion completed！")




# -----------------------------
# Basic I/O helpers
# -----------------------------
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _iter_rows(path: str) -> Iterable[Dict[str, str]]:
    df = pd.read_csv(path)
    if "615_idx" in df.columns:
        test_dataset_index = '615_idx'
    elif "712_idx" in df.columns:
        test_dataset_index = '712_idx'
    elif "1000_idx" in df.columns:
        test_dataset_index = '1000_idx'
        
    for _, row in df.iterrows():
        yield {
            "problem_id": str(row[test_dataset_index]), 
            "description": "None",             
            "best_before": str(row["input"])   
        }


def _is_resume_terminal_status(status: Any) -> bool:
    return _clean_text(status).lower() in {"ok", "no_survivor"}

def _load_existing_ids(out_path: str) -> Set[str]:
    if not os.path.exists(out_path):
        return set()
    seen: Set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("problem_id")
            if pid and _is_resume_terminal_status(obj.get("status")):
                seen.add(pid)
    return seen

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# -----------------------------
# Skill loading
# -----------------------------
def _find_skill_id(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().startswith("skill_id:"):
            return line.split(":", 1)[1].strip()
    return None

def load_skills_index(skills_root: str) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for root, _, files in os.walk(skills_root):
        if "SKILL.md" not in files:
            continue
        path = os.path.join(root, "SKILL.md")
        text = _read_text(path)
        sid = _find_skill_id(text)
        if not sid:
            continue
        index[sid] = {"path": path, "content": text}
    return index

def load_registry(skills_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(skills_root, "registry.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# LLM utilities
# -----------------------------
def create_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

def _hash_payload(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _read_cache(cache_path: str) -> Optional[str]:
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
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_obj: Dict[str, Any] = {"content": content}
    if CACHE_INCLUDE_PAYLOAD:
        cache_obj["payload"] = payload
    else:
        cache_obj["payload_hash"] = _hash_payload(payload)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)

def _is_non_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in {400, 401, 403, 404}

def _is_rate_limited_error(exc: Exception) -> bool:
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
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    _ = cache_dir
    _ = stage

    max_attempts = 4
    last_exc: Optional[Exception] = None
    response = None
    
    for attempt in range(max_attempts):
        try:
            response = generate_data_with_api_function(messages[0]['content'], messages[1]['content'], key_index = key_index, temperature = temperature, max_length = 1024, should_print_prompt = DeBug,)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_non_retryable_error(exc):
                raise
            
            if attempt < max_attempts - 1:
                if _is_rate_limited_error(exc):
                    delay = min(20.0, 2.0 * (attempt + 1))
                else:
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
    if model_name in ["./CodeLlama-34b-Instruct-hf", "./CodeLlama-13b-Instruct-hf"]:
        model_response_text_list = codellama_server_standard_inference_function( model=codellama_model,   
                                                        input_role_prompt=input_role_prompt, 
                                                        input_question_text=input_question_str, 
                                                        generated_code_count=1, 
                                                        temperature=temperature, 
                                                        max_length=max_length,
                                                        should_print_prompt=should_print_prompt,
                                                        token_vocabulary=codellama_tokenizer,
                                                        )

    elif model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = codellama_deepinfra_function(key_index=key_index, 
                                        resolved_model_name=model_name,   
                                        input_role_prompt=input_role_prompt, 
                                        input_question_text=input_question_str, 
                                        generated_code_count=1, 
                                        temperature=temperature, 
                                        max_length=max_length,
                                        should_print_prompt=should_print_prompt,
                                        )

    elif "gemini" in model_name:
        model_response_text_list = gemini_yunwu_api_function(  platform="YunwuAPI", 
                                                key_index = key_index, 
                                                resolved_model_name = model_name, 
                                                input_role_prompt=input_role_prompt, 
                                                input_question_text=input_question_str, 
                                                generated_code_count = 1,  
                                                temperature = temperature, 
                                                max_length = max_length, 
                                                should_print_prompt = should_print_prompt,
                                                )

    elif False and "gemini" in model_name:
        model_response_text_list, average_logprob_list = gemini_official_function(key_index=key_index, 
                                                            resolved_model_name=model_name,   
                                                            input_role_prompt=input_role_prompt, 
                                                            input_question_text=input_question_str, 
                                                            generated_code_count=1, 
                                                            temperature=temperature, 
                                                            max_length="no_limit", 
                                                            thinking_budget=-404,
                                                            return_logprob=False,
                                                            should_print_prompt=should_print_prompt,
                                                            )

    elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4o-mini", "gpt-4.1-nano"]:
        model_response_text_list, average_logprob_list = chatgpt_three_platform_function(platform = "YunwuAPI", 
                                                                key_index = key_index, 
                                                                resolved_model_name = model_name, 
                                                                input_role_prompt=input_role_prompt, 
                                                                input_question_text=input_question_str, 
                                                                generated_code_count = 1, 
                                                                temperature = temperature, 
                                                                max_length = max_length, 
                                                                return_logprob=False, 
                                                                should_print_prompt=should_print_prompt,
                                                                )

    elif model_name in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-V3.2-Exp"]:
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            _platform = "DeepSeekOfficial"
        elif model_name in ["deepseek-ai/DeepSeek-V3.2-Exp"]:
            _platform = "SiliconFlowAPI"
        model_response_text_list, full_model_response_dict, chain_of_thought_text, average_logprob = deepseek_dual_platform_function( platform = _platform, 
                                                                                            key_index = key_index, 
                                                                                            resolved_model_name = model_name, 
                                                                                            input_role_prompt = input_role_prompt, 
                                                                                            input_question_text = input_question_str, 
                                                                                            temperature = temperature, 
                                                                                            max_length = max_length, 
                                                                                            should_print_prompt = should_print_prompt,
                                                                                            )
        
    return model_response_text_list


def log_llm_failure(stage: str, payload: Dict[str, Any], response_text: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{stage}_{ts}_{_hash_payload(payload)[:8]}"
    jpath = os.path.join(out_dir, f"json_fail_{base}.json")
    tpath = os.path.join(out_dir, f"json_fail_{base}.txt")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"payload": payload}, f, ensure_ascii=False, indent=2)
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(response_text)

def extract_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*", "", cleaned).strip("`\n ")
    s_obj = cleaned.find("{")
    s_arr = cleaned.find("[")
    starts = [x for x in (s_obj, s_arr) if x != -1]
    if not starts:
        return None
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        return None
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
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

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

    obj = extract_json(content)
    if obj is not None:
        return obj

    fix_messages = [
        {
            "role": "system",
            "content": "Fix this to valid strict JSON. Return JSON only.",
        },
        {"role": "user", "content": content},
    ]
    fix_payload = {
        "model": model,
        "temperature": 1.0,
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

    obj = extract_json(fix_content)
    if obj is not None:
        return obj

    log_llm_failure(stage + "_json", fix_payload, fix_content, out_fail_dir)
    return None


# -----------------------------
# Prompt builders (C++ specific)
# -----------------------------
def build_s1_messages(meta_skill: str, statement: str, slow_code: str) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        f"""
        You are a C++ code optimization diagnostician.
        Follow the meta skill guidance below.

        {meta_skill}

        Runtime-input constraint:
        - At inference time, you only have the problem statement and baseline code shown below.
        - Do not assume access to trace aggregates, cluster statistics, or historical examples unless they are explicitly present in the prompt.
        - Infer constraints, bottlenecks, and required optimization scope directly from the statement and code.

        Return strict JSON only.
        Required fields:
        - constraints_guess: {{n_max, q_max, notes}}
        - problem_type_tags: [string]
        - dominant_ops: [string]
        - complexity: {{time, space, why}}
        - bottlenecks: [{{kind, detail}}]
        - must_change: "algorithm"|"implementation"|"mixed"
        - risks: [string]
        - confidence: "high"|"medium"|"low"
        """
    ).strip()

    user = textwrap.dedent(
        f"""
        Problem statement:
        {statement}

        Baseline code:
        ```{CODE_FENCE}
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
    system = textwrap.dedent(
        f"""
        You are a skill router for C++ optimization.
        Follow the meta skill guidance below.

        {meta_skill}

        Runtime-input constraint:
        - At inference time, you only have the OptimizationBrief and the operator registry metadata shown below.
        - Do not assume access to global trace stats, hidden task features, or any other side inputs unless they are explicitly present in the prompt.
        - Route using only the provided diagnosis plus registry fields such as description, family, tags, aliases, and triggers.

        Select multiple DIFFERENT skill sets in one pass.
        Return strict JSON only.
        Required fields:
        - skill_sets: [
            {{
              set_id,
              confidence: "high"|"medium"|"low",
              selection_theme,
              selected_skills: [{{skill_id, reason}}]
            }}
          ]
        - global_coverage_gaps: [string]

        Constraints:
        - Produce exactly {num_skill_sets} skill sets.
        - Max {skills_per_set} skills per set.
        - Prefer skill sets that are valid for C++.
        - Match against description, family, tags, aliases, and triggers.
        - Respect must_change from the OptimizationBrief:
          - prefer algorithmic/structural operators when must_change is "algorithm";
          - prefer implementation/constant-factor operators when must_change is "implementation";
          - mix both only when must_change is "mixed".
        - Skill sets should be meaningfully different in optimization mechanism.
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
              plan_style: "conservative"|"balanced"|"aggressive",
              name,
              core_idea,
              target_complexity,
              risks: [string],
              steps: [string]
            }}
          ]

        Constraints:
        - Produce exactly 3 plans: conservative, balanced, aggressive.
        - All plans must preserve the external code interface style.
        Think step by step to get better plan.
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        Problem statement:
        {statement}

        Baseline code:
        ```{CODE_FENCE}
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
            CODE_FENCE=CODE_FENCE,
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
    system = textwrap.dedent(
        """
        Generate optimized C++ code for the given plan.
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
        - Preserve external interface style and exposed entry points exactly.
          Examples:
          - if input is class/method style, keep class/method style;
          - if input is top-level function style, keep function style;
          - if input is main/stdin/stdout style, keep script style.
        - Do not convert between these styles.

        Header/style guidance (soft):
        - Try to preserve the input file header/comments, include directives,
          namespace usage, and overall top-level layout unless a change
          is necessary for correctness or the optimization plan.
        - Avoid unnecessary reformatting or cosmetic rewrites.

        Assume baseline logic is valid and keep semantics.
        Think step by step to get better code.
        """
    ).strip()

    user = (
        textwrap.dedent(
            """
        Problem statement:
        {statement}

        Baseline code:
        ```{CODE_FENCE}
        {slow_code}
        ```

        Plan:
        {plan}
        """
        )
        .strip()
        .format(
            CODE_FENCE=CODE_FENCE,
            statement=statement,
            slow_code=slow_code,
            plan=json.dumps(plan, ensure_ascii=False),
        )
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_sanity_messages(
    statement: str, slow_code: str, candidate_code: str
) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        """
        Perform static sanity cleanup only (no execution).
        Preserve external interface style exactly as baseline.
        Softly preserve file header/comments/include structure from baseline unless
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
        ```{CODE_FENCE}
        {slow_code}
        ```

        Candidate code:
        ```{CODE_FENCE}
        {code}
        ```
        """
        )
        .strip()
        .format(
            CODE_FENCE=CODE_FENCE,
            statement=statement,
            slow_code=slow_code,
            code=candidate_code,
        )
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -----------------------------
# Normalization & C++ AST helpers
# -----------------------------
def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip()
    return re.sub(r"\s+", " ", str(x)).strip()


def _clean_code(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    return x.strip("\n")


def _strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:cpp|c\+\+|cc|python|py)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned



def _is_public_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _remove_cpp_comments_and_strings(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r'R"([^()]*)\((.*?)\)\1"', '""', code, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)
    return code


def _find_matching_brace(code: str, open_idx: int) -> Optional[int]:
    depth = 0
    for idx in range(open_idx, len(code)):
        ch = code[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _normalize_cpp_signature(sig: str) -> str:
    sig = re.sub(r"\s+", " ", sig).strip()
    sig = re.sub(r"\s*([(),*&<>:=])\s*", r"\1", sig)
    return sig


def _extract_cpp_param_block(signature: str) -> str:
    end = signature.rfind(")")
    start = signature.rfind("(", 0, end + 1)
    if start == -1 or end == -1 or end <= start:
        return ""
    return signature[start + 1 : end]


def _extract_cpp_decl_name(signature: str) -> Optional[str]:
    end = signature.rfind(")")
    start = signature.rfind("(", 0, end + 1)
    if start == -1:
        return None
    prefix = signature[:start].strip()
    if not prefix:
        return None
    match = re.search(r"([A-Za-z_~][A-Za-z0-9_:~]*)\s*$", prefix)
    if not match:
        return None
    name = match.group(1).split("::")[-1]
    blocked = {"if", "for", "while", "switch", "catch", "return"}
    if name in blocked:
        return None
    return name


def _extract_top_level_cpp_entities(
    code: str,
) -> Tuple[Dict[str, str], Dict[str, Tuple[bool, str]], bool]:
    cleaned = _remove_cpp_comments_and_strings(code)
    functions: Dict[str, str] = {}
    classes: Dict[str, Tuple[bool, str]] = {}
    has_main = False
    depth = 0
    stmt_start = 0
    idx = 0

    while idx < len(cleaned):
        ch = cleaned[idx]
        if ch == "{":
            if depth == 0:
                prefix = cleaned[stmt_start:idx]
                prefix = re.sub(r"(?m)^\s*#.*$", "", prefix).strip()
                if not prefix:
                    depth += 1
                    idx += 1
                    continue
                lines = [line.strip() for line in prefix.splitlines() if line.strip()]
                last_line = lines[-1] if lines else prefix
                class_match = re.match(r"^(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b", last_line)
                if class_match:
                    name = class_match.group(2)
                    is_struct = class_match.group(1) == "struct"
                    close_idx = _find_matching_brace(cleaned, idx)
                    if close_idx is not None:
                        classes[name] = (is_struct, cleaned[idx + 1 : close_idx])
                elif "(" in prefix and ")" in prefix:
                    name = _extract_cpp_decl_name(prefix)
                    if name and _is_public_name(name):
                        signature = _normalize_cpp_signature(prefix)
                        functions[name] = signature
                        if name == "main":
                            has_main = True
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                stmt_start = idx + 1
        elif ch == ";" and depth == 0:
            stmt_start = idx + 1
        idx += 1

    return functions, classes, has_main


def _extract_cpp_public_methods(class_body: str, *, default_public: bool) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    access_public = default_public
    body = _remove_cpp_comments_and_strings(class_body)

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        access_match = re.fullmatch(r"(public|private|protected)\s*:\s*", line)
        if access_match:
            access_public = access_match.group(1) == "public"
            continue
        if not access_public:
            continue
        if "(" not in line or ")" not in line:
            continue
        if not line.endswith(("{", ";")):
            continue
        name = _extract_cpp_decl_name(line)
        if not name or name in {"if", "for", "while", "switch", "catch"}:
            continue
        if name.startswith("operator"):
            continue
        methods[name] = _normalize_cpp_signature(line.rstrip("{;").strip())

    return methods


def _extract_interface_contract(code: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fence(code)
    if not cleaned.strip():
        return None

    functions, classes_raw, has_main = _extract_top_level_cpp_entities(cleaned)
    classes: Dict[str, Dict[str, str]] = {}
    for name, (is_struct, body) in classes_raw.items():
        classes[name] = _extract_cpp_public_methods(body, default_public=is_struct)

    top_level_functions = {k: v for k, v in functions.items() if k != "main"}
    if classes and not top_level_functions and not has_main:
        style = "class"
    elif top_level_functions and not classes and not has_main:
        style = "function"
    elif has_main and not classes and not top_level_functions:
        style = "script"
    elif has_main:
        style = "script"
    elif not classes and not top_level_functions:
        style = "script"
    else:
        style = "mixed"

    return {
        "style": style,
        "functions": top_level_functions,
        "classes": classes,
        "has_main": has_main,
    }


def _compiler_for_cpp() -> Optional[str]:
    for candidate in ("g++", "clang++", "c++"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _fallback_cpp_syntax_error(code: str) -> Optional[str]:
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = {v: k for k, v in pairs.items()}
    stack: List[Tuple[str, int]] = []
    for idx, ch in enumerate(_remove_cpp_comments_and_strings(code)):
        if ch in pairs:
            stack.append((ch, idx))
        elif ch in closing:
            if not stack or stack[-1][0] != closing[ch]:
                return f"C++ syntax looks malformed near character {idx}: unmatched `{ch}`"
            stack.pop()
    if stack:
        return f"C++ syntax looks malformed near character {stack[-1][1]}: unmatched `{stack[-1][0]}`"
    return None


def _cpp_syntax_error(code: str) -> Optional[str]:
    cleaned = _strip_code_fence(code)
    compiler = _compiler_for_cpp()
    if compiler is None:
        return _fallback_cpp_syntax_error(cleaned)

    try:
        proc = subprocess.run(
            [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
            input=cleaned,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return f"C++ syntax check failed to run: {exc}"

    if proc.returncode == 0:
        return None
    stderr = _clean_text(proc.stderr or proc.stdout)
    if "bits/stdc++.h" in stderr and ("not found" in stderr or "no such file" in stderr.lower()):
        return _fallback_cpp_syntax_error(cleaned)
    return stderr or "C++ syntax check failed"


def _cpp_signature_compatible(expected: str, actual: str) -> bool:
    expected_name = _extract_cpp_decl_name(expected) or ""
    actual_name = _extract_cpp_decl_name(actual) or ""
    if expected_name != actual_name:
        return False
    expected_params = _normalize_cpp_signature(_extract_cpp_param_block(expected))
    actual_params = _normalize_cpp_signature(_extract_cpp_param_block(actual))
    if expected_params != actual_params:
        return False
    expected_const = " const" in f" {expected} "
    actual_const = " const" in f" {actual} "
    return expected_const == actual_const


def _check_interface_match(baseline_code: str, candidate_code: str) -> Tuple[bool, List[str]]:
    baseline = _extract_interface_contract(baseline_code)
    candidate = _extract_interface_contract(candidate_code)

    if baseline is None:
        return True, []
    if candidate is None:
        return False, ["Candidate interface could not be verified from the generated C++ code."]

    issues: List[str] = []
    baseline_style = baseline["style"]
    candidate_style = candidate["style"]

    if baseline.get("has_main") and not candidate.get("has_main"):
        issues.append("Candidate no longer exposes the baseline main()-based script interface.")
    if baseline_style == "class" and not candidate["classes"]:
        issues.append("Candidate no longer exposes the baseline class-style interface.")
    if baseline_style == "function" and not candidate["functions"]:
        issues.append("Candidate no longer exposes the baseline function-style interface.")
    if baseline_style in {"function", "script"} and candidate_style == "class":
        issues.append(f"Candidate changed interface style from {baseline_style} to class.")

    if baseline_style in {"function", "mixed"}:
        for func_name, expected_sig in baseline["functions"].items():
            actual_sig = candidate["functions"].get(func_name)
            if actual_sig is None:
                issues.append(f"Missing top-level function `{func_name}` from baseline interface.")
                continue
            if not _cpp_signature_compatible(expected_sig, actual_sig):
                issues.append(f"Function signature drift for `{func_name}`.")

    if baseline_style in {"class", "mixed"}:
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
                if not _cpp_signature_compatible(expected_sig, actual_sig):
                    issues.append(f"Method signature drift for `{class_name}.{method_name}`.")

    return not issues, issues


def _op_registry_meta(operator_registry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in operator_registry:
        out.append(
            {
                "skill_id": r.get("skill_id"),
                "name": r.get("name"),
                "description": r.get("description"),
                "family": r.get("family"),
                "tags": (r.get("tags") or [])[:10],
                "aliases": (r.get("aliases") or [])[:8],
                "triggers": (r.get("triggers") or [])[:6],
            }
        )
    return out


def _normalize_skill_sets(
    raw: Any,
    valid_operator_ids: Set[str],
    num_skill_sets: int,
    skills_per_set: int,
) -> List[Dict[str, Any]]:
    skill_sets = []
    if isinstance(raw, dict):
        skill_sets = raw.get("skill_sets") or []

    norm: List[Dict[str, Any]] = []
    used_signatures = set()

    for idx, ss in enumerate(skill_sets, start=1):
        if not isinstance(ss, dict):
            continue
        selected = ss.get("selected_skills") or []
        cleaned_selected = []
        seen = set()
        for item in selected:
            if not isinstance(item, dict):
                continue
            sid = _clean_text(item.get("skill_id"))
            if sid not in valid_operator_ids or sid in seen:
                continue
            seen.add(sid)
            cleaned_selected.append(
                {
                    "skill_id": sid,
                    "reason": _clean_text(item.get("reason")) or "matched diagnosis",
                }
            )
            if len(cleaned_selected) >= skills_per_set:
                break

        if not cleaned_selected:
            continue

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
    return norm[:num_skill_sets]


def _normalize_plan_style(x: Any) -> str:
    s = _clean_text(x).lower()
    if "conserv" in s:
        return "conservative"
    if "aggr" in s:
        return "aggressive"
    return "balanced"


def _normalize_plans(raw: Any, set_id: str) -> List[Dict[str, Any]]:
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
                "risks": [_clean_text(x) for x in (p.get("risks") or []) if _clean_text(x)][:6],
                "steps": [_clean_text(x) for x in (p.get("steps") or []) if _clean_text(x)][:8],
            }
        )

    style_to_plan = {p["plan_style"]: p for p in norm}
    for style in ("conservative", "balanced", "aggressive"):
        if style in style_to_plan:
            continue
        norm.append(
            {
                "plan_id": f"{set_id}_{style[:3].upper()}",
                "set_id": set_id,
                "plan_style": style,
                "name": f"{style.title()} Plan",
                "core_idea": "Apply selected skills while preserving interface and semantics.",
                "target_complexity": "(improved vs baseline where applicable)",
                "risks": ["Static-only validation may miss subtle bugs."],
                "steps": ["Apply the selected skills coherently to baseline code."],
            }
        )

    style_order = {"conservative": 0, "balanced": 1, "aggressive": 2}
    norm = sorted(norm, key=lambda p: (style_order.get(p["plan_style"], 9), p["plan_id"]))
    return norm


def _allocate_plan_slots(plans: List[Dict[str, Any]], k: int) -> List[int]:
    if k <= 0 or not plans:
        return []
    n = len(plans)
    alloc = [0] * n
    if n <= k:
        for i in range(n):
            alloc[i] = 1
        remain = k - n
        i = 0
        while remain > 0:
            alloc[i % n] += 1
            i += 1
            remain -= 1
        return alloc

    order = sorted(
        range(n),
        key=lambda i: {"aggressive": 0, "balanced": 1, "conservative": 2}.get(
            plans[i].get("plan_style", "balanced"), 9
        ),
    )
    for i in order[:k]:
        alloc[i] = 1
    return alloc


# #####################################################################################################################🔖💡✅🟨❌
# -----------------------------
# Core problem runner
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
    result: Dict[str, Any] = {"status": "ok"}

    stage_prefix = f"{problem_id or 'unknown'}"

    # S1 Diagnose
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

    # S2 One-pass multi-set retrieval
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

    skill_sets = _normalize_skill_sets(
        s2,
        valid_operator_ids=valid_operator_ids,
        num_skill_sets=num_skill_sets,
        skills_per_set=skills_per_set,
    )
    if not skill_sets:
        return {**result, "status": "failed", "error_stage": "s2_empty"}
    result["skill_sets"] = skill_sets

    # S3 Plans per skill set
    all_plans: List[Dict[str, Any]] = []
    for ss in skill_sets:
        set_id = ss["set_id"]
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

    # S4 Generate candidates
    alloc = _allocate_plan_slots(all_plans, k)
    candidates: List[Dict[str, Any]] = []
    cid = 1

    for plan, cnt in zip(all_plans, alloc):
        for j in range(cnt):
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

            cand["sanity_fixes"] = []
            if sanity_fix:
                sanity = llm_json_call(
                    client,
                    messages=build_sanity_messages(statement, slow_code, cand["code"]),
                    model=model,
                    temperature=0.15,
                    max_tokens=20000,
                    cache_dir=cache_dir,
                    stage=f"{stage_prefix}_s45_sanity_{cand['candidate_id']}",
                    out_fail_dir=fail_dir,
                    key_index=key_index,
                )
                if isinstance(sanity, dict) and _clean_code(sanity.get("code")):
                    cand["code"] = _clean_code(sanity.get("code"))
                    cand["sanity_fixes"] = sanity.get("fixes", [])

            cand["static_verdict"] = _clean_text(gen.get("static_verdict")).lower() or "unknown"
            cand["interface_match_model"] = bool(gen.get("interface_match", False))
            cand["complexity_plausible"] = bool(gen.get("complexity_plausible", False))
            cand["static_issues"] = (
                list(gen.get("static_issues", [])) if isinstance(gen.get("static_issues"), list) else []
            )
            cand["static_confidence"] = gen.get("static_confidence", 0.0)

            local_interface_match, local_interface_issues = _check_interface_match(
                slow_code, cand["code"]
            )
            cand["interface_match_local"] = local_interface_match
            cand["interface_match"] = cand["interface_match_model"] and local_interface_match
            for issue in local_interface_issues:
                if issue not in cand["static_issues"]:
                    cand["static_issues"].append(issue)
            if not local_interface_match:
                cand["static_verdict"] = "fail"

            syntax_err = _cpp_syntax_error(cand["code"])
            cand["syntax_valid"] = syntax_err is None
            if syntax_err:
                if syntax_err not in cand["static_issues"]:
                    cand["static_issues"].append(syntax_err)
                cand["static_verdict"] = "fail"

            candidates.append(cand)
            cid += 1

    if not candidates:
        return {**result, "status": "failed", "error_stage": "s4_empty"}

    # C++ version behavior: Keep diagnostics for analysis, but do not discard candidates at the end.
    survivors = list(candidates)

    result["candidates"] = candidates
    result["survivor_candidates"] = survivors
    result["candidate_count"] = len(candidates)
    result["survivor_count"] = len(survivors)

    return result





# #####################################################################################################################🔖💡✅🟨❌
# -----------------------------
# Dataset/model configuration
# -----------------------------
def resolve_dataset_config(args: argparse.Namespace) -> Tuple[str, str]:
    if args.dataset == "problem_table":
        test_dataset_path = args.data or "problem_table.csv"
        slow_code_column_name = args.slow_column or "best_before"
    elif args.dataset == "canonical_solutions":
        test_dataset_path = args.data or "canonical_solutions.csv"
        slow_code_column_name = args.slow_column or "canonical_solution"
    else:
        if not args.data:
            raise ValueError("--dataset custom requires --data")
        if not args.slow_column:
            raise ValueError("--dataset custom requires --slow-column")
        test_dataset_path = args.data
        slow_code_column_name = args.slow_column
        
    return test_dataset_path, slow_code_column_name


def resolve_model_config(args: argparse.Namespace) -> Tuple[str, str, str]:
    LLM_base_url = (args.base_url or os.environ.get("EFFISKILL_BASE_URL", "")).strip()
    llm_api_key = (args.api_key or os.environ.get("EFFISKILL_API_KEY", "")).strip()
    model_name = (args.model or os.environ.get("EFFISKILL_MODEL", "")).strip()
    if not LLM_base_url:
        raise ValueError("❌❌❌ Missing base URL. Pass --base-url or set EFFISKILL_BASE_URL.")
    if not llm_api_key:
        raise ValueError("❌❌❌ Missing API key. Pass --api-key or set EFFISKILL_API_KEY.")
    if not model_name:
        raise ValueError("❌❌❌ Missing model. Pass --model or set EFFISKILL_MODEL.")
    return LLM_base_url, llm_api_key, model_name






if __name__ == "__main__":
    main()