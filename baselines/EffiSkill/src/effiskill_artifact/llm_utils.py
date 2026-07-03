"""
LLM utilities with caching and failure logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

REQUEST_TIMEOUT_SEC = float(
    os.getenv("EFFISKILL_LLM_TIMEOUT_SEC", os.getenv("SELF_EVOLVE_LLM_TIMEOUT_SEC", "120"))
)
DEFAULT_MODEL_ENV_VAR = "EFFISKILL_MODEL"
_client: Optional[OpenAI] = None


def configure_default_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> None:
    global _client, REQUEST_TIMEOUT_SEC
    if timeout_sec is not None:
        REQUEST_TIMEOUT_SEC = float(timeout_sec)

    resolved_base_url = base_url or os.getenv("EFFISKILL_BASE_URL", "").strip()
    resolved_api_key = api_key or os.getenv("EFFISKILL_API_KEY", "").strip()

    if not resolved_base_url or not resolved_api_key:
        _client = None
        return

    _client = OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=REQUEST_TIMEOUT_SEC,
    )


def _require_client() -> OpenAI:
    global _client
    if _client is None:
        configure_default_client()
    if _client is None:
        raise RuntimeError(
            "LLM client is not configured. Pass --base-url/--api-key to the command "
            "or set EFFISKILL_BASE_URL and EFFISKILL_API_KEY."
        )
    return _client


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    reasoning_effort: str = "low",
) -> str:
    resolved_model = (model or os.getenv(DEFAULT_MODEL_ENV_VAR, "")).strip()
    if not resolved_model:
        raise RuntimeError(
            "Missing model. Pass --model to the command or set EFFISKILL_MODEL."
        )
    kwargs = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
    }
    if resolved_model.startswith("gpt-5") or resolved_model.startswith("o1"):
        kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
    else:
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

    response = _require_client().chat.completions.create(timeout=REQUEST_TIMEOUT_SEC, **kwargs)
    return response.choices[0].message.content


def _hash_payload(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cached_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    cache_dir: str,
    stage: str,
    reasoning_effort: str = "low",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "model": model or os.getenv(DEFAULT_MODEL_ENV_VAR, ""),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
    }
    key = _hash_payload(payload)
    cache_path = os.path.join(cache_dir, f"{stage}_{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        content = cached.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        try:
            os.remove(cache_path)
        except OSError:
            pass

    content = chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty response from LLM")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"content": content, "payload": payload}, f, ensure_ascii=False, indent=2)
    return content


def log_llm_failure(stage: str, payload: Dict[str, Any], response_text: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{stage}_{ts}_{_hash_payload(payload)[:8]}"
    json_path = os.path.join(out_dir, f"json_fail_{base}.json")
    txt_path = os.path.join(out_dir, f"json_fail_{base}.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"payload": payload}, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(response_text)
