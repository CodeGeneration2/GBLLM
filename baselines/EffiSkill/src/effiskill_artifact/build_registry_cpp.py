"""Build registry.json from SKILL.md frontmatter."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import yaml


def _extract_frontmatter(text: str) -> Dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return _fallback_extract_frontmatter(parts[1])


def _fallback_extract_frontmatter(blob: str) -> Dict[str, Any]:
    """Best-effort parser for mildly malformed YAML-like frontmatter.

    This keeps the registry builder usable on imperfect skill libraries where a
    few descriptions or trigger lines contain characters that break strict YAML.
    """
    front: Dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in blob.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("- ") and current_key:
            front.setdefault(current_key, [])
            if isinstance(front[current_key], list):
                front[current_key].append(line[2:].strip())
            continue

        if re.match(r"^[A-Za-z0-9_]+:\s*", line):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                front[key] = value
            else:
                front[key] = []
            current_key = key
            continue

        if current_key:
            existing = front.get(current_key)
            if isinstance(existing, list):
                if line.startswith("  - "):
                    existing.append(line[4:].strip())
                elif line.startswith("  ") and existing:
                    existing[-1] = f"{existing[-1]} {line.strip()}".strip()
            elif isinstance(existing, str):
                front[current_key] = f"{existing} {line.strip()}".strip()

    return front


def build_registry(skills_dir: str, out_path: str) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for root, _, files in os.walk(skills_dir):
        if "SKILL.md" not in files:
            continue
        path = os.path.join(root, "SKILL.md")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        front = _extract_frontmatter(text)
        if not front:
            continue

        card = {
            "skill_id": front.get("skill_id"),
            "type": front.get("type"),
            "language": front.get("language"),
            "name": front.get("name"),
            "description": front.get("description"),
        }
        if front.get("family"):
            card["family"] = front.get("family")
        if front.get("optimization_scope"):
            card["optimization_scope"] = front.get("optimization_scope")
        if front.get("tags"):
            card["tags"] = front.get("tags")
        if front.get("aliases"):
            card["aliases"] = front.get("aliases")
        if front.get("triggers"):
            card["triggers"] = front.get("triggers")
        cards.append(card)

    cards.sort(key=lambda x: (x.get("type", ""), x.get("skill_id", "")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_registry(args.skills_dir, args.out)


if __name__ == "__main__":
    main()
