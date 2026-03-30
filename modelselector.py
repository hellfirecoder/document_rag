from __future__ import annotations

import subprocess
from typing import Iterable

import requests


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _models_from_api(timeout_seconds: int = 5) -> list[str]:
    response = requests.get(OLLAMA_TAGS_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    raw_models = payload.get("models", [])
    names = [m.get("name", "") for m in raw_models if isinstance(m, dict)]
    return _dedupe_preserve_order(names)


def _models_from_cli() -> list[str]:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    models: list[str] = []
    for line in result.stdout.splitlines():
        columns = line.split()
        if not columns:
            continue
        if columns[0].lower() == "name":
            continue
        models.append(columns[0])
    return _dedupe_preserve_order(models)


def discover_ollama_models() -> list[str]:
    """Discover locally available Ollama model names.

    Attempts the Ollama HTTP API first, then falls back to the CLI.
    Returns an empty list if no models can be discovered.
    """
    try:
        models = _models_from_api(timeout_seconds=5)
        if models:
            return models
    except Exception:
        pass

    try:
        return _models_from_cli()
    except Exception:
        return []


def split_ollama_models(models: list[str]) -> tuple[list[str], list[str]]:
    """Split model names into embedding and chat-capable candidates."""
    embed_candidates = [m for m in models if "embed" in m.lower()]
    llm_candidates = [m for m in models if m not in embed_candidates]
    return embed_candidates, llm_candidates
