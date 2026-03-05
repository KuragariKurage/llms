from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_cache_dir

API_URL = "https://models.dev/api.json"
CACHE_DIR = Path(user_cache_dir("llms"))
CACHE_FILE = CACHE_DIR / "api.json"
ETAG_FILE = CACHE_DIR / "api.json.etag"
MAX_AGE_SECONDS = 3600


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < MAX_AGE_SECONDS


def _load_cache() -> dict[str, Any]:
    return json.loads(CACHE_FILE.read_text())


def _save_cache(data: bytes, etag: str | None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_bytes(data)
    if etag:
        ETAG_FILE.write_text(etag)


def _get_etag() -> str | None:
    if ETAG_FILE.exists():
        return ETAG_FILE.read_text().strip()
    return None


def fetch_models(force_refresh: bool = False) -> dict[str, Any]:
    if not force_refresh and _cache_is_fresh():
        return _load_cache()

    headers: dict[str, str] = {}
    etag = _get_etag()
    if etag and CACHE_FILE.exists():
        headers["If-None-Match"] = etag

    try:
        response = httpx.get(API_URL, headers=headers, timeout=30)
        if response.status_code == 304:
            CACHE_FILE.touch()
            return _load_cache()
        response.raise_for_status()
        _save_cache(response.content, response.headers.get("etag"))
        return response.json()
    except httpx.HTTPError:
        if CACHE_FILE.exists():
            return _load_cache()
        raise


def flatten_models(
    data: dict[str, Any],
    provider_filter: str | None = None,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for provider_id, provider_data in data.items():
        if not isinstance(provider_data, dict):
            continue
        if provider_filter and provider_id != provider_filter:
            continue
        provider_name = provider_data.get("name", provider_id)
        provider_models = provider_data.get("models", {})
        if not isinstance(provider_models, dict):
            continue
        for model_id, model_data in provider_models.items():
            if not isinstance(model_data, dict):
                continue
            models.append(
                {
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "full_id": f"{provider_id}/{model_id}",
                    **model_data,
                }
            )
    return models
