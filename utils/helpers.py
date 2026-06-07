from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def ensure_directories(*paths: str | Path) -> None:
    """Create directories (including parents) if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Replace filesystem-unsafe characters and truncate."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:max_length]


def timestamp_str() -> str:
    """UTC timestamp as a filesystem-safe string: 20240601_153045."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def save_json(data: Any, filepath: str | Path, indent: int = 2) -> None:
    """Write *data* to *filepath* as JSON (async, UTF-8)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, path.write_text, json_str, "utf-8")


def load_json(filepath: str | Path) -> Any:
    """Load and parse JSON from *filepath*."""
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def truncate_text(text: Optional[str], max_length: int = 500) -> Optional[str]:
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_domain(url: str) -> Optional[str]:
    try:
        return urllib.parse.urlparse(url).netloc or None
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"
