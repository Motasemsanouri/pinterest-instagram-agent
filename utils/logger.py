from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog


# ─── Human-friendly rendering ──────────────────────────────────────────────────
# Translates structured log events into plain-English status lines, e.g.
#   "07:56:24  Scraping finished - collected 5 pins in 13.78 seconds"
# instead of a raw JSON blob. Falls back to a generic "humanized" line for any
# event that doesn't have a specific phrasing below, so nothing is lost.

_LEVEL_ICONS = {
    "debug": ".",
    "info": "*",
    "warning": "!",
    "error": "X",
    "critical": "X",
}

# Fields that are part of the log envelope, not the message - never shown raw.
_ENVELOPE_KEYS = {"event", "logger", "level", "timestamp", "exception"}

# (logger name or None for "any logger", event name) -> plain-English template.
# Templates use {field} placeholders filled from the event's data.
_FRIENDLY_TEMPLATES: dict[tuple[Optional[str], str], str] = {
    (None, "starting"): 'Starting up - searching for "{query}" (target: {max_pins} pins)',
    (None, "pipeline_starting"): "Kicking off the pipeline...",
    (None, "pipeline_completed"): "All done - the whole run took {elapsed_s} seconds",
    (None, "pipeline_had_errors"): "Finished, but ran into some problems along the way",
    (None, "results_saved"): "Saved the results to {file}",
    (None, "fatal_error"): "Something went badly wrong: {error}",
    (None, "interrupted_by_user"): "Stopped because you pressed Ctrl+C",
    (None, "agent_error"): "{agent} ran into a problem: {error}",

    (None, "node_entering"): "Starting step: {node}",
    (None, "node_exiting"): "Finished step: {node} (took {elapsed_s} seconds)",
    (None, "node_failed"): "Step failed: {node} - {error}",

    ("PinterestAgent", "agent_started"): 'Looking for pins about "{query}" (target: {max_pins})',
    ("PinterestAgent", "agent_completed"): "Scraping finished - collected {total_scraped} pins in {duration_s} seconds",

    ("pinterest_scraper", "navigating_to_search"): "Opening Pinterest and loading the search results...",
    ("pinterest_scraper", "scraping_progress"): "Found {scraped} of {target} pins so far (pass #{scroll})",
    ("pinterest_scraper", "scroll_yielding_no_new_pins"):
        "Not finding any new pins - Pinterest may be slowing us down or blocking the scraper",
    ("pinterest_scraper", "scraping_failed"): "Scraping ran into a problem: {error}",
    ("pinterest_scraper", "attempting_login"): "Logging into Pinterest...",
    ("pinterest_scraper", "login_completed"): "Logged into Pinterest successfully",
    ("pinterest_scraper", "login_failed"): "Could not log into Pinterest - continuing without logging in",
    ("pinterest_scraper", "page_extraction_failed"): "Had trouble reading the search results page",

    ("ImageAnalysisAgent", "agent_started"): "Examining {pin_count} images with AI...",
    ("ImageAnalysisAgent", "agent_completed"):
        "Image review finished - {analyzed} reviewed, {llm_succeeded} got AI descriptions, {failed} had issues",
    ("ImageAnalysisAgent", "no_pins_to_analyze"): "Skipping image review - there are no pins to look at",

    ("image_analyzer", "llm_client_initialized"): "AI image model is ready ({model})",
    ("image_analyzer", "llm_disabled_no_api_key"): "AI image descriptions are turned off - no API key is set",
    ("image_analyzer", "llm_client_init_failed"): "Could not set up the AI image model: {error}",
    ("image_analyzer", "fetch_failed"): "Could not download an image: {error}",
    ("image_analyzer", "local_analysis_failed"): "Could not examine an image: {error}",
    ("image_analyzer", "llm_analysis_failed"): "AI could not describe an image: {error}",

    ("RankingAgent", "agent_started"): "Sorting {pin_count} pins by quality...",
    ("RankingAgent", "agent_completed"): "Sorting finished - the best pin is #{top_pin_id} (score {top_score})",
    ("RankingAgent", "no_pins_to_rank"): "Skipping ranking - there is nothing to sort",
}


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _short(value: Any, limit: int = 160) -> str:
    """Collapse verbose error blobs (stack traces, JSON dumps) to one short phrase."""
    text = str(value)
    text = text.split("\n", 1)[0]
    text = text.split(". {", 1)[0]  # e.g. "404 NOT_FOUND. {'error': {...}}" -> "404 NOT_FOUND"
    text = text.strip()
    return text if len(text) <= limit else text[: limit].rstrip() + "..."


def _format_time(timestamp: str) -> str:
    try:
        parsed = dt.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return str(timestamp)[:8] or "--:--:--"


def _humanize_event_name(event: str) -> str:
    return (event or "log entry").replace("_", " ").strip().capitalize()


def _friendly_message(event_dict: dict[str, Any]) -> str:
    event = event_dict.get("event", "")
    logger_name = event_dict.get("logger", "")

    template = _FRIENDLY_TEMPLATES.get((logger_name, event)) or _FRIENDLY_TEMPLATES.get((None, event))
    if template:
        values = _SafeDict({k: (_short(v) if k == "error" else v) for k, v in event_dict.items()})
        try:
            return template.format_map(values)
        except (KeyError, ValueError, IndexError):
            pass  # fall through to the generic phrasing below

    extras = {
        k: v for k, v in event_dict.items()
        if k not in _ENVELOPE_KEYS and v not in (None, [], {})
    }
    phrase = _humanize_event_name(event)
    if extras:
        details = ", ".join(f"{k.replace('_', ' ')}: {_short(v)}" for k, v in extras.items())
        return f"{phrase} ({details})"
    return phrase


def friendly_renderer(_logger: Any, _name: str, event_dict: dict[str, Any]) -> str:
    """structlog renderer that prints plain-English, single-line status updates."""
    time_str = _format_time(event_dict.get("timestamp", ""))
    icon = _LEVEL_ICONS.get(event_dict.get("level", "info"), "*")
    return f"{time_str} {icon} {_friendly_message(event_dict)}"


# ─── Setup ─────────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """Configure structlog for the application. Call once at startup.

    log_format:
        "friendly" - plain-English single-line status updates (for non-technical readers)
        "console"  - colored key=value developer output
        "json"     - structured JSON, one object per line (for tooling/machine parsing)
    """

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=log_level, handlers=handlers, format="%(message)s")

    # Silence noisy third-party loggers
    for name in ("urllib3", "playwright", "asyncio", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    shared_processors: list = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "friendly":
        renderer = friendly_renderer
    elif log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given component name."""
    return structlog.get_logger(name)
