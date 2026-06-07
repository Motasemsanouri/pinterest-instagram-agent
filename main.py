"""
Pinterest Agent — Phase 1 entry point.

Usage:
    python main.py "home decor ideas"
    python main.py "minimalist living room" --max-pins 100
    python main.py "boho bedroom" --output-file data/boho.json --log-level DEBUG
    python main.py "kitchen design" --no-save
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config.settings import get_settings
from agents.state import create_initial_state
from utils.logger import setup_logging, get_logger
from utils.helpers import ensure_directories, save_json, timestamp_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pinterest Agent — LangGraph-powered Pinterest scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="Pinterest search query")
    parser.add_argument(
        "--max-pins", type=int, default=None,
        help="Maximum pins to scrape (default: MAX_PINS_PER_QUERY from .env)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output file path (default: data/results_<query>_<timestamp>.json)",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
        help="Log level (default: LOG_LEVEL from .env)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving results to disk",
    )
    return parser.parse_args()


async def run_pipeline(query: str, max_pins: int) -> dict:
    """Build the LangGraph pipeline and run it to completion."""
    from graph import build_graph

    logger = get_logger("main")
    logger.info("pipeline_starting", query=query, max_pins=max_pins)

    graph = build_graph()
    initial_state = create_initial_state(query=query, max_pins=max_pins)

    t0 = time.time()
    final_state = await graph.ainvoke(initial_state)
    elapsed = time.time() - t0

    logger.info(
        "pipeline_completed",
        elapsed_s=round(elapsed, 2),
        errors=final_state.get("errors", []),
    )
    return final_state


async def main() -> int:
    args = parse_args()
    settings = get_settings()

    log_level = args.log_level or settings.log_level
    setup_logging(
        level=log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )
    logger = get_logger("main")

    ensure_directories(settings.output_dir, Path(settings.log_file).parent)

    max_pins = args.max_pins or settings.max_pins_per_query

    logger.info("starting", query=args.query, max_pins=max_pins)

    try:
        final_state = await run_pipeline(query=args.query, max_pins=max_pins)

        pinterest_result = final_state.get("pinterest_result")
        errors = final_state.get("errors", [])

        if errors:
            logger.warning("pipeline_had_errors", errors=errors)

        if pinterest_result:
            summary = pinterest_result.summary()
            print("\n" + "=" * 60)
            print("  SCRAPING SUMMARY")
            print("=" * 60)
            for key, value in summary.items():
                print(f"  {key:<22}: {value}")
            print("=" * 60 + "\n")

        should_save = not args.no_save and settings.save_results
        if should_save and pinterest_result:
            slug = re.sub(r"[^\w\-]", "_", args.query)[:50]
            output_file = args.output_file or str(
                Path(settings.output_dir) / f"results_{slug}_{timestamp_str()}.json"
            )
            await save_json(
                {
                    "query": args.query,
                    "max_pins_requested": max_pins,
                    "result": pinterest_result.model_dump(mode="json"),
                    "errors": errors,
                },
                output_file,
            )
            logger.info("results_saved", file=output_file)
            print(f"Results saved to: {output_file}")

        return 1 if errors else 0

    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
        return 130
    except Exception as exc:
        logger.exception("fatal_error", error=str(exc))
        print(f"\nFatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
