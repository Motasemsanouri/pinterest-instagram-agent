"""
Phase 5 — Instagram Publisher Agent.

Takes the pins that passed the quality gate (Phase 4) and prepares a
publish-ready Instagram post for each: image URL, caption, and hashtags.

Captions are built locally from the pin's own metadata (title / alt_text /
query) — no LLM required, so this runs fully offline.

Modes (INSTAGRAM_DRY_RUN in .env):
  - dry_run = True  (default): nothing is posted. Each post is marked
    would_publish=True and the batch is saved to data/ for review.
  - dry_run = False: posts are sent via the Instagram Graph API. The real
    call is isolated in _publish_real() — the only method to implement when
    you wire up live publishing later.

This is the final node; extend the graph here if more phases are added.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from agents.models import InstagramPost, InstagramPublishResult, PinData
from agents.state import AgentState
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("instagram_publisher_agent")

_MAX_CAPTION = 2200          # Instagram caption hard limit
_MAX_HASHTAGS = 30           # Instagram hashtag hard limit
_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "for", "to", "of", "in", "on",
    "this", "that", "is", "are", "your", "next", "ideas", "idea",
}


def _slug_to_hashtag(text: str) -> str:
    """Turn an arbitrary phrase into a single #camelfree hashtag token."""
    cleaned = re.sub(r"[^a-z0-9]+", "", text.lower())
    return f"#{cleaned}" if cleaned else ""


class CaptionBuilder:
    """Builds captions + hashtags from pin metadata. Pure, no I/O."""

    def __init__(self, query: str, base_hashtags: list[str]) -> None:
        self.query = query
        self.base_hashtags = base_hashtags

    def _headline(self, pin: PinData) -> str:
        raw = (pin.title or pin.alt_text or self.query or "").strip()
        # Pinterest titles often bolt a keyword dump on the end; keep it short.
        raw = raw.split(". ")[0]
        if len(raw) > 150:
            raw = raw[:147].rstrip() + "..."
        return raw or self.query

    def _hashtags(self, pin: PinData) -> list[str]:
        tags: list[str] = []
        # query words
        for word in re.split(r"\s+", self.query or ""):
            if len(word) > 2 and word.lower() not in _STOPWORDS:
                h = _slug_to_hashtag(word)
                if h:
                    tags.append(h)
        # a combined query hashtag (e.g. "home decor ideas" -> #homedecorideas)
        combined = _slug_to_hashtag(self.query or "")
        if combined and combined not in tags:
            tags.append(combined)
        # LLM tags if they happen to exist (optional, no dependency)
        ia = pin.image_analysis
        if ia and ia.tags:
            for t in ia.tags:
                h = _slug_to_hashtag(t)
                if h and h not in tags:
                    tags.append(h)
        # configured base hashtags
        for b in self.base_hashtags:
            h = b if b.startswith("#") else f"#{b}"
            if h not in tags:
                tags.append(h)
        # dedup preserving order, cap at limit
        seen: set[str] = set()
        unique = [t for t in tags if not (t in seen or seen.add(t))]
        return unique[:_MAX_HASHTAGS]

    def build(self, pin: PinData) -> tuple[str, list[str]]:
        headline = self._headline(pin)
        hashtags = self._hashtags(pin)
        credit = f"\n\nvia Pinterest: {pin.pin_url}" if pin.pin_url else ""
        caption = f"{headline}{credit}\n\n{' '.join(hashtags)}".strip()
        if len(caption) > _MAX_CAPTION:
            caption = caption[:_MAX_CAPTION].rstrip()
        return caption, hashtags


class InstagramPublisherAgent(BaseAgent):
    """
    LangGraph node (final): prepares (and optionally publishes) Instagram posts.

    Reads from state:  pinterest_result (pins flagged passed_quality),
                       quality_result
    Writes to state:   pinterest_result (pins get instagram_post),
                       instagram_result, current_step, errors
    """

    def __init__(self) -> None:
        super().__init__(name="InstagramPublisherAgent")
        self.settings = get_settings()

    async def validate_inputs(self, state: AgentState) -> bool:
        result = state.get("pinterest_result")
        return bool(result and result.pins)

    def _selected_pins(self, pins: list[PinData], limit: int) -> list[PinData]:
        # prefer pins that passed quality; if none were flagged, fall back to all
        passed = [p for p in pins if p.passed_quality]
        candidates = passed if passed else list(pins)
        candidates.sort(key=lambda p: (p.rank_position or 10**9))
        return candidates[:limit]

    async def _publish_real(self, post: InstagramPost) -> str:
        """
        TODO: implement live publishing via the Instagram Graph API.

        Two-step flow:
          1. POST /{ig_account_id}/media           (image_url + caption) -> creation_id
          2. POST /{ig_account_id}/media_publish   (creation_id)         -> published_id

        Requires settings.instagram_access_token and instagram_account_id.
        Raises on failure so the caller records the error per post.
        """
        raise NotImplementedError("Live Instagram publishing is not enabled yet.")

    async def run(self, state: AgentState) -> dict[str, Any]:
        result = state.get("pinterest_result")

        if not await self.validate_inputs(state):
            self.logger.warning("no_pins_to_publish")
            return {
                "instagram_result": None,
                "current_step": f"{self.name}:skipped",
            }

        try:
            dry_run = bool(getattr(self.settings, "instagram_dry_run", True))
            limit = int(getattr(self.settings, "max_posts_per_run", 3) or 3)
            base = [
                h.strip()
                for h in str(getattr(self.settings, "base_hashtags", "")).split(",")
                if h.strip()
            ]

            selected = self._selected_pins(result.pins, limit)
            builder = CaptionBuilder(result.query, base)
            self.logger.info(
                "agent_started", selected=len(selected), dry_run=dry_run
            )

            posts: list[InstagramPost] = []
            published = 0
            for pin in selected:
                caption, hashtags = builder.build(pin)
                post = InstagramPost(
                    pin_id=pin.pin_id,
                    image_url=pin.original_image_url or pin.image_url,
                    caption=caption,
                    hashtags=hashtags,
                    would_publish=dry_run,
                )
                if dry_run:
                    post.status = "prepared"
                else:
                    try:
                        post.published_id = await self._publish_real(post)
                        post.status = "published"
                        post.would_publish = False
                        published += 1
                    except Exception as exc:
                        post.status = "failed"
                        post.error = f"{type(exc).__name__}: {exc}"
                        self.logger.warning(
                            "publish_failed", pin_id=pin.pin_id, error=str(exc)
                        )
                pin.instagram_post = post
                posts.append(post)

            publish_result = InstagramPublishResult(
                query=result.query,
                dry_run=dry_run,
                total_prepared=len(posts),
                total_published=published,
                posts=posts,
            )

            if dry_run:
                self._save_dry_run(publish_result)

            self.logger.info(
                "agent_completed",
                prepared=len(posts),
                published=published,
                dry_run=dry_run,
            )

            return {
                "pinterest_result": result,
                "instagram_result": publish_result,
                "current_step": f"{self.name}:completed",
            }
        except Exception as exc:
            return self._error_update(state, exc)

    def _save_dry_run(self, publish_result: InstagramPublishResult) -> None:
        """Persist the prepared batch so the user can review it."""
        out_dir = Path(getattr(self.settings, "output_dir", "data"))
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_query = re.sub(r"[^a-z0-9]+", "_", publish_result.query.lower()).strip("_")
        path = out_dir / f"instagram_dryrun_{safe_query}_{int(time.time())}.json"
        path.write_text(
            json.dumps(publish_result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info("dry_run_saved", file=str(path))