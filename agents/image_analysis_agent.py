"""
Phase 2 — Image Analysis Agent (Google Gemini, free tier).

For each scraped pin this agent fetches the image bytes ONCE into memory
(never written to disk) and runs two complementary analyses on those bytes:

1. Local (no API): dimensions, aspect ratio, orientation, dominant colors
   via Pillow.
2. LLM (Gemini vision): a short description, content tags, and a 0-1 quality
   score. Uses Google's free-tier Gemini API.

Provider is isolated behind ImageAnalyzer, so swapping Gemini for another
vision API later only touches this file.

Results are stored inside each PinData.image_analysis so they flow downstream
to the Ranking Agent (Phase 3) without changing the pipeline contract.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from collections import Counter
from typing import Any, Optional

import httpx
from PIL import Image

from agents.base_agent import BaseAgent
from agents.models import ImageAnalysisResult, PinData
from agents.state import AgentState
from config.settings import get_settings
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger("image_analysis_agent")

_DEFAULT_VISION_MODEL = "gemini-2.0-flash"
_FETCH_TIMEOUT = 20.0
_MAX_DIMENSION_FOR_COLORS = 100  # downscale before counting colors (speed)

_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.pinterest.com/",
    "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*",
}

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class ImageAnalyzer:
    """Stateless helper that analyzes a single image (local + optional LLM)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("image_analyzer")
        self._llm_client = self._build_llm_client()

    def _build_llm_client(self) -> Optional[Any]:
        api_key = getattr(self.settings, "gemini_api_key", "") or ""
        model = getattr(self.settings, "vision_model", "") or _DEFAULT_VISION_MODEL
        if not api_key:
            self.logger.warning("llm_disabled_no_api_key")
            return None
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            self.logger.info("llm_client_initialized", model=model, api_key_prefix=api_key[:6])
            return client
        except Exception as exc:  # pragma: no cover - import/runtime guard
            self.logger.warning("llm_client_init_failed", model=model, error=str(exc))
            return None

    # ─── Fetch ──────────────────────────────────────────────────────────────────

    @async_retry(
        max_attempts=3,
        wait_min=1.0,
        wait_max=8.0,
        exceptions=(httpx.TransportError, httpx.TimeoutException),
    )
    async def _get(self, client: httpx.AsyncClient, url: str) -> bytes:
        """Single GET. Retries only on transient network errors, not on 4xx."""
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

    async def _fetch_bytes(self, urls: list[str]) -> bytes:
        """Try each candidate URL in order; first success wins."""
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT, follow_redirects=True, headers=_IMAGE_HEADERS
        ) as client:
            for index, url in enumerate(urls):
                try:
                    raw = await self._get(client, url)
                    self.logger.debug(
                        "image_fetch_succeeded",
                        url=url,
                        candidate_index=index,
                        bytes=len(raw),
                    )
                    return raw
                except Exception as exc:  # noqa: BLE001 - try next candidate
                    last_err = exc
                    self.logger.debug("image_url_failed", url=url, candidate_index=index, error=str(exc))
        raise last_err if last_err else RuntimeError("no image urls to fetch")

    # ─── Local analysis ───────────────────────────────────────────────────────

    def _analyze_local(self, raw: bytes) -> dict[str, Any]:
        """CPU-bound Pillow work; run inside a thread via asyncio.to_thread."""
        with Image.open(io.BytesIO(raw)) as img:
            mime = _PIL_FORMAT_TO_MIME.get(img.format or "", "image/jpeg")
            rgb = img.convert("RGB")
            width, height = rgb.size

            if width > height:
                orientation = "landscape"
            elif height > width:
                orientation = "portrait"
            else:
                orientation = "square"

            thumb = rgb.copy()
            thumb.thumbnail((_MAX_DIMENSION_FOR_COLORS, _MAX_DIMENSION_FOR_COLORS))
            most_common = Counter(thumb.getdata()).most_common(5)
            dominant = ["#{:02x}{:02x}{:02x}".format(*rgb_t) for rgb_t, _ in most_common]

        aspect = round(width / height, 3) if height else None
        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "orientation": orientation,
            "dominant_colors": dominant,
            "_mime": mime,
        }

    # ─── LLM analysis ───────────────────────────────────────────────────────────

    async def _analyze_llm(self, raw: bytes, mime: str) -> dict[str, Any]:
        if self._llm_client is None:
            return {}

        from google.genai import types

        model = getattr(self.settings, "vision_model", "") or _DEFAULT_VISION_MODEL
        prompt = (
            "Analyze this image for use on a social feed. Respond with ONLY a JSON "
            "object, no prose, no markdown fences, with exactly these keys: "
            '"description" (string, max 30 words), "tags" (array of 3-8 lowercase '
            'strings), "quality_score" (number between 0 and 1).'
        )

        t0 = time.time()
        response = await self._llm_client.aio.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=raw, mime_type=mime),
                prompt,
            ],
        )
        elapsed = time.time() - t0

        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)

        score = data.get("quality_score")
        try:
            score = max(0.0, min(1.0, float(score))) if score is not None else None
        except (TypeError, ValueError):
            score = None

        self.logger.debug(
            "llm_analysis_succeeded",
            model=model,
            elapsed_s=round(elapsed, 2),
            tags=list(data.get("tags", []))[:8],
            quality_score=score,
        )

        return {
            "description": data.get("description"),
            "tags": list(data.get("tags", []))[:8],
            "quality_score": score,
            "analyzed_by_llm": True,
        }

    # ─── Orchestration for one pin ──────────────────────────────────────────────

    async def analyze(self, pin: PinData) -> ImageAnalysisResult:
        result = ImageAnalysisResult(pin_id=pin.pin_id)
        t0 = time.time()

        # Candidate URLs in priority order: high-res first, thumbnail as fallback
        # (Pinterest sometimes 403s /originals/ but serves the smaller sizes).
        candidates = [pin.original_image_url, pin.image_url, pin.thumbnail_url]
        seen: set[str] = set()
        urls = [u for u in candidates if u and not (u in seen or seen.add(u))]

        # Fetch once, reuse for both analyses.
        try:
            raw = await self._fetch_bytes(urls)
        except Exception as exc:
            result.error = f"fetch: {type(exc).__name__}: {exc}"
            self.logger.warning("fetch_failed", pin_id=pin.pin_id, error=str(exc))
            return result

        mime = "image/jpeg"
        # Local analysis
        try:
            local = await asyncio.to_thread(self._analyze_local, raw)
            mime = local.pop("_mime", "image/jpeg")
            for key, value in local.items():
                setattr(result, key, value)
        except Exception as exc:
            result.error = f"local: {type(exc).__name__}: {exc}"
            self.logger.warning("local_analysis_failed", pin_id=pin.pin_id, error=str(exc))

        # LLM analysis (independent of local success)
        try:
            llm = await self._analyze_llm(raw, mime)
            for key, value in llm.items():
                setattr(result, key, value)
        except Exception as exc:
            prev = f"{result.error} | " if result.error else ""
            result.error = f"{prev}llm: {type(exc).__name__}: {exc}"
            self.logger.warning("llm_analysis_failed", pin_id=pin.pin_id, error=str(exc))

        self.logger.debug(
            "pin_analysis_completed",
            pin_id=pin.pin_id,
            elapsed_s=round(time.time() - t0, 2),
            has_local=result.width is not None,
            has_llm=result.analyzed_by_llm,
            error=result.error,
        )
        return result


class ImageAnalysisAgent(BaseAgent):
    """
    LangGraph node: enriches each scraped pin with image analysis.

    Reads from state:  pinterest_result
    Writes to state:   pinterest_result (pins mutated in place), images_analyzed,
                       current_step, errors
    """

    def __init__(self) -> None:
        super().__init__(name="ImageAnalysisAgent")
        self.analyzer = ImageAnalyzer()
        self.settings = get_settings()

    async def validate_inputs(self, state: AgentState) -> bool:
        result = state.get("pinterest_result")
        return bool(result and result.pins)

    async def run(self, state: AgentState) -> dict[str, Any]:
        result = state.get("pinterest_result")

        if not await self.validate_inputs(state):
            self.logger.warning("no_pins_to_analyze")
            return {
                "images_analyzed": 0,
                "current_step": f"{self.name}:skipped",
            }

        pins = result.pins
        self.logger.info("agent_started", pin_count=len(pins))

        concurrency = int(getattr(self.settings, "image_analysis_concurrency", 5) or 5)
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(pin: PinData) -> None:
            async with semaphore:
                pin.image_analysis = await self.analyzer.analyze(pin)

        try:
            await asyncio.gather(*(_bounded(pin) for pin in pins))
        except Exception as exc:
            return self._error_update(state, exc)

        analyzed = sum(1 for p in pins if p.image_analysis is not None)
        failed = sum(1 for p in pins if p.image_analysis and p.image_analysis.error)
        llm_succeeded = sum(1 for p in pins if p.image_analysis and p.image_analysis.analyzed_by_llm)
        local_succeeded = sum(1 for p in pins if p.image_analysis and p.image_analysis.width is not None)

        # Bucket failure reasons (fetch/local/llm) so it's obvious which stage breaks.
        failure_stages: Counter = Counter()
        for p in pins:
            if p.image_analysis and p.image_analysis.error:
                for chunk in p.image_analysis.error.split(" | "):
                    stage = chunk.split(":", 1)[0].strip()
                    failure_stages[stage] += 1

        self.logger.info(
            "agent_completed",
            analyzed=analyzed,
            failed=failed,
            local_succeeded=local_succeeded,
            llm_succeeded=llm_succeeded,
            failure_stages=dict(failure_stages),
        )

        return {
            "pinterest_result": result,
            "images_analyzed": analyzed,
            "current_step": f"{self.name}:completed",
        }