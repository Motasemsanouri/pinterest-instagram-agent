"""
Phase 3 — Ranking Agent.

Scores every analyzed pin and orders them best-first, so later phases
(Quality Evaluation, Instagram Publishing) can act on the strongest content.

Scoring is driven by the LOCAL image analysis (always available) and, when
present, the LLM quality score. Each component returns a 0-1 sub-score; the
final score is their weighted sum. If the LLM score is missing, its weight is
redistributed across the remaining components, so disabling the LLM never
penalizes ranking — it just relies on local signals.

Results:
  - each PinData gets rank_score, rank_position, rank_breakdown
  - state.pinterest_result.pins is reordered best-first
  - state.ranking_result holds a summary
"""
from __future__ import annotations

from typing import Any, Optional

from agents.base_agent import BaseAgent
from agents.models import PinData, RankingResult
from agents.state import AgentState
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("ranking_agent")

# Base weights (sum to 1.0). The "llm" weight is redistributed when absent.
_BASE_WEIGHTS: dict[str, float] = {
    "aspect": 0.50,   # how Instagram-friendly the aspect ratio is
    "size": 0.20,     # larger images preferred
    "color": 0.15,    # color variety
    "llm": 0.15,      # LLM quality score (optional)
}

# Instagram sweet spots: portrait 4:5 (=0.8) is ideal, square (=1.0) is great.
_IDEAL_ASPECT = 0.8
_GOOD_MIN, _GOOD_MAX = 0.8, 1.0
# Pins at or above this width are considered full-size.
_GOOD_WIDTH = 600


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _aspect_score(aspect: Optional[float]) -> float:
    """1.0 inside the 4:5–1:1 band, decaying smoothly outside it."""
    if not aspect or aspect <= 0:
        return 0.0
    if _GOOD_MIN <= aspect <= _GOOD_MAX:
        return 1.0
    # distance from the nearest edge of the good band
    edge = _GOOD_MIN if aspect < _GOOD_MIN else _GOOD_MAX
    return _clamp(1.0 - abs(aspect - edge) / _IDEAL_ASPECT)


def _size_score(width: Optional[int]) -> float:
    if not width or width <= 0:
        return 0.0
    return _clamp(width / _GOOD_WIDTH)


def _color_score(colors: list[str]) -> float:
    """More distinct dominant colors -> richer image (cap at 5)."""
    if not colors:
        return 0.0
    return _clamp(len(set(colors)) / 5.0)


class ImageRanker:
    """Pure scoring logic — no I/O, easy to unit test."""

    def score_pin(self, pin: PinData) -> tuple[float, dict[str, float]]:
        ia = pin.image_analysis
        if ia is None:
            return 0.0, {}

        components = {
            "aspect": _aspect_score(ia.aspect_ratio),
            "size": _size_score(ia.width),
            "color": _color_score(ia.dominant_colors),
        }
        has_llm = ia.quality_score is not None
        if has_llm:
            components["llm"] = _clamp(float(ia.quality_score))

        # Build active weights; redistribute "llm" weight if it's missing.
        weights = dict(_BASE_WEIGHTS)
        if not has_llm:
            llm_w = weights.pop("llm")
            active_total = sum(weights.values())
            for key in weights:
                weights[key] += llm_w * (weights[key] / active_total)

        final = sum(components[k] * weights[k] for k in components)
        breakdown = {k: round(components[k], 3) for k in components}
        breakdown["_weighted_total"] = round(final, 4)
        return round(final, 4), breakdown


class RankingAgent(BaseAgent):
    """
    LangGraph node: scores and orders analyzed pins best-first.

    Reads from state:  pinterest_result (with image_analysis on each pin)
    Writes to state:   pinterest_result (pins reordered + scored),
                       ranking_result, current_step, errors
    """

    def __init__(self) -> None:
        super().__init__(name="RankingAgent")
        self.ranker = ImageRanker()
        self.settings = get_settings()

    async def validate_inputs(self, state: AgentState) -> bool:
        result = state.get("pinterest_result")
        return bool(result and result.pins)

    async def run(self, state: AgentState) -> dict[str, Any]:
        result = state.get("pinterest_result")

        if not await self.validate_inputs(state):
            self.logger.warning("no_pins_to_rank")
            return {
                "ranking_result": None,
                "current_step": f"{self.name}:skipped",
            }

        try:
            pins = result.pins
            self.logger.info("agent_started", pin_count=len(pins))

            # Score every pin
            for pin in pins:
                score, breakdown = self.ranker.score_pin(pin)
                pin.rank_score = score
                pin.rank_breakdown = breakdown

            # Order best-first and assign 1-based positions
            pins.sort(key=lambda p: (p.rank_score or 0.0), reverse=True)
            for position, pin in enumerate(pins, start=1):
                pin.rank_position = position

            ranking_result = RankingResult(
                query=result.query,
                total_ranked=len(pins),
                ranked_pin_ids=[p.pin_id for p in pins],
                top_pin_id=pins[0].pin_id if pins else None,
                weights_used=_BASE_WEIGHTS,
            )

            self.logger.info(
                "agent_completed",
                total_ranked=len(pins),
                top_pin_id=ranking_result.top_pin_id,
                top_score=pins[0].rank_score if pins else None,
            )

            return {
                "pinterest_result": result,
                "ranking_result": ranking_result,
                "current_step": f"{self.name}:completed",
            }
        except Exception as exc:
            return self._error_update(state, exc)