"""
Phase 4 — Quality Evaluation Agent.

Acts as a quality gate before publishing. Using each pin's rank_score
(from Phase 3) and a configurable threshold (QUALITY_THRESHOLD in .env),
it:

  1. Filters: flags every pin as passed_quality True/False.
  2. Reports: produces a batch-level QualityEvaluationResult (counts,
     average score, ordered list of passing pins, and whether the batch
     has enough quality content to publish).

Pins are NOT deleted — they are flagged. Downstream phases (Instagram
Publisher) consume only the passing pins, but the full set stays available
for logging/debugging. This keeps the pipeline non-destructive and easy to
audit.
"""
from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.models import QualityEvaluationResult
from agents.state import AgentState
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("quality_evaluation_agent")


class QualityEvaluationAgent(BaseAgent):
    """
    LangGraph node: quality gate over ranked pins.

    Reads from state:  pinterest_result (pins carry rank_score)
    Writes to state:   pinterest_result (pins flagged passed_quality),
                       quality_result, current_step, errors
    """

    def __init__(self) -> None:
        super().__init__(name="QualityEvaluationAgent")
        self.settings = get_settings()

    async def validate_inputs(self, state: AgentState) -> bool:
        result = state.get("pinterest_result")
        return bool(result and result.pins)

    async def run(self, state: AgentState) -> dict[str, Any]:
        result = state.get("pinterest_result")

        if not await self.validate_inputs(state):
            self.logger.warning("no_pins_to_evaluate")
            return {
                "quality_result": None,
                "current_step": f"{self.name}:skipped",
            }

        try:
            threshold = float(getattr(self.settings, "quality_threshold", 0.70) or 0.70)
            min_to_publish = int(
                getattr(self.settings, "min_images_to_publish", 1) or 1
            )
            pins = result.pins
            self.logger.info(
                "agent_started", pin_count=len(pins), threshold=threshold
            )

            scores: list[float] = []
            passed_pins = []
            for pin in pins:
                score = pin.rank_score if pin.rank_score is not None else 0.0
                scores.append(score)
                pin.passed_quality = score >= threshold
                if pin.passed_quality:
                    passed_pins.append(pin)

            # passed_pins keep the best-first order from ranking
            passed_pins.sort(key=lambda p: (p.rank_score or 0.0), reverse=True)

            passed = len(passed_pins)
            rejected = len(pins) - passed
            avg = round(sum(scores) / len(scores), 4) if scores else None
            batch_ready = passed >= min_to_publish

            quality_result = QualityEvaluationResult(
                query=result.query,
                threshold=threshold,
                total_evaluated=len(pins),
                passed=passed,
                rejected=rejected,
                average_score=avg,
                passed_pin_ids=[p.pin_id for p in passed_pins],
                batch_ready=batch_ready,
            )

            self.logger.info(
                "agent_completed",
                passed=passed,
                rejected=rejected,
                average_score=avg,
                batch_ready=batch_ready,
            )

            return {
                "pinterest_result": result,
                "quality_result": quality_result,
                "current_step": f"{self.name}:completed",
            }
        except Exception as exc:
            return self._error_update(state, exc)