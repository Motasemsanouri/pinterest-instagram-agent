from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from agents.models import (
    PinterestScrapingResult,
    RankingResult,
    QualityEvaluationResult,
    InstagramPublishResult,
)


class AgentState(TypedDict):
    """
    Shared state passed through every node in the LangGraph pipeline.

    Each agent reads its required inputs from this state and writes its
    outputs back. LangGraph merges returned dicts into the state automatically.
    """

    # ─── Input parameters ─────────────────────────────────────────────────────
    query: str
    max_pins: int

    # ─── Phase 1: Pinterest Agent output ──────────────────────────────────────
    pinterest_result: Optional[PinterestScrapingResult]

    # ─── Phase 2 — Image Analysis Agent output ────────────────────────────────
    # The analysis is stored inside PinData itself; this is just a summary counter
    images_analyzed: Optional[int]

    # ─── Phase 3 — Image Ranking Agent output ─────────────────────────────────
    ranking_result: Optional[RankingResult]

    # ─── Phase 4 — Quality Evaluation Agent output ────────────────────────────
    quality_result: Optional[QualityEvaluationResult]

    # ─── Phase 5 — Instagram Publisher Agent output ───────────────────────────
    instagram_result: Optional[InstagramPublishResult]

    # ─── Pipeline metadata ────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]  # lists from all nodes are concatenated
    current_step: str
    pipeline_start_time: float
    pipeline_metadata: dict[str, Any]


def create_initial_state(query: str, max_pins: int) -> AgentState:
    return AgentState(
        query=query,
        max_pins=max_pins,
        pinterest_result=None,
        images_analyzed=None,
        ranking_result=None,
        quality_result=None,
        instagram_result=None,
        errors=[],
        current_step="start",
    )