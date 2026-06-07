"""
LangGraph pipeline definition for the Pinterest multi-agent system.

Phase 1 (current):
    START → pinterest_agent → END

Future phases extend this graph:
    START → pinterest_agent → image_analysis_agent → ranking_agent
          → quality_evaluation_agent → instagram_publisher_agent → END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.state import AgentState
from utils.logger import get_logger

logger = get_logger("graph")


# ─── Node functions ────────────────────────────────────────────────────────────
# Each function wraps an agent and returns a dict of state updates.

async def pinterest_node(state: AgentState) -> dict[str, Any]:
    """
    Phase 1: Scrapes Pinterest for pins matching the query.

    Reads:  state.query, state.max_pins
    Writes: state.pinterest_result, state.current_step, state.errors
    """
    from agents.pinterest_agent import PinterestAgent
    return await PinterestAgent().run(state)


# Phase 2 — Image Analysis Node
async def image_analysis_node(state: AgentState) -> dict[str, Any]:
    """Analyzes scraped pin images for content, quality, and relevance."""
    from agents.image_analysis_agent import ImageAnalysisAgent
    return await ImageAnalysisAgent().run(state)


# Phase 3 — Ranking Node
async def ranking_node(state: AgentState) -> dict[str, Any]:
    """Ranks pins by relevance, quality, and engagement potential."""
    from agents.ranking_agent import RankingAgent
    return await RankingAgent().run(state)


# Phase 4 — Quality Evaluation Node
async def quality_evaluation_node(state: AgentState) -> dict[str, Any]:
    """Evaluates pin quality against Instagram publishing standards."""
    from agents.quality_evaluation_agent import QualityEvaluationAgent
    return await QualityEvaluationAgent().run(state)


# Phase 5 — Instagram Publisher Node
async def instagram_publisher_node(state: AgentState) -> dict[str, Any]:
    """Publishes top-ranked pins to Instagram."""
    from agents.instagram_publisher_agent import InstagramPublisherAgent
    return await InstagramPublisherAgent().run(state)


# ─── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build and compile the LangGraph pipeline.

    To add a new phase:
        1. Implement the agent class in agents/<name>_agent.py
        2. Add a node function above
        3. Call workflow.add_node(...) below
        4. Update the edges to route through the new node
    """
    workflow = StateGraph(AgentState)

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    workflow.add_node("pinterest_agent", pinterest_node)

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    workflow.add_node("image_analysis_agent", image_analysis_node)

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    workflow.add_node("ranking_agent", ranking_node)

    # ── Phase 4 ────────────────────────────────────────────────────────────────
    workflow.add_node("quality_evaluation_agent", quality_evaluation_node)

    # ── Phase 5 ────────────────────────────────────────────────────────────────
    workflow.add_node("instagram_publisher_agent", instagram_publisher_node)

    # ── Phase 1 edges (replace these as new phases are added) ──────────────────
    workflow.add_edge(START, "pinterest_agent")
    workflow.add_edge("pinterest_agent", "image_analysis_agent")
    workflow.add_edge("image_analysis_agent", "ranking_agent")
    workflow.add_edge("ranking_agent", "quality_evaluation_agent")
    workflow.add_edge("quality_evaluation_agent", "instagram_publisher_agent")
    workflow.add_edge("instagram_publisher_agent", END)

    return workflow.compile()