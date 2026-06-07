from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from agents.state import AgentState


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the LangGraph pipeline.

    Each agent:
    - Reads required inputs from AgentState
    - Performs its specific task
    - Returns a dict of state field updates (LangGraph merges these)
    - Handles errors gracefully without crashing the pipeline
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = structlog.get_logger(name)

    @abstractmethod
    async def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the agent's core logic.

        Args:
            state: The current shared pipeline state.

        Returns:
            Dict of state field updates. LangGraph merges these into the state.
        """

    async def validate_inputs(self, state: AgentState) -> bool:
        """Validate required inputs are present. Override in subclasses as needed."""
        return True

    def _error_update(self, state: AgentState, error: Exception) -> dict[str, Any]:
        """Build a state update dict that records an agent error."""
        error_msg = f"[{self.name}] {type(error).__name__}: {error}"
        self.logger.error("agent_error", agent=self.name, error=str(error))
        return {
            "errors": [error_msg],
            "current_step": f"{self.name}:error",
        }
