"""
OpenHands-inspired agent loop patterns for Nexus Coder v0.3
===========================================================
Ported & simplified from OpenHands/OpenHands (MIT).

OpenHands models the agent as a loop:
    PLAN → ACT → OBSERVE → REFLECT → PLAN (next)

This module provides a generic agent-loop scaffold with:
  - Planner: decomposes high-level goal into steps
  - Executor: runs a single step (calls a Tool)
  - Observer: parses the result, detects success/failure
  - Reflector: revises the plan if the step failed

It is NOT a replacement for `nexus.agent.agent.NexusAgent` — rather, an
alternative pattern that can be used when the task is well-defined.

Original attribution:
    OpenHands (formerly OpenDevin): an open platform for AI software developers.
    Authors: OpenHands contributors.
    License: MIT
    Source: https://github.com/OpenHands/OpenHands
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any


@dataclass
class AgentStep:
    """A single step in the agent's plan."""
    description: str
    tool: Optional[str] = None                # tool name to invoke
    args: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[str] = None            # what a successful result looks like
    actual: Optional[Any] = None              # observed result (set after execution)
    status: str = "pending"                   # pending | running | done | failed
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2


class Planner:
    """Decomposes a goal into a list of steps.

    Default planner is a thin heuristic wrapper. For real use, replace
    with an LLM-backed planner.
    """

    def __init__(self, llm_planner: Optional[Callable[[str], List[AgentStep]]] = None):
        self.llm_planner = llm_planner

    def plan(self, goal: str) -> List[AgentStep]:
        if self.llm_planner is not None:
            return self.llm_planner(goal)
        # Fallback: single step that just calls chat
        return [AgentStep(
            description=f"Address goal: {goal}",
            tool=None,
            expected="A useful response",
        )]


class Executor:
    """Executes a single step by invoking a tool (or chat as fallback)."""

    def __init__(self, tool_registry=None, chat_callback: Optional[Callable[[str], str]] = None):
        self.tool_registry = tool_registry
        self.chat_callback = chat_callback

    def execute(self, step: AgentStep) -> Any:
        step.status = "running"
        try:
            if step.tool and self.tool_registry is not None:
                result = self.tool_registry.execute(step.tool, step.args)
                step.actual = result.output if hasattr(result, "output") else result
                step.status = "done"
            elif self.chat_callback is not None:
                step.actual = self.chat_callback(step.description)
                step.status = "done"
            else:
                step.actual = "[no executor configured]"
                step.status = "failed"
                step.error = "No executor"
        except Exception as e:
            step.actual = None
            step.error = str(e)
            step.status = "failed"
        return step.actual


class Observer:
    """Parses tool results to decide success/failure."""

    def observe(self, step: AgentStep) -> bool:
        """Return True if step succeeded."""
        if step.status != "done":
            return False
        if step.expected is None:
            return True
        # Naive substring match — replace with LLM check in production
        actual_str = str(step.actual or "").lower()
        return step.expected.lower() in actual_str


class Reflector:
    """Revises the plan when a step fails.

    Default: retry up to max_retries, then mark failed and skip.
    """

    def reflect(self, step: AgentStep, plan: List[AgentStep]) -> List[AgentStep]:
        if step.status == "failed" and step.retries < step.max_retries:
            step.retries += 1
            step.status = "pending"
            step.error = None
        return plan


class AgentLoop:
    """Generic agent loop combining Planner, Executor, Observer, Reflector."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        observer: Optional[Observer] = None,
        reflector: Optional[Reflector] = None,
        max_iterations: int = 20,
    ):
        self.planner = planner or Planner()
        self.executor = executor or Executor()
        self.observer = observer or Observer()
        self.reflector = reflector or Reflector()
        self.max_iterations = max_iterations

    def run(self, goal: str) -> List[AgentStep]:
        """Execute the agent loop until all steps are done or max_iterations reached."""
        plan = self.planner.plan(goal)
        for _ in range(self.max_iterations):
            pending = [s for s in plan if s.status == "pending"]
            if not pending:
                break
            step = pending[0]
            self.executor.execute(step)
            ok = self.observer.observe(step)
            if not ok:
                plan = self.reflector.reflect(step, plan)
        return plan


__all__ = ["AgentStep", "Planner", "Executor", "Observer", "Reflector", "AgentLoop"]
