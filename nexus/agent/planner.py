"""Task Planner - Lập kế hoạch cho multi-step tasks."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Một task trong plan."""
    id: int
    description: str
    skill: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """Một execution plan."""
    id: str
    goal: str
    tasks: List[Task] = field(default_factory=list)
    created_at: str = ""
    status: TaskStatus = TaskStatus.PENDING
    
    def add_task(self, task: Task) -> None:
        self.tasks.append(task)
    
    def get_next_task(self) -> Optional[Task]:
        """Get next pending task whose dependencies are met."""
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            # Check dependencies
            deps_met = all(
                self.tasks[dep_id].status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                for dep_id in task.depends_on
                if dep_id < len(self.tasks)
            )
            if deps_met:
                return task
        return None
    
    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED) for t in self.tasks)
    
    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "is_complete": self.is_complete(),
        }


class TaskPlanner:
    """Lập kế hoạch cho complex multi-step tasks.
    
    Features:
    - Decompose goal thành subtasks
    - Identify dependencies
    - Suggest skills/tools per task
    - Track execution status
    
    Usage:
        planner = TaskPlanner()
        plan = planner.create_plan("Build a REST API for todo app")
        for task in plan.tasks:
            print(f"Task {task.id}: {task.description}")
    """
    
    def __init__(self):
        self._plans: List[Plan] = []
        self._next_plan_id = 1
    
    def create_plan(self, goal: str) -> Plan:
        """Create an execution plan for a goal."""
        plan = Plan(
            id=f"plan_{self._next_plan_id}",
            goal=goal,
            created_at=__import__("datetime").datetime.now().isoformat(),
        )
        self._next_plan_id += 1
        
        # Decompose goal into tasks
        tasks = self._decompose(goal)
        for i, task_def in enumerate(tasks):
            task = Task(
                id=i,
                description=task_def["description"],
                skill=task_def.get("skill"),
                tools=task_def.get("tools", []),
                depends_on=task_def.get("depends_on", []),
            )
            plan.add_task(task)
        
        self._plans.append(plan)
        return plan
    
    def _decompose(self, goal: str) -> List[Dict[str, Any]]:
        """Decompose goal into subtasks.
        
        This is a heuristic-based decomposition.
        In production, this would use the LLM itself.
        """
        goal_lower = goal.lower()
        tasks = []
        
        # Common patterns
        if any(kw in goal_lower for kw in ["build", "create", "develop", "implement"]):
            tasks.extend([
                {
                    "description": f"Analyze requirements for: {goal}",
                    "skill": "reasoning",
                    "tools": [],
                },
                {
                    "description": "Design architecture and data models",
                    "skill": "algorithm_design",
                    "tools": [],
                    "depends_on": [0],
                },
                {
                    "description": "Implement core functionality",
                    "skill": "code_generation",
                    "tools": ["file_write", "python_exec"],
                    "depends_on": [1],
                },
                {
                    "description": "Write tests",
                    "skill": "testing",
                    "tools": ["python_exec", "shell_exec"],
                    "depends_on": [2],
                },
                {
                    "description": "Generate documentation",
                    "skill": "documentation",
                    "tools": ["file_write"],
                    "depends_on": [2],
                },
                {
                    "description": "Review and optimize code",
                    "skill": "code_review",
                    "tools": ["code_search", "code_lint"],
                    "depends_on": [3, 4],
                },
            ])
        elif any(kw in goal_lower for kw in ["debug", "fix", "repair"]):
            tasks.extend([
                {
                    "description": "Reproduce the issue",
                    "skill": "debugging",
                    "tools": ["shell_exec", "python_exec"],
                },
                {
                    "description": "Identify root cause",
                    "skill": "debugging",
                    "tools": ["code_search", "regex_search"],
                    "depends_on": [0],
                },
                {
                    "description": "Implement fix",
                    "skill": "code_generation",
                    "tools": ["file_write"],
                    "depends_on": [1],
                },
                {
                    "description": "Verify fix with tests",
                    "skill": "testing",
                    "tools": ["python_exec"],
                    "depends_on": [2],
                },
            ])
        elif any(kw in goal_lower for kw in ["analyze", "investigate", "understand"]):
            tasks.extend([
                {
                    "description": f"Gather information about: {goal}",
                    "skill": "reasoning",
                    "tools": ["web_search", "web_fetch", "file_read"],
                },
                {
                    "description": "Analyze and synthesize findings",
                    "skill": "data_analysis",
                    "tools": ["python_exec"],
                    "depends_on": [0],
                },
                {
                    "description": "Present insights and recommendations",
                    "skill": "summarization",
                    "tools": [],
                    "depends_on": [1],
                },
            ])
        else:
            # Default: single task
            tasks.append({
                "description": f"Handle: {goal}",
                "skill": None,
                "tools": [],
            })
        
        return tasks
    
    def execute_plan(
        self,
        plan: Plan,
        executor=None,
    ) -> Plan:
        """Execute a plan step by step.
        
        Args:
            plan: Plan to execute
            executor: Function(task) -> result (None = simulation)
        """
        while not plan.is_complete():
            task = plan.get_next_task()
            if task is None:
                break
            
            task.status = TaskStatus.IN_PROGRESS
            try:
                if executor:
                    result = executor(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = "[simulated]"
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = f"Error: {e}"
        
        return plan
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans."""
        return [p.summary() for p in self._plans]
