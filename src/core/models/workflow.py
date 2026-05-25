"""
Workflow management models for TPE system.

Defines workflow orchestration and state management models.
"""

from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator, model_validator

from .base import BaseTPEModel
from .enums import AgentType, SessionStatus, TaskStatus, WorkflowPhase


class WorkflowDefinition(BaseTPEModel):
    """Definition of a workflow with phases and task templates."""

    name: str = Field(description="Name of the workflow")
    description: str = Field(description="Description of the workflow")
    phases: List[WorkflowPhase] = Field(description="Phases in the workflow")
    phase_dependencies: Dict[WorkflowPhase, List[WorkflowPhase]] = Field(
        default_factory=dict, description="Dependencies between phases"
    )
    task_templates: Dict[WorkflowPhase, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Task templates for each phase"
    )
    max_iterations: int = Field(
        default=10, ge=1, description="Maximum workflow iterations"
    )
    timeout_minutes: int = Field(
        default=30, ge=1, description="Workflow timeout in minutes"
    )

    @model_validator(mode="after")
    def validate_workflow_definition(self):
        """Validate workflow definition consistency."""
        # Check that all phases in task_templates exist in phases
        for phase in self.task_templates:
            if phase not in self.phases:
                raise ValueError(f"Task template phase {phase} not in workflow phases")

        # Check that all dependency phases exist
        for phase, dependencies in self.phase_dependencies.items():
            if phase not in self.phases:
                raise ValueError(f"Dependency phase {phase} not in workflow phases")
            for dep in dependencies:
                if dep not in self.phases:
                    raise ValueError(
                        f"Dependency {dep} for phase {phase} not in workflow phases"
                    )

        return self


class WorkflowState(BaseTPEModel):
    """Current state of a workflow execution."""

    workflow_id: str = Field(description="ID of the workflow")
    session_id: str = Field(description="ID of the session")
    current_phase: WorkflowPhase = Field(description="Current workflow phase")
    completed_phases: Set[WorkflowPhase] = Field(
        default_factory=set, description="Completed phases"
    )
    failed_phases: Set[WorkflowPhase] = Field(
        default_factory=set, description="Failed phases"
    )
    phase_start_times: Dict[WorkflowPhase, str] = Field(
        default_factory=dict, description="Phase start times"
    )
    phase_durations: Dict[WorkflowPhase, float] = Field(
        default_factory=dict, description="Phase durations in seconds"
    )
    iteration_count: int = Field(default=0, ge=0, description="Current iteration count")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow context data"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow metrics"
    )

    def can_enter_phase(
        self, phase: WorkflowPhase, workflow_def: WorkflowDefinition
    ) -> bool:
        """Check if workflow can enter a specific phase."""
        # Check if phase is already completed
        if phase in self.completed_phases:
            return True

        # Check dependencies
        if phase in workflow_def.phase_dependencies:
            for dep in workflow_def.phase_dependencies[phase]:
                if dep not in self.completed_phases:
                    return False

        return True

    def mark_phase_started(self, phase: WorkflowPhase) -> None:
        """Mark a phase as started."""
        from datetime import datetime

        self.phase_start_times[phase] = datetime.utcnow().isoformat() + "Z"

    def mark_phase_completed(
        self, phase: WorkflowPhase, duration: Optional[float] = None
    ) -> None:
        """Mark a phase as completed."""
        self.completed_phases.add(phase)
        if phase in self.failed_phases:
            self.failed_phases.remove(phase)

        # Calculate duration if not provided
        if duration is None and phase in self.phase_start_times:
            from datetime import datetime

            start_time = datetime.fromisoformat(
                self.phase_start_times[phase].replace("Z", "+00:00")
            )
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

        if duration is not None:
            self.phase_durations[phase] = duration

    def mark_phase_failed(self, phase: WorkflowPhase) -> None:
        """Mark a phase as failed."""
        self.failed_phases.add(phase)

    def is_workflow_complete(self) -> bool:
        """Check if the entire workflow is complete."""
        return all(phase in self.completed_phases for phase in self.phase_durations)

    def get_progress_percentage(self) -> float:
        """Calculate workflow progress as percentage."""
        total_phases = len(self.phase_durations)
        if total_phases == 0:
            return 0.0
        return len(self.completed_phases) / total_phases * 100.0


class TaskDependency(BaseTPEModel):
    """Dependency between tasks."""

    task_id: str = Field(description="ID of the dependent task")
    depends_on: str = Field(description="ID of the task this depends on")
    dependency_type: str = Field(default="completion", description="Type of dependency")
    condition: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional condition for dependency"
    )


class TaskQueue(BaseTPEModel):
    """Queue of tasks to be executed."""

    queue_id: str = Field(description="ID of the task queue")
    tasks: List[str] = Field(
        default_factory=list, description="List of task IDs in order"
    )
    priority_tasks: List[str] = Field(
        default_factory=list, description="High priority tasks"
    )
    blocked_tasks: Set[str] = Field(default_factory=set, description="Blocked tasks")
    completed_tasks: Set[str] = Field(
        default_factory=set, description="Completed tasks"
    )
    dependencies: List[TaskDependency] = Field(
        default_factory=list, description="Task dependencies"
    )

    def add_task(self, task_id: str, priority: bool = False) -> None:
        """Add a task to the queue."""
        if priority:
            self.priority_tasks.append(task_id)
        else:
            self.tasks.append(task_id)

    def get_next_task(self) -> Optional[str]:
        """Get the next task to execute."""
        # Check priority tasks first
        if self.priority_tasks:
            for task_id in self.priority_tasks:
                if self._can_execute_task(task_id):
                    self.priority_tasks.remove(task_id)
                    return task_id

        # Check regular tasks
        for task_id in self.tasks:
            if self._can_execute_task(task_id):
                self.tasks.remove(task_id)
                return task_id

        return None

    def _can_execute_task(self, task_id: str) -> bool:
        """Check if a task can be executed."""
        if task_id in self.blocked_tasks or task_id in self.completed_tasks:
            return False

        # Check dependencies
        for dep in self.dependencies:
            if dep.task_id == task_id:
                if dep.depends_on not in self.completed_tasks:
                    return False

                # Check optional condition
                if dep.condition:
                    # Simplified condition checking - in practice, this would be more complex
                    pass

        return True

    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        self.completed_tasks.add(task_id)

    def block_task(self, task_id: str) -> None:
        """Block a task from execution."""
        self.blocked_tasks.add(task_id)

    def unblock_task(self, task_id: str) -> None:
        """Unblock a task for execution."""
        self.blocked_tasks.discard(task_id)


class WorkflowMetrics(BaseTPEModel):
    """Metrics collected during workflow execution."""

    workflow_id: str = Field(description="ID of the workflow")
    session_id: str = Field(description="ID of the session")
    start_time: str = Field(description="Workflow start time")
    end_time: Optional[str] = Field(default=None, description="Workflow end time")
    total_duration: Optional[float] = Field(
        default=None, description="Total duration in seconds"
    )
    phase_metrics: Dict[WorkflowPhase, Dict[str, Any]] = Field(
        default_factory=dict, description="Metrics per phase"
    )
    task_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Metrics per task"
    )
    agent_metrics: Dict[AgentType, Dict[str, Any]] = Field(
        default_factory=dict, description="Metrics per agent type"
    )
    error_counts: Dict[str, int] = Field(
        default_factory=dict, description="Error counts by type"
    )
    performance_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Key performance indicators"
    )

    def record_phase_start(self, phase: WorkflowPhase) -> None:
        """Record the start of a phase."""
        from datetime import datetime

        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = {}
        self.phase_metrics[phase]["start_time"] = datetime.utcnow().isoformat() + "Z"

    def record_phase_completion(self, phase: WorkflowPhase, success: bool) -> None:
        """Record the completion of a phase."""
        from datetime import datetime

        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = {}

        end_time = datetime.utcnow()
        self.phase_metrics[phase]["end_time"] = end_time.isoformat() + "Z"
        self.phase_metrics[phase]["success"] = success

        # Calculate duration
        if "start_time" in self.phase_metrics[phase]:
            start_time = datetime.fromisoformat(
                self.phase_metrics[phase]["start_time"].replace("Z", "+00:00")
            )
            duration = (end_time - start_time).total_seconds()
            self.phase_metrics[phase]["duration"] = duration

    def record_task_completion(
        self, task_id: str, agent_type: AgentType, success: bool, duration: float
    ) -> None:
        """Record the completion of a task."""
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = {}

        self.task_metrics[task_id].update(
            {"agent_type": agent_type, "success": success, "duration": duration}
        )

        # Update agent metrics
        if agent_type not in self.agent_metrics:
            self.agent_metrics[agent_type] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_duration": 0.0,
                "success_rate": 0.0,
            }

        agent_metrics = self.agent_metrics[agent_type]
        agent_metrics["total_tasks"] += 1
        agent_metrics["total_duration"] += duration

        if success:
            agent_metrics["successful_tasks"] += 1

        agent_metrics["success_rate"] = (
            agent_metrics["successful_tasks"] / agent_metrics["total_tasks"]
        )
        agent_metrics["average_duration"] = (
            agent_metrics["total_duration"] / agent_metrics["total_tasks"]
        )

    def finalize_metrics(self) -> None:
        """Finalize workflow metrics."""
        from datetime import datetime

        self.end_time = datetime.utcnow().isoformat() + "Z"

        # Calculate total duration
        if self.start_time:
            start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.total_duration = (end - start).total_seconds()

        # Calculate key performance indicators
        total_tasks = sum(
            metrics.get("total_tasks", 0) for metrics in self.agent_metrics.values()
        )
        successful_tasks = sum(
            metrics.get("successful_tasks", 0)
            for metrics in self.agent_metrics.values()
        )

        if total_tasks > 0:
            self.performance_indicators["overall_success_rate"] = (
                successful_tasks / total_tasks
            )
            self.performance_indicators["average_task_duration"] = (
                sum(
                    metrics.get("total_duration", 0)
                    for metrics in self.agent_metrics.values()
                )
                / total_tasks
            )

        self.performance_indicators["total_errors"] = sum(self.error_counts.values())
