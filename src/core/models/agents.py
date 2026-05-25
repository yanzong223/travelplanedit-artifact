"""
Agent coordination data models for TPE system.

Defines models for agent tasks, workflow sessions, and coordination.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import AuditEntry, BaseTPEModel
from .enums import AgentType, SessionStatus, TaskStatus, WorkflowPhase


class AgentTask(BaseTPEModel):
    """Task assigned to specific agent."""

    agent_type: AgentType = Field(description="Type of agent to execute this task")
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the task"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Output data from the task"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status of the task"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if task failed"
    )
    workflow_phase: WorkflowPhase = Field(
        description="Workflow phase this task belongs to"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Task priority (1-10, higher is more important)",
    )
    timeout_seconds: int = Field(
        default=300, ge=1, description="Task timeout in seconds"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retries allowed"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of tasks this task depends on"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional task metadata"
    )

    @field_validator("status")
    @classmethod
    def validate_status_transition(cls, v, info):
        """Ensure status transitions are valid."""
        # This is a simplified validation - in practice, you'd want to check
        # the previous status and ensure the transition is valid
        return v

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def is_completed(self) -> bool:
        """Check if task is completed (successfully or permanently failed)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.IN_PROGRESS

    def mark_in_progress(self) -> None:
        """Mark task as in progress."""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, output_data: Dict[str, Any]) -> None:
        """Mark task as completed with output data."""
        self.status = TaskStatus.COMPLETED
        self.output_data = output_data

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed and increment retry count."""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.retry_count += 1

    def reset_for_retry(self) -> None:
        """Reset task status for retry."""
        if self.can_retry():
            self.status = TaskStatus.PENDING
            self.error_message = None


class WorkflowSession(BaseTPEModel):
    """Orchestrated workflow for edit processing."""

    plan_id: str = Field(description="ID of the travel plan being edited")
    edit_request_id: str = Field(description="ID of the edit request being processed")
    tasks: List[AgentTask] = Field(
        default_factory=list, description="Tasks in this workflow"
    )
    current_phase: WorkflowPhase = Field(description="Current workflow phase")
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE, description="Current session status"
    )
    iterations: int = Field(
        default=0, ge=0, description="Number of iterations completed"
    )
    final_edit: Optional[str] = Field(
        default=None, description="ID of the final edit operation"
    )
    audit_trail: List[AuditEntry] = Field(
        default_factory=list, description="Audit trail for this session"
    )
    session_data: Dict[str, Any] = Field(
        default_factory=dict, description="Session-specific data"
    )
    error_summary: Dict[str, int] = Field(
        default_factory=dict, description="Summary of errors by type"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )

    @field_validator("current_phase")
    @classmethod
    def validate_phase_transition(cls, v, info):
        """Ensure phase transitions are valid."""
        # Simplified validation - in practice, check the workflow state machine
        return v

    def add_task(self, task: AgentTask) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)

    def get_tasks_by_phase(self, phase: WorkflowPhase) -> List[AgentTask]:
        """Get all tasks in a specific phase."""
        return [task for task in self.tasks if task.workflow_phase == phase]

    def get_tasks_by_status(self, status: TaskStatus) -> List[AgentTask]:
        """Get all tasks with a specific status."""
        return [task for task in self.tasks if task.status == status]

    def get_pending_tasks(self) -> List[AgentTask]:
        """Get all pending tasks."""
        return self.get_tasks_by_status(TaskStatus.PENDING)

    def get_failed_tasks(self) -> List[AgentTask]:
        """Get all failed tasks."""
        return self.get_tasks_by_status(TaskStatus.FAILED)

    def is_phase_complete(self, phase: WorkflowPhase) -> bool:
        """Check if all tasks in a phase are completed."""
        phase_tasks = self.get_tasks_by_phase(phase)
        return all(task.is_completed() for task in phase_tasks)

    def can_advance_to_next_phase(self) -> bool:
        """Check if workflow can advance to the next phase."""
        current_phase_tasks = self.get_tasks_by_phase(self.current_phase)
        return all(task.status == TaskStatus.COMPLETED for task in current_phase_tasks)

    def advance_phase(self, next_phase: WorkflowPhase) -> None:
        """Advance to the next workflow phase."""
        if self.can_advance_to_next_phase():
            self.current_phase = next_phase

    def add_audit_entry(self, entry: AuditEntry) -> None:
        """Add an audit entry to the session."""
        self.audit_trail.append(entry)

    def record_error(self, error_type: str, count: int = 1) -> None:
        """Record an error occurrence."""
        self.error_summary[error_type] = self.error_summary.get(error_type, 0) + count

    def get_completion_rate(self) -> float:
        """Calculate the completion rate of tasks."""
        if not self.tasks:
            return 0.0
        completed_tasks = len(self.get_tasks_by_status(TaskStatus.COMPLETED))
        return completed_tasks / len(self.tasks)

    def get_average_task_duration(self) -> float:
        """Calculate average task duration in seconds."""
        completed_tasks = self.get_tasks_by_status(TaskStatus.COMPLETED)
        if not completed_tasks:
            return 0.0

        total_duration = 0.0
        for task in completed_tasks:
            if task.updated_at and task.created_at:
                duration = (task.updated_at - task.created_at).total_seconds()
                total_duration += duration

        return total_duration / len(completed_tasks)

    def mark_converged(self, final_edit_id: Optional[str] = None) -> None:
        """Mark the session as converged."""
        self.status = SessionStatus.CONVERGED
        self.final_edit = final_edit_id

    def mark_failed(self, reason: str) -> None:
        """Mark the session as failed."""
        self.status = SessionStatus.FAILED
        self.session_data["failure_reason"] = reason

    def mark_escalated(self, reason: str) -> None:
        """Mark the session as escalated."""
        self.status = SessionStatus.ESCALATED
        self.session_data["escalation_reason"] = reason


class EditRequest(BaseTPEModel):
    """User request for plan modification.
    
    Compatible with TPE dataset episode format:
    - episode.edit_request.natural_language -> natural_language
    - episode.edit_request.priority -> priority  
    - episode.edit_request.allowed_operations -> allowed_operations
    - episode.edit_request.max_candidates -> max_candidates
    - episode.edit_request.metadata -> metadata (scene_type, query_id, etc.)
    """

    plan_id: str = Field(default="", description="ID of the travel plan to modify")
    natural_language: str = Field(
        description="Natural language description of the edit request"
    )
    priority: str = Field(
        default="MEDIUM", description="Priority level (HIGH, MEDIUM, LOW)"
    )
    requested_by: str = Field(
        default="user", description="User or system making the request"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Additional constraints for the edit"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User preferences"
    )
    allowed_operations: Optional[List[str]] = Field(
        default=None, description="Allowed operation types (insert, replace, adjust_time, etc.)"
    )
    max_candidates: Optional[int] = Field(
        default=5, ge=1, le=10, description="Maximum number of edit candidates"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Episode metadata (scene_type, query_id, language, etc.)"
    )
    # Original query from ChinaTravel for complete constraint extraction
    original_query: Optional[str] = Field(
        default=None,
        description="Original ChinaTravel query text for extracting base constraints"
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        """Validate priority value."""
        valid_priorities = ["HIGH", "MEDIUM", "LOW"]
        if v.upper() not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}")
        return v.upper()
    
    @classmethod
    def from_episode(cls, episode_data: Dict[str, Any], plan_id: str = "") -> "EditRequest":
        """Create EditRequest from TPE dataset episode format.
        
        Args:
            episode_data: Episode JSON data containing edit_request field
            plan_id: Plan ID (can be extracted from episode or provided)
            
        Returns:
            EditRequest instance
        """
        edit_req = episode_data.get("edit_request", {})
        return cls(
            plan_id=plan_id or episode_data.get("plan_id", ""),
            natural_language=edit_req.get("natural_language", ""),
            priority=edit_req.get("priority", "MEDIUM"),
            allowed_operations=edit_req.get("allowed_operations"),
            max_candidates=edit_req.get("max_candidates", 5),
            metadata=edit_req.get("metadata", {}),
        )


class EditOperation(BaseTPEModel):
    """Minimal edit Δ to resolve violations or implement requests."""

    plan_id: str = Field(description="ID of the travel plan")
    operation_type: str = Field(description="Type of edit operation")
    target_variables: List[str] = Field(description="Variables this operation affects")
    changes: Dict[str, Any] = Field(description="Changes to apply")
    rationale: Optional[str] = Field(
        default=None, description="Explanation for the edit"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    proof: Optional[Dict[str, Any]] = Field(
        default=None, description="Proof or justification"
    )
    audit_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Audit trail data"
    )
    applied_at: Optional[str] = Field(
        default=None, description="Timestamp when edit was applied"
    )
    applied_by: Optional[str] = Field(
        default=None, description="ID of agent that applied the edit"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def mark_applied(self, applied_by: str) -> None:
        """Mark the edit as applied."""
        from datetime import datetime

        self.applied_at = datetime.utcnow().isoformat() + "Z"
        self.applied_by = applied_by


class AgentState(BaseTPEModel):
    """State information for an agent."""

    agent_type: AgentType = Field(description="Type of agent")
    is_active: bool = Field(
        default=True, description="Whether the agent is currently active"
    )
    current_task_id: Optional[str] = Field(
        default=None, description="ID of current task being processed"
    )
    total_tasks_processed: int = Field(
        default=0, ge=0, description="Total number of tasks processed"
    )
    successful_tasks: int = Field(
        default=0, ge=0, description="Number of successful tasks"
    )
    failed_tasks: int = Field(default=0, ge=0, description="Number of failed tasks")
    average_processing_time: float = Field(
        default=0.0, ge=0.0, description="Average processing time in seconds"
    )
    last_activity: Optional[str] = Field(
        default=None, description="Timestamp of last activity"
    )
    configuration: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration"
    )

    def get_success_rate(self) -> float:
        """Calculate the success rate of the agent."""
        total = self.successful_tasks + self.failed_tasks
        if total == 0:
            return 0.0
        return self.successful_tasks / total

    def update_task_completion(self, success: bool, processing_time: float) -> None:
        """Update agent statistics after task completion."""
        self.total_tasks_processed += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1

        # Update average processing time
        if self.total_tasks_processed == 1:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (
                self.average_processing_time * (self.total_tasks_processed - 1)
                + processing_time
            ) / self.total_tasks_processed

        from datetime import datetime

        self.last_activity = datetime.utcnow().isoformat() + "Z"
