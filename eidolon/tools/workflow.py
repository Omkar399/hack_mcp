"""
Workflow Engine for Eidolon Tool Orchestration Framework

Provides workflow definition, execution, and management capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ..utils.logging import get_component_logger
from .base import ToolResult, ToolError
from .registry import ToolRegistry

logger = get_component_logger("tools.workflow")


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Types of workflow steps."""
    TOOL = "tool"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    DELAY = "delay"
    MANUAL = "manual"


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    id: str
    name: str
    step_type: StepType
    
    # Tool execution
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    
    # Flow control
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression
    on_success: Optional[str] = None  # Next step on success
    on_failure: Optional[str] = None  # Next step on failure
    
    # Loop control
    loop_condition: Optional[str] = None
    loop_max_iterations: int = 100
    
    # Parallel execution
    parallel_steps: List[str] = field(default_factory=list)
    
    # Delay
    delay_seconds: Optional[float] = None
    
    # Manual step
    manual_instructions: Optional[str] = None
    
    # Execution state
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class Workflow(BaseModel):
    """A workflow definition."""
    id: str = Field(description="Workflow ID")
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    # Steps
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict, description="Workflow steps")
    entry_point: str = Field(description="ID of first step to execute")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system")
    tags: List[str] = Field(default_factory=list)
    
    # Execution settings
    timeout_seconds: Optional[float] = Field(default=None, description="Overall workflow timeout")
    auto_retry: bool = Field(default=True, description="Auto-retry failed steps")
    stop_on_failure: bool = Field(default=False, description="Stop workflow on any failure")
    
    # Variables and context
    variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output schema")


@dataclass
class WorkflowExecution:
    """Runtime state of workflow execution."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # State
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    
    # Context
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, ToolResult] = field(default_factory=dict)
    
    # Execution metadata
    triggered_by: str = "system"
    execution_context: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """
    Engine for executing workflows with tool orchestration.
    """
    
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """Initialize the workflow engine."""
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Workflow storage
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Execution control
        self.max_concurrent_workflows = 5
        self.active_workflows: Dict[str, asyncio.Task] = {}
        
        # Built-in functions for conditions
        self.condition_functions = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all
        }
        
        logger.info("Workflow engine initialized")
    
    def register_workflow(self, workflow: Workflow) -> bool:
        """
        Register a workflow definition.
        
        Args:
            workflow: Workflow to register
            
        Returns:
            True if registration was successful
        """
        try:
            # Validate workflow
            validation_result = self._validate_workflow(workflow)
            if not validation_result["valid"]:
                logger.error(f"Workflow validation failed: {validation_result['errors']}")
                return False
            
            self.workflows[workflow.id] = workflow
            logger.info(f"Registered workflow: {workflow.name} (ID: {workflow.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register workflow: {e}")
            return False
    
    def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow."""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Unregistered workflow: {workflow_id}")
            return True
        return False
    
    async def execute_workflow(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            inputs: Input parameters
            context: Execution context
            execution_id: Custom execution ID
            
        Returns:
            Workflow execution state
            
        Raises:
            ToolError: If workflow execution fails
        """
        if workflow_id not in self.workflows:
            raise ToolError(f"Workflow not found: {workflow_id}")
        
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise ToolError("Maximum concurrent workflows reached")
        
        workflow = self.workflows[workflow_id]
        
        # Create execution state
        if execution_id is None:
            execution_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now(),
            variables={**workflow.variables, **(inputs or {})},
            execution_context=context or {}
        )
        
        self.executions[execution_id] = execution
        
        # Start execution
        try:
            logger.info(f"Starting workflow execution: {workflow.name} (ID: {execution_id})")
            
            execution_task = asyncio.create_task(
                self._execute_workflow_internal(workflow, execution)
            )
            self.active_workflows[execution_id] = execution_task
            
            # Wait for completion
            await execution_task
            
            logger.info(f"Workflow execution completed: {execution_id} (Status: {execution.status})")
            return execution
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            raise ToolError(f"Workflow execution failed: {str(e)}")
            
        finally:
            if execution_id in self.active_workflows:
                del self.active_workflows[execution_id]
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        if execution_id in self.active_workflows:
            self.active_workflows[execution_id].cancel()
            
            if execution_id in self.executions:
                self.executions[execution_id].status = WorkflowStatus.CANCELLED
                self.executions[execution_id].completed_at = datetime.now()
            
            logger.info(f"Cancelled workflow execution: {execution_id}")
            return True
        
        return False
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow (not implemented in this version)."""
        # Would require more complex state management
        logger.warning("Workflow pause not implemented yet")
        return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow definition by ID."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """List all registered workflows."""
        return list(self.workflows.values())
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        return self.executions.get(execution_id)
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[WorkflowExecution]:
        """List workflow executions with optional filters."""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return list(self.active_workflows.keys())
    
    async def _execute_workflow_internal(self, workflow: Workflow, execution: WorkflowExecution) -> None:
        """Internal workflow execution logic."""
        execution.status = WorkflowStatus.RUNNING
        execution.current_step = workflow.entry_point
        
        # Set overall timeout
        if workflow.timeout_seconds:
            timeout_task = asyncio.create_task(
                asyncio.sleep(workflow.timeout_seconds)
            )
        else:
            timeout_task = None
        
        try:
            # Execute steps starting from entry point
            await self._execute_step(workflow, execution, workflow.entry_point)
            
            # Check if workflow completed successfully
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
            
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
            raise
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            logger.error(f"Workflow execution error: {e}")
            raise
        finally:
            execution.completed_at = datetime.now()
            
            if timeout_task:
                timeout_task.cancel()
    
    async def _execute_step(self, workflow: Workflow, execution: WorkflowExecution, step_id: str) -> None:
        """Execute a single workflow step."""
        if step_id not in workflow.steps:
            logger.warning(f"Step not found: {step_id}")
            return
        
        step = workflow.steps[step_id]
        
        # Check dependencies
        if not self._are_dependencies_met(step, execution):
            logger.debug(f"Dependencies not met for step: {step_id}")
            return
        
        # Check condition
        if step.condition and not self._evaluate_condition(step.condition, execution):
            step.status = StepStatus.SKIPPED
            logger.debug(f"Step condition not met: {step_id}")
            return
        
        logger.debug(f"Executing step: {step.name} (ID: {step_id})")
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        execution.current_step = step_id
        
        try:
            # Execute based on step type
            if step.step_type == StepType.TOOL:
                await self._execute_tool_step(workflow, execution, step)
            elif step.step_type == StepType.DELAY:
                await self._execute_delay_step(step)
            elif step.step_type == StepType.CONDITION:
                await self._execute_condition_step(workflow, execution, step)
            elif step.step_type == StepType.LOOP:
                await self._execute_loop_step(workflow, execution, step)
            elif step.step_type == StepType.PARALLEL:
                await self._execute_parallel_step(workflow, execution, step)
            elif step.step_type == StepType.MANUAL:
                await self._execute_manual_step(step)
            else:
                raise ToolError(f"Unknown step type: {step.step_type}")
            
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            execution.completed_steps.append(step_id)
            
            # Determine next step
            next_step = step.on_success
            if next_step:
                await self._execute_step(workflow, execution, next_step)
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.completed_at = datetime.now()
            execution.failed_steps.append(step_id)
            
            logger.error(f"Step execution failed: {step.name} - {e}")
            
            # Handle failure
            if step.on_failure:
                await self._execute_step(workflow, execution, step.on_failure)
            elif workflow.stop_on_failure:
                raise
            else:
                # Continue to next step if available
                next_step = step.on_success
                if next_step:
                    await self._execute_step(workflow, execution, next_step)
    
    async def _execute_tool_step(self, workflow: Workflow, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a tool step."""
        if not step.tool_name:
            raise ToolError("Tool step missing tool_name")
        
        # Resolve parameters with variables
        resolved_params = self._resolve_parameters(step.parameters, execution)
        
        # Execute tool
        result = await self.tool_registry.execute_tool(
            tool_name=step.tool_name,
            parameters=resolved_params,
            context=execution.execution_context,
            timeout=step.timeout
        )
        
        step.result = result
        execution.step_results[step.id] = result
        
        # Update variables with result data
        if result.success and result.data:
            execution.variables.update(result.data)
    
    async def _execute_delay_step(self, step: WorkflowStep) -> None:
        """Execute a delay step."""
        if step.delay_seconds:
            await asyncio.sleep(step.delay_seconds)
    
    async def _execute_condition_step(self, workflow: Workflow, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a conditional step."""
        # Condition steps just control flow, no actual execution
        pass
    
    async def _execute_loop_step(self, workflow: Workflow, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a loop step."""
        iteration = 0
        
        while iteration < step.loop_max_iterations:
            if step.loop_condition and not self._evaluate_condition(step.loop_condition, execution):
                break
            
            # Execute loop body (would need to define loop body steps)
            # This is a simplified implementation
            iteration += 1
    
    async def _execute_parallel_step(self, workflow: Workflow, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute parallel steps."""
        if not step.parallel_steps:
            return
        
        # Execute all parallel steps concurrently
        tasks = []
        for parallel_step_id in step.parallel_steps:
            task = asyncio.create_task(
                self._execute_step(workflow, execution, parallel_step_id)
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_manual_step(self, step: WorkflowStep) -> None:
        """Execute a manual step (requires user intervention)."""
        # In a real implementation, this would notify the user and wait for confirmation
        logger.info(f"Manual step: {step.manual_instructions or 'Manual intervention required'}")
        # For now, just complete immediately
    
    def _are_dependencies_met(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Check if step dependencies are met."""
        for dep_id in step.depends_on:
            if dep_id not in execution.completed_steps:
                return False
        return True
    
    def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate a condition expression."""
        try:
            # Create safe evaluation context
            eval_context = {
                **self.condition_functions,
                "variables": execution.variables,
                "results": execution.step_results
            }
            
            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} - {e}")
            return False
    
    def _resolve_parameters(self, parameters: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Resolve parameter values with variable substitution."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Variable substitution
                var_name = value[2:-1]
                if var_name in execution.variables:
                    resolved[key] = execution.variables[var_name]
                else:
                    resolved[key] = value  # Keep original if not found
            else:
                resolved[key] = value
        
        return resolved
    
    def _validate_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Validate workflow definition."""
        errors = []
        
        # Check entry point exists
        if workflow.entry_point not in workflow.steps:
            errors.append(f"Entry point step not found: {workflow.entry_point}")
        
        # Check step references
        for step_id, step in workflow.steps.items():
            # Check dependencies
            for dep_id in step.depends_on:
                if dep_id not in workflow.steps:
                    errors.append(f"Step {step_id} dependency not found: {dep_id}")
            
            # Check next step references
            if step.on_success and step.on_success not in workflow.steps:
                errors.append(f"Step {step_id} on_success step not found: {step.on_success}")
            
            if step.on_failure and step.on_failure not in workflow.steps:
                errors.append(f"Step {step_id} on_failure step not found: {step.on_failure}")
            
            # Check tool existence for tool steps
            if step.step_type == StepType.TOOL and step.tool_name:
                if not self.tool_registry.get_tool(step.tool_name):
                    errors.append(f"Step {step_id} tool not found: {step.tool_name}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


def load_workflow_from_file(file_path: Union[str, Path]) -> Optional[Workflow]:
    """Load workflow from JSON or YAML file."""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Workflow file not found: {file_path}")
            return None
        
        content = file_path.read_text()
        
        if file_path.suffix.lower() == '.json':
            data = json.loads(content)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            data = yaml.safe_load(content)
        else:
            logger.error(f"Unsupported workflow file format: {file_path.suffix}")
            return None
        
        # Convert step data to WorkflowStep objects
        steps = {}
        for step_id, step_data in data.get("steps", {}).items():
            steps[step_id] = WorkflowStep(id=step_id, **step_data)
        
        data["steps"] = steps
        
        return Workflow(**data)
        
    except Exception as e:
        logger.error(f"Failed to load workflow from {file_path}: {e}")
        return None


def save_workflow_to_file(workflow: Workflow, file_path: Union[str, Path]) -> bool:
    """Save workflow to JSON or YAML file."""
    try:
        file_path = Path(file_path)
        
        # Convert to serializable dict
        data = workflow.dict()
        
        # Convert WorkflowStep objects to dicts
        steps_data = {}
        for step_id, step in data["steps"].items():
            steps_data[step_id] = step
        data["steps"] = steps_data
        
        if file_path.suffix.lower() == '.json':
            content = json.dumps(data, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            content = yaml.dump(data, indent=2, default_flow_style=False)
        else:
            logger.error(f"Unsupported workflow file format: {file_path.suffix}")
            return False
        
        file_path.write_text(content)
        logger.info(f"Saved workflow to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save workflow to {file_path}: {e}")
        return False