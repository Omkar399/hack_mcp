"""
Plan Execution system for Eidolon AI Personal Assistant

Handles execution of complex task plans with adaptive scheduling, resource management,
and real-time optimization capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from .task_planner import Task, TaskPlan, TaskStatus, TaskPriority
from .dependency_analyzer import DependencyAnalyzer
from .resource_manager import ResourceManager


class ExecutionMode(Enum):
    """Plan execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    OPTIMIZED = "optimized"


class ExecutionStrategy(Enum):
    """Execution strategies for handling failures and conflicts"""
    FAIL_FAST = "fail_fast"
    CONTINUE_ON_ERROR = "continue_on_error"
    RETRY_FAILED = "retry_failed"
    ADAPTIVE_RECOVERY = "adaptive_recovery"


@dataclass
class ExecutionContext:
    """Context information for task execution"""
    plan_id: str
    execution_id: str
    mode: ExecutionMode
    strategy: ExecutionStrategy
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    current_phase: str = "initialization"
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Individual task execution tracking"""
    task_id: str
    execution_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    resource_allocations: List[str] = field(default_factory=list)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


class PlanExecutor:
    """Advanced plan execution engine with adaptive capabilities"""
    
    def __init__(self):
        self.logger = get_component_logger("plan_executor")
        self.config = get_config()
        
        # Dependencies
        self.dependency_analyzer = DependencyAnalyzer()
        self.resource_manager = ResourceManager()
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.task_executions: Dict[str, TaskExecution] = {}
        self.execution_history: List[ExecutionContext] = []
        
        # Configuration
        self.max_parallel_tasks = 5
        self.max_retry_attempts = 3
        self.execution_timeout = timedelta(hours=24)
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        
        self.task_handlers.update({
            "information_gathering": self._handle_information_gathering,
            "file_operation": self._handle_file_operation,
            "communication": self._handle_communication,
            "automation": self._handle_automation,
            "analysis": self._handle_analysis,
            "creative": self._handle_creative,
            "system_task": self._handle_system_task
        })
    
    @log_exceptions()
    async def execute_plan(
        self,
        plan: TaskPlan,
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE_RECOVERY
    ) -> Dict[str, Any]:
        """Execute a complete task plan"""
        
        execution_id = f"exec_{plan.id}_{int(datetime.now().timestamp())}"
        self.logger.info(f"Starting execution {execution_id} for plan {plan.id}")
        
        # Create execution context
        context = ExecutionContext(
            plan_id=plan.id,
            execution_id=execution_id,
            mode=mode,
            strategy=strategy,
            started_at=datetime.now()
        )
        
        self.active_executions[execution_id] = context
        
        try:
            # Pre-execution analysis and optimization
            await self._prepare_execution(plan, context)
            
            # Execute based on mode
            if mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(plan, context)
            elif mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(plan, context)
            elif mode == ExecutionMode.ADAPTIVE:
                result = await self._execute_adaptive(plan, context)
            elif mode == ExecutionMode.OPTIMIZED:
                result = await self._execute_optimized(plan, context)
            else:
                raise ValueError(f"Unknown execution mode: {mode}")
            
            # Post-execution analysis
            await self._finalize_execution(plan, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution {execution_id} failed: {e}")
            context.current_phase = "failed"
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "completed_tasks": len(context.completed_tasks),
                "failed_tasks": len(context.failed_tasks)
            }
        finally:
            # Clean up
            if execution_id in self.active_executions:
                self.execution_history.append(context)
                del self.active_executions[execution_id]
    
    async def _prepare_execution(self, plan: TaskPlan, context: ExecutionContext):
        """Prepare plan for execution"""
        
        context.current_phase = "preparation"
        
        # Analyze dependencies
        dependency_analysis = await self.dependency_analyzer.analyze_plan_dependencies(plan)
        
        # Check resource availability
        resource_conflicts = await self.resource_manager.check_resource_conflicts(plan)
        
        # Optimize if needed
        if resource_conflicts:
            optimization_result = await self.resource_manager.optimize_resource_allocation(plan)
            context.metrics["optimization_applied"] = optimization_result.get("optimization_applied", False)
        
        # Estimate completion time
        total_duration = sum(
            (task.estimated_duration or timedelta(minutes=30)).total_seconds()
            for task in plan.tasks.values()
        )
        
        if context.mode in [ExecutionMode.PARALLEL, ExecutionMode.ADAPTIVE]:
            # Parallel execution can reduce time
            parallelization_factor = dependency_analysis.get("metrics", {}).get("parallelization_potential", 0.3)
            total_duration *= (1 - parallelization_factor * 0.5)
        
        context.estimated_completion = context.started_at + timedelta(seconds=total_duration)
        context.metrics.update({
            "dependency_complexity": dependency_analysis.get("metrics", {}).get("total_dependencies", 0),
            "resource_conflicts": len(resource_conflicts),
            "estimated_duration_seconds": total_duration
        })
    
    async def _execute_sequential(self, plan: TaskPlan, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan sequentially"""
        
        context.current_phase = "sequential_execution"
        
        for task_id in plan.execution_order:
            if task_id not in plan.tasks:
                continue
            
            task = plan.tasks[task_id]
            
            # Execute task
            execution_result = await self._execute_single_task(task, context)
            
            if not execution_result["success"] and context.strategy == ExecutionStrategy.FAIL_FAST:
                return {
                    "success": False,
                    "execution_id": context.execution_id,
                    "failed_at_task": task_id,
                    "completed_tasks": len(context.completed_tasks),
                    "total_tasks": len(plan.tasks)
                }
        
        return {
            "success": len(context.failed_tasks) == 0,
            "execution_id": context.execution_id,
            "completed_tasks": len(context.completed_tasks),
            "failed_tasks": len(context.failed_tasks),
            "total_tasks": len(plan.tasks),
            "execution_time": (datetime.now() - context.started_at).total_seconds()
        }
    
    async def _execute_parallel(self, plan: TaskPlan, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan with parallel task execution"""
        
        context.current_phase = "parallel_execution"
        
        # Group tasks that can run in parallel
        parallel_groups = self._create_execution_groups(plan)
        
        for group in parallel_groups:
            # Execute all tasks in group concurrently
            tasks = [plan.tasks[task_id] for task_id in group if task_id in plan.tasks]
            
            if len(tasks) <= self.max_parallel_tasks:
                # Execute all tasks in parallel
                await self._execute_task_group(tasks, context)
            else:
                # Split into smaller batches
                for i in range(0, len(tasks), self.max_parallel_tasks):
                    batch = tasks[i:i + self.max_parallel_tasks]
                    await self._execute_task_group(batch, context)
        
        return {
            "success": len(context.failed_tasks) == 0,
            "execution_id": context.execution_id,
            "completed_tasks": len(context.completed_tasks),
            "failed_tasks": len(context.failed_tasks),
            "total_tasks": len(plan.tasks),
            "parallel_groups": len(parallel_groups),
            "execution_time": (datetime.now() - context.started_at).total_seconds()
        }
    
    async def _execute_adaptive(self, plan: TaskPlan, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan with adaptive strategy"""
        
        context.current_phase = "adaptive_execution"
        
        # Start with ready tasks
        remaining_tasks = set(plan.tasks.keys())
        
        while remaining_tasks:
            # Find tasks ready to execute
            ready_tasks = self._get_ready_tasks(plan, remaining_tasks, context)
            
            if not ready_tasks:
                # Check for deadlock
                if context.active_tasks:
                    # Wait for active tasks to complete
                    await asyncio.sleep(1)
                    continue
                else:
                    # All remaining tasks are blocked
                    self.logger.warning(f"Execution deadlock detected with {len(remaining_tasks)} tasks remaining")
                    break
            
            # Determine optimal execution approach for ready tasks
            if len(ready_tasks) == 1:
                # Single task - execute directly
                task = plan.tasks[ready_tasks[0]]
                await self._execute_single_task(task, context)
                remaining_tasks.discard(ready_tasks[0])
            else:
                # Multiple ready tasks - execute in parallel up to limit
                tasks_to_execute = ready_tasks[:self.max_parallel_tasks]
                tasks = [plan.tasks[task_id] for task_id in tasks_to_execute]
                await self._execute_task_group(tasks, context)
                remaining_tasks -= set(tasks_to_execute)
        
        return {
            "success": len(context.failed_tasks) == 0,
            "execution_id": context.execution_id,
            "completed_tasks": len(context.completed_tasks),
            "failed_tasks": len(context.failed_tasks),
            "blocked_tasks": len(context.blocked_tasks),
            "total_tasks": len(plan.tasks),
            "execution_time": (datetime.now() - context.started_at).total_seconds()
        }
    
    async def _execute_optimized(self, plan: TaskPlan, context: ExecutionContext) -> Dict[str, Any]:
        """Execute plan with optimization"""
        
        context.current_phase = "optimized_execution"
        
        # Apply optimizations before execution
        optimization_result = await self.resource_manager.optimize_resource_allocation(plan)
        
        # Use adaptive execution with optimizations
        return await self._execute_adaptive(plan, context)
    
    def _create_execution_groups(self, plan: TaskPlan) -> List[List[str]]:
        """Create groups of tasks that can execute in parallel"""
        
        groups = []
        remaining_tasks = set(plan.tasks.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies on remaining tasks
            ready_tasks = []
            for task_id in remaining_tasks:
                task = plan.tasks[task_id]
                if all(dep not in remaining_tasks for dep in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Break circular dependencies by taking one task
                ready_tasks = [list(remaining_tasks)[0]]
            
            groups.append(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return groups
    
    def _get_ready_tasks(self, plan: TaskPlan, remaining_tasks: set, context: ExecutionContext) -> List[str]:
        """Get tasks that are ready to execute"""
        
        ready_tasks = []
        
        for task_id in remaining_tasks:
            if task_id in context.active_tasks:
                continue  # Already executing
            
            task = plan.tasks[task_id]
            
            # Check if all dependencies are completed
            deps_completed = all(
                dep_id in context.completed_tasks or dep_id not in plan.tasks
                for dep_id in task.dependencies
            )
            
            if deps_completed:
                ready_tasks.append(task_id)
        
        # Sort by priority
        ready_tasks.sort(key=lambda tid: plan.tasks[tid].priority.value)
        
        return ready_tasks
    
    async def _execute_task_group(self, tasks: List[Task], context: ExecutionContext):
        """Execute a group of tasks concurrently"""
        
        # Create execution coroutines
        execution_coroutines = [
            self._execute_single_task(task, context)
            for task in tasks
        ]
        
        # Execute all tasks concurrently
        await asyncio.gather(*execution_coroutines, return_exceptions=True)
    
    async def _execute_single_task(self, task: Task, context: ExecutionContext) -> Dict[str, Any]:
        """Execute a single task"""
        
        task_execution = TaskExecution(
            task_id=task.id,
            execution_id=context.execution_id,
            status=TaskStatus.IN_PROGRESS
        )
        
        self.task_executions[task.id] = task_execution
        context.active_tasks.append(task.id)
        
        try:
            self.logger.info(f"Starting execution of task {task.id}: {task.title}")
            
            # Allocate resources
            allocation_result = await self.resource_manager.allocate_resources(task)
            
            if not allocation_result["success"]:
                # Resource allocation failed
                task_execution.status = TaskStatus.BLOCKED
                task_execution.error = f"Resource allocation failed: {allocation_result.get('conflicts', [])}"
                context.blocked_tasks.append(task.id)
                return {"success": False, "error": "Resource allocation failed"}
            
            task_execution.resource_allocations = allocation_result.get("allocation_ids", [])
            
            # Start resource usage
            await self.resource_manager.start_resource_usage(task.id)
            
            # Execute task based on type
            task_execution.started_at = datetime.now()
            task_execution.status = TaskStatus.IN_PROGRESS
            
            handler = self.task_handlers.get(task.task_type.value, self._handle_default_task)
            result = await handler(task, task_execution)
            
            # Update task execution
            task_execution.completed_at = datetime.now()
            task_execution.progress = 1.0
            task_execution.result = result
            task_execution.status = TaskStatus.COMPLETED
            
            # Release resources
            await self.resource_manager.release_resources(task.id)
            
            # Update context
            context.completed_tasks.append(task.id)
            
            self.logger.info(f"Completed task {task.id}")
            return {"success": True, "result": result}
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            
            # Handle failure based on strategy
            task_execution.error = str(e)
            task_execution.status = TaskStatus.FAILED
            
            # Release resources
            await self.resource_manager.release_resources(task.id)
            
            # Retry logic
            if (task_execution.retry_count < self.max_retry_attempts and 
                context.strategy in [ExecutionStrategy.RETRY_FAILED, ExecutionStrategy.ADAPTIVE_RECOVERY]):
                
                task_execution.retry_count += 1
                self.logger.info(f"Retrying task {task.id} (attempt {task_execution.retry_count})")
                
                # Wait before retry
                await asyncio.sleep(2 ** task_execution.retry_count)  # Exponential backoff
                
                # Retry
                return await self._execute_single_task(task, context)
            
            context.failed_tasks.append(task.id)
            return {"success": False, "error": str(e)}
            
        finally:
            # Clean up
            if task.id in context.active_tasks:
                context.active_tasks.remove(task.id)
    
    # Task Handlers
    async def _handle_information_gathering(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle information gathering tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "information_gathering_started",
            "description": task.description
        })
        
        # Simulate information gathering
        await asyncio.sleep(2)  # Simulate research time
        
        result = {
            "task_type": "information_gathering",
            "summary": f"Gathered information for: {task.description}",
            "sources": ["source1", "source2", "source3"],
            "findings": f"Key findings related to {task.title}",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_file_operation(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle file operation tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "file_operation_started",
            "description": task.description
        })
        
        # Simulate file operation
        await asyncio.sleep(1)
        
        result = {
            "task_type": "file_operation",
            "operation": task.description,
            "files_processed": ["file1", "file2"],
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_communication(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle communication tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "communication_started",
            "description": task.description
        })
        
        # Simulate communication
        await asyncio.sleep(1.5)
        
        result = {
            "task_type": "communication",
            "message": task.description,
            "recipients": ["recipient1", "recipient2"],
            "status": "sent",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_automation(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle automation tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "automation_started",
            "description": task.description
        })
        
        # Simulate automation
        await asyncio.sleep(3)
        
        result = {
            "task_type": "automation",
            "automation": task.description,
            "actions_executed": ["action1", "action2", "action3"],
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_analysis(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle analysis tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "analysis_started",
            "description": task.description
        })
        
        # Simulate analysis
        await asyncio.sleep(4)
        
        result = {
            "task_type": "analysis",
            "analysis": task.description,
            "insights": f"Key insights from {task.title}",
            "recommendations": ["recommendation1", "recommendation2"],
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_creative(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle creative tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "creative_task_started",
            "description": task.description
        })
        
        # Simulate creative work
        await asyncio.sleep(5)
        
        result = {
            "task_type": "creative",
            "creation": task.description,
            "output": f"Creative output for {task.title}",
            "quality_score": 0.9,
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_system_task(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Handle system tasks"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "system_task_started",
            "description": task.description
        })
        
        # Simulate system task
        await asyncio.sleep(1)
        
        result = {
            "task_type": "system_task",
            "task": task.description,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _handle_default_task(self, task: Task, execution: TaskExecution) -> Dict[str, Any]:
        """Default handler for unknown task types"""
        
        execution.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "default_handler",
            "description": f"Handling unknown task type: {task.task_type.value}"
        })
        
        # Simulate generic task execution
        await asyncio.sleep(2)
        
        result = {
            "task_type": task.task_type.value,
            "description": task.description,
            "status": "completed_with_default_handler",
            "timestamp": datetime.now().isoformat()
        }
        
        execution.progress = 1.0
        return result
    
    async def _finalize_execution(self, plan: TaskPlan, context: ExecutionContext):
        """Finalize execution and cleanup"""
        
        context.current_phase = "finalization"
        
        # Update plan status
        if len(context.failed_tasks) == 0:
            plan.status = TaskStatus.COMPLETED
        elif len(context.completed_tasks) > 0:
            plan.status = TaskStatus.FAILED  # Partial completion
        else:
            plan.status = TaskStatus.FAILED
        
        # Calculate final metrics
        total_time = (datetime.now() - context.started_at).total_seconds()
        context.metrics.update({
            "actual_duration_seconds": total_time,
            "completion_rate": len(context.completed_tasks) / len(plan.tasks) if plan.tasks else 0,
            "failure_rate": len(context.failed_tasks) / len(plan.tasks) if plan.tasks else 0,
            "efficiency_score": self._calculate_efficiency_score(context, plan)
        })
        
        self.logger.info(
            f"Execution {context.execution_id} completed: "
            f"{len(context.completed_tasks)}/{len(plan.tasks)} tasks successful"
        )
    
    def _calculate_efficiency_score(self, context: ExecutionContext, plan: TaskPlan) -> float:
        """Calculate execution efficiency score"""
        
        completion_rate = len(context.completed_tasks) / len(plan.tasks) if plan.tasks else 0
        
        # Time efficiency
        estimated_duration = context.metrics.get("estimated_duration_seconds", 0)
        actual_duration = context.metrics.get("actual_duration_seconds", 0)
        
        time_efficiency = 1.0
        if estimated_duration > 0 and actual_duration > 0:
            time_efficiency = min(1.0, estimated_duration / actual_duration)
        
        # Resource efficiency (simplified)
        resource_efficiency = 0.8  # Default value
        
        # Overall efficiency
        return (completion_rate * 0.5 + time_efficiency * 0.3 + resource_efficiency * 0.2)
    
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of an active execution"""
        
        if execution_id not in self.active_executions:
            return {"error": "Execution not found"}
        
        context = self.active_executions[execution_id]
        
        # Calculate progress
        total_tasks = len(context.completed_tasks) + len(context.failed_tasks) + len(context.active_tasks) + len(context.blocked_tasks)
        progress = len(context.completed_tasks) / total_tasks if total_tasks > 0 else 0
        
        return {
            "execution_id": execution_id,
            "plan_id": context.plan_id,
            "status": "active",
            "current_phase": context.current_phase,
            "progress": progress,
            "started_at": context.started_at.isoformat(),
            "estimated_completion": context.estimated_completion.isoformat() if context.estimated_completion else None,
            "active_tasks": len(context.active_tasks),
            "completed_tasks": len(context.completed_tasks),
            "failed_tasks": len(context.failed_tasks),
            "blocked_tasks": len(context.blocked_tasks),
            "metrics": context.metrics
        }
    
    async def cancel_execution(self, execution_id: str) -> Dict[str, Any]:
        """Cancel an active execution"""
        
        if execution_id not in self.active_executions:
            return {"success": False, "error": "Execution not found"}
        
        context = self.active_executions[execution_id]
        
        # Stop active tasks (simplified - in reality would need more sophisticated cancellation)
        for task_id in context.active_tasks:
            if task_id in self.task_executions:
                task_execution = self.task_executions[task_id]
                task_execution.status = TaskStatus.CANCELLED
                
                # Release resources
                await self.resource_manager.release_resources(task_id)
        
        # Move to history
        context.current_phase = "cancelled"
        self.execution_history.append(context)
        del self.active_executions[execution_id]
        
        return {
            "success": True,
            "execution_id": execution_id,
            "cancelled_tasks": len(context.active_tasks),
            "completed_tasks": len(context.completed_tasks)
        }
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a custom task handler"""
        
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    async def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history"""
        
        return [
            {
                "execution_id": context.execution_id,
                "plan_id": context.plan_id,
                "mode": context.mode.value,
                "strategy": context.strategy.value,
                "started_at": context.started_at.isoformat(),
                "completed_tasks": len(context.completed_tasks),
                "failed_tasks": len(context.failed_tasks),
                "metrics": context.metrics
            }
            for context in self.execution_history[-limit:]
        ]