"""
Autonomous Task System for Eidolon AI Personal Assistant

Provides autonomous task planning, execution, and workflow automation with
safety mechanisms and user control.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from pydantic import BaseModel, Field

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.safety import SafetyManager, ActionApproval, RiskLevel
from ..core.observer import Observer
from ..core.analyzer import Analyzer
from ..core.memory import MemorySystem
from ..models.cloud_api import CloudAPIManager

# Initialize logger
logger = get_component_logger("agent")


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_APPROVAL = "requires_approval"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskType(str, Enum):
    """Types of tasks the agent can handle."""
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    COMMUNICATION = "communication"
    FILE_OPERATION = "file_operation"
    SYSTEM_OPERATION = "system_operation"
    WEB_OPERATION = "web_operation"
    CUSTOM = "custom"


@dataclass
class TaskContext:
    """Context information for task execution."""
    user_request: str
    screen_context: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class Task(BaseModel):
    """A task that can be executed by the autonomous agent."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(description="Human-readable task title")
    description: str = Field(description="Detailed task description")
    task_type: TaskType = Field(description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    
    # Task execution
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="List of actions to execute")
    dependencies: List[str] = Field(default_factory=list, description="Task IDs this task depends on")
    context: Optional[TaskContext] = Field(default=None, description="Execution context")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = Field(default=None, description="When to execute the task")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Results
    result: Optional[TaskResult] = Field(default=None)
    approval: Optional[ActionApproval] = Field(default=None)
    
    # Retry logic
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    
    # Safety and permissions
    requires_approval: bool = Field(default=True, description="Whether task requires user approval")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    approved_by: Optional[str] = Field(default=None)


class TaskQueue:
    """Queue for managing task execution order and dependencies."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: Task) -> None:
        """Add a task to the queue."""
        async with self._lock:
            self.tasks[task.id] = task
            self._update_execution_order()
    
    async def get_next_task(self) -> Optional[Task]:
        """Get the next task ready for execution."""
        async with self._lock:
            for task_id in self.execution_order:
                task = self.tasks[task_id]
                
                # Check if task is ready to execute
                if (task.status in [TaskStatus.APPROVED, TaskStatus.PENDING] and
                    self._are_dependencies_met(task) and
                    (task.scheduled_at is None or task.scheduled_at <= datetime.now())):
                    
                    return task
            
            return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, result: Optional[TaskResult] = None) -> None:
        """Update task status and result."""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                
                if status == TaskStatus.RUNNING:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
                
                if result:
                    task.result = result
    
    async def get_pending_tasks(self) -> List[Task]:
        """Get all tasks requiring approval."""
        async with self._lock:
            return [task for task in self.tasks.values() 
                   if task.status == TaskStatus.REQUIRES_APPROVAL]
    
    async def get_active_tasks(self) -> List[Task]:
        """Get all active tasks."""
        async with self._lock:
            return [task for task in self.tasks.values() 
                   if task.status in [TaskStatus.RUNNING, TaskStatus.APPROVED, TaskStatus.PENDING]]
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue."""
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._update_execution_order()
                return True
            return False
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _update_execution_order(self) -> None:
        """Update task execution order based on priority and dependencies."""
        # Simple topological sort with priority
        pending_tasks = [(task.priority.value, task.created_at, task.id) 
                        for task in self.tasks.values() 
                        if task.status in [TaskStatus.PENDING, TaskStatus.APPROVED]]
        
        # Sort by priority (urgent first) then by creation time
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        pending_tasks.sort(key=lambda x: (priority_order.get(x[0], 3), x[1]))
        
        self.execution_order = [task_id for _, _, task_id in pending_tasks]


class AutonomousAgent:
    """
    Main autonomous agent that plans and executes tasks with safety controls.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the autonomous agent."""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        # Core components
        self.observer: Optional[Observer] = None
        self.analyzer: Optional[Analyzer] = None
        self.memory: Optional[MemorySystem] = None
        self.cloud_api: Optional[CloudAPIManager] = None
        self.safety_manager = SafetyManager()
        
        # Task management
        self.task_queue = TaskQueue()
        self.running = False
        self.max_concurrent_tasks = 3
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Tool registry
        self.tools: Dict[str, Callable] = {}
        self._register_builtin_tools()
        
        # Performance tracking
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "started_at": None
        }
    
    async def initialize(self) -> None:
        """Initialize agent components."""
        logger.info("Initializing autonomous agent...")
        
        try:
            # Initialize core components
            self.observer = Observer()
            self.analyzer = Analyzer()
            self.memory = MemorySystem()
            
            # Initialize cloud API if available
            try:
                self.cloud_api = CloudAPIManager()
                if not self.cloud_api.get_available_providers():
                    self.cloud_api = None
                    logger.warning("No cloud AI providers available")
            except Exception as e:
                logger.warning(f"Cloud AI initialization failed: {e}")
                self.cloud_api = None
            
            logger.info("Autonomous agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the autonomous agent."""
        if self.running:
            logger.warning("Agent is already running")
            return
        
        if not self.observer:
            await self.initialize()
        
        self.running = True
        self.stats["started_at"] = datetime.now()
        
        logger.info("Starting autonomous agent...")
        
        # Start main execution loop
        asyncio.create_task(self._execution_loop())
        
        logger.info("Autonomous agent started")
    
    async def stop(self) -> None:
        """Stop the autonomous agent."""
        if not self.running:
            return
        
        logger.info("Stopping autonomous agent...")
        
        self.running = False
        
        # Cancel active tasks
        for task_id, async_task in list(self.active_tasks.items()):
            async_task.cancel()
            try:
                await async_task
            except asyncio.CancelledError:
                pass
        
        self.active_tasks.clear()
        
        logger.info("Autonomous agent stopped")
    
    async def create_task(
        self,
        title: str,
        description: str,
        task_type: TaskType,
        actions: List[Dict[str, Any]],
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[TaskContext] = None,
        requires_approval: bool = True,
        scheduled_at: Optional[datetime] = None
    ) -> Task:
        """Create a new task."""
        
        # Assess risk level based on actions
        risk_level = await self.safety_manager.assess_risk(actions)
        
        task = Task(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            actions=actions,
            context=context,
            requires_approval=requires_approval,
            risk_level=risk_level,
            scheduled_at=scheduled_at
        )
        
        # Add to queue
        await self.task_queue.add_task(task)
        
        logger.info(f"Created task: {task.title} (ID: {task.id})")
        
        return task
    
    async def approve_task(self, task_id: str, user_id: str = "user") -> bool:
        """Approve a task for execution."""
        if task_id not in self.task_queue.tasks:
            return False
        
        task = self.task_queue.tasks[task_id]
        
        # Create approval record
        approval = ActionApproval(
            approved=True,
            approved_by=user_id,
            approved_at=datetime.now(),
            conditions=[]
        )
        
        task.approval = approval
        task.approved_by = user_id
        await self.task_queue.update_task_status(task_id, TaskStatus.APPROVED)
        
        logger.info(f"Task approved: {task.title} (ID: {task_id})")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.task_queue.tasks:
            return False
        
        task = self.task_queue.tasks[task_id]
        
        # Cancel if running
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]
        
        await self.task_queue.update_task_status(task_id, TaskStatus.CANCELLED)
        
        logger.info(f"Task cancelled: {task.title} (ID: {task_id})")
        return True
    
    async def get_pending_approvals(self) -> List[Task]:
        """Get tasks requiring approval."""
        return await self.task_queue.get_pending_tasks()
    
    async def get_active_tasks(self) -> List[Task]:
        """Get currently active tasks."""
        return await self.task_queue.get_active_tasks()
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a task."""
        if task_id not in self.task_queue.tasks:
            return None
        
        task = self.task_queue.tasks[task_id]
        
        return {
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "priority": task.priority,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result.dict() if task.result else None,
            "risk_level": task.risk_level,
            "requires_approval": task.requires_approval,
            "approved_by": task.approved_by
        }
    
    async def suggest_task(self, user_request: str) -> List[Task]:
        """Analyze user request and suggest tasks."""
        try:
            # Get current screen context
            screen_context = None
            if self.observer:
                screenshot = self.observer.capture_screenshot()
                if screenshot and self.analyzer:
                    text_result = self.analyzer.extract_text(screenshot.image)
                    screen_context = {
                        "text": text_result.text if text_result else "",
                        "timestamp": screenshot.timestamp
                    }
            
            # Use AI to analyze request and suggest tasks
            if self.cloud_api:
                prompt = f"""
                Analyze this user request and suggest specific tasks that could be automated:
                
                User Request: {user_request}
                
                Current Screen Context: {screen_context.get('text', 'No context available') if screen_context else 'No context available'}
                
                Suggest 1-3 specific, actionable tasks that could help the user. For each task:
                1. Provide a clear title
                2. Describe what it would do
                3. Suggest the task type (analysis, automation, monitoring, communication, file_operation, system_operation, web_operation)
                4. Estimate risk level (low, medium, high, critical)
                5. List specific actions needed
                
                Return as JSON array of task suggestions.
                """
                
                response = await self.cloud_api.analyze_text(prompt, analysis_type="task_planning")
                
                if response and response.content:
                    try:
                        # Parse AI response into task suggestions
                        task_data = json.loads(response.content)
                        
                        suggested_tasks = []
                        for suggestion in task_data:
                            task = await self.create_task(
                                title=suggestion.get("title", "AI Suggested Task"),
                                description=suggestion.get("description", user_request),
                                task_type=TaskType(suggestion.get("task_type", "custom")),
                                actions=suggestion.get("actions", []),
                                priority=TaskPriority.MEDIUM,
                                context=TaskContext(
                                    user_request=user_request,
                                    screen_context=screen_context
                                ),
                                requires_approval=True
                            )
                            suggested_tasks.append(task)
                        
                        return suggested_tasks
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse AI task suggestions: {e}")
            
            # Fallback: create a generic analysis task
            task = await self.create_task(
                title="Analyze Request",
                description=f"Analyze user request: {user_request}",
                task_type=TaskType.ANALYSIS,
                actions=[
                    {
                        "type": "analyze_request",
                        "params": {"request": user_request}
                    }
                ],
                context=TaskContext(
                    user_request=user_request,
                    screen_context=screen_context
                )
            )
            
            return [task]
            
        except Exception as e:
            logger.error(f"Failed to suggest tasks: {e}")
            return []
    
    async def _execution_loop(self) -> None:
        """Main execution loop for the agent."""
        logger.info("Agent execution loop started")
        
        while self.running:
            try:
                # Clean up completed tasks
                completed_tasks = []
                for task_id, async_task in list(self.active_tasks.items()):
                    if async_task.done():
                        completed_tasks.append(task_id)
                
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                # Check for new tasks to execute
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    next_task = await self.task_queue.get_next_task()
                    
                    if next_task:
                        # Check if task requires approval
                        if next_task.requires_approval and not next_task.approval:
                            await self.task_queue.update_task_status(
                                next_task.id, 
                                TaskStatus.REQUIRES_APPROVAL
                            )
                            logger.info(f"Task requires approval: {next_task.title}")
                        else:
                            # Execute task
                            async_task = asyncio.create_task(
                                self._execute_task(next_task)
                            )
                            self.active_tasks[next_task.id] = async_task
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Agent execution loop stopped")
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        logger.info(f"Executing task: {task.title} (ID: {task.id})")
        
        start_time = time.time()
        
        try:
            await self.task_queue.update_task_status(task.id, TaskStatus.RUNNING)
            
            # Execute each action in sequence
            results = []
            side_effects = []
            
            for action in task.actions:
                action_type = action.get("type")
                action_params = action.get("params", {})
                
                if action_type in self.tools:
                    tool = self.tools[action_type]
                    action_result = await tool(action_params, task.context)
                    results.append(action_result)
                    
                    # Track side effects
                    if hasattr(action_result, 'side_effects'):
                        side_effects.extend(action_result.side_effects)
                else:
                    logger.warning(f"Unknown action type: {action_type}")
                    results.append({"error": f"Unknown action type: {action_type}"})
            
            execution_time = time.time() - start_time
            
            # Create result
            result = TaskResult(
                success=True,
                message="Task completed successfully",
                data={"results": results},
                execution_time=execution_time,
                side_effects=side_effects
            )
            
            await self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED, result)
            
            # Update statistics
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            
            logger.info(f"Task completed: {task.title} (Time: {execution_time:.2f}s)")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"Task execution failed: {task.title} - {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await self.task_queue.update_task_status(task.id, TaskStatus.PENDING)
                logger.info(f"Retrying task: {task.title} (Attempt {task.retry_count})")
            else:
                result = TaskResult(
                    success=False,
                    message=f"Task failed: {str(e)}",
                    execution_time=execution_time
                )
                
                await self.task_queue.update_task_status(task.id, TaskStatus.FAILED, result)
                self.stats["tasks_failed"] += 1
            
            return result
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools for task execution."""
        
        async def analyze_request(params: Dict[str, Any], context: Optional[TaskContext]) -> Dict[str, Any]:
            """Analyze user request tool."""
            request = params.get("request", "")
            
            if self.cloud_api:
                try:
                    response = await self.cloud_api.analyze_text(
                        f"Analyze this user request and provide insights: {request}",
                        analysis_type="general"
                    )
                    
                    return {
                        "analysis": response.content if response else "Analysis failed",
                        "confidence": response.confidence if response else 0.0
                    }
                except Exception as e:
                    return {"error": f"Analysis failed: {e}"}
            
            return {"analysis": f"Basic analysis of: {request}"}
        
        async def capture_screen(params: Dict[str, Any], context: Optional[TaskContext]) -> Dict[str, Any]:
            """Capture screen tool."""
            if not self.observer:
                return {"error": "Observer not available"}
            
            try:
                screenshot = self.observer.capture_screenshot()
                if screenshot:
                    return {
                        "success": True,
                        "timestamp": screenshot.timestamp,
                        "hash": screenshot.hash
                    }
                else:
                    return {"error": "Failed to capture screenshot"}
            except Exception as e:
                return {"error": f"Screen capture failed: {e}"}
        
        async def search_memory(params: Dict[str, Any], context: Optional[TaskContext]) -> Dict[str, Any]:
            """Search memory tool."""
            query = params.get("query", "")
            
            if not self.memory:
                return {"error": "Memory system not available"}
            
            try:
                results = await self.memory.search(query)
                return {
                    "results": [result.dict() for result in results],
                    "count": len(results)
                }
            except Exception as e:
                return {"error": f"Memory search failed: {e}"}
        
        # Register tools
        self.tools.update({
            "analyze_request": analyze_request,
            "capture_screen": capture_screen,
            "search_memory": search_memory
        })
    
    def register_tool(self, name: str, tool_func: Callable) -> None:
        """Register a custom tool."""
        self.tools[name] = tool_func
        logger.info(f"Registered tool: {name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        uptime = (datetime.now() - self.stats["started_at"]).total_seconds() if self.stats["started_at"] else 0
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "active_tasks": len(self.active_tasks),
            "queue_size": len(self.task_queue.tasks),
            "average_execution_time": (
                self.stats["total_execution_time"] / max(self.stats["tasks_completed"], 1)
            )
        }