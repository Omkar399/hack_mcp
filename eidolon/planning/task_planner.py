"""
Complex Task Planning system for Eidolon AI Personal Assistant

Handles multi-step task decomposition, dependency analysis, and adaptive planning
for autonomous task execution.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..models.cloud_api import CloudAPIManager
from ..core.memory import MemorySystem


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Types of tasks the system can handle"""
    INFORMATION_GATHERING = "information_gathering"
    FILE_OPERATION = "file_operation"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    SYSTEM_TASK = "system_task"


@dataclass
class TaskResource:
    """Resources required for task execution"""
    type: str  # 'api', 'file', 'application', 'network', 'cpu', 'memory'
    identifier: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    availability_window: Optional[tuple] = None


@dataclass
class Task:
    """Individual task with execution details"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    task_type: TaskType = TaskType.SYSTEM_TASK
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    
    # Resources and constraints
    required_resources: List[TaskResource] = field(default_factory=list)
    estimated_duration: Optional[timedelta] = None
    deadline: Optional[datetime] = None
    
    # Execution details
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'blocks': self.blocks,
            'subtasks': self.subtasks,
            'parent_task': self.parent_task,
            'required_resources': [
                {
                    'type': r.type,
                    'identifier': r.identifier,
                    'amount': r.amount,
                    'unit': r.unit,
                    'availability_window': r.availability_window
                } for r in self.required_resources
            ],
            'estimated_duration': self.estimated_duration.total_seconds() if self.estimated_duration else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'context': self.context,
            'metadata': self.metadata,
            'execution_log': self.execution_log,
            'result': self.result,
            'error': self.error
        }


@dataclass
class TaskPlan:
    """Complete execution plan for a complex task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    objective: str = ""
    
    # Tasks and structure
    tasks: Dict[str, Task] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    
    # Plan metadata
    created_at: datetime = field(default_factory=datetime.now)
    estimated_total_duration: Optional[timedelta] = None
    target_completion: Optional[datetime] = None
    
    # Execution state
    current_phase: str = "planning"
    progress: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    
    # Adaptation and learning
    success_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    adaptation_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.READY:
                ready_tasks.append(task)
            elif task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    task.status = TaskStatus.READY
                    ready_tasks.append(task)
        
        # Sort by priority and then by creation time
        ready_tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        return ready_tasks
    
    def get_blocked_tasks(self) -> List[Task]:
        """Get tasks that are blocked"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.BLOCKED]
    
    def calculate_progress(self) -> float:
        """Calculate overall plan progress"""
        if not self.tasks:
            return 0.0
        
        total_progress = sum(task.progress for task in self.tasks.values())
        return total_progress / len(self.tasks)
    
    def update_progress(self):
        """Update plan progress and status"""
        self.progress = self.calculate_progress()
        
        # Update status based on task states
        task_statuses = [task.status for task in self.tasks.values()]
        if all(status == TaskStatus.COMPLETED for status in task_statuses):
            self.status = TaskStatus.COMPLETED
        elif any(status == TaskStatus.FAILED for status in task_statuses):
            self.status = TaskStatus.FAILED
        elif any(status == TaskStatus.IN_PROGRESS for status in task_statuses):
            self.status = TaskStatus.IN_PROGRESS
        elif any(status == TaskStatus.READY for status in task_statuses):
            self.status = TaskStatus.READY


class TaskPlanner:
    """Advanced task planning system with AI-powered decomposition"""
    
    def __init__(self):
        self.logger = get_component_logger("task_planner")
        self.config = get_config()
        self.memory = MemorySystem()
        self.cloud_api = CloudAPIManager()
        
        # Planning state
        self.active_plans: Dict[str, TaskPlan] = {}
        self.plan_templates: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Planning patterns learned from user behavior
        self.user_patterns: Dict[str, Any] = {}
        self.success_patterns: Dict[str, Any] = {}
        
        # Load planning templates and patterns
        self._load_planning_templates()
        self._load_user_patterns()
    
    @log_exceptions()
    async def create_plan(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        """Create a comprehensive task plan from a high-level objective"""
        
        self.logger.info(f"Creating plan for objective: {objective}")
        
        # Initialize plan
        plan = TaskPlan(
            title=f"Plan: {objective[:50]}...",
            description=f"Automated plan for: {objective}",
            objective=objective
        )
        
        # Decompose objective into tasks using AI
        decomposition = await self._decompose_objective(
            objective, context, constraints, user_preferences
        )
        
        # Create tasks from decomposition
        await self._create_tasks_from_decomposition(plan, decomposition)
        
        # Analyze dependencies
        await self._analyze_task_dependencies(plan)
        
        # Optimize execution order
        await self._optimize_execution_order(plan)
        
        # Estimate durations and resources
        await self._estimate_task_resources(plan)
        
        # Calculate critical path
        await self._calculate_critical_path(plan)
        
        # Store plan
        self.active_plans[plan.id] = plan
        
        self.logger.info(f"Created plan {plan.id} with {len(plan.tasks)} tasks")
        return plan
    
    @log_exceptions()
    async def _decompose_objective(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use AI to decompose high-level objective into specific tasks"""
        
        # Build context for AI analysis
        analysis_context = {
            "objective": objective,
            "context": context or {},
            "constraints": constraints or {},
            "user_preferences": user_preferences or {},
            "user_patterns": self.user_patterns,
            "available_tools": self._get_available_tools(),
            "historical_plans": self._get_similar_historical_plans(objective)
        }
        
        # Create decomposition prompt
        prompt = self._build_decomposition_prompt(analysis_context)
        
        try:
            # Use cloud AI for intelligent decomposition
            response = await self.cloud_api.analyze_complex_request(
                prompt,
                context="task_planning"
            )
            
            # Parse AI response into structured format
            decomposition = self._parse_decomposition_response(response)
            
            return decomposition
            
        except Exception as e:
            self.logger.warning(f"Cloud AI decomposition failed: {e}, using fallback")
            return self._fallback_decomposition(objective, context)
    
    def _build_decomposition_prompt(self, context: Dict[str, Any]) -> str:
        """Build AI prompt for task decomposition"""
        
        prompt = f"""
You are an expert task planner helping to decompose a complex objective into executable tasks.

OBJECTIVE: {context['objective']}

CONTEXT:
{json.dumps(context.get('context', {}), indent=2)}

CONSTRAINTS:
{json.dumps(context.get('constraints', {}), indent=2)}

USER PREFERENCES:
{json.dumps(context.get('user_preferences', {}), indent=2)}

AVAILABLE TOOLS:
{', '.join(context.get('available_tools', []))}

Please decompose this objective into a structured plan with the following format:

{{
  "main_phases": [
    {{
      "name": "Phase Name",
      "description": "Phase description",
      "tasks": [
        {{
          "title": "Task title",
          "description": "Detailed task description", 
          "type": "information_gathering|file_operation|communication|automation|analysis|creative|system_task",
          "priority": "critical|high|medium|low",
          "estimated_duration_minutes": 30,
          "dependencies": ["task_id1", "task_id2"],
          "required_resources": [
            {{
              "type": "api|file|application|network",
              "identifier": "resource_name",
              "amount": 1
            }}
          ],
          "success_criteria": ["criterion1", "criterion2"],
          "potential_issues": ["issue1", "issue2"]
        }}
      ]
    }}
  ],
  "overall_strategy": "High-level strategy description",
  "key_risks": ["risk1", "risk2"],
  "success_factors": ["factor1", "factor2"]
}}

Consider:
1. Break down into logical phases and granular tasks
2. Identify all dependencies between tasks
3. Consider resource requirements and availability
4. Account for potential failure points and alternatives
5. Prioritize based on impact and urgency
6. Include verification and quality check tasks
7. Factor in user's working patterns and preferences

Respond with only the JSON structure.
"""
        return prompt
    
    def _parse_decomposition_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into structured decomposition"""
        try:
            if isinstance(response.get('content'), str):
                # Try to extract JSON from text response
                content = response['content']
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    return json.loads(json_str)
            
            return response.get('content', {})
            
        except Exception as e:
            self.logger.error(f"Failed to parse decomposition response: {e}")
            return {
                "main_phases": [{
                    "name": "Manual Planning Required",
                    "description": "AI decomposition failed, manual planning needed",
                    "tasks": []
                }],
                "overall_strategy": "Fallback to manual planning",
                "key_risks": ["AI decomposition failure"],
                "success_factors": ["Manual oversight"]
            }
    
    def _fallback_decomposition(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback decomposition when AI is unavailable"""
        
        # Simple rule-based decomposition
        tasks = []
        
        # Common task patterns
        if "research" in objective.lower() or "find" in objective.lower():
            tasks.append({
                "title": "Information Gathering",
                "description": f"Research and gather information about: {objective}",
                "type": "information_gathering",
                "priority": "high",
                "estimated_duration_minutes": 30
            })
        
        if "create" in objective.lower() or "write" in objective.lower():
            tasks.append({
                "title": "Content Creation",
                "description": f"Create content for: {objective}",
                "type": "creative",
                "priority": "high",
                "estimated_duration_minutes": 60
            })
        
        if "send" in objective.lower() or "email" in objective.lower():
            tasks.append({
                "title": "Communication",
                "description": f"Send communication regarding: {objective}",
                "type": "communication",
                "priority": "medium",
                "estimated_duration_minutes": 15
            })
        
        # Default analysis task
        tasks.append({
            "title": "Review and Verify",
            "description": "Review completed work and verify objectives are met",
            "type": "analysis",
            "priority": "medium",
            "estimated_duration_minutes": 15
        })
        
        return {
            "main_phases": [{
                "name": "Execution Phase",
                "description": "Complete the requested objective",
                "tasks": tasks
            }],
            "overall_strategy": "Direct execution approach",
            "key_risks": ["Limited AI assistance"],
            "success_factors": ["Clear objective definition"]
        }
    
    async def _create_tasks_from_decomposition(
        self,
        plan: TaskPlan,
        decomposition: Dict[str, Any]
    ):
        """Create Task objects from decomposition structure"""
        
        task_id_counter = 0
        
        for phase in decomposition.get('main_phases', []):
            for task_data in phase.get('tasks', []):
                task_id_counter += 1
                task_id = f"task_{task_id_counter:03d}"
                
                # Create task resource objects
                resources = []
                for res_data in task_data.get('required_resources', []):
                    resource = TaskResource(
                        type=res_data.get('type', 'unknown'),
                        identifier=res_data.get('identifier', ''),
                        amount=res_data.get('amount'),
                        unit=res_data.get('unit')
                    )
                    resources.append(resource)
                
                # Create task
                task = Task(
                    id=task_id,
                    title=task_data.get('title', f'Task {task_id_counter}'),
                    description=task_data.get('description', ''),
                    task_type=TaskType(task_data.get('type', 'system_task')),
                    priority=TaskPriority(task_data.get('priority', 'medium')),
                    required_resources=resources,
                    estimated_duration=timedelta(
                        minutes=task_data.get('estimated_duration_minutes', 30)
                    ),
                    metadata={
                        'phase': phase.get('name'),
                        'success_criteria': task_data.get('success_criteria', []),
                        'potential_issues': task_data.get('potential_issues', [])
                    }
                )
                
                plan.tasks[task_id] = task
        
        # Store decomposition metadata
        plan.success_factors = decomposition.get('success_factors', [])
        plan.risk_factors = decomposition.get('key_risks', [])
    
    async def _analyze_task_dependencies(self, plan: TaskPlan):
        """Analyze and set up task dependencies"""
        
        # For now, implement simple sequential dependencies within phases
        # In a real implementation, this would use AI to analyze task relationships
        
        phase_tasks = {}
        for task in plan.tasks.values():
            phase = task.metadata.get('phase', 'default')
            if phase not in phase_tasks:
                phase_tasks[phase] = []
            phase_tasks[phase].append(task)
        
        # Create sequential dependencies within each phase
        for phase, tasks in phase_tasks.items():
            tasks.sort(key=lambda t: t.created_at)
            for i in range(1, len(tasks)):
                tasks[i].dependencies.append(tasks[i-1].id)
                tasks[i-1].blocks.append(tasks[i].id)
    
    async def _optimize_execution_order(self, plan: TaskPlan):
        """Optimize task execution order considering dependencies and resources"""
        
        # Topological sort considering dependencies
        execution_order = []
        completed = set()
        
        def can_execute(task: Task) -> bool:
            return all(dep in completed for dep in task.dependencies)
        
        while len(completed) < len(plan.tasks):
            # Find tasks that can be executed
            ready_tasks = [
                task for task in plan.tasks.values() 
                if task.id not in completed and can_execute(task)
            ]
            
            if not ready_tasks:
                # Detect circular dependency or other issue
                remaining = [t for t in plan.tasks.values() if t.id not in completed]
                self.logger.warning(f"Circular dependency detected in plan {plan.id}")
                # Add remaining tasks anyway
                for task in remaining:
                    execution_order.append(task.id)
                    completed.add(task.id)
                break
            
            # Sort by priority and add to execution order
            ready_tasks.sort(key=lambda t: (t.priority.value, t.created_at))
            
            for task in ready_tasks:
                execution_order.append(task.id)
                completed.add(task.id)
        
        plan.execution_order = execution_order
    
    async def _estimate_task_resources(self, plan: TaskPlan):
        """Estimate resource requirements for tasks"""
        
        total_duration = timedelta()
        
        for task in plan.tasks.values():
            if task.estimated_duration:
                total_duration += task.estimated_duration
        
        plan.estimated_total_duration = total_duration
        plan.target_completion = datetime.now() + total_duration
    
    async def _calculate_critical_path(self, plan: TaskPlan):
        """Calculate critical path through the task network"""
        
        # Simplified critical path calculation
        # In reality, this would implement proper CPM algorithm
        
        critical_path = []
        max_duration = timedelta()
        
        # Find the longest path through dependencies
        def calculate_path_duration(task_id: str, visited: set = None) -> timedelta:
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return timedelta()  # Avoid cycles
            
            visited.add(task_id)
            task = plan.tasks[task_id]
            
            if not task.blocks:
                return task.estimated_duration or timedelta()
            
            max_child_duration = max(
                calculate_path_duration(child_id, visited.copy())
                for child_id in task.blocks
            )
            
            return (task.estimated_duration or timedelta()) + max_child_duration
        
        # Find task with longest total path
        for task_id in plan.tasks:
            path_duration = calculate_path_duration(task_id)
            if path_duration > max_duration:
                max_duration = path_duration
                critical_path = self._build_critical_path(plan, task_id)
        
        plan.critical_path = critical_path
    
    def _build_critical_path(self, plan: TaskPlan, start_task_id: str) -> List[str]:
        """Build critical path starting from given task"""
        path = [start_task_id]
        current = plan.tasks[start_task_id]
        
        while current.blocks:
            # Find child with longest duration
            max_duration = timedelta()
            next_task_id = None
            
            for child_id in current.blocks:
                child = plan.tasks[child_id]
                if (child.estimated_duration or timedelta()) > max_duration:
                    max_duration = child.estimated_duration or timedelta()
                    next_task_id = child_id
            
            if next_task_id:
                path.append(next_task_id)
                current = plan.tasks[next_task_id]
            else:
                break
        
        return path
    
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools and capabilities"""
        return [
            "web_search", "file_operations", "email", "calendar",
            "document_creation", "data_analysis", "image_processing",
            "code_execution", "api_calls", "database_queries"
        ]
    
    def _get_similar_historical_plans(self, objective: str) -> List[Dict[str, Any]]:
        """Get similar historical plans for context"""
        # In a real implementation, this would use semantic search
        # on the execution_history to find similar past plans
        return []
    
    def _load_planning_templates(self):
        """Load predefined planning templates"""
        # This would load from configuration or database
        self.plan_templates = {
            "research_project": {
                "phases": ["information_gathering", "analysis", "synthesis", "documentation"],
                "common_tasks": ["literature_review", "data_collection", "analysis", "report_writing"]
            },
            "content_creation": {
                "phases": ["planning", "creation", "review", "publication"],
                "common_tasks": ["outline", "draft", "edit", "format", "publish"]
            },
            "communication": {
                "phases": ["planning", "composition", "review", "sending"],
                "common_tasks": ["recipient_analysis", "message_draft", "review", "send"]
            }
        }
    
    def _load_user_patterns(self):
        """Load learned user patterns and preferences"""
        # This would load from user's historical data
        self.user_patterns = {
            "preferred_working_hours": {"start": 9, "end": 17},
            "task_preferences": {"batch_similar_tasks": True, "minimize_context_switching": True},
            "communication_style": {"formal": False, "detailed": True},
            "typical_task_durations": {"email": 5, "research": 30, "writing": 60}
        }
    
    async def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get plan by ID"""
        return self.active_plans.get(plan_id)
    
    async def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ):
        """Update task status and progress"""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        task = plan.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan {plan_id}")
        
        # Update task
        old_status = task.status
        task.status = status
        
        if progress is not None:
            task.progress = max(0.0, min(1.0, progress))
        
        if result is not None:
            task.result = result
        
        if error is not None:
            task.error = error
        
        # Update timestamps
        if status == TaskStatus.IN_PROGRESS and old_status != TaskStatus.IN_PROGRESS:
            task.started_at = datetime.now()
        elif status == TaskStatus.COMPLETED and old_status != TaskStatus.COMPLETED:
            task.completed_at = datetime.now()
            task.progress = 1.0
        
        # Log status change
        task.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "old_status": old_status.value,
            "new_status": status.value,
            "progress": task.progress,
            "result": result,
            "error": error
        })
        
        # Update plan progress
        plan.update_progress()
        
        self.logger.info(f"Updated task {task_id} status: {old_status.value} -> {status.value}")
    
    async def adapt_plan(
        self,
        plan_id: str,
        changes: Dict[str, Any],
        reason: str
    ):
        """Adapt plan based on execution feedback or changing requirements"""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "changes": changes,
            "plan_state_before": {
                "progress": plan.progress,
                "status": plan.status.value,
                "task_count": len(plan.tasks)
            }
        }
        
        # Apply changes
        if "add_tasks" in changes:
            for task_data in changes["add_tasks"]:
                task = Task(**task_data)
                plan.tasks[task.id] = task
        
        if "modify_tasks" in changes:
            for task_id, modifications in changes["modify_tasks"].items():
                if task_id in plan.tasks:
                    task = plan.tasks[task_id]
                    for key, value in modifications.items():
                        if hasattr(task, key):
                            setattr(task, key, value)
        
        if "remove_tasks" in changes:
            for task_id in changes["remove_tasks"]:
                if task_id in plan.tasks:
                    del plan.tasks[task_id]
        
        # Recalculate execution order and dependencies
        await self._analyze_task_dependencies(plan)
        await self._optimize_execution_order(plan)
        await self._calculate_critical_path(plan)
        
        # Log adaptation
        adaptation_record["plan_state_after"] = {
            "progress": plan.progress,
            "status": plan.status.value,
            "task_count": len(plan.tasks)
        }
        
        plan.adaptation_log.append(adaptation_record)
        
        self.logger.info(f"Adapted plan {plan_id}: {reason}")
    
    async def get_execution_recommendations(self, plan_id: str) -> Dict[str, Any]:
        """Get recommendations for optimal plan execution"""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        
        ready_tasks = plan.get_ready_tasks()
        blocked_tasks = plan.get_blocked_tasks()
        
        recommendations = {
            "next_actions": [],
            "optimizations": [],
            "warnings": [],
            "resource_needs": []
        }
        
        # Recommend next tasks to execute
        if ready_tasks:
            recommendations["next_actions"] = [
                {
                    "task_id": task.id,
                    "title": task.title,
                    "priority": task.priority.value,
                    "estimated_duration": task.estimated_duration.total_seconds() if task.estimated_duration else None
                }
                for task in ready_tasks[:3]  # Top 3 recommendations
            ]
        
        # Identify potential optimizations
        if len(ready_tasks) > 1:
            recommendations["optimizations"].append(
                "Multiple tasks are ready - consider parallel execution"
            )
        
        # Check for blocked tasks
        if blocked_tasks:
            recommendations["warnings"].append(
                f"{len(blocked_tasks)} tasks are blocked - review dependencies"
            )
        
        # Resource analysis
        resource_needs = {}
        for task in ready_tasks:
            for resource in task.required_resources:
                if resource.type not in resource_needs:
                    resource_needs[resource.type] = []
                resource_needs[resource.type].append(resource.identifier)
        
        recommendations["resource_needs"] = resource_needs
        
        return recommendations