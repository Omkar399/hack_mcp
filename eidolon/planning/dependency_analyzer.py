"""
Dependency Analysis system for Eidolon AI Personal Assistant

Analyzes task dependencies, identifies potential conflicts, and optimizes
execution paths for complex multi-step workflows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from .task_planner import Task, TaskPlan, TaskStatus


class DependencyType(Enum):
    """Types of dependencies between tasks"""
    REQUIRED = "required"  # Task B cannot start until Task A completes
    PREFERRED = "preferred"  # Task B should wait for Task A when possible
    RESOURCE = "resource"  # Tasks share limited resources
    DATA = "data"  # Task B needs output from Task A
    TEMPORAL = "temporal"  # Time-based dependency
    CONDITIONAL = "conditional"  # Dependency based on task outcome


class ConflictType(Enum):
    """Types of conflicts that can occur"""
    CIRCULAR_DEPENDENCY = "circular_dependency"
    RESOURCE_CONTENTION = "resource_contention"
    TEMPORAL_IMPOSSIBILITY = "temporal_impossibility"
    PREREQUISITE_FAILURE = "prerequisite_failure"
    CAPACITY_OVERFLOW = "capacity_overflow"


@dataclass
class Dependency:
    """Represents a dependency between two tasks"""
    source_task_id: str
    target_task_id: str
    dependency_type: DependencyType
    strength: float  # 0.0 to 1.0, how critical this dependency is
    description: str
    conditions: Optional[Dict[str, Any]] = None
    estimated_delay: Optional[timedelta] = None


@dataclass
class Conflict:
    """Represents a conflict in the task plan"""
    conflict_type: ConflictType
    involved_tasks: List[str]
    severity: float  # 0.0 to 1.0
    description: str
    suggested_resolution: str
    impact_assessment: Dict[str, Any]


@dataclass
class ExecutionPath:
    """Represents an optimized execution path"""
    task_sequence: List[str]
    parallel_groups: List[List[str]]
    estimated_duration: timedelta
    resource_utilization: Dict[str, float]
    risk_score: float
    efficiency_score: float


class DependencyAnalyzer:
    """Advanced dependency analysis and optimization system"""
    
    def __init__(self):
        self.logger = get_component_logger("dependency_analyzer")
        
        # Dependency tracking
        self.dependency_graph: nx.DiGraph = nx.DiGraph()
        self.resource_graph: nx.Graph = nx.Graph()
        
        # Analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Patterns and learning
        self.dependency_patterns: Dict[str, List[Dependency]] = {}
        self.conflict_history: List[Conflict] = []
        
    @log_exceptions()
    async def analyze_plan_dependencies(self, plan: TaskPlan) -> Dict[str, Any]:
        """Comprehensive dependency analysis for a task plan"""
        
        self.logger.info(f"Analyzing dependencies for plan {plan.id}")
        
        # Build dependency graph
        await self._build_dependency_graph(plan)
        
        # Detect conflicts
        conflicts = await self._detect_conflicts(plan)
        
        # Analyze critical path
        critical_path_analysis = await self._analyze_critical_path(plan)
        
        # Resource dependency analysis
        resource_analysis = await self._analyze_resource_dependencies(plan)
        
        # Generate optimization recommendations
        optimizations = await self._generate_optimizations(plan, conflicts)
        
        # Calculate execution paths
        execution_paths = await self._calculate_execution_paths(plan)
        
        analysis_result = {
            "plan_id": plan.id,
            "timestamp": datetime.now().isoformat(),
            "dependency_graph": self._export_dependency_graph(),
            "conflicts": [self._conflict_to_dict(c) for c in conflicts],
            "critical_path": critical_path_analysis,
            "resource_analysis": resource_analysis,
            "optimizations": optimizations,
            "execution_paths": [self._path_to_dict(p) for p in execution_paths],
            "metrics": {
                "total_dependencies": len(self.dependency_graph.edges),
                "conflict_count": len(conflicts),
                "high_severity_conflicts": len([c for c in conflicts if c.severity > 0.7]),
                "parallelization_potential": self._calculate_parallelization_potential(plan)
            }
        }
        
        # Cache analysis
        self.analysis_cache[plan.id] = analysis_result
        
        return analysis_result
    
    async def _build_dependency_graph(self, plan: TaskPlan):
        """Build a detailed dependency graph from the task plan"""
        
        # Clear existing graph
        self.dependency_graph.clear()
        self.resource_graph.clear()
        
        # Add all tasks as nodes
        for task_id, task in plan.tasks.items():
            self.dependency_graph.add_node(
                task_id,
                task=task,
                priority=task.priority.value,
                duration=task.estimated_duration,
                resources=[r.identifier for r in task.required_resources]
            )
        
        # Add explicit dependencies
        for task_id, task in plan.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in plan.tasks:
                    dependency = Dependency(
                        source_task_id=dep_id,
                        target_task_id=task_id,
                        dependency_type=DependencyType.REQUIRED,
                        strength=1.0,
                        description=f"Explicit dependency: {dep_id} -> {task_id}"
                    )
                    
                    self.dependency_graph.add_edge(
                        dep_id, task_id,
                        dependency=dependency,
                        weight=dependency.strength
                    )
        
        # Detect implicit dependencies
        await self._detect_implicit_dependencies(plan)
        
        # Build resource dependency graph
        await self._build_resource_graph(plan)
    
    async def _detect_implicit_dependencies(self, plan: TaskPlan):
        """Detect implicit dependencies based on task characteristics"""
        
        tasks = list(plan.tasks.values())
        
        for i, task_a in enumerate(tasks):
            for task_b in tasks[i+1:]:
                # Check for resource conflicts
                shared_resources = set(r.identifier for r in task_a.required_resources) & \
                                 set(r.identifier for r in task_b.required_resources)
                
                if shared_resources:
                    # Create resource dependency
                    dependency = Dependency(
                        source_task_id=task_a.id,
                        target_task_id=task_b.id,
                        dependency_type=DependencyType.RESOURCE,
                        strength=0.7,
                        description=f"Resource conflict: {shared_resources}"
                    )
                    
                    # Decide direction based on priority
                    if task_a.priority.value < task_b.priority.value:  # Higher priority first
                        self.dependency_graph.add_edge(
                            task_b.id, task_a.id,
                            dependency=dependency,
                            weight=dependency.strength
                        )
                    else:
                        self.dependency_graph.add_edge(
                            task_a.id, task_b.id,
                            dependency=dependency,
                            weight=dependency.strength
                        )
                
                # Check for data dependencies based on task types and descriptions
                if self._has_data_dependency(task_a, task_b):
                    dependency = Dependency(
                        source_task_id=task_a.id,
                        target_task_id=task_b.id,
                        dependency_type=DependencyType.DATA,
                        strength=0.8,
                        description="Detected data dependency"
                    )
                    
                    self.dependency_graph.add_edge(
                        task_a.id, task_b.id,
                        dependency=dependency,
                        weight=dependency.strength
                    )
    
    def _has_data_dependency(self, task_a: Task, task_b: Task) -> bool:
        """Check if task_b depends on data from task_a"""
        
        # Simple heuristics for data dependency detection
        output_keywords = ["create", "generate", "produce", "write", "build"]
        input_keywords = ["use", "analyze", "process", "review", "edit"]
        
        task_a_produces = any(keyword in task_a.description.lower() for keyword in output_keywords)
        task_b_consumes = any(keyword in task_b.description.lower() for keyword in input_keywords)
        
        # Check if they work with similar subjects
        task_a_words = set(task_a.description.lower().split())
        task_b_words = set(task_b.description.lower().split())
        common_words = task_a_words & task_b_words
        
        return task_a_produces and task_b_consumes and len(common_words) > 2
    
    async def _build_resource_graph(self, plan: TaskPlan):
        """Build resource conflict graph"""
        
        # Group tasks by shared resources
        resource_tasks: Dict[str, List[str]] = {}
        
        for task_id, task in plan.tasks.items():
            for resource in task.required_resources:
                if resource.identifier not in resource_tasks:
                    resource_tasks[resource.identifier] = []
                resource_tasks[resource.identifier].append(task_id)
        
        # Create edges between tasks that share resources
        for resource_id, task_ids in resource_tasks.items():
            for i, task_a in enumerate(task_ids):
                for task_b in task_ids[i+1:]:
                    self.resource_graph.add_edge(
                        task_a, task_b,
                        resource=resource_id,
                        conflict_type="resource_contention"
                    )
    
    async def _detect_conflicts(self, plan: TaskPlan) -> List[Conflict]:
        """Detect various types of conflicts in the plan"""
        
        conflicts = []
        
        # Detect circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            for cycle in cycles:
                conflict = Conflict(
                    conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                    involved_tasks=cycle,
                    severity=1.0,
                    description=f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}",
                    suggested_resolution="Remove or modify one of the dependencies in the cycle",
                    impact_assessment={"blocks_execution": True, "affects_all_tasks": True}
                )
                conflicts.append(conflict)
        except nx.NetworkXError:
            pass  # No cycles found
        
        # Detect resource contentions
        resource_conflicts = await self._detect_resource_conflicts(plan)
        conflicts.extend(resource_conflicts)
        
        # Detect temporal impossibilities
        temporal_conflicts = await self._detect_temporal_conflicts(plan)
        conflicts.extend(temporal_conflicts)
        
        # Detect capacity overflows
        capacity_conflicts = await self._detect_capacity_conflicts(plan)
        conflicts.extend(capacity_conflicts)
        
        return conflicts
    
    async def _detect_resource_conflicts(self, plan: TaskPlan) -> List[Conflict]:
        """Detect resource contention conflicts"""
        
        conflicts = []
        
        # Analyze resource usage over time
        resource_schedule: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
        
        for task_id, task in plan.tasks.items():
            if not task.estimated_duration:
                continue
                
            # Estimate task start time based on dependencies
            start_time = await self._estimate_task_start_time(task, plan)
            end_time = start_time + task.estimated_duration
            
            for resource in task.required_resources:
                if resource.identifier not in resource_schedule:
                    resource_schedule[resource.identifier] = []
                
                resource_schedule[resource.identifier].append((start_time, end_time, task_id))
        
        # Check for overlapping resource usage
        for resource_id, schedule in resource_schedule.items():
            schedule.sort(key=lambda x: x[0])  # Sort by start time
            
            for i in range(len(schedule) - 1):
                current_end = schedule[i][1]
                next_start = schedule[i + 1][0]
                
                if current_end > next_start:  # Overlap detected
                    conflict = Conflict(
                        conflict_type=ConflictType.RESOURCE_CONTENTION,
                        involved_tasks=[schedule[i][2], schedule[i + 1][2]],
                        severity=0.8,
                        description=f"Resource '{resource_id}' contention between tasks",
                        suggested_resolution="Reschedule tasks or add resource capacity",
                        impact_assessment={
                            "delays_execution": True,
                            "resource": resource_id,
                            "overlap_duration": (current_end - next_start).total_seconds()
                        }
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_temporal_conflicts(self, plan: TaskPlan) -> List[Conflict]:
        """Detect temporal impossibility conflicts"""
        
        conflicts = []
        
        for task_id, task in plan.tasks.items():
            if not task.deadline:
                continue
            
            # Calculate earliest possible completion time
            earliest_completion = await self._calculate_earliest_completion(task, plan)
            
            if earliest_completion > task.deadline:
                conflict = Conflict(
                    conflict_type=ConflictType.TEMPORAL_IMPOSSIBILITY,
                    involved_tasks=[task_id],
                    severity=0.9,
                    description=f"Task cannot meet deadline: earliest completion {earliest_completion}, deadline {task.deadline}",
                    suggested_resolution="Extend deadline, reduce scope, or prioritize dependencies",
                    impact_assessment={
                        "deadline_miss": True,
                        "delay_amount": (earliest_completion - task.deadline).total_seconds()
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_capacity_conflicts(self, plan: TaskPlan) -> List[Conflict]:
        """Detect capacity overflow conflicts"""
        
        conflicts = []
        
        # Simulate execution timeline
        timeline = await self._simulate_execution_timeline(plan)
        
        # Check for capacity violations at each time point
        for time_point, active_tasks in timeline.items():
            total_cpu = sum(
                getattr(plan.tasks[task_id], 'cpu_requirement', 1.0)
                for task_id in active_tasks
            )
            total_memory = sum(
                getattr(plan.tasks[task_id], 'memory_requirement', 1.0)
                for task_id in active_tasks
            )
            
            # Check against system limits
            if total_cpu > 8.0:  # Assuming 8 CPU cores
                conflict = Conflict(
                    conflict_type=ConflictType.CAPACITY_OVERFLOW,
                    involved_tasks=active_tasks,
                    severity=0.7,
                    description=f"CPU capacity overflow at {time_point}: {total_cpu} > 8.0",
                    suggested_resolution="Serialize CPU-intensive tasks or increase capacity",
                    impact_assessment={"resource_type": "cpu", "overflow_amount": total_cpu - 8.0}
                )
                conflicts.append(conflict)
            
            if total_memory > 32.0:  # Assuming 32GB memory
                conflict = Conflict(
                    conflict_type=ConflictType.CAPACITY_OVERFLOW,
                    involved_tasks=active_tasks,
                    severity=0.8,
                    description=f"Memory capacity overflow at {time_point}: {total_memory} > 32.0",
                    suggested_resolution="Serialize memory-intensive tasks or increase capacity",
                    impact_assessment={"resource_type": "memory", "overflow_amount": total_memory - 32.0}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _analyze_critical_path(self, plan: TaskPlan) -> Dict[str, Any]:
        """Analyze critical path and bottlenecks"""
        
        # Calculate critical path using networkx
        try:
            # Use longest path for critical path analysis
            critical_path = nx.dag_longest_path(
                self.dependency_graph,
                weight='weight'
            )
            
            # Calculate critical path duration
            total_duration = timedelta()
            for task_id in critical_path:
                task = plan.tasks[task_id]
                if task.estimated_duration:
                    total_duration += task.estimated_duration
            
            # Identify bottlenecks
            bottlenecks = []
            for task_id in critical_path:
                task = plan.tasks[task_id]
                in_degree = self.dependency_graph.in_degree(task_id)
                out_degree = self.dependency_graph.out_degree(task_id)
                
                if in_degree > 1 or out_degree > 1:
                    bottlenecks.append({
                        "task_id": task_id,
                        "title": task.title,
                        "in_degree": in_degree,
                        "out_degree": out_degree,
                        "duration": task.estimated_duration.total_seconds() if task.estimated_duration else 0
                    })
            
            return {
                "critical_path": critical_path,
                "total_duration_seconds": total_duration.total_seconds(),
                "bottlenecks": bottlenecks,
                "parallelization_opportunities": self._find_parallelization_opportunities()
            }
            
        except nx.NetworkXError as e:
            self.logger.error(f"Critical path analysis failed: {e}")
            return {
                "error": str(e),
                "critical_path": [],
                "total_duration_seconds": 0,
                "bottlenecks": []
            }
    
    def _find_parallelization_opportunities(self) -> List[Dict[str, Any]]:
        """Find tasks that can be executed in parallel"""
        
        opportunities = []
        
        # Find tasks with no dependencies between them
        for node_set in nx.weakly_connected_components(self.dependency_graph):
            if len(node_set) > 1:
                # Check if tasks in this component can run in parallel
                subgraph = self.dependency_graph.subgraph(node_set)
                
                # Find nodes with no incoming edges (can start immediately)
                ready_nodes = [n for n in node_set if subgraph.in_degree(n) == 0]
                
                if len(ready_nodes) > 1:
                    opportunities.append({
                        "parallel_tasks": ready_nodes,
                        "potential_time_savings": "high",
                        "complexity": "low" if len(ready_nodes) <= 3 else "medium"
                    })
        
        return opportunities
    
    async def _analyze_resource_dependencies(self, plan: TaskPlan) -> Dict[str, Any]:
        """Analyze resource usage patterns and dependencies"""
        
        resource_analysis = {
            "resource_types": {},
            "contention_points": [],
            "utilization_forecast": {},
            "optimization_suggestions": []
        }
        
        # Analyze each resource type
        resource_usage: Dict[str, List[Dict[str, Any]]] = {}
        
        for task_id, task in plan.tasks.items():
            for resource in task.required_resources:
                if resource.type not in resource_usage:
                    resource_usage[resource.type] = []
                
                resource_usage[resource.type].append({
                    "task_id": task_id,
                    "identifier": resource.identifier,
                    "amount": resource.amount or 1.0,
                    "duration": task.estimated_duration
                })
        
        # Analyze each resource type
        for resource_type, usage_list in resource_usage.items():
            total_demand = sum(item["amount"] for item in usage_list)
            unique_resources = len(set(item["identifier"] for item in usage_list))
            
            resource_analysis["resource_types"][resource_type] = {
                "total_demand": total_demand,
                "unique_resources": unique_resources,
                "average_demand_per_task": total_demand / len(usage_list) if usage_list else 0,
                "peak_concurrent_demand": self._calculate_peak_demand(usage_list)
            }
            
            # Check for contention
            identifier_counts = {}
            for item in usage_list:
                identifier = item["identifier"]
                if identifier not in identifier_counts:
                    identifier_counts[identifier] = 0
                identifier_counts[identifier] += 1
            
            for identifier, count in identifier_counts.items():
                if count > 1:
                    resource_analysis["contention_points"].append({
                        "resource_type": resource_type,
                        "identifier": identifier,
                        "conflicting_tasks": count,
                        "severity": min(1.0, count / 5.0)  # Normalize to 0-1
                    })
        
        return resource_analysis
    
    def _calculate_peak_demand(self, usage_list: List[Dict[str, Any]]) -> float:
        """Calculate peak concurrent demand for a resource"""
        # Simplified calculation - in reality would need timeline simulation
        return max(item["amount"] for item in usage_list) if usage_list else 0.0
    
    async def _generate_optimizations(
        self,
        plan: TaskPlan,
        conflicts: List[Conflict]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        
        optimizations = []
        
        # Optimization based on conflicts
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.RESOURCE_CONTENTION:
                optimizations.append({
                    "type": "resource_optimization",
                    "description": "Serialize conflicting tasks",
                    "impact": "Reduces resource contention",
                    "tasks_affected": conflict.involved_tasks,
                    "implementation": "Add sequential dependencies"
                })
            
            elif conflict.conflict_type == ConflictType.CIRCULAR_DEPENDENCY:
                optimizations.append({
                    "type": "dependency_optimization", 
                    "description": "Break circular dependency",
                    "impact": "Enables plan execution",
                    "tasks_affected": conflict.involved_tasks,
                    "implementation": "Remove weakest dependency link"
                })
        
        # General optimizations
        parallel_groups = self._find_parallelization_opportunities()
        if parallel_groups:
            optimizations.append({
                "type": "parallelization",
                "description": f"Execute {len(parallel_groups)} task groups in parallel",
                "impact": "Reduces overall execution time",
                "potential_time_savings": "20-40%",
                "implementation": "Schedule parallel execution"
            })
        
        # Resource pooling optimization
        if len(set(r.identifier for task in plan.tasks.values() for r in task.required_resources)) > 10:
            optimizations.append({
                "type": "resource_pooling",
                "description": "Consolidate similar resources",
                "impact": "Improves resource utilization",
                "implementation": "Group tasks by resource type"
            })
        
        return optimizations
    
    async def _calculate_execution_paths(self, plan: TaskPlan) -> List[ExecutionPath]:
        """Calculate optimized execution paths"""
        
        paths = []
        
        # Sequential path (baseline)
        sequential_path = ExecutionPath(
            task_sequence=plan.execution_order,
            parallel_groups=[],
            estimated_duration=sum(
                (task.estimated_duration or timedelta())
                for task in plan.tasks.values()
            ),
            resource_utilization={"cpu": 1.0, "memory": 1.0},
            risk_score=0.2,
            efficiency_score=0.5
        )
        paths.append(sequential_path)
        
        # Optimized parallel path
        parallel_groups = self._create_parallel_groups(plan)
        if parallel_groups:
            # Calculate duration for parallel execution
            max_group_duration = timedelta()
            for group in parallel_groups:
                group_duration = max(
                    plan.tasks[task_id].estimated_duration or timedelta()
                    for task_id in group
                )
                max_group_duration += group_duration
            
            parallel_path = ExecutionPath(
                task_sequence=[],
                parallel_groups=parallel_groups,
                estimated_duration=max_group_duration,
                resource_utilization={"cpu": 0.8, "memory": 0.7},
                risk_score=0.4,
                efficiency_score=0.8
            )
            paths.append(parallel_path)
        
        return paths
    
    def _create_parallel_groups(self, plan: TaskPlan) -> List[List[str]]:
        """Create groups of tasks that can execute in parallel"""
        
        # Use topological sorting with parallelization
        groups = []
        remaining_tasks = set(plan.tasks.keys())
        
        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = plan.tasks[task_id]
                if all(dep not in remaining_tasks for dep in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Handle remaining tasks (shouldn't happen with proper dependency resolution)
                ready_tasks = list(remaining_tasks)
            
            groups.append(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return groups
    
    async def _estimate_task_start_time(self, task: Task, plan: TaskPlan) -> datetime:
        """Estimate when a task can start based on dependencies"""
        
        base_time = datetime.now()
        
        if not task.dependencies:
            return base_time
        
        # Find latest completion time of dependencies
        latest_completion = base_time
        for dep_id in task.dependencies:
            if dep_id in plan.tasks:
                dep_task = plan.tasks[dep_id]
                dep_start = await self._estimate_task_start_time(dep_task, plan)
                dep_completion = dep_start + (dep_task.estimated_duration or timedelta())
                latest_completion = max(latest_completion, dep_completion)
        
        return latest_completion
    
    async def _calculate_earliest_completion(self, task: Task, plan: TaskPlan) -> datetime:
        """Calculate earliest possible completion time for a task"""
        
        start_time = await self._estimate_task_start_time(task, plan)
        return start_time + (task.estimated_duration or timedelta())
    
    async def _simulate_execution_timeline(self, plan: TaskPlan) -> Dict[datetime, List[str]]:
        """Simulate execution timeline to identify resource conflicts"""
        
        timeline = {}
        current_time = datetime.now()
        
        # Simple simulation - in reality would be more sophisticated
        for task_id in plan.execution_order:
            task = plan.tasks[task_id]
            if task.estimated_duration:
                end_time = current_time + task.estimated_duration
                timeline[current_time] = [task_id]
                current_time = end_time
        
        return timeline
    
    def _calculate_parallelization_potential(self, plan: TaskPlan) -> float:
        """Calculate how much the plan can benefit from parallelization"""
        
        if not plan.tasks:
            return 0.0
        
        # Count tasks that have no dependencies on each other
        independent_tasks = 0
        total_tasks = len(plan.tasks)
        
        for task_id in plan.tasks:
            has_dependencies = bool(plan.tasks[task_id].dependencies)
            if not has_dependencies:
                independent_tasks += 1
        
        return independent_tasks / total_tasks
    
    def _export_dependency_graph(self) -> Dict[str, Any]:
        """Export dependency graph for analysis"""
        
        return {
            "nodes": [
                {
                    "id": node,
                    "task_title": self.dependency_graph.nodes[node].get("task", Task()).title,
                    "priority": self.dependency_graph.nodes[node].get("priority", "medium")
                }
                for node in self.dependency_graph.nodes()
            ],
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "dependency_type": self.dependency_graph.edges[edge].get("dependency", Dependency("", "", DependencyType.REQUIRED, 1.0, "")).dependency_type.value,
                    "strength": self.dependency_graph.edges[edge].get("weight", 1.0)
                }
                for edge in self.dependency_graph.edges()
            ]
        }
    
    def _conflict_to_dict(self, conflict: Conflict) -> Dict[str, Any]:
        """Convert conflict to dictionary"""
        return {
            "type": conflict.conflict_type.value,
            "involved_tasks": conflict.involved_tasks,
            "severity": conflict.severity,
            "description": conflict.description,
            "suggested_resolution": conflict.suggested_resolution,
            "impact_assessment": conflict.impact_assessment
        }
    
    def _path_to_dict(self, path: ExecutionPath) -> Dict[str, Any]:
        """Convert execution path to dictionary"""
        return {
            "task_sequence": path.task_sequence,
            "parallel_groups": path.parallel_groups,
            "estimated_duration_seconds": path.estimated_duration.total_seconds(),
            "resource_utilization": path.resource_utilization,
            "risk_score": path.risk_score,
            "efficiency_score": path.efficiency_score
        }
    
    async def get_dependency_insights(self, plan_id: str) -> Dict[str, Any]:
        """Get insights about dependencies for a specific plan"""
        
        if plan_id not in self.analysis_cache:
            return {"error": "Plan analysis not found"}
        
        analysis = self.analysis_cache[plan_id]
        
        insights = {
            "complexity_score": self._calculate_complexity_score(analysis),
            "execution_efficiency": self._calculate_execution_efficiency(analysis),
            "risk_assessment": self._assess_execution_risks(analysis),
            "recommendations": self._generate_execution_recommendations(analysis)
        }
        
        return insights
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate plan complexity score based on dependencies"""
        
        dependency_count = analysis["metrics"]["total_dependencies"]
        conflict_count = analysis["metrics"]["conflict_count"]
        
        # Simple complexity calculation
        base_complexity = min(1.0, dependency_count / 20.0)
        conflict_penalty = min(0.5, conflict_count / 10.0)
        
        return min(1.0, base_complexity + conflict_penalty)
    
    def _calculate_execution_efficiency(self, analysis: Dict[str, Any]) -> float:
        """Calculate potential execution efficiency"""
        
        parallelization_potential = analysis["metrics"]["parallelization_potential"]
        conflict_impact = analysis["metrics"]["conflict_count"] / 10.0
        
        return max(0.0, parallelization_potential - conflict_impact)
    
    def _assess_execution_risks(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks in plan execution"""
        
        high_severity_conflicts = analysis["metrics"]["high_severity_conflicts"]
        total_conflicts = analysis["metrics"]["conflict_count"]
        
        return {
            "overall_risk": "high" if high_severity_conflicts > 0 else "medium" if total_conflicts > 3 else "low",
            "critical_risks": high_severity_conflicts,
            "manageable_risks": total_conflicts - high_severity_conflicts,
            "risk_factors": [
                "Circular dependencies" if any(c["type"] == "circular_dependency" for c in analysis["conflicts"]) else None,
                "Resource contention" if any(c["type"] == "resource_contention" for c in analysis["conflicts"]) else None,
                "Temporal constraints" if any(c["type"] == "temporal_impossibility" for c in analysis["conflicts"]) else None
            ]
        }
    
    def _generate_execution_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific execution recommendations"""
        
        recommendations = []
        
        if analysis["metrics"]["parallelization_potential"] > 0.5:
            recommendations.append("Consider parallel execution to reduce overall time")
        
        if analysis["metrics"]["high_severity_conflicts"] > 0:
            recommendations.append("Resolve high-severity conflicts before execution")
        
        if len(analysis["execution_paths"]) > 1:
            recommendations.append("Multiple execution paths available - choose based on priorities")
        
        if analysis["metrics"]["conflict_count"] > 5:
            recommendations.append("High conflict count - consider plan simplification")
        
        return recommendations