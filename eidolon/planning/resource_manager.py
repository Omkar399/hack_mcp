"""
Resource Management system for Eidolon AI Personal Assistant

Handles resource allocation, scheduling, conflict resolution, and optimization
for complex task execution with limited resources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from .task_planner import Task, TaskPlan, TaskResource, TaskStatus


class ResourceStatus(Enum):
    """Resource availability status"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    RESERVED = "reserved"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    DEADLINE_AWARE = "deadline_aware"


@dataclass
class ResourcePool:
    """Represents a pool of similar resources"""
    resource_type: str
    identifier: str
    capacity: float
    current_usage: float = 0.0
    reserved_usage: float = 0.0
    status: ResourceStatus = ResourceStatus.AVAILABLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def available_capacity(self) -> float:
        """Get currently available capacity"""
        return max(0.0, self.capacity - self.current_usage - self.reserved_usage)
    
    @property
    def utilization_rate(self) -> float:
        """Get current utilization as percentage"""
        if self.capacity == 0:
            return 0.0
        return (self.current_usage + self.reserved_usage) / self.capacity


@dataclass
class ResourceAllocation:
    """Represents an allocation of resources to a task"""
    allocation_id: str
    task_id: str
    resource_type: str
    resource_identifier: str
    amount: float
    start_time: datetime
    end_time: datetime
    status: ResourceStatus = ResourceStatus.RESERVED
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConflict:
    """Represents a resource conflict between tasks"""
    conflict_id: str
    resource_type: str
    resource_identifier: str
    conflicting_tasks: List[str]
    total_demand: float
    available_capacity: float
    severity: float  # 0.0 to 1.0
    suggested_resolution: str
    impact_assessment: Dict[str, Any]


class ResourceManager:
    """Advanced resource management system with intelligent allocation"""
    
    def __init__(self):
        self.logger = get_component_logger("resource_manager")
        self.config = get_config()
        
        # Resource tracking
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_queue: List[ResourceAllocation] = []
        
        # Scheduling and optimization
        self.allocation_strategy = AllocationStrategy.PRIORITY_BASED
        self.optimization_enabled = True
        
        # Monitoring and analytics
        self.usage_metrics: Dict[str, Dict[str, Any]] = {}
        self.conflict_history: List[ResourceConflict] = []
        self.allocation_history: List[ResourceAllocation] = []
        
        # Initialize default resource pools
        self._initialize_default_resources()
    
    def _initialize_default_resources(self):
        """Initialize default system resource pools"""
        
        # CPU resources
        self.resource_pools["cpu"] = ResourcePool(
            resource_type="cpu",
            identifier="system_cpu",
            capacity=8.0,  # 8 CPU cores
            metadata={"unit": "cores", "description": "System CPU cores"}
        )
        
        # Memory resources
        self.resource_pools["memory"] = ResourcePool(
            resource_type="memory",
            identifier="system_memory",
            capacity=32.0,  # 32GB memory
            metadata={"unit": "GB", "description": "System memory"}
        )
        
        # Network bandwidth
        self.resource_pools["network"] = ResourcePool(
            resource_type="network",
            identifier="system_network",
            capacity=1000.0,  # 1Gbps
            metadata={"unit": "Mbps", "description": "Network bandwidth"}
        )
        
        # API rate limits
        self.resource_pools["api_calls"] = ResourcePool(
            resource_type="api",
            identifier="cloud_api_calls",
            capacity=1000.0,  # 1000 calls per hour
            metadata={"unit": "calls/hour", "description": "Cloud API rate limit"}
        )
        
        # File system operations
        self.resource_pools["file_operations"] = ResourcePool(
            resource_type="file",
            identifier="file_system",
            capacity=100.0,  # 100 concurrent operations
            metadata={"unit": "operations", "description": "File system operations"}
        )
    
    @log_exceptions()
    async def allocate_resources(
        self,
        task: Task,
        allocation_strategy: Optional[AllocationStrategy] = None
    ) -> Dict[str, Any]:
        """Allocate resources for a task"""
        
        self.logger.info(f"Allocating resources for task {task.id}")
        
        strategy = allocation_strategy or self.allocation_strategy
        allocations = []
        conflicts = []
        
        # Process each required resource
        for resource_req in task.required_resources:
            result = await self._allocate_single_resource(
                task, resource_req, strategy
            )
            
            if result["success"]:
                allocations.append(result["allocation"])
            else:
                conflicts.append(result["conflict"])
        
        # If any allocation failed, release successful ones
        if conflicts:
            for allocation in allocations:
                await self._release_allocation(allocation.allocation_id)
            
            return {
                "success": False,
                "allocations": [],
                "conflicts": conflicts,
                "total_conflicts": len(conflicts),
                "suggested_actions": self._generate_conflict_resolutions(conflicts)
            }
        
        # All allocations successful
        for allocation in allocations:
            self.active_allocations[allocation.allocation_id] = allocation
        
        return {
            "success": True,
            "allocations": allocations,
            "conflicts": [],
            "allocation_ids": [a.allocation_id for a in allocations]
        }
    
    async def _allocate_single_resource(
        self,
        task: Task,
        resource_req: TaskResource,
        strategy: AllocationStrategy
    ) -> Dict[str, Any]:
        """Allocate a single resource for a task"""
        
        # Find matching resource pool
        pool = self._find_resource_pool(resource_req)
        if not pool:
            # Create new resource pool if needed
            pool = await self._create_resource_pool(resource_req)
        
        required_amount = resource_req.amount or 1.0
        
        # Check availability
        if pool.available_capacity < required_amount:
            conflict = ResourceConflict(
                conflict_id=f"conflict_{task.id}_{resource_req.identifier}",
                resource_type=resource_req.type,
                resource_identifier=resource_req.identifier,
                conflicting_tasks=[task.id],
                total_demand=required_amount,
                available_capacity=pool.available_capacity,
                severity=min(1.0, required_amount / pool.available_capacity),
                suggested_resolution=self._suggest_conflict_resolution(pool, required_amount),
                impact_assessment={"blocks_task_execution": True}
            )
            
            return {"success": False, "conflict": conflict}
        
        # Calculate allocation timing
        start_time = await self._calculate_allocation_start_time(task)
        end_time = start_time + (task.estimated_duration or timedelta(hours=1))
        
        # Create allocation
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{task.id}_{resource_req.identifier}_{int(start_time.timestamp())}",
            task_id=task.id,
            resource_type=resource_req.type,
            resource_identifier=resource_req.identifier,
            amount=required_amount,
            start_time=start_time,
            end_time=end_time,
            priority=self._calculate_allocation_priority(task),
            metadata={
                "task_title": task.title,
                "task_priority": task.priority.value,
                "allocation_strategy": strategy.value
            }
        )
        
        # Reserve the resources
        pool.reserved_usage += required_amount
        
        return {"success": True, "allocation": allocation}
    
    def _find_resource_pool(self, resource_req: TaskResource) -> Optional[ResourcePool]:
        """Find matching resource pool for a requirement"""
        
        # Direct identifier match
        if resource_req.identifier in self.resource_pools:
            return self.resource_pools[resource_req.identifier]
        
        # Type-based match
        for pool in self.resource_pools.values():
            if pool.resource_type == resource_req.type:
                return pool
        
        return None
    
    async def _create_resource_pool(self, resource_req: TaskResource) -> ResourcePool:
        """Create new resource pool for unknown resource"""
        
        # Default capacity based on resource type
        default_capacities = {
            "api": 100.0,
            "file": 50.0,
            "application": 10.0,
            "network": 100.0,
            "cpu": 1.0,
            "memory": 1.0
        }
        
        capacity = default_capacities.get(resource_req.type, 10.0)
        
        pool = ResourcePool(
            resource_type=resource_req.type,
            identifier=resource_req.identifier,
            capacity=capacity,
            metadata={"auto_created": True, "source": "resource_requirement"}
        )
        
        self.resource_pools[resource_req.identifier] = pool
        self.logger.info(f"Created new resource pool: {resource_req.identifier}")
        
        return pool
    
    async def _calculate_allocation_start_time(self, task: Task) -> datetime:
        """Calculate when allocation should start for a task"""
        
        # For now, use current time
        # In a more sophisticated system, this would consider:
        # - Task dependencies
        # - Resource availability windows
        # - User preferences
        # - System load patterns
        
        base_time = datetime.now()
        
        # Add small delay for task preparation
        return base_time + timedelta(minutes=1)
    
    def _calculate_allocation_priority(self, task: Task) -> int:
        """Calculate allocation priority for a task"""
        
        # Convert task priority to numeric value
        priority_values = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25
        }
        
        base_priority = priority_values.get(task.priority.value, 50)
        
        # Adjust based on deadline urgency
        if task.deadline:
            time_to_deadline = task.deadline - datetime.now()
            if time_to_deadline < timedelta(hours=1):
                base_priority += 25
            elif time_to_deadline < timedelta(hours=6):
                base_priority += 10
        
        return min(100, base_priority)
    
    def _suggest_conflict_resolution(self, pool: ResourcePool, required_amount: float) -> str:
        """Suggest resolution for resource conflict"""
        
        shortage = required_amount - pool.available_capacity
        utilization = pool.utilization_rate
        
        if utilization > 0.9:
            return "Consider scheduling task for later when resources are less utilized"
        elif shortage < pool.capacity * 0.1:
            return "Minor shortage - consider reducing resource requirements or waiting briefly"
        else:
            return "Significant resource shortage - consider task decomposition or resource scaling"
    
    @log_exceptions()
    async def release_resources(self, task_id: str) -> Dict[str, Any]:
        """Release all resources allocated to a task"""
        
        self.logger.info(f"Releasing resources for task {task_id}")
        
        released_allocations = []
        
        # Find and release all allocations for this task
        for allocation_id, allocation in list(self.active_allocations.items()):
            if allocation.task_id == task_id:
                await self._release_allocation(allocation_id)
                released_allocations.append(allocation)
        
        return {
            "success": True,
            "released_count": len(released_allocations),
            "released_allocations": [a.allocation_id for a in released_allocations]
        }
    
    async def _release_allocation(self, allocation_id: str):
        """Release a specific resource allocation"""
        
        if allocation_id not in self.active_allocations:
            self.logger.warning(f"Allocation {allocation_id} not found")
            return
        
        allocation = self.active_allocations[allocation_id]
        
        # Find resource pool and release capacity
        pool = self.resource_pools.get(allocation.resource_identifier)
        if pool:
            if allocation.status == ResourceStatus.RESERVED:
                pool.reserved_usage = max(0, pool.reserved_usage - allocation.amount)
            elif allocation.status == ResourceStatus.IN_USE:
                pool.current_usage = max(0, pool.current_usage - allocation.amount)
        
        # Move to history
        allocation.status = ResourceStatus.AVAILABLE
        self.allocation_history.append(allocation)
        del self.active_allocations[allocation_id]
        
        self.logger.debug(f"Released allocation {allocation_id}")
    
    @log_exceptions()
    async def start_resource_usage(self, task_id: str) -> Dict[str, Any]:
        """Mark resources as actively in use for a task"""
        
        activated_allocations = []
        
        for allocation in self.active_allocations.values():
            if allocation.task_id == task_id and allocation.status == ResourceStatus.RESERVED:
                # Move from reserved to active usage
                pool = self.resource_pools.get(allocation.resource_identifier)
                if pool:
                    pool.reserved_usage = max(0, pool.reserved_usage - allocation.amount)
                    pool.current_usage += allocation.amount
                    
                    allocation.status = ResourceStatus.IN_USE
                    activated_allocations.append(allocation)
        
        return {
            "success": True,
            "activated_count": len(activated_allocations),
            "activated_allocations": [a.allocation_id for a in activated_allocations]
        }
    
    @log_exceptions()
    async def check_resource_conflicts(self, plan: TaskPlan) -> List[ResourceConflict]:
        """Check for resource conflicts in a task plan"""
        
        self.logger.info(f"Checking resource conflicts for plan {plan.id}")
        
        conflicts = []
        
        # Simulate resource usage over time
        timeline = await self._simulate_resource_timeline(plan)
        
        # Check each time point for conflicts
        for time_point, resource_demands in timeline.items():
            conflicts.extend(await self._detect_conflicts_at_time(time_point, resource_demands))
        
        return conflicts
    
    async def _simulate_resource_timeline(self, plan: TaskPlan) -> Dict[datetime, Dict[str, float]]:
        """Simulate resource usage timeline for a plan"""
        
        timeline = defaultdict(lambda: defaultdict(float))
        
        for task in plan.tasks.values():
            if not task.estimated_duration:
                continue
            
            start_time = await self._estimate_task_start_time(task, plan)
            end_time = start_time + task.estimated_duration
            
            # Add resource demands at start time
            for resource in task.required_resources:
                resource_key = f"{resource.type}:{resource.identifier}"
                timeline[start_time][resource_key] += resource.amount or 1.0
        
        return dict(timeline)
    
    async def _estimate_task_start_time(self, task: Task, plan: TaskPlan) -> datetime:
        """Estimate when a task will start in the plan"""
        
        # Simple estimation based on dependencies
        base_time = datetime.now()
        
        if not task.dependencies:
            return base_time
        
        # Find latest dependency completion
        latest_completion = base_time
        for dep_id in task.dependencies:
            if dep_id in plan.tasks:
                dep_task = plan.tasks[dep_id]
                dep_start = await self._estimate_task_start_time(dep_task, plan)
                dep_completion = dep_start + (dep_task.estimated_duration or timedelta(hours=1))
                latest_completion = max(latest_completion, dep_completion)
        
        return latest_completion
    
    async def _detect_conflicts_at_time(
        self,
        time_point: datetime,
        resource_demands: Dict[str, float]
    ) -> List[ResourceConflict]:
        """Detect resource conflicts at a specific time point"""
        
        conflicts = []
        
        for resource_key, total_demand in resource_demands.items():
            resource_type, identifier = resource_key.split(":", 1)
            
            pool = self.resource_pools.get(identifier)
            if not pool:
                continue
            
            if total_demand > pool.capacity:
                conflict = ResourceConflict(
                    conflict_id=f"conflict_{identifier}_{int(time_point.timestamp())}",
                    resource_type=resource_type,
                    resource_identifier=identifier,
                    conflicting_tasks=[],  # Would need to track which tasks contribute
                    total_demand=total_demand,
                    available_capacity=pool.capacity,
                    severity=min(1.0, total_demand / pool.capacity),
                    suggested_resolution=f"Increase {resource_type} capacity or serialize tasks",
                    impact_assessment={
                        "time_point": time_point.isoformat(),
                        "overflow_amount": total_demand - pool.capacity
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _generate_conflict_resolutions(self, conflicts: List[ResourceConflict]) -> List[str]:
        """Generate specific actions to resolve conflicts"""
        
        resolutions = []
        
        for conflict in conflicts:
            if conflict.severity > 0.8:
                resolutions.append(f"Critical: Increase {conflict.resource_type} capacity immediately")
            elif conflict.severity > 0.5:
                resolutions.append(f"Moderate: Consider task rescheduling for {conflict.resource_identifier}")
            else:
                resolutions.append(f"Minor: Monitor {conflict.resource_identifier} usage")
        
        # Add general recommendations
        if len(conflicts) > 3:
            resolutions.append("Consider overall resource optimization and task prioritization")
        
        return resolutions
    
    @log_exceptions()
    async def optimize_resource_allocation(self, plan: TaskPlan) -> Dict[str, Any]:
        """Optimize resource allocation for better efficiency"""
        
        self.logger.info(f"Optimizing resource allocation for plan {plan.id}")
        
        if not self.optimization_enabled:
            return {"optimization_applied": False, "reason": "Optimization disabled"}
        
        # Analyze current allocation efficiency
        efficiency_metrics = await self._calculate_allocation_efficiency(plan)
        
        optimizations = []
        
        # Resource pooling optimization
        if efficiency_metrics["fragmentation_score"] > 0.7:
            optimizations.append({
                "type": "resource_pooling",
                "description": "Consolidate similar resource requirements",
                "potential_improvement": "20-30% better utilization"
            })
        
        # Temporal optimization
        if efficiency_metrics["temporal_conflicts"] > 0:
            optimizations.append({
                "type": "temporal_optimization",
                "description": "Reschedule tasks to reduce resource conflicts",
                "potential_improvement": f"Resolve {efficiency_metrics['temporal_conflicts']} conflicts"
            })
        
        # Load balancing
        max_utilization = max(pool.utilization_rate for pool in self.resource_pools.values())
        if max_utilization > 0.9:
            optimizations.append({
                "type": "load_balancing",
                "description": "Distribute load across available resources",
                "potential_improvement": "Reduce peak utilization"
            })
        
        return {
            "optimization_applied": len(optimizations) > 0,
            "efficiency_metrics": efficiency_metrics,
            "optimizations": optimizations,
            "estimated_improvement": self._estimate_optimization_impact(optimizations)
        }
    
    async def _calculate_allocation_efficiency(self, plan: TaskPlan) -> Dict[str, Any]:
        """Calculate resource allocation efficiency metrics"""
        
        total_resources = len(self.resource_pools)
        utilized_resources = len([p for p in self.resource_pools.values() if p.utilization_rate > 0.1])
        
        # Calculate fragmentation (unused capacity in partially used pools)
        fragmentation_score = 0.0
        if total_resources > 0:
            fragmented_pools = [
                p for p in self.resource_pools.values()
                if 0.1 < p.utilization_rate < 0.8
            ]
            fragmentation_score = len(fragmented_pools) / total_resources
        
        # Check for temporal conflicts
        conflicts = await self.check_resource_conflicts(plan)
        temporal_conflicts = len([c for c in conflicts if c.severity > 0.5])
        
        return {
            "resource_utilization": utilized_resources / total_resources if total_resources > 0 else 0,
            "fragmentation_score": fragmentation_score,
            "temporal_conflicts": temporal_conflicts,
            "average_utilization": sum(p.utilization_rate for p in self.resource_pools.values()) / total_resources if total_resources > 0 else 0
        }
    
    def _estimate_optimization_impact(self, optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate the impact of proposed optimizations"""
        
        if not optimizations:
            return {"time_savings": 0, "efficiency_gain": 0, "conflict_reduction": 0}
        
        # Simple estimation based on optimization types
        time_savings = 0
        efficiency_gain = 0
        conflict_reduction = 0
        
        for opt in optimizations:
            if opt["type"] == "temporal_optimization":
                time_savings += 15  # 15% time savings
                conflict_reduction += 50  # 50% conflict reduction
            elif opt["type"] == "resource_pooling":
                efficiency_gain += 25  # 25% efficiency gain
            elif opt["type"] == "load_balancing":
                efficiency_gain += 15  # 15% efficiency gain
        
        return {
            "time_savings": min(50, time_savings),  # Cap at 50%
            "efficiency_gain": min(40, efficiency_gain),  # Cap at 40%
            "conflict_reduction": min(80, conflict_reduction)  # Cap at 80%
        }
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "resource_pools": {
                identifier: {
                    "type": pool.resource_type,
                    "capacity": pool.capacity,
                    "current_usage": pool.current_usage,
                    "reserved_usage": pool.reserved_usage,
                    "available_capacity": pool.available_capacity,
                    "utilization_rate": pool.utilization_rate,
                    "status": pool.status.value
                }
                for identifier, pool in self.resource_pools.items()
            },
            "active_allocations": len(self.active_allocations),
            "total_allocations_today": len([
                a for a in self.allocation_history
                if a.start_time.date() == datetime.now().date()
            ]),
            "system_metrics": {
                "memory_pools": len([p for p in self.resource_pools.values() if p.resource_type == "memory"]),
                "cpu_pools": len([p for p in self.resource_pools.values() if p.resource_type == "cpu"]),
                "api_pools": len([p for p in self.resource_pools.values() if p.resource_type == "api"]),
                "average_utilization": sum(p.utilization_rate for p in self.resource_pools.values()) / len(self.resource_pools) if self.resource_pools else 0
            }
        }
    
    async def get_allocation_recommendations(self, task: Task) -> Dict[str, Any]:
        """Get recommendations for resource allocation strategy"""
        
        recommendations = []
        
        # Analyze task resource requirements
        total_resource_demand = sum(r.amount or 1.0 for r in task.required_resources)
        resource_types = set(r.type for r in task.required_resources)
        
        # Check current system load
        avg_utilization = sum(p.utilization_rate for p in self.resource_pools.values()) / len(self.resource_pools) if self.resource_pools else 0
        
        if avg_utilization > 0.8:
            recommendations.append({
                "type": "scheduling",
                "priority": "high",
                "message": "System under high load - consider scheduling for later"
            })
        
        if total_resource_demand > 10:
            recommendations.append({
                "type": "decomposition",
                "priority": "medium",
                "message": "High resource demand - consider breaking into smaller tasks"
            })
        
        if len(resource_types) > 3:
            recommendations.append({
                "type": "optimization",
                "priority": "low",
                "message": "Multiple resource types - ensure efficient allocation strategy"
            })
        
        # Suggest optimal allocation strategy
        suggested_strategy = AllocationStrategy.PRIORITY_BASED
        if task.deadline and (task.deadline - datetime.now()) < timedelta(hours=2):
            suggested_strategy = AllocationStrategy.DEADLINE_AWARE
        elif len(task.required_resources) > 5:
            suggested_strategy = AllocationStrategy.LOAD_BALANCED
        
        return {
            "task_id": task.id,
            "recommendations": recommendations,
            "suggested_strategy": suggested_strategy.value,
            "estimated_allocation_time": "immediate" if avg_utilization < 0.5 else "within 10 minutes",
            "resource_availability": {
                resource_type: len([p for p in self.resource_pools.values() if p.resource_type == resource_type and p.utilization_rate < 0.8])
                for resource_type in resource_types
            }
        }