"""
Workflow Optimizer for Eidolon AI Personal Assistant

Analyzes user workflows to identify optimization opportunities, bottlenecks,
and automation possibilities for improved productivity and efficiency.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.memory import MemorySystem
from ..storage.metadata_db import MetadataDatabase
from ..models.cloud_api import CloudAPIManager
from .pattern_recognizer import PatternRecognizer, UserPattern, PatternType


class OptimizationType(Enum):
    """Types of workflow optimizations"""
    CONTEXT_SWITCHING = "context_switching"
    AUTOMATION = "automation"
    BATCHING = "batching"
    SCHEDULING = "scheduling"
    TOOL_SELECTION = "tool_selection"
    EFFICIENCY = "efficiency"
    DISTRACTION_REDUCTION = "distraction_reduction"
    ERROR_PREVENTION = "error_prevention"


class OptimizationPriority(Enum):
    """Priority levels for optimizations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowInsight:
    """Represents a workflow optimization insight"""
    id: str
    optimization_type: OptimizationType
    title: str
    description: str
    impact_score: float  # 0.0 to 1.0 - potential impact
    confidence: float  # 0.0 to 1.0 - confidence in recommendation
    priority: OptimizationPriority
    
    # Optimization details
    current_workflow: str
    proposed_optimization: str
    expected_benefits: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    # Supporting data
    supporting_patterns: List[str] = field(default_factory=list)
    affected_apps: List[str] = field(default_factory=list)
    time_savings_estimate: Optional[timedelta] = None
    effort_estimate: str = "medium"  # low, medium, high
    
    # Metrics and evidence
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    inefficiency_indicators: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Outcome tracking
    implemented: bool = False
    implementation_date: Optional[datetime] = None
    actual_benefits: List[str] = field(default_factory=list)
    user_feedback: Optional[str] = None
    
    def get_priority_score(self) -> float:
        """Calculate priority score based on impact and confidence"""
        priority_weights = {
            OptimizationPriority.LOW: 0.25,
            OptimizationPriority.MEDIUM: 0.5,
            OptimizationPriority.HIGH: 0.75,
            OptimizationPriority.CRITICAL: 1.0
        }
        
        base_score = self.impact_score * self.confidence
        priority_multiplier = priority_weights[self.priority]
        
        return base_score * priority_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary"""
        return {
            'id': self.id,
            'optimization_type': self.optimization_type.value,
            'title': self.title,
            'description': self.description,
            'impact_score': self.impact_score,
            'confidence': self.confidence,
            'priority': self.priority.value,
            'priority_score': self.get_priority_score(),
            'current_workflow': self.current_workflow,
            'proposed_optimization': self.proposed_optimization,
            'expected_benefits': self.expected_benefits,
            'implementation_steps': self.implementation_steps,
            'supporting_patterns': self.supporting_patterns,
            'affected_apps': self.affected_apps,
            'time_savings_estimate': self.time_savings_estimate.total_seconds() if self.time_savings_estimate else None,
            'effort_estimate': self.effort_estimate,
            'baseline_metrics': self.baseline_metrics,
            'inefficiency_indicators': self.inefficiency_indicators,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'implemented': self.implemented,
            'implementation_date': self.implementation_date.isoformat() if self.implementation_date else None,
            'actual_benefits': self.actual_benefits,
            'user_feedback': self.user_feedback
        }


class WorkflowOptimizer:
    """Advanced workflow optimization system for productivity enhancement"""
    
    def __init__(self):
        self.logger = get_component_logger("workflow_optimizer")
        self.config = get_config()
        self.memory = MemorySystem()
        self.database = MetadataDatabase()
        self.cloud_api = CloudAPIManager()
        self.pattern_recognizer = PatternRecognizer()
        
        # Optimization state
        self.optimization_insights: Dict[str, WorkflowInsight] = {}
        self.optimization_history: List[WorkflowInsight] = []
        
        # Analysis parameters
        self.analysis_window = timedelta(days=14)  # Default analysis window
        self.min_pattern_strength = 0.5  # Minimum pattern strength for optimization
        self.efficiency_threshold = 0.7  # Efficiency threshold for identifying issues
        
        # Load existing optimizations
        asyncio.create_task(self._load_existing_optimizations())
    
    @log_exceptions()
    async def analyze_workflow_patterns(
        self,
        time_range_days: int = 7
    ) -> List[WorkflowInsight]:
        """Analyze recent workflow patterns and generate optimization insights."""
        
        self.logger.info(f"Analyzing workflow patterns for last {time_range_days} days")
        
        # Get patterns from pattern recognizer
        patterns = await self.pattern_recognizer.get_recent_patterns(
            hours=time_range_days * 24
        )
        
        insights = []
        
        # Analyze different aspects of workflow
        insights.extend(await self._analyze_time_management(patterns))
        insights.extend(await self._analyze_task_ordering(patterns))
        insights.extend(await self._analyze_tool_usage(patterns))
        insights.extend(await self._analyze_context_switching(patterns))
        insights.extend(await self._analyze_break_patterns(patterns))
        insights.extend(await self._analyze_automation_opportunities(patterns))
        
        # Sort by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        # Store insights
        self.optimization_history.extend(insights)
        self.metrics["optimizations_suggested"] += len(insights)
        self.metrics["last_analysis"] = datetime.now()
        
        self.logger.info(f"Generated {len(insights)} workflow optimization insights")
        return insights
    
    @log_exceptions()
    async def _analyze_time_management(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze time management patterns and suggest optimizations."""
        
        insights = []
        
        # Analyze productive vs unproductive time patterns
        productive_patterns = [p for p in patterns if p.productivity_score > 0.7]
        unproductive_patterns = [p for p in patterns if p.productivity_score < 0.3]
        
        if len(unproductive_patterns) > len(productive_patterns) * 0.3:
            insights.append(WorkflowInsight(
                type=OptimizationType.TIME_MANAGEMENT,
                title="High Unproductive Time Detected",
                description="Analysis shows significant time spent on low-productivity activities",
                impact_score=0.8,
                confidence=0.9,
                suggested_actions=[
                    "Schedule focused work blocks",
                    "Use time-blocking techniques",
                    "Set productivity goals for each work session"
                ],
                supporting_data={
                    "unproductive_ratio": len(unproductive_patterns) / len(patterns),
                    "main_distractions": self._get_common_apps(unproductive_patterns)
                },
                estimated_time_saved=60
            ))
        
        # Analyze peak productivity hours
        if productive_patterns:
            peak_hours = self._find_peak_hours(productive_patterns)
            insights.append(WorkflowInsight(
                type=OptimizationType.TIME_MANAGEMENT,
                title="Optimize Peak Productivity Hours",
                description=f"Your peak productivity hours are {peak_hours}",
                impact_score=0.7,
                confidence=0.8,
                suggested_actions=[
                    f"Schedule important tasks during {peak_hours}",
                    "Block calendar during peak hours",
                    "Minimize meetings during peak productivity time"
                ],
                supporting_data={"peak_hours": peak_hours},
                estimated_time_saved=30
            ))
        
        return insights
    
    @log_exceptions()
    async def _analyze_task_ordering(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze task ordering and suggest optimizations."""
        
        insights = []
        
        # Analyze context switching frequency
        context_switches = self._count_context_switches(patterns)
        
        if context_switches > 20:  # Arbitrary threshold
            insights.append(WorkflowInsight(
                type=OptimizationType.TASK_ORDERING,
                title="Reduce Context Switching",
                description=f"High context switching detected ({context_switches} switches)",
                impact_score=0.6,
                confidence=0.8,
                suggested_actions=[
                    "Batch similar tasks together",
                    "Use time-blocking for focused work",
                    "Minimize tool switching within work blocks"
                ],
                supporting_data={"context_switches": context_switches},
                estimated_time_saved=25
            ))
        
        return insights
    
    @log_exceptions()
    async def _analyze_tool_usage(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze tool usage patterns and suggest optimizations."""
        
        insights = []
        
        # Count app usage
        app_usage = Counter()
        for pattern in patterns:
            if hasattr(pattern, 'app_name') and pattern.app_name:
                app_usage[pattern.app_name] += 1
        
        # Find underutilized productivity tools
        productivity_tools = ['VS Code', 'Terminal', 'Sublime Text', 'Vim']
        total_usage = sum(app_usage.values())
        
        for tool in productivity_tools:
            if tool in app_usage and app_usage[tool] / total_usage < 0.1:
                insights.append(WorkflowInsight(
                    type=OptimizationType.TOOL_USAGE,
                    title=f"Underutilized Tool: {tool}",
                    description=f"{tool} usage is low compared to other development tools",
                    impact_score=0.4,
                    confidence=0.6,
                    suggested_actions=[
                        f"Explore {tool} features and shortcuts",
                        "Consider workflow integration opportunities",
                        "Set up efficient {tool} configurations"
                    ],
                    supporting_data={"usage_ratio": app_usage[tool] / total_usage}
                ))
        
        return insights
    
    @log_exceptions()
    async def _analyze_context_switching(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze context switching patterns."""
        
        insights = []
        
        # Calculate average time between app switches
        switch_times = []
        for i in range(1, len(patterns)):
            if hasattr(patterns[i], 'timestamp') and hasattr(patterns[i-1], 'timestamp'):
                time_diff = (patterns[i].timestamp - patterns[i-1].timestamp).total_seconds()
                if time_diff < 600:  # Less than 10 minutes
                    switch_times.append(time_diff)
        
        if switch_times and sum(switch_times) / len(switch_times) < 120:  # Average < 2 minutes
            insights.append(WorkflowInsight(
                type=OptimizationType.CONTEXT_SWITCHING,
                title="Frequent Context Switching",
                description="Very frequent switching between applications detected",
                impact_score=0.7,
                confidence=0.8,
                suggested_actions=[
                    "Use focus mode or app blockers",
                    "Implement the Pomodoro technique",
                    "Create dedicated workspaces for different tasks"
                ],
                supporting_data={"avg_switch_time": sum(switch_times) / len(switch_times)},
                estimated_time_saved=40
            ))
        
        return insights
    
    @log_exceptions()
    async def _analyze_break_patterns(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze break patterns and suggest optimizations."""
        
        insights = []
        
        # Simple break analysis (would need more sophisticated logic in real implementation)
        work_patterns = [p for p in patterns if hasattr(p, 'productivity_score') and p.productivity_score > 0.5]
        
        if len(work_patterns) > 50:  # Long work session without breaks
            insights.append(WorkflowInsight(
                type=OptimizationType.BREAK_SCHEDULING,
                title="Schedule Regular Breaks",
                description="Long work sessions detected without adequate breaks",
                impact_score=0.5,
                confidence=0.7,
                suggested_actions=[
                    "Schedule 15-minute breaks every 2 hours",
                    "Use break reminder apps",
                    "Take walks or do light exercises during breaks"
                ],
                supporting_data={"continuous_work_patterns": len(work_patterns)},
                estimated_time_saved=0  # Breaks improve quality, not save time
            ))
        
        return insights
    
    @log_exceptions()
    async def _analyze_automation_opportunities(
        self, 
        patterns: List[UserPattern]
    ) -> List[WorkflowInsight]:
        """Analyze patterns for automation opportunities."""
        
        insights = []
        
        # Look for repetitive patterns
        repetitive_sequences = self._find_repetitive_sequences(patterns)
        
        if repetitive_sequences:
            insights.append(WorkflowInsight(
                type=OptimizationType.AUTOMATION,
                title="Automation Opportunities Found",
                description="Repetitive workflow patterns detected that could be automated",
                impact_score=0.9,
                confidence=0.7,
                suggested_actions=[
                    "Create scripts for repetitive tasks",
                    "Use workflow automation tools",
                    "Set up keyboard shortcuts for common actions"
                ],
                supporting_data={"repetitive_patterns": len(repetitive_sequences)},
                estimated_time_saved=90
            ))
        
        return insights
    
    def _get_common_apps(self, patterns: List[UserPattern]) -> List[str]:
        """Get most common apps from patterns."""
        app_counter = Counter()
        for pattern in patterns:
            if hasattr(pattern, 'app_name') and pattern.app_name:
                app_counter[pattern.app_name] += 1
        return [app for app, _ in app_counter.most_common(3)]
    
    def _find_peak_hours(self, patterns: List[UserPattern]) -> str:
        """Find peak productivity hours."""
        hour_scores = defaultdict(list)
        
        for pattern in patterns:
            if hasattr(pattern, 'timestamp') and hasattr(pattern, 'productivity_score'):
                hour = pattern.timestamp.hour
                hour_scores[hour].append(pattern.productivity_score)
        
        # Calculate average productivity by hour
        avg_scores = {}
        for hour, scores in hour_scores.items():
            avg_scores[hour] = sum(scores) / len(scores)
        
        # Find peak hour
        if avg_scores:
            peak_hour = max(avg_scores, key=avg_scores.get)
            return f"{peak_hour}:00-{peak_hour+1}:00"
        
        return "9:00-11:00"  # Default
    
    def _count_context_switches(self, patterns: List[UserPattern]) -> int:
        """Count context switches in patterns."""
        switches = 0
        current_app = None
        
        for pattern in patterns:
            if hasattr(pattern, 'app_name'):
                if current_app and current_app != pattern.app_name:
                    switches += 1
                current_app = pattern.app_name
        
        return switches
    
    def _find_repetitive_sequences(self, patterns: List[UserPattern]) -> List[List[UserPattern]]:
        """Find repetitive sequences in patterns."""
        # Simplified implementation - would need more sophisticated pattern matching
        sequences = []
        
        # Look for repeated app sequences of length 3
        for i in range(len(patterns) - 5):
            sequence1 = patterns[i:i+3]
            for j in range(i+3, len(patterns) - 2):
                sequence2 = patterns[j:j+3]
                
                # Check if sequences match (simplified comparison)
                if self._sequences_match(sequence1, sequence2):
                    sequences.append(sequence1)
                    break
        
        return sequences
    
    def _sequences_match(self, seq1: List[UserPattern], seq2: List[UserPattern]) -> bool:
        """Check if two sequences match."""
        if len(seq1) != len(seq2):
            return False
        
        for p1, p2 in zip(seq1, seq2):
            if (hasattr(p1, 'app_name') and hasattr(p2, 'app_name') and 
                p1.app_name != p2.app_name):
                return False
        
        return True
    
    @log_exceptions()
    async def get_daily_recommendations(self) -> List[WorkflowInsight]:
        """Get daily workflow recommendations."""
        
        # Analyze last 24 hours
        insights = await self.analyze_workflow_patterns(time_range_days=1)
        
        # Return top 3 most impactful insights
        return insights[:3]
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics and statistics."""
        return {
            **self.metrics,
            "optimization_history_count": len(self.optimization_history),
            "applied_optimizations_count": len(self.applied_optimizations)
        }