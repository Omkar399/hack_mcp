"""
Productivity Insights Engine for Eidolon AI Personal Assistant

Generates actionable productivity insights, recommendations, and optimization 
suggestions based on user activity patterns and analytics data.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import asyncio

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..storage.metadata_db import MetadataDatabase
from .analytics import AnalyticsEngine, ProductivityMetrics, Habit


@dataclass
class Insight:
    """Represents a single productivity insight."""
    id: str
    title: str
    description: str
    category: str  # 'productivity', 'time_management', 'focus', 'habits', 'tools'
    priority: str  # 'high', 'medium', 'low'
    confidence: float  # 0-1
    impact_score: float  # 0-100
    actionable: bool
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class ProductivityReport:
    """Comprehensive productivity report."""
    period: Dict[str, str]
    overall_score: float
    key_insights: List[Insight]
    metrics_summary: Dict[str, Any]
    trends: Dict[str, Any]
    recommendations: List[str]
    comparative_analysis: Optional[Dict[str, Any]] = None
    generated_at: datetime = None


class InsightGenerator:
    """Generates specific types of productivity insights."""
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics = analytics_engine
        self.logger = get_component_logger("insights.generator")
    
    def generate_focus_insights(
        self, 
        daily_metrics: List[ProductivityMetrics],
        habits: List[Habit]
    ) -> List[Insight]:
        """Generate insights about focus patterns and interruptions."""
        insights = []
        
        if not daily_metrics:
            return insights
        
        # Analyze focus sessions
        all_focus_sessions = []
        for day_metrics in daily_metrics:
            all_focus_sessions.extend(day_metrics.focus_sessions)
        
        if all_focus_sessions:
            # Average focus session length
            focus_durations = [session['duration'].total_seconds() / 60 for session in all_focus_sessions]
            avg_focus_time = statistics.mean(focus_durations)
            
            if avg_focus_time < 25:  # Less than Pomodoro time
                insights.append(Insight(
                    id="short_focus_sessions",
                    title="Short Focus Sessions Detected",
                    description=f"Your average focus session is {avg_focus_time:.1f} minutes, which may limit deep work productivity.",
                    category="focus",
                    priority="medium",
                    confidence=0.8,
                    impact_score=60,
                    actionable=True,
                    recommendations=[
                        "Try the Pomodoro Technique: 25-minute focused work blocks",
                        "Identify and eliminate common interruptions",
                        "Use focus apps or website blockers during work sessions",
                        "Set specific times for checking messages and emails"
                    ],
                    supporting_data={
                        "average_focus_minutes": avg_focus_time,
                        "total_sessions": len(all_focus_sessions),
                        "focus_duration_distribution": {
                            "under_15min": len([d for d in focus_durations if d < 15]),
                            "15_30min": len([d for d in focus_durations if 15 <= d < 30]),
                            "30_60min": len([d for d in focus_durations if 30 <= d < 60]),
                            "over_60min": len([d for d in focus_durations if d >= 60])
                        }
                    },
                    created_at=datetime.now()
                ))
            
            elif avg_focus_time > 90:  # Very long sessions
                insights.append(Insight(
                    id="long_focus_sessions",
                    title="Excellent Deep Focus Capability",
                    description=f"Your average focus session of {avg_focus_time:.1f} minutes shows excellent deep work capability.",
                    category="focus",
                    priority="low",
                    confidence=0.9,
                    impact_score=80,
                    actionable=True,
                    recommendations=[
                        "Maintain your excellent focus patterns",
                        "Consider taking short breaks every 90 minutes to prevent fatigue",
                        "Document what helps you maintain long focus sessions",
                        "Share your focus techniques with others"
                    ],
                    supporting_data={
                        "average_focus_minutes": avg_focus_time,
                        "total_sessions": len(all_focus_sessions)
                    },
                    created_at=datetime.now()
                ))
        
        # Analyze context switches
        context_switches = [day.context_switches for day in daily_metrics if day.context_switches > 0]
        if context_switches:
            avg_switches = statistics.mean(context_switches)
            
            if avg_switches > 50:  # High context switching
                insights.append(Insight(
                    id="high_context_switching",
                    title="High Application Context Switching",
                    description=f"You average {avg_switches:.1f} application switches per day, which may fragment your attention.",
                    category="focus",
                    priority="high",
                    confidence=0.85,
                    impact_score=75,
                    actionable=True,
                    recommendations=[
                        "Batch similar tasks together",
                        "Use dedicated time blocks for specific applications",
                        "Close unnecessary applications and browser tabs",
                        "Set specific times for checking communication apps"
                    ],
                    supporting_data={
                        "average_daily_switches": avg_switches,
                        "max_switches_day": max(context_switches),
                        "min_switches_day": min(context_switches)
                    },
                    created_at=datetime.now()
                ))
        
        return insights
    
    def generate_time_insights(
        self, 
        daily_metrics: List[ProductivityMetrics]
    ) -> List[Insight]:
        """Generate insights about time usage patterns."""
        insights = []
        
        if not daily_metrics:
            return insights
        
        # Analyze productive vs non-productive time
        productive_ratios = []
        for day_metrics in daily_metrics:
            if day_metrics.total_active_time.total_seconds() > 0:
                ratio = day_metrics.productive_time.total_seconds() / day_metrics.total_active_time.total_seconds()
                productive_ratios.append(ratio)
        
        if productive_ratios:
            avg_productive_ratio = statistics.mean(productive_ratios)
            
            if avg_productive_ratio < 0.6:  # Less than 60% productive
                insights.append(Insight(
                    id="low_productivity_ratio",
                    title="Opportunity to Increase Productive Time",
                    description=f"Only {avg_productive_ratio:.1%} of your active time is spent on productive activities.",
                    category="productivity",
                    priority="high",
                    confidence=0.8,
                    impact_score=85,
                    actionable=True,
                    recommendations=[
                        "Identify time-wasting activities and reduce them",
                        "Set clear daily goals and priorities",
                        "Use time-blocking techniques",
                        "Review and optimize your daily routine"
                    ],
                    supporting_data={
                        "productive_ratio": avg_productive_ratio,
                        "total_days_analyzed": len(daily_metrics)
                    },
                    created_at=datetime.now()
                ))
            
            elif avg_productive_ratio > 0.8:  # Very high productivity
                insights.append(Insight(
                    id="high_productivity_ratio",
                    title="Excellent Productivity Ratio",
                    description=f"You maintain {avg_productive_ratio:.1%} productive time - excellent work habits!",
                    category="productivity",
                    priority="low",
                    confidence=0.9,
                    impact_score=90,
                    actionable=True,
                    recommendations=[
                        "Document your productivity strategies",
                        "Ensure you're taking adequate breaks",
                        "Consider sharing your methods with others",
                        "Monitor for signs of burnout"
                    ],
                    supporting_data={
                        "productive_ratio": avg_productive_ratio,
                        "consistency": statistics.stdev(productive_ratios) if len(productive_ratios) > 1 else 0
                    },
                    created_at=datetime.now()
                ))
        
        # Analyze break patterns
        all_breaks = []
        for day_metrics in daily_metrics:
            all_breaks.extend(day_metrics.break_patterns)
        
        if all_breaks:
            break_durations = [b['duration'] for b in all_breaks]
            avg_break_duration = statistics.mean(break_durations)
            
            short_breaks = len([b for b in break_durations if b < 900])  # Less than 15 minutes
            long_breaks = len([b for b in break_durations if b > 3600])  # More than 1 hour
            
            if short_breaks / len(break_durations) > 0.8:
                insights.append(Insight(
                    id="good_break_pattern",
                    title="Healthy Break Patterns",
                    description="You take regular short breaks, which is excellent for maintaining focus and energy.",
                    category="time_management",
                    priority="low",
                    confidence=0.7,
                    impact_score=70,
                    actionable=True,
                    recommendations=[
                        "Continue your excellent break discipline",
                        "Consider using breaks for light physical activity",
                        "Step away from screens during breaks",
                        "Use breaks for mindfulness or relaxation"
                    ],
                    supporting_data={
                        "average_break_minutes": avg_break_duration / 60,
                        "total_breaks": len(all_breaks),
                        "short_breaks_percentage": short_breaks / len(break_durations)
                    },
                    created_at=datetime.now()
                ))
        
        return insights
    
    def generate_habit_insights(self, habits: List[Habit]) -> List[Insight]:
        """Generate insights about user habits."""
        insights = []
        
        if not habits:
            return insights
        
        # Analyze positive vs negative habits
        positive_habits = [h for h in habits if h.habit_type == 'positive']
        negative_habits = [h for h in habits if h.habit_type == 'negative']
        
        if positive_habits:
            strong_positive = [h for h in positive_habits if h.strength > 0.7]
            if strong_positive:
                insights.append(Insight(
                    id="strong_positive_habits",
                    title="Strong Positive Habits Identified",
                    description=f"You have {len(strong_positive)} well-established positive habits that support your productivity.",
                    category="habits",
                    priority="low",
                    confidence=0.8,
                    impact_score=75,
                    actionable=True,
                    recommendations=[
                        "Continue reinforcing these positive patterns",
                        "Document what triggers these habits",
                        "Consider expanding these habits to other areas",
                        "Use these as examples when building new habits"
                    ],
                    supporting_data={
                        "positive_habits_count": len(positive_habits),
                        "strong_habits": [h.name for h in strong_positive]
                    },
                    created_at=datetime.now()
                ))
        
        if negative_habits:
            actionable_negative = [h for h in negative_habits if h.strength < 0.5]
            if actionable_negative:
                insights.append(Insight(
                    id="addressable_negative_habits",
                    title="Negative Habits Can Be Improved",
                    description=f"You have {len(actionable_negative)} negative habits that are not yet strongly established.",
                    category="habits",
                    priority="medium",
                    confidence=0.7,
                    impact_score=65,
                    actionable=True,
                    recommendations=[
                        "Focus on the habit with the lowest strength first",
                        "Identify triggers that lead to these behaviors",
                        "Replace negative habits with positive alternatives",
                        "Use habit-tracking tools for awareness"
                    ],
                    supporting_data={
                        "negative_habits_count": len(negative_habits),
                        "actionable_habits": [h.name for h in actionable_negative]
                    },
                    created_at=datetime.now()
                ))
        
        # Analyze habit triggers
        all_triggers = []
        for habit in habits:
            all_triggers.extend(habit.triggers)
        
        trigger_counts = Counter(all_triggers)
        common_triggers = trigger_counts.most_common(3)
        
        if common_triggers:
            insights.append(Insight(
                id="common_habit_triggers",
                title="Common Habit Triggers Identified",
                description=f"Your most common habit triggers are: {', '.join([t[0] for t in common_triggers])}",
                category="habits",
                priority="medium",
                confidence=0.75,
                impact_score=60,
                actionable=True,
                recommendations=[
                    "Use these triggers to build new positive habits",
                    "Be mindful of triggers that lead to negative habits",
                    "Design your environment to support positive triggers",
                    "Create habit stacks around these common triggers"
                ],
                supporting_data={
                    "trigger_frequency": dict(common_triggers),
                    "total_triggers_analyzed": len(all_triggers)
                },
                created_at=datetime.now()
            ))
        
        return insights
    
    def generate_tool_insights(
        self, 
        daily_metrics: List[ProductivityMetrics]
    ) -> List[Insight]:
        """Generate insights about tool and application usage."""
        insights = []
        
        if not daily_metrics:
            return insights
        
        # Analyze application usage patterns
        app_usage = defaultdict(int)
        for day_metrics in daily_metrics:
            for app in day_metrics.applications_used:
                app_usage[app] += 1
        
        if app_usage:
            most_used_apps = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Check for productivity tool usage
            productivity_apps = ['vscode', 'pycharm', 'sublime', 'atom', 'vim', 'emacs', 'terminal', 'iterm']
            distraction_apps = ['youtube', 'netflix', 'facebook', 'twitter', 'instagram', 'tiktok']
            
            productive_app_usage = sum(count for app, count in most_used_apps 
                                     if any(prod_app in app.lower() for prod_app in productivity_apps))
            
            distraction_app_usage = sum(count for app, count in most_used_apps 
                                      if any(dist_app in app.lower() for dist_app in distraction_apps))
            
            if productive_app_usage > distraction_app_usage * 2:
                insights.append(Insight(
                    id="good_tool_choices",
                    title="Excellent Tool Selection",
                    description="Your application usage shows a strong preference for productivity tools.",
                    category="tools",
                    priority="low",
                    confidence=0.8,
                    impact_score=70,
                    actionable=True,
                    recommendations=[
                        "Continue using these productive tools",
                        "Explore advanced features of your most-used tools",
                        "Consider automation opportunities",
                        "Share tool recommendations with colleagues"
                    ],
                    supporting_data={
                        "most_used_apps": most_used_apps,
                        "productivity_ratio": productive_app_usage / (productive_app_usage + distraction_app_usage)
                    },
                    created_at=datetime.now()
                ))
            
            elif distraction_app_usage > productive_app_usage:
                insights.append(Insight(
                    id="tool_optimization_needed",
                    title="Tool Usage Could Be Optimized",
                    description="Consider reducing time spent on distracting applications.",
                    category="tools",
                    priority="medium",
                    confidence=0.7,
                    impact_score=65,
                    actionable=True,
                    recommendations=[
                        "Set specific times for entertainment apps",
                        "Use app timers or website blockers",
                        "Replace distracting apps with productive alternatives",
                        "Create app-free zones or times"
                    ],
                    supporting_data={
                        "most_used_apps": most_used_apps,
                        "distraction_usage": distraction_app_usage
                    },
                    created_at=datetime.now()
                ))
        
        return insights


class ProductivityInsightsEngine:
    """Main engine for generating comprehensive productivity insights."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("insights")
        
        # Initialize components
        self.analytics_engine = AnalyticsEngine()
        self.insight_generator = InsightGenerator(self.analytics_engine)
        
        # Insight cache
        self.insights_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        self.logger.info("Productivity insights engine initialized")
    
    @log_performance
    async def generate_insights(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> List[Insight]:
        """
        Generate comprehensive productivity insights for a time period.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            categories: Filter insights by categories
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of insights ordered by priority and impact
        """
        self.logger.info(f"Generating insights for period {start_date} to {end_date}")
        
        # Check cache
        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
        if cache_key in self.insights_cache:
            cached_insights, cache_time = self.insights_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return self._filter_insights(cached_insights, categories, min_confidence)
        
        # Get analytics data
        daily_metrics = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            date_dt = datetime.combine(current_date, datetime.min.time())
            try:
                metrics = await asyncio.to_thread(
                    self.analytics_engine.analyze_productivity_patterns,
                    date_dt
                )
                daily_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Failed to get metrics for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Get habits
        try:
            habits = await asyncio.to_thread(
                self.analytics_engine.identify_habits,
                start_date,
                end_date
            )
        except Exception as e:
            self.logger.warning(f"Failed to get habits: {e}")
            habits = []
        
        # Generate insights
        all_insights = []
        
        try:
            focus_insights = self.insight_generator.generate_focus_insights(daily_metrics, habits)
            all_insights.extend(focus_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate focus insights: {e}")
        
        try:
            time_insights = self.insight_generator.generate_time_insights(daily_metrics)
            all_insights.extend(time_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate time insights: {e}")
        
        try:
            habit_insights = self.insight_generator.generate_habit_insights(habits)
            all_insights.extend(habit_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate habit insights: {e}")
        
        try:
            tool_insights = self.insight_generator.generate_tool_insights(daily_metrics)
            all_insights.extend(tool_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate tool insights: {e}")
        
        # Sort by priority and impact
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        all_insights.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.impact_score),
            reverse=True
        )
        
        # Cache results
        self.insights_cache[cache_key] = (all_insights, datetime.now())
        
        self.logger.info(f"Generated {len(all_insights)} insights")
        return self._filter_insights(all_insights, categories, min_confidence)
    
    @log_performance
    async def generate_productivity_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_comparison: bool = True
    ) -> ProductivityReport:
        """
        Generate a comprehensive productivity report.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            include_comparison: Whether to include comparison with previous period
            
        Returns:
            Complete productivity report
        """
        self.logger.info(f"Generating productivity report for {start_date} to {end_date}")
        
        # Get insights
        insights = await self.generate_insights(start_date, end_date)
        key_insights = [i for i in insights if i.priority in ['high', 'medium']][:10]
        
        # Get analytics summary
        try:
            analytics_summary = await asyncio.to_thread(
                self.analytics_engine.get_analytics_summary,
                start_date,
                end_date
            )
        except Exception as e:
            self.logger.error(f"Failed to get analytics summary: {e}")
            analytics_summary = {}
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(analytics_summary, insights)
        
        # Generate trends
        trends = await self._analyze_trends(start_date, end_date)
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations(insights, analytics_summary)
        
        # Comparison with previous period
        comparative_analysis = None
        if include_comparison:
            try:
                comparative_analysis = await self._generate_comparison(start_date, end_date)
            except Exception as e:
                self.logger.warning(f"Failed to generate comparison: {e}")
        
        return ProductivityReport(
            period={
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days + 1
            },
            overall_score=overall_score,
            key_insights=key_insights,
            metrics_summary=analytics_summary.get('summary', {}),
            trends=trends,
            recommendations=recommendations,
            comparative_analysis=comparative_analysis,
            generated_at=datetime.now()
        )
    
    def _filter_insights(
        self,
        insights: List[Insight],
        categories: Optional[List[str]],
        min_confidence: float
    ) -> List[Insight]:
        """Filter insights by categories and confidence."""
        filtered = insights
        
        if categories:
            filtered = [i for i in filtered if i.category in categories]
        
        if min_confidence > 0:
            filtered = [i for i in filtered if i.confidence >= min_confidence]
        
        return filtered
    
    def _calculate_overall_score(
        self,
        analytics_summary: Dict[str, Any],
        insights: List[Insight]
    ) -> float:
        """Calculate overall productivity score."""
        base_score = analytics_summary.get('summary', {}).get('average_productivity_score', 50)
        
        # Adjust based on insights
        high_priority_negative = len([i for i in insights if i.priority == 'high' and i.impact_score < 50])
        high_priority_positive = len([i for i in insights if i.priority == 'high' and i.impact_score >= 70])
        
        adjustment = (high_priority_positive * 5) - (high_priority_negative * 10)
        
        return min(100, max(0, base_score + adjustment))
    
    async def _analyze_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze productivity trends over the period."""
        trends = {
            'productivity_trend': 'stable',
            'focus_trend': 'stable',
            'application_diversity': 'stable'
        }
        
        try:
            # Simple trend analysis - could be enhanced with more sophisticated algorithms
            period_days = (end_date - start_date).days + 1
            
            if period_days >= 7:
                # Analyze first half vs second half
                mid_date = start_date + timedelta(days=period_days // 2)
                
                # Get metrics for both halves
                first_half_summary = await asyncio.to_thread(
                    self.analytics_engine.get_analytics_summary,
                    start_date,
                    mid_date
                )
                
                second_half_summary = await asyncio.to_thread(
                    self.analytics_engine.get_analytics_summary,
                    mid_date,
                    end_date
                )
                
                # Compare productivity scores
                first_score = first_half_summary.get('summary', {}).get('average_productivity_score', 50)
                second_score = second_half_summary.get('summary', {}).get('average_productivity_score', 50)
                
                if second_score > first_score + 5:
                    trends['productivity_trend'] = 'improving'
                elif second_score < first_score - 5:
                    trends['productivity_trend'] = 'declining'
                
                trends['score_change'] = second_score - first_score
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze trends: {e}")
        
        return trends
    
    def _generate_overall_recommendations(
        self,
        insights: List[Insight],
        analytics_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate overall recommendations based on insights."""
        recommendations = []
        
        # Get recommendations from high-priority insights
        high_priority_insights = [i for i in insights if i.priority == 'high']
        for insight in high_priority_insights[:3]:
            if insight.recommendations:
                recommendations.extend(insight.recommendations[:2])
        
        # Add general recommendations based on overall patterns
        avg_productivity = analytics_summary.get('summary', {}).get('average_productivity_score', 50)
        
        if avg_productivity < 60:
            recommendations.extend([
                "Set clear daily goals and priorities",
                "Use time-blocking techniques to structure your day",
                "Minimize distractions during work periods"
            ])
        elif avg_productivity > 80:
            recommendations.extend([
                "You're doing great! Focus on maintaining consistency",
                "Consider sharing your productivity strategies",
                "Monitor for signs of overwork or burnout"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:8]  # Limit to 8 recommendations
    
    async def _generate_comparison(
        self,
        current_start: datetime,
        current_end: datetime
    ) -> Dict[str, Any]:
        """Generate comparison with previous period."""
        period_length = current_end - current_start
        previous_start = current_start - period_length
        previous_end = current_start
        
        try:
            current_summary = await asyncio.to_thread(
                self.analytics_engine.get_analytics_summary,
                current_start,
                current_end
            )
            
            previous_summary = await asyncio.to_thread(
                self.analytics_engine.get_analytics_summary,
                previous_start,
                previous_end
            )
            
            current_score = current_summary.get('summary', {}).get('average_productivity_score', 50)
            previous_score = previous_summary.get('summary', {}).get('average_productivity_score', 50)
            
            return {
                'previous_period': {
                    'start': previous_start.isoformat(),
                    'end': previous_end.isoformat()
                },
                'productivity_change': current_score - previous_score,
                'productivity_change_percent': ((current_score - previous_score) / max(1, previous_score)) * 100,
                'current_score': current_score,
                'previous_score': previous_score,
                'interpretation': self._interpret_change(current_score - previous_score)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison: {e}")
            return {}
    
    def _interpret_change(self, change: float) -> str:
        """Interpret productivity score change."""
        if change > 10:
            return "Significant improvement"
        elif change > 5:
            return "Moderate improvement"
        elif change > -5:
            return "Stable performance"
        elif change > -10:
            return "Slight decline"
        else:
            return "Concerning decline"
    
    def clear_cache(self):
        """Clear the insights cache."""
        self.insights_cache.clear()
        self.logger.info("Insights cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(
            1 for _, cache_time in self.insights_cache.values()
            if now - cache_time < self.cache_ttl
        )
        
        return {
            'total_entries': len(self.insights_cache),
            'valid_entries': valid_entries,
            'cache_hit_rate': valid_entries / max(1, len(self.insights_cache))
        }