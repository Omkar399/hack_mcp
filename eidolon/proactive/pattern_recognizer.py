"""
Pattern Recognition system for Eidolon AI Personal Assistant

Identifies user behavior patterns, habits, and preferences through analysis
of screen activity, application usage, and temporal patterns.
"""

import asyncio
import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics
import re

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.memory import MemorySystem
from ..storage.metadata_db import MetadataDatabase
from ..models.cloud_api import CloudAPIManager


class PatternType(Enum):
    """Types of patterns that can be recognized"""
    TEMPORAL = "temporal"  # Time-based patterns
    WORKFLOW = "workflow"  # Work sequence patterns
    APPLICATION = "application"  # App usage patterns
    COMMUNICATION = "communication"  # Communication habits
    CONTENT = "content"  # Content interaction patterns
    PRODUCTIVITY = "productivity"  # Productivity patterns
    BREAK = "break"  # Break and rest patterns
    ERROR = "error"  # Error and problem patterns


class PatternStrength(Enum):
    """Strength/confidence of pattern recognition"""
    WEAK = "weak"  # 0.3-0.5
    MODERATE = "moderate"  # 0.5-0.7
    STRONG = "strong"  # 0.7-0.9
    VERY_STRONG = "very_strong"  # 0.9+


@dataclass
class UserPattern:
    """Represents a recognized user pattern"""
    id: str
    pattern_type: PatternType
    title: str
    description: str
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    # Pattern specifics
    triggers: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    typical_duration: Optional[timedelta] = None
    frequency: str = "unknown"  # daily, weekly, occasional, rare
    
    # Evidence and data
    occurrences: List[datetime] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    counter_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    context_factors: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    
    # Metadata
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Predictions and recommendations
    predictions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_strength_level(self) -> PatternStrength:
        """Get categorical strength level"""
        if self.strength >= 0.9:
            return PatternStrength.VERY_STRONG
        elif self.strength >= 0.7:
            return PatternStrength.STRONG
        elif self.strength >= 0.5:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            'id': self.id,
            'pattern_type': self.pattern_type.value,
            'title': self.title,
            'description': self.description,
            'strength': self.strength,
            'confidence': self.confidence,
            'strength_level': self.get_strength_level().value,
            'triggers': self.triggers,
            'conditions': self.conditions,
            'typical_duration': self.typical_duration.total_seconds() if self.typical_duration else None,
            'frequency': self.frequency,
            'occurrence_count': len(self.occurrences),
            'context_factors': self.context_factors,
            'related_patterns': self.related_patterns,
            'first_observed': self.first_observed.isoformat() if self.first_observed else None,
            'last_observed': self.last_observed.isoformat() if self.last_observed else None,
            'predictions': self.predictions,
            'recommendations': self.recommendations
        }


class PatternRecognizer:
    """Advanced pattern recognition system for user behavior analysis"""
    
    def __init__(self):
        self.logger = get_component_logger("pattern_recognizer")
        self.config = get_config()
        self.memory = MemorySystem()
        self.database = MetadataDatabase()
        self.cloud_api = CloudAPIManager()
        
        # Pattern storage
        self.recognized_patterns: Dict[str, UserPattern] = {}
        self.pattern_templates: Dict[str, Dict[str, Any]] = {}
        
        # Analysis state
        self.analysis_window = timedelta(days=7)  # Default analysis window
        self.min_occurrences = 3  # Minimum occurrences to recognize pattern
        self.confidence_threshold = 0.6  # Minimum confidence for pattern
        
        # Load pattern templates and existing patterns
        self._load_pattern_templates()
        asyncio.create_task(self._load_existing_patterns())
    
    @log_exceptions
    async def analyze_user_patterns(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern_types: Optional[List[PatternType]] = None
    ) -> List[UserPattern]:
        """Comprehensive analysis of user patterns across all data"""
        
        self.logger.info("Starting comprehensive pattern analysis")
        
        # Set analysis window
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - self.analysis_window
        
        # Get data for analysis
        activity_data = await self._get_activity_data(start_date, end_date)
        
        if not activity_data:
            self.logger.warning("No activity data available for pattern analysis")
            return []
        
        # Analyze different pattern types
        detected_patterns = []
        
        pattern_types_to_analyze = pattern_types or list(PatternType)
        
        for pattern_type in pattern_types_to_analyze:
            patterns = await self._analyze_pattern_type(pattern_type, activity_data)
            detected_patterns.extend(patterns)
        
        # Cross-reference and correlate patterns
        correlated_patterns = await self._correlate_patterns(detected_patterns)
        
        # Validate and filter patterns
        validated_patterns = await self._validate_patterns(correlated_patterns)
        
        # Update pattern database
        await self._update_pattern_database(validated_patterns)
        
        self.logger.info(f"Detected {len(validated_patterns)} patterns")
        return validated_patterns
    
    async def _get_activity_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get activity data from the specified time range"""
        
        # Get screenshots and metadata
        screenshots = await self.database.get_screenshots_by_date_range(start_date, end_date)
        
        activity_data = []
        for screenshot in screenshots:
            # Get OCR text and analysis
            ocr_text = await self.database.get_ocr_text(screenshot['id'])
            analysis = await self.database.get_analysis(screenshot['id'])
            
            activity_data.append({
                'id': screenshot['id'],
                'timestamp': datetime.fromisoformat(screenshot['timestamp']),
                'file_path': screenshot['file_path'],
                'hash': screenshot['hash'],
                'window_title': screenshot.get('window_title'),
                'app_name': screenshot.get('app_name'),
                'ocr_text': ocr_text.get('text', '') if ocr_text else '',
                'analysis': analysis,
                'metadata': screenshot
            })
        
        return activity_data
    
    async def _analyze_pattern_type(
        self,
        pattern_type: PatternType,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze specific type of pattern"""
        
        self.logger.debug(f"Analyzing {pattern_type.value} patterns")
        
        if pattern_type == PatternType.TEMPORAL:
            return await self._analyze_temporal_patterns(activity_data)
        elif pattern_type == PatternType.WORKFLOW:
            return await self._analyze_workflow_patterns(activity_data)
        elif pattern_type == PatternType.APPLICATION:
            return await self._analyze_application_patterns(activity_data)
        elif pattern_type == PatternType.COMMUNICATION:
            return await self._analyze_communication_patterns(activity_data)
        elif pattern_type == PatternType.CONTENT:
            return await self._analyze_content_patterns(activity_data)
        elif pattern_type == PatternType.PRODUCTIVITY:
            return await self._analyze_productivity_patterns(activity_data)
        elif pattern_type == PatternType.BREAK:
            return await self._analyze_break_patterns(activity_data)
        elif pattern_type == PatternType.ERROR:
            return await self._analyze_error_patterns(activity_data)
        
        return []
    
    async def _analyze_temporal_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze time-based activity patterns"""
        
        patterns = []
        
        # Group activities by hour of day
        hourly_activity = defaultdict(list)
        for activity in activity_data:
            hour = activity['timestamp'].hour
            hourly_activity[hour].append(activity)
        
        # Find peak activity hours
        peak_hours = []
        avg_activity = len(activity_data) / 24 if activity_data else 0
        
        for hour, activities in hourly_activity.items():
            if len(activities) > avg_activity * 1.5:  # 50% above average
                peak_hours.append(hour)
        
        if peak_hours:
            # Create temporal pattern for peak hours
            peak_pattern = UserPattern(
                id=f"temporal_peak_{datetime.now().strftime('%Y%m%d')}",
                pattern_type=PatternType.TEMPORAL,
                title="Peak Activity Hours",
                description=f"High activity typically occurs during hours: {', '.join(map(str, peak_hours))}",
                strength=0.7,
                confidence=0.8,
                triggers=[f"hour_{hour}" for hour in peak_hours],
                conditions={"peak_hours": peak_hours},
                frequency="daily",
                context_factors=["work_schedule", "productivity_rhythm"]
            )
            
            # Add occurrences
            for hour in peak_hours:
                for activity in hourly_activity[hour]:
                    peak_pattern.occurrences.append(activity['timestamp'])
            
            patterns.append(peak_pattern)
        
        # Analyze day-of-week patterns
        weekday_activity = defaultdict(list)
        for activity in activity_data:
            weekday = activity['timestamp'].weekday()
            weekday_activity[weekday].append(activity)
        
        # Check for weekend vs weekday patterns
        weekday_count = sum(len(activities) for day, activities in weekday_activity.items() if day < 5)
        weekend_count = sum(len(activities) for day, activities in weekday_activity.items() if day >= 5)
        
        if weekday_count > 0 and weekend_count > 0:
            weekday_ratio = weekday_count / (weekday_count + weekend_count)
            
            if weekday_ratio > 0.7:  # Primarily weekday activity
                pattern = UserPattern(
                    id=f"temporal_weekday_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.TEMPORAL,
                    title="Weekday Work Pattern",
                    description="Primary activity occurs during weekdays",
                    strength=weekday_ratio,
                    confidence=0.75,
                    frequency="weekly",
                    context_factors=["work_schedule", "professional_activity"]
                )
                patterns.append(pattern)
            elif weekday_ratio < 0.3:  # Primarily weekend activity
                pattern = UserPattern(
                    id=f"temporal_weekend_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.TEMPORAL,
                    title="Weekend Activity Pattern",
                    description="Primary activity occurs during weekends",
                    strength=1 - weekday_ratio,
                    confidence=0.75,
                    frequency="weekly",
                    context_factors=["personal_time", "leisure_activity"]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_workflow_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze workflow and task sequence patterns"""
        
        patterns = []
        
        # Group activities by application transitions
        app_sequences = []
        current_sequence = []
        last_app = None
        
        for activity in sorted(activity_data, key=lambda x: x['timestamp']):
            app = activity.get('app_name', 'unknown')
            
            if app != last_app:
                if len(current_sequence) >= 2:
                    app_sequences.append(current_sequence.copy())
                current_sequence = [app]
                last_app = app
            else:
                if not current_sequence or current_sequence[-1] != app:
                    current_sequence.append(app)
        
        # Find common sequences
        sequence_counts = Counter()
        for sequence in app_sequences:
            if len(sequence) >= 2:
                for i in range(len(sequence) - 1):
                    transition = f"{sequence[i]} -> {sequence[i+1]}"
                    sequence_counts[transition] += 1
        
        # Create patterns for common transitions
        for transition, count in sequence_counts.most_common(5):
            if count >= self.min_occurrences:
                apps = transition.split(' -> ')
                pattern = UserPattern(
                    id=f"workflow_{hash(transition)}",
                    pattern_type=PatternType.WORKFLOW,
                    title=f"Common Transition: {transition}",
                    description=f"Frequently switches from {apps[0]} to {apps[1]}",
                    strength=min(1.0, count / len(app_sequences)),
                    confidence=0.7,
                    triggers=[apps[0]],
                    conditions={"source_app": apps[0], "target_app": apps[1]},
                    frequency="frequent" if count > 10 else "occasional"
                )
                patterns.append(pattern)
        
        # Analyze task duration patterns
        await self._analyze_task_duration_patterns(activity_data, patterns)
        
        return patterns
    
    async def _analyze_task_duration_patterns(
        self,
        activity_data: List[Dict[str, Any]],
        patterns: List[UserPattern]
    ):
        """Analyze how long user typically spends on different tasks"""
        
        # Group by application and calculate session durations
        app_sessions = defaultdict(list)
        current_app = None
        session_start = None
        
        for activity in sorted(activity_data, key=lambda x: x['timestamp']):
            app = activity.get('app_name', 'unknown')
            timestamp = activity['timestamp']
            
            if app != current_app:
                # End previous session
                if current_app and session_start:
                    duration = timestamp - session_start
                    if duration.total_seconds() > 60:  # Only sessions > 1 minute
                        app_sessions[current_app].append(duration)
                
                # Start new session
                current_app = app
                session_start = timestamp
        
        # Analyze duration patterns for each app
        for app, durations in app_sessions.items():
            if len(durations) >= 3:  # Need at least 3 sessions
                avg_duration = statistics.mean(d.total_seconds() for d in durations)
                std_duration = statistics.stdev(d.total_seconds() for d in durations) if len(durations) > 1 else 0
                
                # Check for consistent duration pattern
                consistency = 1 - (std_duration / avg_duration) if avg_duration > 0 else 0
                
                if consistency > 0.3:  # Reasonably consistent
                    pattern = UserPattern(
                        id=f"duration_{app}_{hash(app)}",
                        pattern_type=PatternType.WORKFLOW,
                        title=f"Typical {app} Session Duration",
                        description=f"Usually spends {avg_duration/60:.1f} minutes in {app}",
                        strength=consistency,
                        confidence=0.6,
                        typical_duration=timedelta(seconds=avg_duration),
                        conditions={"app": app, "avg_duration": avg_duration},
                        frequency="regular"
                    )
                    patterns.append(pattern)
    
    async def _analyze_application_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze application usage patterns"""
        
        patterns = []
        
        # Count application usage
        app_counts = Counter()
        app_times = defaultdict(list)
        
        for activity in activity_data:
            app = activity.get('app_name', 'unknown')
            if app != 'unknown':
                app_counts[app] += 1
                app_times[app].append(activity['timestamp'])
        
        total_activities = len(activity_data)
        
        # Find dominant applications
        for app, count in app_counts.most_common(10):
            usage_ratio = count / total_activities
            
            if usage_ratio > 0.1:  # App used in >10% of captures
                # Analyze temporal distribution
                times = app_times[app]
                hours = [t.hour for t in times]
                
                if len(set(hours)) < 8:  # Used in specific time windows
                    pattern = UserPattern(
                        id=f"app_{app}_{hash(app)}",
                        pattern_type=PatternType.APPLICATION,
                        title=f"Heavy {app} Usage",
                        description=f"{app} is used frequently ({usage_ratio:.1%} of time)",
                        strength=usage_ratio,
                        confidence=0.8,
                        triggers=[app],
                        conditions={"app": app, "usage_ratio": usage_ratio},
                        frequency="daily" if usage_ratio > 0.3 else "frequent",
                        context_factors=["primary_tool", "workflow_dependency"]
                    )
                    
                    pattern.occurrences = times
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_communication_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze communication habits and patterns"""
        
        patterns = []
        
        # Define communication apps and keywords
        comm_apps = ['mail', 'slack', 'teams', 'zoom', 'skype', 'discord', 'telegram', 'whatsapp']
        comm_keywords = ['email', 'message', 'chat', 'call', 'meeting', 'conference']
        
        # Find communication activities
        comm_activities = []
        for activity in activity_data:
            app = activity.get('app_name', '').lower()
            text = activity.get('ocr_text', '').lower()
            
            is_comm = any(comm_app in app for comm_app in comm_apps) or \
                     any(keyword in text for keyword in comm_keywords)
            
            if is_comm:
                comm_activities.append(activity)
        
        if len(comm_activities) >= self.min_occurrences:
            # Analyze communication timing
            comm_hours = [activity['timestamp'].hour for activity in comm_activities]
            peak_comm_hours = [hour for hour, count in Counter(comm_hours).items() 
                             if count > len(comm_activities) / 12]  # Above average
            
            if peak_comm_hours:
                pattern = UserPattern(
                    id=f"communication_timing_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.COMMUNICATION,
                    title="Communication Peak Hours",
                    description=f"Most communication occurs during: {', '.join(map(str, peak_comm_hours))}",
                    strength=0.7,
                    confidence=0.75,
                    triggers=["communication_app", "message_notification"],
                    conditions={"peak_hours": peak_comm_hours},
                    frequency="daily",
                    context_factors=["work_schedule", "collaboration_patterns"]
                )
                
                pattern.occurrences = [a['timestamp'] for a in comm_activities]
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_content_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze content interaction patterns"""
        
        patterns = []
        
        # Analyze text content for topics and themes
        text_content = []
        for activity in activity_data:
            text = activity.get('ocr_text', '')
            if len(text) > 50:  # Substantial text content
                text_content.append({
                    'text': text,
                    'timestamp': activity['timestamp'],
                    'app': activity.get('app_name', 'unknown')
                })
        
        if len(text_content) >= 10:  # Need substantial content for analysis
            # Use AI to analyze content themes
            content_analysis = await self._analyze_content_themes(text_content)
            
            if content_analysis.get('themes'):
                for theme in content_analysis['themes']:
                    if theme.get('confidence', 0) > 0.6:
                        pattern = UserPattern(
                            id=f"content_{theme['name']}_{hash(theme['name'])}",
                            pattern_type=PatternType.CONTENT,
                            title=f"Content Focus: {theme['name']}",
                            description=theme.get('description', f"Frequently engages with {theme['name']} content"),
                            strength=theme.get('confidence', 0.6),
                            confidence=theme.get('confidence', 0.6),
                            triggers=theme.get('keywords', []),
                            conditions={"theme": theme['name']},
                            frequency="regular",
                            context_factors=["interests", "work_domain"]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _analyze_content_themes(self, text_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to analyze content themes"""
        
        try:
            # Prepare content for analysis
            content_sample = "\n\n".join([
                f"[{content['timestamp'].strftime('%H:%M')}] {content['text'][:200]}"
                for content in text_content[:20]  # Sample of content
            ])
            
            prompt = f"""
Analyze the following user screen content and identify the main themes and topics of interest:

{content_sample}

Identify:
1. Main themes/topics (programming, business, design, research, etc.)
2. Technical domains (web development, data science, marketing, etc.)  
3. Recurring keywords and concepts
4. Content patterns and focus areas

Respond with JSON:
{{
  "themes": [
    {{
      "name": "theme_name",
      "description": "brief description",
      "confidence": 0.8,
      "keywords": ["keyword1", "keyword2"],
      "category": "work|personal|learning|entertainment"
    }}
  ]
}}
"""
            
            response = await self.cloud_api.analyze_complex_request(
                prompt,
                context="content_analysis"
            )
            
            # Parse response
            if isinstance(response.get('content'), str):
                import json
                try:
                    return json.loads(response['content'])
                except:
                    pass
            
            return response.get('content', {})
            
        except Exception as e:
            self.logger.error(f"Content theme analysis failed: {e}")
            return {}
    
    async def _analyze_productivity_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze productivity patterns and focus periods"""
        
        patterns = []
        
        # Define productivity indicators
        productive_apps = ['vscode', 'pycharm', 'intellij', 'sublime', 'atom', 'vim', 'emacs',
                          'excel', 'word', 'powerpoint', 'keynote', 'figma', 'sketch']
        
        # Find productive activities
        productive_activities = []
        for activity in activity_data:
            app = activity.get('app_name', '').lower()
            
            # Check if it's a productive app or contains code/work content
            is_productive = any(prod_app in app for prod_app in productive_apps)
            
            # Also check text content for programming/work indicators
            text = activity.get('ocr_text', '').lower()
            work_indicators = ['function', 'class', 'import', 'def ', 'var ', 'const ', 
                             'project', 'task', 'deadline', 'meeting', 'report']
            has_work_content = any(indicator in text for indicator in work_indicators)
            
            if is_productive or has_work_content:
                productive_activities.append(activity)
        
        if len(productive_activities) >= 5:
            # Analyze focus sessions (continuous productive activity)
            focus_sessions = []
            current_session = []
            last_time = None
            
            for activity in sorted(productive_activities, key=lambda x: x['timestamp']):
                timestamp = activity['timestamp']
                
                if last_time and (timestamp - last_time).total_seconds() > 1800:  # 30 min gap
                    if len(current_session) >= 3:  # At least 3 activities
                        focus_sessions.append(current_session)
                    current_session = [activity]
                else:
                    current_session.append(activity)
                
                last_time = timestamp
            
            if current_session and len(current_session) >= 3:
                focus_sessions.append(current_session)
            
            if focus_sessions:
                # Calculate average focus session duration
                session_durations = []
                for session in focus_sessions:
                    duration = session[-1]['timestamp'] - session[0]['timestamp']
                    session_durations.append(duration)
                
                avg_duration = statistics.mean(d.total_seconds() for d in session_durations)
                
                pattern = UserPattern(
                    id=f"productivity_focus_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.PRODUCTIVITY,
                    title="Focus Session Pattern",
                    description=f"Typical focus sessions last {avg_duration/60:.1f} minutes",
                    strength=0.8,
                    confidence=0.7,
                    typical_duration=timedelta(seconds=avg_duration),
                    frequency="daily",
                    context_factors=["deep_work", "concentration_periods"],
                    recommendations=[
                        "Schedule uninterrupted blocks for deep work",
                        "Use focus techniques like Pomodoro",
                        "Minimize distractions during peak focus times"
                    ]
                )
                
                pattern.occurrences = [session[0]['timestamp'] for session in focus_sessions]
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_break_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze break and rest patterns"""
        
        patterns = []
        
        # Define break indicators
        break_apps = ['safari', 'chrome', 'firefox', 'youtube', 'netflix', 'spotify', 'music']
        break_keywords = ['youtube', 'news', 'social', 'entertainment', 'video', 'music']
        
        # Find break activities
        break_activities = []
        for activity in activity_data:
            app = activity.get('app_name', '').lower()
            text = activity.get('ocr_text', '').lower()
            
            is_break = any(break_app in app for break_app in break_apps) or \
                      any(keyword in text for keyword in break_keywords)
            
            if is_break:
                break_activities.append(activity)
        
        if len(break_activities) >= 3:
            # Analyze break timing
            break_hours = [activity['timestamp'].hour for activity in break_activities]
            
            # Find common break times
            break_time_counts = Counter(break_hours)
            common_break_times = [hour for hour, count in break_time_counts.items() 
                                if count >= 2]
            
            if common_break_times:
                pattern = UserPattern(
                    id=f"break_timing_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.BREAK,
                    title="Regular Break Times",
                    description=f"Typically takes breaks around: {', '.join(map(str, common_break_times))}",
                    strength=0.6,
                    confidence=0.7,
                    triggers=["break_time", "fatigue_indicator"],
                    conditions={"break_hours": common_break_times},
                    frequency="daily",
                    context_factors=["work_rhythm", "energy_levels"],
                    recommendations=[
                        "Schedule regular breaks to maintain productivity",
                        "Consider more structured break intervals",
                        "Use breaks for physical movement or relaxation"
                    ]
                )
                
                pattern.occurrences = [a['timestamp'] for a in break_activities]
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_error_patterns(
        self,
        activity_data: List[Dict[str, Any]]
    ) -> List[UserPattern]:
        """Analyze error occurrence and problem-solving patterns"""
        
        patterns = []
        
        # Define error indicators
        error_keywords = ['error', 'exception', 'failed', 'bug', 'issue', 'problem', 
                         'not working', 'broken', 'crash', 'debug']
        
        # Find error-related activities
        error_activities = []
        for activity in activity_data:
            text = activity.get('ocr_text', '').lower()
            
            if any(keyword in text for keyword in error_keywords):
                error_activities.append(activity)
        
        if len(error_activities) >= 3:
            # Analyze error timing and context
            error_times = [activity['timestamp'] for activity in error_activities]
            error_apps = [activity.get('app_name', 'unknown') for activity in error_activities]
            
            # Find most common error contexts
            app_counts = Counter(error_apps)
            problematic_apps = [app for app, count in app_counts.items() if count >= 2]
            
            if problematic_apps:
                pattern = UserPattern(
                    id=f"error_context_{datetime.now().strftime('%Y%m%d')}",
                    pattern_type=PatternType.ERROR,
                    title="Common Error Contexts",
                    description=f"Errors frequently occur in: {', '.join(problematic_apps)}",
                    strength=0.7,
                    confidence=0.6,
                    triggers=["error_message", "debugging_activity"],
                    conditions={"problematic_apps": problematic_apps},
                    frequency="occasional",
                    context_factors=["technical_challenges", "learning_areas"],
                    recommendations=[
                        "Keep error logs for debugging",
                        "Consider additional training for problematic tools",
                        "Create troubleshooting documentation"
                    ]
                )
                
                pattern.occurrences = error_times
                patterns.append(pattern)
        
        return patterns
    
    async def _correlate_patterns(self, patterns: List[UserPattern]) -> List[UserPattern]:
        """Find correlations and relationships between patterns"""
        
        # Simple correlation analysis
        for i, pattern_a in enumerate(patterns):
            for pattern_b in patterns[i+1:]:
                correlation = await self._calculate_pattern_correlation(pattern_a, pattern_b)
                
                if correlation > 0.5:  # Significant correlation
                    pattern_a.related_patterns.append(pattern_b.id)
                    pattern_b.related_patterns.append(pattern_a.id)
        
        return patterns
    
    async def _calculate_pattern_correlation(
        self,
        pattern_a: UserPattern,
        pattern_b: UserPattern
    ) -> float:
        """Calculate correlation between two patterns"""
        
        # Simple correlation based on temporal overlap
        if not pattern_a.occurrences or not pattern_b.occurrences:
            return 0.0
        
        # Count co-occurrences within time windows
        co_occurrences = 0
        time_window = timedelta(hours=1)
        
        for time_a in pattern_a.occurrences:
            for time_b in pattern_b.occurrences:
                if abs(time_a - time_b) <= time_window:
                    co_occurrences += 1
                    break
        
        max_possible = min(len(pattern_a.occurrences), len(pattern_b.occurrences))
        
        return co_occurrences / max_possible if max_possible > 0 else 0.0
    
    async def _validate_patterns(self, patterns: List[UserPattern]) -> List[UserPattern]:
        """Validate and filter patterns based on confidence and evidence"""
        
        validated = []
        
        for pattern in patterns:
            # Check minimum requirements
            if (pattern.confidence >= self.confidence_threshold and
                len(pattern.occurrences) >= self.min_occurrences):
                
                # Update pattern metadata
                pattern.first_observed = min(pattern.occurrences) if pattern.occurrences else None
                pattern.last_observed = max(pattern.occurrences) if pattern.occurrences else None
                pattern.updated_at = datetime.now()
                
                validated.append(pattern)
        
        return validated
    
    async def _update_pattern_database(self, patterns: List[UserPattern]):
        """Update the pattern database with new and updated patterns"""
        
        for pattern in patterns:
            self.recognized_patterns[pattern.id] = pattern
            
            # Store pattern in memory system for future reference
            await self.memory.store_content(
                content_id=pattern.id,
                content=json.dumps(pattern.to_dict()),
                content_type="user_pattern",
                metadata={
                    "pattern_type": pattern.pattern_type.value,
                    "strength": pattern.strength,
                    "confidence": pattern.confidence
                }
            )
    
    def _load_pattern_templates(self):
        """Load predefined pattern templates"""
        
        self.pattern_templates = {
            "morning_routine": {
                "type": PatternType.TEMPORAL,
                "triggers": ["morning_hours"],
                "typical_apps": ["mail", "news", "calendar"]
            },
            "coding_session": {
                "type": PatternType.WORKFLOW,
                "triggers": ["ide_app"],
                "typical_sequence": ["editor", "terminal", "browser"]
            },
            "meeting_preparation": {
                "type": PatternType.WORKFLOW,
                "triggers": ["calendar_event"],
                "typical_sequence": ["calendar", "docs", "presentation"]
            }
        }
    
    async def _load_existing_patterns(self):
        """Load previously recognized patterns"""
        
        try:
            # Query memory system for stored patterns
            pattern_results = await self.memory.search_content(
                "user_pattern",
                filters={"content_type": "user_pattern"}
            )
            
            for result in pattern_results:
                try:
                    pattern_data = json.loads(result.content)
                    pattern = UserPattern(**pattern_data)
                    self.recognized_patterns[pattern.id] = pattern
                except Exception as e:
                    self.logger.warning(f"Failed to load pattern: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load existing patterns: {e}")
    
    async def get_patterns_by_type(self, pattern_type: PatternType) -> List[UserPattern]:
        """Get patterns of a specific type"""
        
        return [
            pattern for pattern in self.recognized_patterns.values()
            if pattern.pattern_type == pattern_type
        ]
    
    async def get_patterns_by_strength(self, min_strength: float = 0.7) -> List[UserPattern]:
        """Get patterns above a certain strength threshold"""
        
        return [
            pattern for pattern in self.recognized_patterns.values()
            if pattern.strength >= min_strength
        ]
    
    async def predict_next_activity(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely next activities based on patterns"""
        
        predictions = []
        current_time = datetime.now()
        current_hour = current_time.hour
        current_app = current_context.get('current_app')
        
        # Check temporal patterns
        for pattern in self.recognized_patterns.values():
            if pattern.pattern_type == PatternType.TEMPORAL:
                if current_hour in pattern.conditions.get('peak_hours', []):
                    predictions.append({
                        "type": "temporal_pattern",
                        "pattern_id": pattern.id,
                        "description": pattern.description,
                        "confidence": pattern.confidence * 0.8,  # Slight discount for prediction
                        "recommendations": pattern.recommendations
                    })
        
        # Check workflow patterns
        if current_app:
            for pattern in self.recognized_patterns.values():
                if pattern.pattern_type == PatternType.WORKFLOW:
                    if current_app in pattern.triggers:
                        predictions.append({
                            "type": "workflow_pattern",
                            "pattern_id": pattern.id,
                            "description": f"After {current_app}, you typically: {pattern.description}",
                            "confidence": pattern.confidence * 0.7,
                            "next_action": pattern.conditions.get('target_app')
                        })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions[:5]  # Top 5 predictions
    
    async def get_pattern_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about recognized patterns"""
        
        insights = {
            "summary": {
                "total_patterns": len(self.recognized_patterns),
                "by_type": {},
                "by_strength": {},
                "most_reliable": [],
                "recent_patterns": []
            },
            "productivity_insights": [],
            "behavioral_insights": [],
            "recommendations": []
        }
        
        # Count by type
        for pattern in self.recognized_patterns.values():
            pattern_type = pattern.pattern_type.value
            if pattern_type not in insights["summary"]["by_type"]:
                insights["summary"]["by_type"][pattern_type] = 0
            insights["summary"]["by_type"][pattern_type] += 1
            
            # Count by strength
            strength_level = pattern.get_strength_level().value
            if strength_level not in insights["summary"]["by_strength"]:
                insights["summary"]["by_strength"][strength_level] = 0
            insights["summary"]["by_strength"][strength_level] += 1
        
        # Most reliable patterns
        reliable_patterns = sorted(
            self.recognized_patterns.values(),
            key=lambda p: p.confidence * p.strength,
            reverse=True
        )[:5]
        
        insights["summary"]["most_reliable"] = [
            {
                "title": p.title,
                "type": p.pattern_type.value,
                "strength": p.strength,
                "confidence": p.confidence
            }
            for p in reliable_patterns
        ]
        
        # Recent patterns (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_patterns = [
            p for p in self.recognized_patterns.values()
            if p.created_at >= week_ago
        ]
        
        insights["summary"]["recent_patterns"] = [
            {
                "title": p.title,
                "type": p.pattern_type.value,
                "created": p.created_at.isoformat()
            }
            for p in recent_patterns
        ]
        
        # Generate insights and recommendations
        productivity_patterns = await self.get_patterns_by_type(PatternType.PRODUCTIVITY)
        if productivity_patterns:
            insights["productivity_insights"] = [
                f"You have {len(productivity_patterns)} productivity patterns identified",
                "Focus sessions are most effective during your peak hours",
                "Consider scheduling deep work during high-productivity periods"
            ]
        
        temporal_patterns = await self.get_patterns_by_type(PatternType.TEMPORAL)
        if temporal_patterns:
            insights["behavioral_insights"] = [
                f"Your activity shows {len(temporal_patterns)} clear temporal patterns",
                "Consistent timing patterns indicate good routine establishment",
                "Consider leveraging natural rhythms for optimal performance"
            ]
        
        # General recommendations
        insights["recommendations"] = [
            "Review patterns weekly to identify new opportunities",
            "Use pattern predictions to optimize your workflow",
            "Consider automating routine tasks that follow clear patterns",
            "Pay attention to error patterns to improve troubleshooting"
        ]
        
        return insights