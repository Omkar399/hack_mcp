"""
Advanced Analytics Engine for Eidolon AI Personal Assistant

Provides deep contextual understanding, productivity insights, and advanced analytics
capabilities for comprehensive personal activity analysis.
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from collections import defaultdict, Counter
import statistics

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..storage.metadata_db import MetadataDatabase
from ..storage.vector_db import VectorDatabase


@dataclass
class TimelineEvent:
    """Represents a single event in a project timeline."""
    timestamp: datetime
    event_type: str  # 'code', 'document', 'meeting', 'email', 'break'
    application: str
    title: str
    description: str
    project_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ProjectTimeline:
    """Represents a complete project timeline."""
    project_id: str
    name: str
    start_date: datetime
    end_date: Optional[datetime]
    events: List[TimelineEvent]
    total_time: timedelta
    languages: List[str]
    main_application: str
    productivity_score: float
    metadata: Dict[str, Any] = None


@dataclass
class ProductivityMetrics:
    """Comprehensive productivity metrics."""
    date: datetime
    total_active_time: timedelta
    productive_time: timedelta
    focus_time: timedelta
    distraction_time: timedelta
    context_switches: int
    applications_used: List[str]
    productivity_score: float  # 0-100
    focus_sessions: List[Dict[str, Any]]
    break_patterns: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None


@dataclass
class Habit:
    """Represents an identified user habit."""
    name: str
    habit_type: str  # 'positive', 'negative', 'neutral'
    strength: float  # 0-1, how established the habit is
    frequency: str  # 'daily', 'weekly', 'occasional'
    triggers: List[str]
    context: Dict[str, Any]
    first_observed: datetime
    last_observed: datetime
    recommendation: str
    confidence: float


class AnalyticsEngine:
    """Advanced analytics engine for deep insights and pattern recognition."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("analytics")
        
        # Database connections
        self.metadata_db = MetadataDatabase()
        self.vector_db = VectorDatabase()
        
        # Analytics configuration
        self.min_session_duration = timedelta(minutes=5)
        self.max_idle_time = timedelta(minutes=10)
        self.productivity_threshold = 0.7
        
        # Programming language patterns
        self.code_patterns = {
            'python': ['.py', 'python', 'pip', 'conda', 'pytest'],
            'javascript': ['.js', '.ts', '.jsx', '.tsx', 'npm', 'node', 'yarn'],
            'java': ['.java', 'gradle', 'maven', 'spring'],
            'cpp': ['.cpp', '.c', '.h', 'cmake', 'gcc', 'clang'],
            'go': ['.go', 'go mod', 'go build', 'golang'],
            'rust': ['.rs', 'cargo', 'rustc'],
            'sql': ['.sql', 'select', 'insert', 'update', 'delete']
        }
        
        # Application categories
        self.app_categories = {
            'development': ['vscode', 'pycharm', 'intellij', 'xcode', 'terminal', 'iterm'],
            'communication': ['slack', 'teams', 'zoom', 'discord', 'mail'],
            'browser': ['chrome', 'firefox', 'safari', 'edge'],
            'design': ['figma', 'sketch', 'photoshop', 'illustrator'],
            'documentation': ['notion', 'obsidian', 'word', 'pages', 'docs'],
            'entertainment': ['youtube', 'netflix', 'spotify', 'games']
        }
        
        self.logger.info("Analytics engine initialized")
    
    @log_performance
    def analyze_project_timelines(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[ProjectTimeline]:
        """
        Reconstruct project timelines from activity data.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            List of identified project timelines
        """
        self.logger.info(f"Analyzing project timelines from {start_date} to {end_date}")
        
        # Get all screenshots and metadata in date range
        screenshots = self.metadata_db.get_screenshots_by_timerange(start_date, end_date)
        
        # Group activities by potential projects
        project_clusters = self._cluster_activities_by_project(screenshots)
        
        # Build timelines for each project
        timelines = []
        for project_id, activities in project_clusters.items():
            timeline = self._build_project_timeline(project_id, activities)
            if timeline:
                timelines.append(timeline)
        
        self.logger.info(f"Identified {len(timelines)} project timelines")
        return timelines
    
    def _cluster_activities_by_project(self, screenshots: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster activities into potential projects using ML and heuristics."""
        clusters = defaultdict(list)
        
        for screenshot in screenshots:
            # Extract features for clustering
            features = self._extract_activity_features(screenshot)
            
            # Determine project ID using various signals
            project_id = self._determine_project_id(features, screenshot)
            clusters[project_id].append(screenshot)
        
        return dict(clusters)
    
    def _extract_activity_features(self, screenshot: Dict) -> Dict[str, Any]:
        """Extract features from a screenshot for analysis."""
        features = {
            'application': screenshot.get('window_info', {}).get('title', '').lower(),
            'file_path': '',
            'language': 'unknown',
            'activity_type': 'unknown',
            'keywords': []
        }
        
        # Extract text content
        text_content = screenshot.get('ocr_text', '')
        if text_content:
            features['keywords'] = self._extract_keywords(text_content)
            features['language'] = self._detect_programming_language(text_content)
            features['activity_type'] = self._classify_activity_type(text_content, features['application'])
        
        # Extract file path if visible
        if 'file' in text_content.lower() or '/' in text_content:
            features['file_path'] = self._extract_file_path(text_content)
        
        return features
    
    def _determine_project_id(self, features: Dict, screenshot: Dict) -> str:
        """Determine project ID based on features and context."""
        # Use file path as primary indicator
        if features['file_path']:
            path_parts = Path(features['file_path']).parts
            if len(path_parts) > 1:
                return path_parts[0]  # Root directory as project
        
        # Use application + language combination
        app = features['application']
        lang = features['language']
        
        if lang != 'unknown':
            return f"{app}_{lang}_project"
        
        # Use keywords to identify project
        keywords = features['keywords']
        if keywords:
            # Look for project-like keywords
            for keyword in keywords:
                if len(keyword) > 5 and keyword.isalpha():
                    return f"project_{keyword}"
        
        # Default to application-based clustering
        return f"general_{app}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common code artifacts
        text = re.sub(r'[{}()\[\]<>]', ' ', text)
        text = re.sub(r'[0-9]+', '', text)
        
        # Split and filter words
        words = text.lower().split()
        keywords = [
            word for word in words 
            if len(word) > 3 and word.isalpha() and word not in {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'day'
            }
        ]
        
        # Return most common keywords
        return [word for word, count in Counter(keywords).most_common(10)]
    
    def _detect_programming_language(self, text: str) -> str:
        """Detect programming language from text content."""
        text_lower = text.lower()
        
        # Count language-specific indicators
        language_scores = {}
        for lang, patterns in self.code_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return 'unknown'
    
    def _classify_activity_type(self, text: str, application: str) -> str:
        """Classify the type of activity based on content and application."""
        text_lower = text.lower()
        app_lower = application.lower()
        
        # Check application category first
        for category, apps in self.app_categories.items():
            if any(app in app_lower for app in apps):
                return category
        
        # Content-based classification
        if any(pattern in text_lower for pattern in ['def ', 'function', 'class ', 'import ', 'from ']):
            return 'development'
        
        if any(pattern in text_lower for pattern in ['email', 'meeting', 'call', 'message']):
            return 'communication'
        
        if any(pattern in text_lower for pattern in ['document', 'notes', 'draft', 'writing']):
            return 'documentation'
        
        return 'general'
    
    def _extract_file_path(self, text: str) -> str:
        """Extract file path from text content."""
        # Look for common file path patterns
        patterns = [
            r'/[a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+',  # Unix paths
            r'[A-Z]:\\[a-zA-Z0-9_\\.-]+\.[a-zA-Z0-9]+',  # Windows paths
            r'~/[a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+'  # Home directory paths
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return ''
    
    def _build_project_timeline(self, project_id: str, activities: List[Dict]) -> Optional[ProjectTimeline]:
        """Build a complete timeline for a project."""
        if len(activities) < 3:  # Minimum activities for a project
            return None
        
        # Sort activities by timestamp
        activities.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        # Create timeline events
        events = []
        for activity in activities:
            event = TimelineEvent(
                timestamp=datetime.fromisoformat(activity['timestamp']),
                event_type=self._classify_activity_type(
                    activity.get('ocr_text', ''),
                    activity.get('window_info', {}).get('title', '')
                ),
                application=activity.get('window_info', {}).get('title', ''),
                title=self._generate_event_title(activity),
                description=activity.get('ocr_text', '')[:200],
                project_id=project_id,
                confidence=0.8,
                metadata={'screenshot_id': activity.get('id')}
            )
            events.append(event)
        
        # Calculate timeline metadata
        start_date = events[0].timestamp
        end_date = events[-1].timestamp
        total_time = end_date - start_date
        
        # Identify languages used
        languages = list(set([
            self._detect_programming_language(event.description)
            for event in events
        ]))
        languages = [lang for lang in languages if lang != 'unknown']
        
        # Find main application
        app_counts = Counter([event.application for event in events])
        main_application = app_counts.most_common(1)[0][0] if app_counts else 'unknown'
        
        # Calculate productivity score
        productivity_score = self._calculate_project_productivity_score(events)
        
        return ProjectTimeline(
            project_id=project_id,
            name=self._generate_project_name(project_id, events),
            start_date=start_date,
            end_date=end_date,
            events=events,
            total_time=total_time,
            languages=languages,
            main_application=main_application,
            productivity_score=productivity_score,
            metadata={
                'total_events': len(events),
                'unique_applications': len(set([e.application for e in events]))
            }
        )
    
    def _generate_event_title(self, activity: Dict) -> str:
        """Generate a meaningful title for a timeline event."""
        app_title = activity.get('window_info', {}).get('title', '')
        ocr_text = activity.get('ocr_text', '')
        
        # Extract meaningful title from content
        if ocr_text:
            lines = ocr_text.split('\n')
            for line in lines[:3]:  # Check first few lines
                line = line.strip()
                if len(line) > 10 and len(line) < 80:
                    return line
        
        return app_title or 'Activity'
    
    def _generate_project_name(self, project_id: str, events: List[TimelineEvent]) -> str:
        """Generate a human-readable project name."""
        # Try to extract from file paths or keywords
        keywords = []
        for event in events[:5]:  # Sample first few events
            keywords.extend(self._extract_keywords(event.description))
        
        if keywords:
            return f"Project: {Counter(keywords).most_common(1)[0][0].title()}"
        
        return project_id.replace('_', ' ').title()
    
    def _calculate_project_productivity_score(self, events: List[TimelineEvent]) -> float:
        """Calculate productivity score for a project."""
        if not events:
            return 0.0
        
        # Factors: consistency, duration, focus
        total_events = len(events)
        
        # Time consistency (regular intervals)
        time_gaps = []
        for i in range(1, len(events)):
            gap = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            time_gaps.append(gap)
        
        consistency_score = 0.5
        if time_gaps:
            avg_gap = sum(time_gaps) / len(time_gaps)
            gap_variance = statistics.variance(time_gaps) if len(time_gaps) > 1 else 0
            consistency_score = max(0, 1 - (gap_variance / (avg_gap ** 2)))
        
        # Activity density
        total_duration = (events[-1].timestamp - events[0].timestamp).total_seconds()
        density_score = min(1.0, total_events / (total_duration / 3600))  # Events per hour
        
        # Focus score (fewer application switches)
        apps = [event.application for event in events]
        unique_apps = len(set(apps))
        focus_score = max(0, 1 - (unique_apps / total_events))
        
        # Combine scores
        productivity_score = (consistency_score * 0.3 + density_score * 0.4 + focus_score * 0.3)
        return min(100.0, productivity_score * 100)
    
    @log_performance
    def analyze_productivity_patterns(
        self, 
        date: datetime
    ) -> ProductivityMetrics:
        """
        Analyze productivity patterns for a specific date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Comprehensive productivity metrics
        """
        self.logger.info(f"Analyzing productivity patterns for {date.date()}")
        
        # Get day's activities
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        screenshots = self.metadata_db.get_screenshots_by_timerange(start_time, end_time)
        
        # Calculate time metrics
        total_active_time = self._calculate_active_time(screenshots)
        focus_sessions = self._identify_focus_sessions(screenshots)
        context_switches = self._count_context_switches(screenshots)
        
        # Calculate derived metrics
        focus_time = sum([session['duration'] for session in focus_sessions], timedelta())
        productive_time = self._calculate_productive_time(screenshots)
        distraction_time = total_active_time - productive_time
        
        # Application usage
        applications_used = []
        for s in screenshots:
            try:
                if isinstance(s, dict) and s.get('window_info'):
                    title = s.get('window_info', {}).get('title', '')
                    if title:
                        applications_used.append(title)
            except (AttributeError, TypeError):
                continue
        applications_used = list(set(applications_used))
        
        # Overall productivity score
        productivity_score = self._calculate_daily_productivity_score(
            total_active_time, productive_time, focus_time, context_switches
        )
        
        # Break patterns
        break_patterns = self._identify_break_patterns(screenshots)
        
        return ProductivityMetrics(
            date=date,
            total_active_time=total_active_time,
            productive_time=productive_time,
            focus_time=focus_time,
            distraction_time=distraction_time,
            context_switches=context_switches,
            applications_used=applications_used,
            productivity_score=productivity_score,
            focus_sessions=[asdict(session) for session in focus_sessions],
            break_patterns=break_patterns,
            metadata={
                'screenshots_analyzed': len(screenshots),
                'unique_applications': len(applications_used)
            }
        )
    
    def _calculate_active_time(self, screenshots: List[Dict]) -> timedelta:
        """Calculate total active time from screenshots."""
        if not screenshots:
            return timedelta()
        
        # Filter to valid dictionaries and sort by timestamp
        valid_screenshots = [s for s in screenshots if isinstance(s, dict) and 'timestamp' in s]
        if not valid_screenshots:
            return timedelta()
        
        valid_screenshots.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        total_time = timedelta()
        last_timestamp = None
        
        for screenshot in valid_screenshots:
            current_timestamp = datetime.fromisoformat(screenshot['timestamp'])
            
            if last_timestamp:
                gap = current_timestamp - last_timestamp
                if gap <= self.max_idle_time:  # Not a break
                    total_time += gap
            
            last_timestamp = current_timestamp
        
        return total_time
    
    def _identify_focus_sessions(self, screenshots: List[Dict]) -> List[Dict[str, Any]]:
        """Identify periods of focused work."""
        if not screenshots:
            return []
        
        # Filter to valid dictionaries
        valid_screenshots = [s for s in screenshots if isinstance(s, dict) and 'timestamp' in s]
        if not valid_screenshots:
            return []
        
        valid_screenshots.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        sessions = []
        current_session = None
        
        for screenshot in valid_screenshots:
            timestamp = datetime.fromisoformat(screenshot['timestamp'])
            app = screenshot.get('window_info', {}).get('title', '')
            
            if current_session is None:
                current_session = {
                    'start': timestamp,
                    'end': timestamp,
                    'application': app,
                    'duration': timedelta(),
                    'activity_count': 1
                }
            else:
                gap = timestamp - current_session['end']
                
                if (gap <= self.max_idle_time and 
                    app == current_session['application']):
                    # Continue session
                    current_session['end'] = timestamp
                    current_session['duration'] = current_session['end'] - current_session['start']
                    current_session['activity_count'] += 1
                else:
                    # End current session if it's long enough
                    if current_session['duration'] >= self.min_session_duration:
                        sessions.append(current_session)
                    
                    # Start new session
                    current_session = {
                        'start': timestamp,
                        'end': timestamp,
                        'application': app,
                        'duration': timedelta(),
                        'activity_count': 1
                    }
        
        # Add final session
        if (current_session and 
            current_session['duration'] >= self.min_session_duration):
            sessions.append(current_session)
        
        return sessions
    
    def _count_context_switches(self, screenshots: List[Dict]) -> int:
        """Count application context switches."""
        if not screenshots:
            return 0
        
        # Filter to valid dictionaries
        valid_screenshots = [s for s in screenshots if isinstance(s, dict) and 'timestamp' in s]
        if not valid_screenshots:
            return 0
        
        valid_screenshots.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        switches = 0
        last_app = None
        
        for screenshot in valid_screenshots:
            app = screenshot.get('window_info', {}).get('title', '')
            if last_app and app != last_app:
                switches += 1
            last_app = app
        
        return switches
    
    def _calculate_productive_time(self, screenshots: List[Dict]) -> timedelta:
        """Calculate time spent on productive activities."""
        productive_time = timedelta()
        
        for screenshot in screenshots:
            app = screenshot.get('window_info', {}).get('title', '').lower()
            
            # Check if activity is productive
            is_productive = any(
                category in ['development', 'documentation', 'communication']
                for category, apps in self.app_categories.items()
                if any(productive_app in app for productive_app in apps)
            )
            
            if is_productive:
                # Estimate 10 seconds per screenshot for productive time
                productive_time += timedelta(seconds=10)
        
        return productive_time
    
    def _calculate_daily_productivity_score(
        self,
        total_active: timedelta,
        productive: timedelta, 
        focus: timedelta,
        context_switches: int
    ) -> float:
        """Calculate overall daily productivity score."""
        if total_active.total_seconds() == 0:
            return 0.0
        
        # Productivity ratio
        productivity_ratio = productive.total_seconds() / total_active.total_seconds()
        
        # Focus ratio
        focus_ratio = focus.total_seconds() / total_active.total_seconds()
        
        # Context switch penalty
        hours_active = total_active.total_seconds() / 3600
        switches_per_hour = context_switches / max(1, hours_active)
        switch_penalty = min(0.5, switches_per_hour / 20)  # Penalty for excessive switching
        
        # Combine scores
        score = (productivity_ratio * 0.5 + focus_ratio * 0.3 + (1 - switch_penalty) * 0.2)
        return min(100.0, score * 100)
    
    def _identify_break_patterns(self, screenshots: List[Dict]) -> List[Dict[str, Any]]:
        """Identify break patterns and timing."""
        if not screenshots:
            return []
        
        screenshots.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        breaks = []
        last_timestamp = None
        
        for screenshot in screenshots:
            current_timestamp = datetime.fromisoformat(screenshot['timestamp'])
            
            if last_timestamp:
                gap = current_timestamp - last_timestamp
                if gap > self.max_idle_time:  # This is a break
                    breaks.append({
                        'start': last_timestamp,
                        'end': current_timestamp,
                        'duration': gap.total_seconds(),
                        'type': self._classify_break_type(gap)
                    })
            
            last_timestamp = current_timestamp
        
        return breaks
    
    def _classify_break_type(self, duration: timedelta) -> str:
        """Classify type of break based on duration."""
        minutes = duration.total_seconds() / 60
        
        if minutes < 15:
            return 'short_break'
        elif minutes < 60:
            return 'medium_break'
        elif minutes < 240:  # 4 hours
            return 'long_break'
        else:
            return 'extended_break'
    
    @log_performance
    def identify_habits(
        self,
        start_date: datetime,
        end_date: datetime,
        min_occurrences: int = 5
    ) -> List[Habit]:
        """
        Identify user habits from activity patterns.
        
        Args:
            start_date: Analysis period start
            end_date: Analysis period end
            min_occurrences: Minimum occurrences to consider a habit
            
        Returns:
            List of identified habits
        """
        self.logger.info(f"Identifying habits from {start_date} to {end_date}")
        
        # Get all activities in period
        screenshots = self.metadata_db.get_screenshots_by_timerange(start_date, end_date)
        
        # Analyze patterns
        time_patterns = self._analyze_time_patterns(screenshots)
        app_patterns = self._analyze_application_patterns(screenshots)
        sequence_patterns = self._analyze_sequence_patterns(screenshots)
        
        habits = []
        
        # Time-based habits
        for pattern_name, data in time_patterns.items():
            if data['occurrences'] >= min_occurrences:
                habit = Habit(
                    name=f"Regular {pattern_name}",
                    habit_type=self._classify_habit_type(pattern_name, data),
                    strength=min(1.0, data['occurrences'] / 30),  # Normalized to monthly
                    frequency=data['frequency'],
                    triggers=data['triggers'],
                    context=data,
                    first_observed=data['first_seen'],
                    last_observed=data['last_seen'],
                    recommendation=self._generate_habit_recommendation(pattern_name, data),
                    confidence=data['confidence']
                )
                habits.append(habit)
        
        # Application usage habits
        for app, data in app_patterns.items():
            if data['occurrences'] >= min_occurrences:
                habit = Habit(
                    name=f"{app} usage pattern",
                    habit_type=self._classify_app_habit_type(app, data),
                    strength=min(1.0, data['occurrences'] / 50),
                    frequency=data['frequency'],
                    triggers=data['triggers'],
                    context=data,
                    first_observed=data['first_seen'],
                    last_observed=data['last_seen'],
                    recommendation=self._generate_app_habit_recommendation(app, data),
                    confidence=data['confidence']
                )
                habits.append(habit)
        
        self.logger.info(f"Identified {len(habits)} habits")
        return habits
    
    def _analyze_time_patterns(self, screenshots: List[Dict]) -> Dict[str, Any]:
        """Analyze time-based activity patterns."""
        patterns = {}
        
        # Group by hour of day
        hourly_activity = defaultdict(list)
        for screenshot in screenshots:
            timestamp = datetime.fromisoformat(screenshot['timestamp'])
            hour = timestamp.hour
            hourly_activity[hour].append(screenshot)
        
        # Identify patterns
        for hour, activities in hourly_activity.items():
            if len(activities) >= 5:  # Minimum for pattern
                pattern_name = f"{hour:02d}:00 activity"
                
                timestamps = [datetime.fromisoformat(s['timestamp']) for s in activities]
                
                patterns[pattern_name] = {
                    'occurrences': len(activities),
                    'frequency': self._calculate_frequency(timestamps),
                    'triggers': [f"time:{hour:02d}:00"],
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'confidence': min(1.0, len(activities) / 50),
                    'hour': hour,
                    'activities': activities
                }
        
        return patterns
    
    def _analyze_application_patterns(self, screenshots: List[Dict]) -> Dict[str, Any]:
        """Analyze application usage patterns."""
        app_usage = defaultdict(list)
        
        for screenshot in screenshots:
            app = screenshot.get('window_info', {}).get('title', '')
            if app:
                app_usage[app].append(screenshot)
        
        patterns = {}
        for app, activities in app_usage.items():
            if len(activities) >= 5:
                timestamps = [datetime.fromisoformat(s['timestamp']) for s in activities]
                
                patterns[app] = {
                    'occurrences': len(activities),
                    'frequency': self._calculate_frequency(timestamps),
                    'triggers': self._identify_app_triggers(activities),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'confidence': min(1.0, len(activities) / 100),
                    'activities': activities
                }
        
        return patterns
    
    def _analyze_sequence_patterns(self, screenshots: List[Dict]) -> Dict[str, Any]:
        """Analyze sequential activity patterns."""
        # Group by sessions (activities within 30 minutes)
        sessions = []
        current_session = []
        
        screenshots.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        for screenshot in screenshots:
            timestamp = datetime.fromisoformat(screenshot['timestamp'])
            
            if (current_session and 
                timestamp - datetime.fromisoformat(current_session[-1]['timestamp']) > timedelta(minutes=30)):
                if len(current_session) > 2:
                    sessions.append(current_session)
                current_session = [screenshot]
            else:
                current_session.append(screenshot)
        
        if len(current_session) > 2:
            sessions.append(current_session)
        
        # Analyze common sequences
        sequences = defaultdict(int)
        for session in sessions:
            apps = [s.get('window_info', {}).get('title', '') for s in session]
            # Look for 3-app sequences
            for i in range(len(apps) - 2):
                sequence = tuple(apps[i:i+3])
                sequences[sequence] += 1
        
        return {f"sequence_{i}": {'apps': seq, 'count': count} 
                for i, (seq, count) in enumerate(sequences.items()) if count >= 3}
    
    def _calculate_frequency(self, timestamps: List[datetime]) -> str:
        """Calculate frequency pattern from timestamps."""
        if not timestamps:
            return 'unknown'
        
        # Calculate days between occurrences
        timestamps.sort()
        days_span = (timestamps[-1] - timestamps[0]).days + 1
        occurrences = len(timestamps)
        
        if days_span <= 1:
            return 'multiple_daily'
        elif occurrences / days_span >= 1:
            return 'daily'
        elif occurrences / days_span >= 0.5:
            return 'frequent'
        elif occurrences / days_span >= 0.14:  # ~weekly
            return 'weekly'
        else:
            return 'occasional'
    
    def _identify_app_triggers(self, activities: List[Dict]) -> List[str]:
        """Identify triggers for application usage."""
        triggers = []
        
        # Time-based triggers
        hours = [datetime.fromisoformat(a['timestamp']).hour for a in activities]
        common_hours = [hour for hour, count in Counter(hours).most_common(3)]
        triggers.extend([f"time:{hour:02d}:00" for hour in common_hours])
        
        # Day-based triggers
        days = [datetime.fromisoformat(a['timestamp']).weekday() for a in activities]
        common_days = [day for day, count in Counter(days).most_common(2)]
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        triggers.extend([f"day:{day_names[day]}" for day in common_days])
        
        return triggers
    
    def _classify_habit_type(self, pattern_name: str, data: Dict) -> str:
        """Classify habit as positive, negative, or neutral."""
        hour = data.get('hour', 12)
        
        # Early morning activities are generally positive
        if 5 <= hour <= 9:
            return 'positive'
        
        # Late night activities might be negative
        if hour >= 23 or hour <= 2:
            return 'negative'
        
        return 'neutral'
    
    def _classify_app_habit_type(self, app: str, data: Dict) -> str:
        """Classify application habit type."""
        app_lower = app.lower()
        
        # Productive apps
        if any(category in ['development', 'documentation'] 
               for category, apps in self.app_categories.items()
               if any(prod_app in app_lower for prod_app in apps)):
            return 'positive'
        
        # Entertainment apps
        if any(ent_app in app_lower for ent_app in self.app_categories.get('entertainment', [])):
            # Too much entertainment might be negative
            if data['occurrences'] > 100:
                return 'negative'
        
        return 'neutral'
    
    def _generate_habit_recommendation(self, pattern_name: str, data: Dict) -> str:
        """Generate recommendations for habits."""
        hour = data.get('hour', 12)
        frequency = data.get('frequency', 'unknown')
        
        if 'activity' in pattern_name:
            if 5 <= hour <= 9:
                return f"Great! Your {frequency} early morning routine is excellent for productivity."
            elif hour >= 22:
                return f"Consider shifting your {frequency} late-night activity earlier for better sleep."
        
        return f"Monitor this {frequency} pattern to understand its impact on your productivity."
    
    def _generate_app_habit_recommendation(self, app: str, data: Dict) -> str:
        """Generate app-specific habit recommendations."""
        occurrences = data['occurrences']
        frequency = data['frequency']
        
        if 'entertainment' in app.lower() or any(
            ent in app.lower() for ent in ['youtube', 'netflix', 'game']
        ):
            if occurrences > 50:
                return f"Consider limiting {app} usage - currently {frequency}. Try setting specific times for entertainment."
        
        if any(dev in app.lower() for dev in ['code', 'terminal', 'editor']):
            return f"Excellent! Your {frequency} use of {app} shows strong development habits."
        
        return f"Your {frequency} use of {app} appears to be part of your regular workflow."
    
    def get_analytics_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary for a period.
        
        Args:
            start_date: Analysis period start
            end_date: Analysis period end
            
        Returns:
            Comprehensive analytics summary
        """
        self.logger.info(f"Generating analytics summary from {start_date} to {end_date}")
        
        # Get project timelines
        timelines = self.analyze_project_timelines(start_date, end_date)
        
        # Get daily productivity metrics
        daily_metrics = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            date_dt = datetime.combine(current_date, datetime.min.time())
            metrics = self.analyze_productivity_patterns(date_dt)
            daily_metrics.append(metrics)
            current_date += timedelta(days=1)
        
        # Get habits
        habits = self.identify_habits(start_date, end_date)
        
        # Calculate summary statistics
        total_projects = len(timelines)
        avg_productivity = statistics.mean([m.productivity_score for m in daily_metrics])
        total_active_time = sum([m.total_active_time for m in daily_metrics], timedelta())
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days + 1
            },
            'summary': {
                'total_projects': total_projects,
                'average_productivity_score': round(avg_productivity, 2),
                'total_active_time_hours': round(total_active_time.total_seconds() / 3600, 2),
                'habits_identified': len(habits)
            },
            'projects': [asdict(timeline) for timeline in timelines[:10]],  # Top 10
            'daily_productivity': [asdict(metric) for metric in daily_metrics],
            'habits': [asdict(habit) for habit in habits[:20]],  # Top 20
            'generated_at': datetime.now().isoformat()
        }