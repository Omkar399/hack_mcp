"""
Predictive Assistant for Eidolon AI Personal Assistant

Provides predictive assistance by anticipating user needs, suggesting actions,
and proactively offering help based on learned patterns and context.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.memory import MemorySystem
from ..models.cloud_api import CloudAPIManager
from .pattern_recognizer import PatternRecognizer, UserPattern, PatternType


class PredictionType(Enum):
    """Types of predictions the assistant can make"""
    NEXT_ACTION = "next_action"
    TASK_COMPLETION = "task_completion"
    BREAK_NEEDED = "break_needed"
    CONTEXT_SWITCH = "context_switch"
    PROBLEM_SOLVING = "problem_solving"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    RESOURCE_NEED = "resource_need"
    DEADLINE_RISK = "deadline_risk"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"  # 0.3-0.5
    MEDIUM = "medium"  # 0.5-0.7
    HIGH = "high"  # 0.7-0.9
    VERY_HIGH = "very_high"  # 0.9+


@dataclass
class Prediction:
    """Represents a prediction made by the assistant"""
    id: str
    prediction_type: PredictionType
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    urgency: float  # 0.0 to 1.0
    
    # Prediction details
    suggested_action: str
    rationale: str
    supporting_patterns: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    predicted_time: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Outcome tracking
    was_correct: Optional[bool] = None
    user_response: Optional[str] = None
    actual_outcome: Optional[str] = None
    
    def get_confidence_level(self) -> PredictionConfidence:
        """Get categorical confidence level"""
        if self.confidence >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif self.confidence >= 0.7:
            return PredictionConfidence.HIGH
        elif self.confidence >= 0.5:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'prediction_type': self.prediction_type.value,
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'confidence_level': self.get_confidence_level().value,
            'urgency': self.urgency,
            'suggested_action': self.suggested_action,
            'rationale': self.rationale,
            'supporting_patterns': self.supporting_patterns,
            'context': self.context,
            'predicted_time': self.predicted_time.isoformat() if self.predicted_time else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'created_at': self.created_at.isoformat(),
            'was_correct': self.was_correct,
            'user_response': self.user_response,
            'actual_outcome': self.actual_outcome
        }


class PredictiveAssistant:
    """Advanced predictive assistance system"""
    
    def __init__(self):
        self.logger = get_component_logger("predictive_assistant")
        self.config = get_config()
        self.memory = MemorySystem()
        self.cloud_api = CloudAPIManager()
        self.pattern_recognizer = PatternRecognizer()
        
        # Prediction state
        self.active_predictions: Dict[str, Prediction] = {}
        self.prediction_history: List[Prediction] = []
        
        # Learning and adaptation
        self.prediction_accuracy: Dict[PredictionType, float] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # Prediction thresholds
        self.min_confidence = 0.5
        self.prediction_window = timedelta(hours=4)  # How far ahead to predict
        
        # Load existing data
        asyncio.create_task(self._load_prediction_history())
        asyncio.create_task(self._load_user_preferences())
    
    @log_exceptions
    async def generate_predictions(
        self,
        current_context: Dict[str, Any]
    ) -> List[Prediction]:
        """Generate predictions based on current context and patterns"""
        
        self.logger.info("Generating predictive assistance")
        
        predictions = []
        
        # Get relevant patterns for prediction
        patterns = await self._get_relevant_patterns(current_context)
        
        # Generate different types of predictions
        next_action_preds = await self._predict_next_actions(current_context, patterns)
        predictions.extend(next_action_preds)
        
        break_preds = await self._predict_break_needs(current_context, patterns)
        predictions.extend(break_preds)
        
        workflow_preds = await self._predict_workflow_optimizations(current_context, patterns)
        predictions.extend(workflow_preds)
        
        problem_preds = await self._predict_problems(current_context, patterns)
        predictions.extend(problem_preds)
        
        resource_preds = await self._predict_resource_needs(current_context, patterns)
        predictions.extend(resource_preds)
        
        # Filter and rank predictions
        filtered_predictions = await self._filter_and_rank_predictions(predictions)
        
        # Store active predictions
        for prediction in filtered_predictions:
            self.active_predictions[prediction.id] = prediction
        
        # Clean up expired predictions
        await self._cleanup_expired_predictions()
        
        self.logger.info(f"Generated {len(filtered_predictions)} predictions")
        return filtered_predictions
    
    async def _get_relevant_patterns(
        self,
        current_context: Dict[str, Any]
    ) -> List[UserPattern]:
        """Get patterns relevant to current context"""
        
        relevant_patterns = []
        current_time = datetime.now()
        current_hour = current_time.hour
        current_app = current_context.get('current_app')
        
        # Get all patterns
        all_patterns = self.pattern_recognizer.recognized_patterns.values()
        
        for pattern in all_patterns:
            relevance_score = 0.0
            
            # Temporal relevance
            if pattern.pattern_type == PatternType.TEMPORAL:
                if current_hour in pattern.conditions.get('peak_hours', []):
                    relevance_score += 0.8
            
            # Application relevance
            if current_app and current_app in pattern.triggers:
                relevance_score += 0.7
            
            # Recent activity relevance
            if pattern.last_observed:
                time_since = current_time - pattern.last_observed
                if time_since < timedelta(hours=24):
                    relevance_score += 0.5
            
            # Pattern strength
            relevance_score *= pattern.strength
            
            if relevance_score > 0.3:
                relevant_patterns.append(pattern)
        
        return sorted(relevant_patterns, key=lambda p: p.strength, reverse=True)
    
    async def _predict_next_actions(
        self,
        current_context: Dict[str, Any],
        patterns: List[UserPattern]
    ) -> List[Prediction]:
        """Predict likely next actions"""
        
        predictions = []
        current_app = current_context.get('current_app')
        current_time = datetime.now()
        
        # Look for workflow patterns
        workflow_patterns = [p for p in patterns if p.pattern_type == PatternType.WORKFLOW]
        
        for pattern in workflow_patterns:
            if current_app in pattern.triggers:
                target_app = pattern.conditions.get('target_app')
                if target_app:
                    prediction = Prediction(
                        id=f"next_action_{pattern.id}_{int(current_time.timestamp())}",
                        prediction_type=PredictionType.NEXT_ACTION,
                        title=f"Switch to {target_app}",
                        description=f"Based on your pattern, you typically move to {target_app} after {current_app}",
                        confidence=pattern.confidence * 0.8,
                        urgency=0.3,
                        suggested_action=f"Open {target_app}",
                        rationale=f"Workflow pattern shows 80% likelihood of switching to {target_app}",
                        supporting_patterns=[pattern.id],
                        context={"current_app": current_app, "target_app": target_app},
                        predicted_time=current_time + timedelta(minutes=15),
                        valid_until=current_time + timedelta(hours=1)
                    )
                    predictions.append(prediction)
        
        # Temporal predictions
        temporal_patterns = [p for p in patterns if p.pattern_type == PatternType.TEMPORAL]
        
        for pattern in temporal_patterns:
            current_hour = current_time.hour
            if current_hour in pattern.conditions.get('peak_hours', []):
                # Predict activities typical for this time
                prediction = Prediction(
                    id=f"temporal_action_{pattern.id}_{int(current_time.timestamp())}",
                    prediction_type=PredictionType.NEXT_ACTION,
                    title=f"Peak Activity Time",
                    description=f"This is typically a high-activity period for you",
                    confidence=pattern.confidence,
                    urgency=0.5,
                    suggested_action="Focus on important tasks",
                    rationale=f"Historical data shows high productivity at this time",
                    supporting_patterns=[pattern.id],
                    context={"time_period": "peak_hours"},
                    predicted_time=current_time,
                    valid_until=current_time + timedelta(hours=2)
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _predict_break_needs(
        self,
        current_context: Dict[str, Any],
        patterns: List[UserPattern]
    ) -> List[Prediction]:
        """Predict when user might need a break"""
        
        predictions = []
        current_time = datetime.now()
        
        # Check productivity patterns for break indicators
        productivity_patterns = [p for p in patterns if p.pattern_type == PatternType.PRODUCTIVITY]
        break_patterns = [p for p in patterns if p.pattern_type == PatternType.BREAK]
        
        # Analyze current work session length
        work_start = current_context.get('current_session_start')
        if work_start:
            session_length = current_time - work_start
            
            # Check against typical focus session durations
            for pattern in productivity_patterns:
                if pattern.typical_duration:
                    if session_length >= pattern.typical_duration * 0.9:  # 90% of typical duration
                        prediction = Prediction(
                            id=f"break_needed_{int(current_time.timestamp())}",
                            prediction_type=PredictionType.BREAK_NEEDED,
                            title="Break Recommended",
                            description=f"You've been working for {session_length.total_seconds()/60:.0f} minutes",
                            confidence=0.8,
                            urgency=0.7,
                            suggested_action="Take a 10-15 minute break",
                            rationale="Extended work sessions can reduce productivity",
                            supporting_patterns=[pattern.id],
                            context={"session_length": session_length.total_seconds()},
                            predicted_time=current_time + timedelta(minutes=10),
                            valid_until=current_time + timedelta(minutes=30)
                        )
                        predictions.append(prediction)
        
        # Check regular break times
        for pattern in break_patterns:
            break_hours = pattern.conditions.get('break_hours', [])
            current_hour = current_time.hour
            
            if current_hour in break_hours:
                prediction = Prediction(
                    id=f"scheduled_break_{int(current_time.timestamp())}",
                    prediction_type=PredictionType.BREAK_NEEDED,
                    title="Regular Break Time",
                    description="This is typically when you take breaks",
                    confidence=pattern.confidence,
                    urgency=0.5,
                    suggested_action="Consider taking your regular break",
                    rationale="Maintaining consistent break schedule improves well-being",
                    supporting_patterns=[pattern.id],
                    context={"break_type": "scheduled"},
                    predicted_time=current_time,
                    valid_until=current_time + timedelta(hours=1)
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _predict_workflow_optimizations(
        self,
        current_context: Dict[str, Any],
        patterns: List[UserPattern]
    ) -> List[Prediction]:
        """Predict workflow optimization opportunities"""
        
        predictions = []
        current_time = datetime.now()
        
        # Look for inefficient patterns
        workflow_patterns = [p for p in patterns if p.pattern_type == PatternType.WORKFLOW]
        
        # Check for frequent context switches
        frequent_switches = []
        for pattern in workflow_patterns:
            if "frequent" in pattern.frequency and "switch" in pattern.description.lower():
                frequent_switches.append(pattern)
        
        if frequent_switches:
            prediction = Prediction(
                id=f"workflow_opt_{int(current_time.timestamp())}",
                prediction_type=PredictionType.WORKFLOW_OPTIMIZATION,
                title="Reduce Context Switching",
                description="You frequently switch between applications",
                confidence=0.7,
                urgency=0.4,
                suggested_action="Consider batching similar tasks together",
                rationale="Frequent context switches reduce productivity",
                supporting_patterns=[p.id for p in frequent_switches],
                context={"optimization_type": "context_switching"},
                predicted_time=current_time,
                valid_until=current_time + timedelta(hours=4)
            )
            predictions.append(prediction)
        
        # Check for automation opportunities
        repetitive_patterns = [
            p for p in workflow_patterns 
            if p.strength > 0.8 and "frequent" in p.frequency
        ]
        
        if repetitive_patterns:
            prediction = Prediction(
                id=f"automation_opp_{int(current_time.timestamp())}",
                prediction_type=PredictionType.WORKFLOW_OPTIMIZATION,
                title="Automation Opportunity",
                description="Detected highly repetitive workflow patterns",
                confidence=0.6,
                urgency=0.3,
                suggested_action="Consider automating repetitive tasks",
                rationale="Automation can save time and reduce errors",
                supporting_patterns=[p.id for p in repetitive_patterns],
                context={"optimization_type": "automation"},
                predicted_time=current_time,
                valid_until=current_time + timedelta(days=1)
            )
            predictions.append(prediction)
        
        return predictions
    
    async def _predict_problems(
        self,
        current_context: Dict[str, Any],
        patterns: List[UserPattern]
    ) -> List[Prediction]:
        """Predict potential problems or issues"""
        
        predictions = []
        current_time = datetime.now()
        current_app = current_context.get('current_app')
        
        # Check error patterns
        error_patterns = [p for p in patterns if p.pattern_type == PatternType.ERROR]
        
        for pattern in error_patterns:
            problematic_apps = pattern.conditions.get('problematic_apps', [])
            
            if current_app and current_app in problematic_apps:
                prediction = Prediction(
                    id=f"problem_risk_{pattern.id}_{int(current_time.timestamp())}",
                    prediction_type=PredictionType.PROBLEM_SOLVING,
                    title=f"Potential Issues in {current_app}",
                    description=f"You've previously encountered errors in {current_app}",
                    confidence=pattern.confidence * 0.6,
                    urgency=0.4,
                    suggested_action="Keep error logs ready, save work frequently",
                    rationale="Historical error patterns suggest potential issues",
                    supporting_patterns=[pattern.id],
                    context={"app": current_app, "risk_type": "errors"},
                    predicted_time=current_time + timedelta(minutes=30),
                    valid_until=current_time + timedelta(hours=2)
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _predict_resource_needs(
        self,
        current_context: Dict[str, Any],
        patterns: List[UserPattern]
    ) -> List[Prediction]:
        """Predict resource needs"""
        
        predictions = []
        current_time = datetime.now()
        
        # Check for patterns that indicate resource-intensive activities
        productivity_patterns = [p for p in patterns if p.pattern_type == PatternType.PRODUCTIVITY]
        
        for pattern in productivity_patterns:
            if pattern.typical_duration and pattern.typical_duration > timedelta(hours=1):
                # Long focus sessions might need resources
                prediction = Prediction(
                    id=f"resource_need_{pattern.id}_{int(current_time.timestamp())}",
                    prediction_type=PredictionType.RESOURCE_NEED,
                    title="Extended Work Session Expected",
                    description="Based on patterns, you might have a long work session",
                    confidence=pattern.confidence * 0.7,
                    urgency=0.3,
                    suggested_action="Prepare workspace, hydration, minimize distractions",
                    rationale="Long work sessions benefit from preparation",
                    supporting_patterns=[pattern.id],
                    context={"resource_type": "workspace_preparation"},
                    predicted_time=current_time,
                    valid_until=current_time + timedelta(hours=1)
                )
                predictions.append(prediction)
        
        return predictions
    
    async def _filter_and_rank_predictions(
        self,
        predictions: List[Prediction]
    ) -> List[Prediction]:
        """Filter and rank predictions by relevance and confidence"""
        
        # Filter by minimum confidence
        filtered = [p for p in predictions if p.confidence >= self.min_confidence]
        
        # Remove duplicates (same type and similar context)
        unique_predictions = []
        seen_contexts = set()
        
        for prediction in filtered:
            context_key = (
                prediction.prediction_type.value,
                str(sorted(prediction.context.items()))
            )
            
            if context_key not in seen_contexts:
                unique_predictions.append(prediction)
                seen_contexts.add(context_key)
        
        # Rank by composite score (confidence + urgency + recency)
        def prediction_score(p: Prediction) -> float:
            time_factor = 1.0  # Could add time-based weighting
            return (p.confidence * 0.6 + p.urgency * 0.3 + time_factor * 0.1)
        
        ranked_predictions = sorted(
            unique_predictions,
            key=prediction_score,
            reverse=True
        )
        
        # Limit to top predictions
        return ranked_predictions[:10]
    
    async def _cleanup_expired_predictions(self):
        """Remove expired predictions"""
        
        current_time = datetime.now()
        expired_ids = []
        
        for prediction_id, prediction in self.active_predictions.items():
            if prediction.valid_until and current_time > prediction.valid_until:
                expired_ids.append(prediction_id)
        
        for prediction_id in expired_ids:
            expired_prediction = self.active_predictions.pop(prediction_id)
            self.prediction_history.append(expired_prediction)
    
    async def provide_proactive_assistance(
        self,
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide comprehensive proactive assistance"""
        
        # Generate predictions
        predictions = await self.generate_predictions(current_context)
        
        # Get contextual recommendations
        recommendations = await self._generate_contextual_recommendations(current_context)
        
        # Check for urgent assistance needs
        urgent_assistance = await self._check_urgent_assistance(current_context)
        
        # Compile assistance response
        assistance = {
            "predictions": [p.to_dict() for p in predictions],
            "recommendations": recommendations,
            "urgent_assistance": urgent_assistance,
            "context_analysis": await self._analyze_current_context(current_context),
            "generated_at": datetime.now().isoformat()
        }
        
        return assistance
    
    async def _generate_contextual_recommendations(
        self,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on current context"""
        
        recommendations = []
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Time-based recommendations
        if 9 <= current_hour <= 11:  # Morning
            recommendations.append({
                "type": "time_based",
                "title": "Morning Productivity",
                "description": "Consider tackling your most important tasks during peak morning hours",
                "action": "Review and prioritize today's tasks",
                "priority": "medium"
            })
        elif 14 <= current_hour <= 16:  # Afternoon
            recommendations.append({
                "type": "time_based",
                "title": "Afternoon Focus",
                "description": "Good time for collaborative work and communication",
                "action": "Schedule meetings or team discussions",
                "priority": "low"
            })
        
        # Context-specific recommendations
        current_app = current_context.get('current_app')
        if current_app:
            if 'code' in current_app.lower() or 'editor' in current_app.lower():
                recommendations.append({
                    "type": "app_specific",
                    "title": "Coding Session",
                    "description": "You're in a development environment",
                    "action": "Consider running tests regularly and saving work frequently",
                    "priority": "medium"
                })
            elif 'mail' in current_app.lower():
                recommendations.append({
                    "type": "app_specific",
                    "title": "Email Management",
                    "description": "Processing emails",
                    "action": "Use inbox zero technique - decide, do, defer, or delete",
                    "priority": "low"
                })
        
        return recommendations
    
    async def _check_urgent_assistance(
        self,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for urgent assistance needs"""
        
        urgent_items = []
        
        # Check for high-urgency predictions
        high_urgency_predictions = [
            p for p in self.active_predictions.values()
            if p.urgency > 0.7
        ]
        
        for prediction in high_urgency_predictions:
            urgent_items.append({
                "type": "prediction",
                "title": prediction.title,
                "description": prediction.description,
                "action": prediction.suggested_action,
                "urgency": prediction.urgency
            })
        
        # Check for error indicators in current context
        current_text = current_context.get('current_text', '').lower()
        error_indicators = ['error', 'exception', 'failed', 'crash', 'not working']
        
        if any(indicator in current_text for indicator in error_indicators):
            urgent_items.append({
                "type": "error_detection",
                "title": "Potential Error Detected",
                "description": "Error-related text detected in current screen",
                "action": "Check for errors and consider troubleshooting steps",
                "urgency": 0.8
            })
        
        return urgent_items
    
    async def _analyze_current_context(
        self,
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze current context for insights"""
        
        analysis = {
            "activity_type": "unknown",
            "focus_level": "medium",
            "productivity_indicators": [],
            "suggestions": []
        }
        
        current_app = current_context.get('current_app', '').lower()
        current_text = current_context.get('current_text', '').lower()
        
        # Determine activity type
        if any(dev_app in current_app for dev_app in ['code', 'editor', 'terminal', 'ide']):
            analysis["activity_type"] = "development"
            analysis["productivity_indicators"].append("Using development tools")
        elif any(comm_app in current_app for comm_app in ['mail', 'slack', 'teams', 'zoom']):
            analysis["activity_type"] = "communication"
            analysis["productivity_indicators"].append("Engaging in communication")
        elif any(browser in current_app for browser in ['chrome', 'safari', 'firefox']):
            analysis["activity_type"] = "research" if len(current_text) > 100 else "browsing"
        
        # Assess focus level based on text density and app type
        text_density = len(current_text.split()) if current_text else 0
        
        if text_density > 50 and analysis["activity_type"] in ["development", "research"]:
            analysis["focus_level"] = "high"
            analysis["productivity_indicators"].append("High text density indicates focus")
        elif text_density < 10:
            analysis["focus_level"] = "low"
        
        # Generate suggestions based on analysis
        if analysis["activity_type"] == "development":
            analysis["suggestions"].append("Consider using version control frequently")
            analysis["suggestions"].append("Take breaks every 45-60 minutes")
        elif analysis["activity_type"] == "communication":
            analysis["suggestions"].append("Batch email processing for efficiency")
            analysis["suggestions"].append("Use templates for common responses")
        
        return analysis
    
    async def update_prediction_outcome(
        self,
        prediction_id: str,
        was_correct: bool,
        user_response: Optional[str] = None,
        actual_outcome: Optional[str] = None
    ):
        """Update prediction with actual outcome for learning"""
        
        # Find prediction in active or history
        prediction = self.active_predictions.get(prediction_id)
        if not prediction:
            prediction = next(
                (p for p in self.prediction_history if p.id == prediction_id),
                None
            )
        
        if prediction:
            prediction.was_correct = was_correct
            prediction.user_response = user_response
            prediction.actual_outcome = actual_outcome
            
            # Update accuracy tracking
            pred_type = prediction.prediction_type
            if pred_type not in self.prediction_accuracy:
                self.prediction_accuracy[pred_type] = []
            
            self.prediction_accuracy[pred_type].append(was_correct)
            
            # Keep only recent accuracy data
            if len(self.prediction_accuracy[pred_type]) > 100:
                self.prediction_accuracy[pred_type] = self.prediction_accuracy[pred_type][-100:]
            
            self.logger.info(f"Updated prediction {prediction_id} outcome: {was_correct}")
    
    async def get_prediction_analytics(self) -> Dict[str, Any]:
        """Get analytics about prediction performance"""
        
        analytics = {
            "total_predictions": len(self.prediction_history) + len(self.active_predictions),
            "active_predictions": len(self.active_predictions),
            "accuracy_by_type": {},
            "most_accurate_types": [],
            "improvement_areas": [],
            "user_engagement": {}
        }
        
        # Calculate accuracy by prediction type
        for pred_type, outcomes in self.prediction_accuracy.items():
            if outcomes:
                accuracy = sum(outcomes) / len(outcomes)
                analytics["accuracy_by_type"][pred_type.value] = {
                    "accuracy": accuracy,
                    "total_predictions": len(outcomes)
                }
        
        # Find most/least accurate types
        if analytics["accuracy_by_type"]:
            sorted_by_accuracy = sorted(
                analytics["accuracy_by_type"].items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )
            
            analytics["most_accurate_types"] = [
                {"type": t, "accuracy": data["accuracy"]}
                for t, data in sorted_by_accuracy[:3]
            ]
            
            analytics["improvement_areas"] = [
                {"type": t, "accuracy": data["accuracy"]}
                for t, data in sorted_by_accuracy[-3:]
                if data["accuracy"] < 0.7
            ]
        
        # User engagement analysis
        responded_predictions = [
            p for p in self.prediction_history
            if p.user_response is not None
        ]
        
        analytics["user_engagement"] = {
            "response_rate": len(responded_predictions) / len(self.prediction_history) if self.prediction_history else 0,
            "positive_responses": len([p for p in responded_predictions if "positive" in (p.user_response or "")]),
            "ignored_predictions": len([p for p in self.prediction_history if p.user_response is None])
        }
        
        return analytics
    
    async def _load_prediction_history(self):
        """Load prediction history from storage"""
        
        try:
            # Query memory system for stored predictions
            prediction_results = await self.memory.search_content(
                "prediction_history",
                filters={"content_type": "prediction"}
            )
            
            for result in prediction_results:
                try:
                    prediction_data = json.loads(result.content)
                    prediction = Prediction(**prediction_data)
                    self.prediction_history.append(prediction)
                except Exception as e:
                    self.logger.warning(f"Failed to load prediction: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load prediction history: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences for predictions"""
        
        # Default preferences
        self.user_preferences = {
            "notification_frequency": "moderate",  # low, moderate, high
            "prediction_types": list(PredictionType),  # All types enabled
            "min_confidence": 0.6,
            "urgent_only": False,
            "learning_mode": True  # Allow system to learn from outcomes
        }
        
        try:
            # Load from memory if available
            prefs_results = await self.memory.search_content(
                "user_preferences",
                filters={"content_type": "preferences"}
            )
            
            if prefs_results:
                stored_prefs = json.loads(prefs_results[0].content)
                self.user_preferences.update(stored_prefs)
                
        except Exception as e:
            self.logger.error(f"Failed to load user preferences: {e}")
    
    async def save_prediction_state(self):
        """Save current prediction state"""
        
        try:
            # Save active predictions
            for prediction in self.active_predictions.values():
                await self.memory.store_content(
                    content_id=f"prediction_{prediction.id}",
                    content=json.dumps(prediction.to_dict()),
                    content_type="prediction",
                    metadata={
                        "prediction_type": prediction.prediction_type.value,
                        "confidence": prediction.confidence,
                        "urgency": prediction.urgency
                    }
                )
            
            # Save user preferences
            await self.memory.store_content(
                content_id="user_preferences_predictions",
                content=json.dumps(self.user_preferences),
                content_type="preferences",
                metadata={"subsystem": "predictive_assistant"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save prediction state: {e}")
    
    async def configure_predictions(self, preferences: Dict[str, Any]):
        """Configure prediction preferences"""
        
        self.user_preferences.update(preferences)
        
        # Update thresholds based on preferences
        if "min_confidence" in preferences:
            self.min_confidence = preferences["min_confidence"]
        
        # Save updated preferences
        await self.save_prediction_state()
        
        self.logger.info("Updated prediction preferences")
    
    async def get_active_predictions(self) -> List[Dict[str, Any]]:
        """Get currently active predictions"""
        
        return [p.to_dict() for p in self.active_predictions.values()]
    
    async def dismiss_prediction(self, prediction_id: str, reason: str = "dismissed"):
        """Dismiss a prediction"""
        
        if prediction_id in self.active_predictions:
            prediction = self.active_predictions.pop(prediction_id)
            prediction.user_response = reason
            self.prediction_history.append(prediction)
            
            self.logger.info(f"Dismissed prediction {prediction_id}: {reason}")