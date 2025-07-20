"""
Digital Twin Engine for Eidolon AI Personal Assistant

Core engine that orchestrates all digital twin capabilities, providing a
comprehensive model of the user that can act autonomously and predictively.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.memory import MemorySystem
from ..models.cloud_api import CloudAPIManager
from ..planning.task_planner import TaskPlanner, TaskPlan
from ..proactive.pattern_recognizer import PatternRecognizer
from ..proactive.predictive_assistant import PredictiveAssistant
from ..personality.style_replicator import StyleReplicator, ResponseType


class TwinCapability(Enum):
    """Capabilities of the digital twin"""
    PATTERN_RECOGNITION = "pattern_recognition"
    PREDICTIVE_ASSISTANCE = "predictive_assistance"
    TASK_PLANNING = "task_planning"
    STYLE_REPLICATION = "style_replication"
    AUTONOMOUS_ACTION = "autonomous_action"
    LEARNING_ADAPTATION = "learning_adaptation"
    CONTEXT_AWARENESS = "context_awareness"
    GOAL_PURSUIT = "goal_pursuit"
    DECISION_MAKING = "decision_making"
    PROACTIVE_SUPPORT = "proactive_support"


class TwinPersonality(Enum):
    """Personality modes for the digital twin"""
    PROFESSIONAL = "professional"  # Work-focused, efficient
    CREATIVE = "creative"  # Innovative, exploratory
    ANALYTICAL = "analytical"  # Data-driven, systematic
    COLLABORATIVE = "collaborative"  # Team-oriented, communicative
    PERSONAL = "personal"  # Informal, friendly
    ADAPTIVE = "adaptive"  # Changes based on context


class TwinState(Enum):
    """Current state of the digital twin"""
    LEARNING = "learning"  # Gathering data and patterns
    ACTIVE = "active"  # Fully operational
    ASSISTING = "assisting"  # Currently helping user
    PLANNING = "planning"  # Working on task plans
    MONITORING = "monitoring"  # Observing and waiting
    UPDATING = "updating"  # Improving models


@dataclass
class TwinAction:
    """Action taken or planned by the digital twin"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""
    description: str = ""
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    urgency: float = 0.0
    
    # Execution details
    scheduled_time: Optional[datetime] = None
    executed_time: Optional[datetime] = None
    result: Optional[str] = None
    success: Optional[bool] = None
    
    # Context
    trigger: Optional[str] = None
    reasoning: str = ""
    learned_from: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'action_type': self.action_type,
            'description': self.description,
            'target': self.target,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'urgency': self.urgency,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'executed_time': self.executed_time.isoformat() if self.executed_time else None,
            'result': self.result,
            'success': self.success,
            'trigger': self.trigger,
            'reasoning': self.reasoning,
            'learned_from': self.learned_from
        }


@dataclass
class TwinPersonality:
    """Personality configuration for the digital twin"""
    personality_type: TwinPersonality
    traits: Dict[str, float] = field(default_factory=dict)  # 0.0 to 1.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    communication_style: Dict[str, float] = field(default_factory=dict)
    decision_factors: Dict[str, float] = field(default_factory=dict)


class DigitalTwinEngine:
    """Comprehensive digital twin engine orchestrating all capabilities"""
    
    def __init__(self):
        self.logger = get_component_logger("digital_twin_engine")
        self.config = get_config()
        self.memory = MemorySystem()
        self.cloud_api = CloudAPIManager()
        
        # Core components
        self.task_planner = TaskPlanner()
        self.pattern_recognizer = PatternRecognizer()
        self.predictive_assistant = PredictiveAssistant()
        self.style_replicator = StyleReplicator()
        
        # Twin state
        self.twin_id = str(uuid.uuid4())
        self.state = TwinState.LEARNING
        self.capabilities: List[TwinCapability] = []
        self.personality: Optional[TwinPersonality] = None
        
        # Twin knowledge and memory
        self.user_model: Dict[str, Any] = {}
        self.behavior_patterns: Dict[str, Any] = {}
        self.goal_hierarchy: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
        
        # Active operations
        self.active_plans: Dict[str, TaskPlan] = {}
        self.scheduled_actions: List[TwinAction] = []
        self.learning_queue: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.action_success_rate: float = 0.0
        self.prediction_accuracy: float = 0.0
        self.user_satisfaction: float = 0.0
        self.adaptation_count: int = 0
        
        # Initialize twin
        asyncio.create_task(self._initialize_twin())
    
    @log_exceptions
    async def _initialize_twin(self):
        """Initialize the digital twin with basic capabilities"""
        
        self.logger.info("Initializing digital twin engine")
        
        # Load or create user model
        await self._load_user_model()
        
        # Initialize capabilities
        await self._initialize_capabilities()
        
        # Set default personality
        await self._initialize_personality()
        
        # Load existing patterns and knowledge
        await self._load_twin_knowledge()
        
        # Start monitoring and learning
        await self._start_continuous_learning()
        
        self.state = TwinState.ACTIVE
        self.logger.info(f"Digital twin {self.twin_id} initialized and active")
    
    async def _load_user_model(self):
        """Load or create comprehensive user model"""
        
        try:
            # Load from memory
            results = await self.memory.search_content(
                "user_model",
                filters={"content_type": "user_model"}
            )
            
            if results:
                self.user_model = json.loads(results[0].content)
            else:
                # Create default user model
                self.user_model = {
                    "preferences": {},
                    "goals": [],
                    "habits": {},
                    "skills": {},
                    "interests": [],
                    "work_patterns": {},
                    "communication_style": {},
                    "decision_patterns": {},
                    "learning_style": {},
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                
                await self._save_user_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load user model: {e}")
            self.user_model = {}
    
    async def _initialize_capabilities(self):
        """Initialize twin capabilities based on available components"""
        
        self.capabilities = [
            TwinCapability.PATTERN_RECOGNITION,
            TwinCapability.PREDICTIVE_ASSISTANCE,
            TwinCapability.TASK_PLANNING,
            TwinCapability.STYLE_REPLICATION,
            TwinCapability.CONTEXT_AWARENESS,
            TwinCapability.LEARNING_ADAPTATION,
            TwinCapability.PROACTIVE_SUPPORT
        ]
        
        # Check if components are ready for advanced capabilities
        if hasattr(self.style_replicator, 'style_model') and self.style_replicator.style_model:
            self.capabilities.append(TwinCapability.AUTONOMOUS_ACTION)
        
        if len(self.pattern_recognizer.recognized_patterns) > 5:
            self.capabilities.append(TwinCapability.DECISION_MAKING)
            self.capabilities.append(TwinCapability.GOAL_PURSUIT)
        
        self.logger.info(f"Initialized with capabilities: {[c.value for c in self.capabilities]}")
    
    async def _initialize_personality(self):
        """Initialize twin personality based on user patterns"""
        
        # Analyze user patterns to determine personality type
        personality_type = await self._determine_personality_type()
        
        self.personality = TwinPersonality(
            personality_type=personality_type,
            traits={
                "proactivity": 0.7,
                "helpfulness": 0.9,
                "efficiency": 0.8,
                "adaptability": 0.8,
                "privacy_respect": 1.0
            },
            preferences={
                "notification_frequency": "moderate",
                "intervention_level": "helpful",
                "learning_speed": "adaptive"
            },
            communication_style={
                "formality": 0.6,
                "verbosity": 0.5,
                "technical_depth": 0.7
            },
            decision_factors={
                "user_preference": 0.9,
                "efficiency": 0.7,
                "safety": 1.0,
                "learning_opportunity": 0.6
            }
        )
    
    async def _determine_personality_type(self) -> TwinPersonality:
        """Determine appropriate personality type based on user patterns"""
        
        # Analyze user patterns to infer appropriate personality
        patterns = list(self.pattern_recognizer.recognized_patterns.values())
        
        if not patterns:
            return TwinPersonality.ADAPTIVE
        
        # Count different types of patterns
        work_patterns = len([p for p in patterns if 'work' in p.description.lower() or 'productive' in p.description.lower()])
        creative_patterns = len([p for p in patterns if 'creative' in p.description.lower() or 'design' in p.description.lower()])
        technical_patterns = len([p for p in patterns if 'code' in p.description.lower() or 'technical' in p.description.lower()])
        social_patterns = len([p for p in patterns if 'communication' in p.description.lower() or 'social' in p.description.lower()])
        
        # Determine dominant personality type
        if technical_patterns > work_patterns and technical_patterns > creative_patterns:
            return TwinPersonality.ANALYTICAL
        elif creative_patterns > work_patterns and creative_patterns > technical_patterns:
            return TwinPersonality.CREATIVE
        elif social_patterns > len(patterns) / 3:
            return TwinPersonality.COLLABORATIVE
        elif work_patterns > len(patterns) / 2:
            return TwinPersonality.PROFESSIONAL
        else:
            return TwinPersonality.ADAPTIVE
    
    async def _load_twin_knowledge(self):
        """Load existing twin knowledge and patterns"""
        
        try:
            # Load behavior patterns from pattern recognizer
            await self.pattern_recognizer.analyze_user_patterns()
            
            # Load prediction history from predictive assistant
            predictions = await self.predictive_assistant.get_active_predictions()
            
            # Load style model from style replicator
            style_model = self.style_replicator.get_style_model()
            
            # Update user model with gathered knowledge
            self.user_model.update({
                "pattern_count": len(self.pattern_recognizer.recognized_patterns),
                "style_confidence": style_model.confidence if style_model else 0.0,
                "prediction_history_count": len(predictions)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to load twin knowledge: {e}")
    
    async def _start_continuous_learning(self):
        """Start continuous learning and adaptation processes"""
        
        # Schedule regular learning updates
        async def learning_loop():
            while True:
                try:
                    await self._perform_learning_cycle()
                    await asyncio.sleep(3600)  # Learn every hour
                except Exception as e:
                    self.logger.error(f"Learning cycle error: {e}")
                    await asyncio.sleep(300)  # Retry in 5 minutes
        
        # Start learning loop in background
        asyncio.create_task(learning_loop())
    
    async def _perform_learning_cycle(self):
        """Perform a single learning and adaptation cycle"""
        
        self.state = TwinState.UPDATING
        
        try:
            # Update patterns
            await self.pattern_recognizer.analyze_user_patterns()
            
            # Update predictions based on recent accuracy
            prediction_analytics = await self.predictive_assistant.get_prediction_analytics()
            
            # Update style model if new communication samples are available
            # This would need to be triggered by new data
            
            # Adapt personality based on user feedback and interaction patterns
            await self._adapt_personality()
            
            # Update user model
            await self._update_user_model()
            
            # Clean up old data
            await self._cleanup_old_data()
            
            self.adaptation_count += 1
            
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {e}")
        finally:
            self.state = TwinState.ACTIVE
    
    async def _adapt_personality(self):
        """Adapt personality based on user interactions and feedback"""
        
        if not self.personality:
            return
        
        # Analyze recent user interactions
        recent_interactions = self.context_history[-50:] if len(self.context_history) > 50 else self.context_history
        
        if not recent_interactions:
            return
        
        # Count positive vs negative feedback
        positive_feedback = sum(1 for i in recent_interactions if i.get('user_satisfaction', 0.5) > 0.7)
        total_feedback = len([i for i in recent_interactions if 'user_satisfaction' in i])
        
        if total_feedback > 0:
            satisfaction_rate = positive_feedback / total_feedback
            
            # Adjust traits based on satisfaction
            if satisfaction_rate > 0.8:
                # Increase current behavior patterns
                self.personality.traits["proactivity"] = min(1.0, self.personality.traits.get("proactivity", 0.7) + 0.05)
            elif satisfaction_rate < 0.5:
                # Reduce intervention levels
                self.personality.traits["proactivity"] = max(0.3, self.personality.traits.get("proactivity", 0.7) - 0.1)
    
    async def _update_user_model(self):
        """Update comprehensive user model"""
        
        # Get latest insights from all components
        pattern_insights = await self.pattern_recognizer.get_pattern_insights()
        prediction_analytics = await self.predictive_assistant.get_prediction_analytics()
        style_analytics = await self.style_replicator.get_style_analytics()
        
        # Update user model
        self.user_model.update({
            "patterns": pattern_insights,
            "predictions": prediction_analytics,
            "style": style_analytics,
            "twin_adaptation_count": self.adaptation_count,
            "last_updated": datetime.now().isoformat()
        })
        
        await self._save_user_model()
    
    async def _cleanup_old_data(self):
        """Clean up old data to maintain performance"""
        
        # Keep only recent context history
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-500:]
        
        # Clean up completed actions older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        self.scheduled_actions = [
            action for action in self.scheduled_actions
            if not action.executed_time or action.executed_time > cutoff_date
        ]
    
    @log_exceptions
    async def process_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process current context and provide comprehensive twin response"""
        
        self.state = TwinState.ASSISTING
        
        try:
            # Store context in history
            context_entry = {
                **context,
                "timestamp": datetime.now().isoformat(),
                "twin_state": self.state.value
            }
            self.context_history.append(context_entry)
            
            # Generate comprehensive response
            response = {
                "twin_id": self.twin_id,
                "timestamp": datetime.now().isoformat(),
                "context_analysis": {},
                "predictions": [],
                "recommendations": [],
                "planned_actions": [],
                "learning_insights": {},
                "personality_response": {}
            }
            
            # Context analysis
            response["context_analysis"] = await self._analyze_context(context)
            
            # Generate predictions
            if TwinCapability.PREDICTIVE_ASSISTANCE in self.capabilities:
                predictions = await self.predictive_assistant.generate_predictions(context)
                response["predictions"] = [p.to_dict() for p in predictions]
            
            # Generate recommendations
            response["recommendations"] = await self._generate_recommendations(context)
            
            # Plan actions if appropriate
            if TwinCapability.AUTONOMOUS_ACTION in self.capabilities:
                planned_actions = await self._plan_autonomous_actions(context)
                response["planned_actions"] = [a.to_dict() for a in planned_actions]
            
            # Provide learning insights
            response["learning_insights"] = await self._generate_learning_insights(context)
            
            # Personality-specific response
            response["personality_response"] = await self._generate_personality_response(context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process context: {e}")
            return {
                "twin_id": self.twin_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.state = TwinState.ACTIVE
    
    async def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current context using all available capabilities"""
        
        analysis = {
            "activity_type": "unknown",
            "focus_level": "medium",
            "productivity_indicators": [],
            "pattern_matches": [],
            "anomalies": [],
            "opportunities": []
        }
        
        current_app = context.get('current_app', '').lower()
        current_text = context.get('current_text', '').lower()
        current_time = datetime.now()
        
        # Activity type analysis
        if any(dev_app in current_app for dev_app in ['code', 'editor', 'terminal', 'ide']):
            analysis["activity_type"] = "development"
            analysis["productivity_indicators"].append("Using development tools")
        elif any(comm_app in current_app for comm_app in ['mail', 'slack', 'teams', 'zoom']):
            analysis["activity_type"] = "communication"
        elif any(browser in current_app for browser in ['chrome', 'safari', 'firefox']):
            analysis["activity_type"] = "research" if len(current_text) > 100 else "browsing"
        
        # Pattern matching
        relevant_patterns = []
        for pattern in self.pattern_recognizer.recognized_patterns.values():
            if pattern.pattern_type.value == "temporal":
                if current_time.hour in pattern.conditions.get('peak_hours', []):
                    relevant_patterns.append(pattern.title)
            elif pattern.pattern_type.value == "application":
                if current_app in pattern.triggers:
                    relevant_patterns.append(pattern.title)
        
        analysis["pattern_matches"] = relevant_patterns
        
        # Anomaly detection
        if analysis["activity_type"] == "unknown" and len(current_text) > 50:
            analysis["anomalies"].append("Unusual activity pattern detected")
        
        # Opportunity identification
        if analysis["activity_type"] == "development" and "error" in current_text:
            analysis["opportunities"].append("Debugging assistance opportunity")
        
        return analysis
    
    async def _generate_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contextual recommendations"""
        
        recommendations = []
        
        # Get recommendations from predictive assistant
        proactive_assistance = await self.predictive_assistant.provide_proactive_assistance(context)
        recommendations.extend(proactive_assistance.get('recommendations', []))
        
        # Add twin-specific recommendations
        current_hour = datetime.now().hour
        
        # Time-based recommendations
        if 9 <= current_hour <= 11:  # Morning productivity
            recommendations.append({
                "type": "productivity",
                "title": "Morning Focus Time",
                "description": "This is typically your most productive time",
                "action": "Consider tackling your most important tasks now",
                "confidence": 0.8,
                "source": "digital_twin"
            })
        
        # Pattern-based recommendations
        for pattern in self.pattern_recognizer.recognized_patterns.values():
            if pattern.pattern_type.value == "productivity" and pattern.strength > 0.7:
                recommendations.append({
                    "type": "pattern_optimization",
                    "title": f"Leverage {pattern.title}",
                    "description": pattern.description,
                    "action": "Continue this productive pattern",
                    "confidence": pattern.confidence,
                    "source": "pattern_analysis"
                })
                break  # Only add one pattern recommendation
        
        return recommendations
    
    async def _plan_autonomous_actions(self, context: Dict[str, Any]) -> List[TwinAction]:
        """Plan autonomous actions based on context and capabilities"""
        
        actions = []
        current_time = datetime.now()
        
        # Only plan actions if user seems to need help
        if "error" in context.get('current_text', '').lower():
            action = TwinAction(
                action_type="offer_assistance",
                description="Offer debugging help",
                confidence=0.7,
                urgency=0.6,
                scheduled_time=current_time + timedelta(minutes=2),
                trigger="error_detection",
                reasoning="Error text detected, user might need assistance"
            )
            actions.append(action)
        
        # Check if it's time for a proactive check-in
        last_interaction = self.context_history[-1].get('timestamp') if self.context_history else None
        if last_interaction:
            last_time = datetime.fromisoformat(last_interaction)
            if current_time - last_time > timedelta(hours=2):
                action = TwinAction(
                    action_type="proactive_checkin",
                    description="Check if user needs assistance",
                    confidence=0.5,
                    urgency=0.3,
                    scheduled_time=current_time + timedelta(minutes=30),
                    trigger="time_based",
                    reasoning="Haven't interacted for over 2 hours"
                )
                actions.append(action)
        
        return actions
    
    async def _generate_learning_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about what the twin is learning"""
        
        insights = {
            "new_patterns_detected": 0,
            "model_confidence_changes": {},
            "adaptation_opportunities": [],
            "learning_progress": {}
        }
        
        # Check for new patterns in recent analysis
        recent_patterns = [
            p for p in self.pattern_recognizer.recognized_patterns.values()
            if (datetime.now() - p.created_at).days < 1
        ]
        insights["new_patterns_detected"] = len(recent_patterns)
        
        # Model confidence tracking
        style_model = self.style_replicator.get_style_model()
        if style_model:
            insights["model_confidence_changes"]["style_model"] = style_model.confidence
        
        # Learning progress
        insights["learning_progress"] = {
            "total_patterns": len(self.pattern_recognizer.recognized_patterns),
            "adaptation_cycles": self.adaptation_count,
            "context_history_size": len(self.context_history)
        }
        
        return insights
    
    async def _generate_personality_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on twin personality"""
        
        if not self.personality:
            return {}
        
        personality_response = {
            "personality_type": self.personality.personality_type.value,
            "response_style": {},
            "interaction_approach": "",
            "decision_factors": self.personality.decision_factors
        }
        
        # Personality-specific response style
        if self.personality.personality_type == TwinPersonality.PROFESSIONAL:
            personality_response["response_style"] = {
                "tone": "professional",
                "focus": "efficiency and results",
                "communication": "direct and clear"
            }
            personality_response["interaction_approach"] = "Focus on task completion and productivity optimization"
        
        elif self.personality.personality_type == TwinPersonality.CREATIVE:
            personality_response["response_style"] = {
                "tone": "inspirational",
                "focus": "innovation and possibilities", 
                "communication": "encouraging and exploratory"
            }
            personality_response["interaction_approach"] = "Encourage creative exploration and new approaches"
        
        elif self.personality.personality_type == TwinPersonality.ANALYTICAL:
            personality_response["response_style"] = {
                "tone": "logical",
                "focus": "data and patterns",
                "communication": "detailed and systematic"
            }
            personality_response["interaction_approach"] = "Provide data-driven insights and systematic analysis"
        
        elif self.personality.personality_type == TwinPersonality.COLLABORATIVE:
            personality_response["response_style"] = {
                "tone": "supportive",
                "focus": "teamwork and communication",
                "communication": "inclusive and helpful"
            }
            personality_response["interaction_approach"] = "Facilitate collaboration and team productivity"
        
        else:  # ADAPTIVE or PERSONAL
            personality_response["response_style"] = {
                "tone": "adaptive",
                "focus": "user preferences",
                "communication": "personalized"
            }
            personality_response["interaction_approach"] = "Adapt to current context and user needs"
        
        return personality_response
    
    @log_exceptions
    async def execute_autonomous_action(self, action: TwinAction) -> Dict[str, Any]:
        """Execute an autonomous action"""
        
        self.logger.info(f"Executing autonomous action: {action.action_type}")
        
        action.executed_time = datetime.now()
        result = {"success": False, "message": "", "action_id": action.id}
        
        try:
            if action.action_type == "offer_assistance":
                # Generate helpful response
                if TwinCapability.STYLE_REPLICATION in self.capabilities:
                    styled_response = await self.style_replicator.generate_styled_response(
                        "I noticed you might be encountering an issue. Would you like me to help?",
                        ResponseType.MESSAGE,
                        context={"proactive": True}
                    )
                    result["message"] = styled_response.get("response", "Can I help you with anything?")
                else:
                    result["message"] = "I noticed you might be encountering an issue. Can I help?"
                
                result["success"] = True
                action.success = True
                
            elif action.action_type == "proactive_checkin":
                # Generate check-in message
                if TwinCapability.STYLE_REPLICATION in self.capabilities:
                    styled_response = await self.style_replicator.generate_styled_response(
                        "How are things going? Is there anything I can help you with?",
                        ResponseType.MESSAGE,
                        context={"proactive": True}
                    )
                    result["message"] = styled_response.get("response", "How are things going?")
                else:
                    result["message"] = "How are things going? Is there anything I can help you with?"
                
                result["success"] = True
                action.success = True
                
            else:
                result["message"] = f"Unknown action type: {action.action_type}"
                action.success = False
            
            action.result = result["message"]
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {action.id}: {e}")
            result["message"] = f"Failed to execute action: {str(e)}"
            action.success = False
            action.result = str(e)
        
        # Update action success rate
        successful_actions = len([a for a in self.scheduled_actions if a.success is True])
        total_actions = len([a for a in self.scheduled_actions if a.success is not None])
        if total_actions > 0:
            self.action_success_rate = successful_actions / total_actions
        
        return result
    
    async def create_task_plan(
        self,
        objective: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        """Create a task plan using the twin's capabilities"""
        
        if TwinCapability.TASK_PLANNING not in self.capabilities:
            raise ValueError("Task planning capability not available")
        
        self.state = TwinState.PLANNING
        
        try:
            # Enhance context with twin knowledge
            enhanced_context = context or {}
            enhanced_context.update({
                "user_patterns": self.pattern_recognizer.recognized_patterns,
                "twin_personality": self.personality.personality_type.value if self.personality else "adaptive",
                "user_preferences": self.user_model.get("preferences", {})
            })
            
            # Create plan
            plan = await self.task_planner.create_plan(objective, enhanced_context)
            
            # Store active plan
            self.active_plans[plan.id] = plan
            
            return plan
            
        finally:
            self.state = TwinState.ACTIVE
    
    async def get_twin_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the digital twin"""
        
        status = {
            "twin_id": self.twin_id,
            "state": self.state.value,
            "capabilities": [c.value for c in self.capabilities],
            "personality": {
                "type": self.personality.personality_type.value if self.personality else None,
                "traits": self.personality.traits if self.personality else {}
            },
            "performance": {
                "action_success_rate": self.action_success_rate,
                "prediction_accuracy": self.prediction_accuracy,
                "user_satisfaction": self.user_satisfaction,
                "adaptation_count": self.adaptation_count
            },
            "knowledge": {
                "patterns_learned": len(self.pattern_recognizer.recognized_patterns),
                "style_confidence": self.style_replicator.get_style_model().confidence if self.style_replicator.get_style_model() else 0.0,
                "context_history_size": len(self.context_history),
                "active_plans": len(self.active_plans)
            },
            "recent_activity": {
                "scheduled_actions": len(self.scheduled_actions),
                "learning_queue_size": len(self.learning_queue),
                "last_adaptation": self.user_model.get("last_updated", "unknown")
            }
        }
        
        return status
    
    async def update_user_feedback(
        self,
        feedback_type: str,
        rating: float,
        comments: Optional[str] = None
    ):
        """Update twin based on user feedback"""
        
        feedback_entry = {
            "type": feedback_type,
            "rating": rating,  # 0.0 to 1.0
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store feedback in context history
        self.context_history.append({
            "type": "user_feedback",
            "feedback": feedback_entry,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update user satisfaction
        recent_feedback = [
            entry["feedback"]["rating"] for entry in self.context_history[-20:]
            if entry.get("type") == "user_feedback"
        ]
        
        if recent_feedback:
            self.user_satisfaction = sum(recent_feedback) / len(recent_feedback)
        
        # Trigger adaptation if feedback is significantly negative
        if rating < 0.3:
            await self._adapt_to_negative_feedback(feedback_type, comments)
        
        self.logger.info(f"Updated user feedback: {feedback_type} = {rating}")
    
    async def _adapt_to_negative_feedback(
        self,
        feedback_type: str,
        comments: Optional[str]
    ):
        """Adapt twin behavior based on negative feedback"""
        
        if feedback_type == "proactivity" and self.personality:
            # Reduce proactive behavior
            self.personality.traits["proactivity"] = max(0.1, self.personality.traits.get("proactivity", 0.7) - 0.2)
        
        elif feedback_type == "communication_style":
            # Adjust communication style
            if comments and "too formal" in comments.lower():
                if self.personality:
                    self.personality.communication_style["formality"] = max(0.0, self.personality.communication_style.get("formality", 0.6) - 0.2)
        
        elif feedback_type == "predictions":
            # Adjust prediction thresholds
            await self.predictive_assistant.configure_predictions({
                "min_confidence": min(0.9, self.predictive_assistant.min_confidence + 0.1)
            })
        
        # Save updated models
        await self._save_user_model()
    
    async def _save_user_model(self):
        """Save user model to memory"""
        
        try:
            await self.memory.store_content(
                content_id="user_model",
                content=json.dumps(self.user_model),
                content_type="user_model",
                metadata={
                    "twin_id": self.twin_id,
                    "last_updated": datetime.now().isoformat()
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to save user model: {e}")
    
    async def simulate_future_scenario(
        self,
        scenario: Dict[str, Any],
        time_horizon: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Simulate how the user might behave in a future scenario"""
        
        simulation_result = {
            "scenario": scenario,
            "time_horizon": time_horizon.total_seconds(),
            "predicted_outcomes": [],
            "confidence": 0.0,
            "recommendations": []
        }
        
        try:
            # Use patterns to predict behavior
            relevant_patterns = []
            for pattern in self.pattern_recognizer.recognized_patterns.values():
                if any(keyword in scenario.get("description", "").lower() 
                      for keyword in pattern.triggers):
                    relevant_patterns.append(pattern)
            
            # Generate predictions based on patterns
            for pattern in relevant_patterns:
                prediction = {
                    "pattern_based": pattern.title,
                    "likelihood": pattern.confidence,
                    "description": f"Based on {pattern.title}, user likely to {pattern.description}"
                }
                simulation_result["predicted_outcomes"].append(prediction)
            
            # Calculate overall confidence
            if relevant_patterns:
                simulation_result["confidence"] = sum(p.confidence for p in relevant_patterns) / len(relevant_patterns)
            
            # Generate recommendations
            simulation_result["recommendations"] = [
                "Prepare resources for anticipated tasks",
                "Schedule reminders for important actions",
                "Optimize environment for predicted activities"
            ]
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            simulation_result["error"] = str(e)
        
        return simulation_result
    
    def get_twin_id(self) -> str:
        """Get the unique identifier for this digital twin"""
        return self.twin_id
    
    def get_capabilities(self) -> List[TwinCapability]:
        """Get list of current twin capabilities"""
        return self.capabilities.copy()
    
    def get_personality(self) -> Optional[TwinPersonality]:
        """Get current twin personality"""
        return self.personality