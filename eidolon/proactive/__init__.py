"""
Proactive Assistance module for Eidolon AI Personal Assistant - Phase 7

Provides pattern recognition, predictive assistance, workflow optimization,
and context-aware notifications for proactive user support.
"""

from .pattern_recognizer import PatternRecognizer, UserPattern, PatternType
from .predictive_assistant import PredictiveAssistant, Prediction, PredictionType
from .workflow_optimizer import WorkflowOptimizer, WorkflowInsight
from .notification_engine import NotificationEngine, Notification, NotificationPriority

__all__ = [
    'PatternRecognizer',
    'UserPattern',
    'PatternType',
    'PredictiveAssistant', 
    'Prediction',
    'PredictionType',
    'WorkflowOptimizer',
    'WorkflowInsight',
    'NotificationEngine',
    'Notification',
    'NotificationPriority'
]