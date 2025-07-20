"""
Personality module for Eidolon AI Personal Assistant - Phase 7

Provides communication style analysis, writing pattern learning, response style
adaptation, and personal preference modeling for digital twin capabilities.
"""

from .style_analyzer import StyleAnalyzer, CommunicationStyle, WritingPattern
from .style_replicator import StyleReplicator, StyleModel, ResponseAdaptation
from .personality_model import PersonalityModel, PersonalityTrait, PersonalityProfile
from .preference_engine import PreferenceEngine, UserPreference, PreferenceType

__all__ = [
    'StyleAnalyzer',
    'CommunicationStyle',
    'WritingPattern',
    'StyleReplicator',
    'StyleModel',
    'ResponseAdaptation',
    'PersonalityModel',
    'PersonalityTrait',
    'PersonalityProfile',
    'PreferenceEngine',
    'UserPreference',
    'PreferenceType'
]