"""
Digital Twin module for Eidolon AI Personal Assistant - Phase 7

Provides comprehensive digital twin capabilities including behavior modeling,
predictive simulation, personal assistant functions, and continuous learning.
"""

from .digital_twin_engine import DigitalTwinEngine, TwinCapability, TwinPersonality
from .behavior_model import BehaviorModel, BehaviorPattern, DecisionModel
from .personal_assistant import PersonalAssistant, AssistantAction, AssistantCapability
from .twin_simulator import TwinSimulator, SimulationScenario, SimulationResult

__all__ = [
    'DigitalTwinEngine',
    'TwinCapability',
    'TwinPersonality',
    'BehaviorModel',
    'BehaviorPattern',
    'DecisionModel',
    'PersonalAssistant',
    'AssistantAction',
    'AssistantCapability',
    'TwinSimulator',
    'SimulationScenario',
    'SimulationResult'
]