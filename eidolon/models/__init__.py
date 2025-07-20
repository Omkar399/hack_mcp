"""
AI Models module for Eidolon

Contains local and cloud AI model integrations:
- Local vision models (Florence-2, CLIP)
- Cloud API integrations (Gemini, Claude, GPT-4V)
- Decision engine for routing between local and cloud
"""

from .local_vision import LocalVisionModel
from .cloud_api import CloudAPIManager
from .decision_engine import DecisionEngine

__all__ = ["LocalVisionModel", "CloudAPIManager", "DecisionEngine"]