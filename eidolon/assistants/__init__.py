"""
Assistant Modules for Eidolon AI Personal Assistant

Provides specialized assistants for different domains like email, documents,
calendar, and office automation.
"""

from .email_assistant import EmailAssistant
from .document_assistant import DocumentAssistant
from .office_assistant import OfficeAssistant

__all__ = [
    "EmailAssistant",
    "DocumentAssistant", 
    "OfficeAssistant"
]