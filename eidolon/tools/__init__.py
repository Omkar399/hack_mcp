"""
Tool Orchestration Framework for Eidolon

Provides a registry system for tools, workflow execution, and tool chaining
for autonomous operations.
"""

from .registry import ToolRegistry, ToolResult, ToolMetadata
from .workflow import WorkflowEngine, Workflow, WorkflowStep
from .base import BaseTool, ToolError, ToolTimeout
from .file_ops import FileOperationsTool
from .web_ops import WebOperationsTool
from .system_ops import SystemOperationsTool
from .communication import CommunicationTool

__all__ = [
    "ToolRegistry",
    "ToolResult",
    "ToolMetadata",
    "WorkflowEngine",
    "Workflow",
    "WorkflowStep",
    "BaseTool",
    "ToolError",
    "ToolTimeout",
    "FileOperationsTool",
    "WebOperationsTool", 
    "SystemOperationsTool",
    "CommunicationTool"
]