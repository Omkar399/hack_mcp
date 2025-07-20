"""
Base Tool Classes for Eidolon Tool Orchestration Framework

Provides base classes and interfaces for all tools in the system.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from ..utils.logging import get_component_logger
from ..core.safety import RiskLevel, ActionCategory

logger = get_component_logger("tools.base")


class ToolStatus(str, Enum):
    """Tool execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(self, message: str, tool_name: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details or {}


class ToolTimeout(ToolError):
    """Exception raised when tool execution times out."""
    
    def __init__(self, timeout_seconds: float, tool_name: str = ""):
        super().__init__(f"Tool execution timed out after {timeout_seconds}s", tool_name)
        self.timeout_seconds = timeout_seconds


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    data: Dict[str, Any]
    message: str = ""
    execution_time: float = 0.0
    side_effects: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.side_effects is None:
            self.side_effects = []
        if self.metadata is None:
            self.metadata = {}


class ToolMetadata(BaseModel):
    """Metadata for tool registration."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    category: str = Field(description="Tool category")
    version: str = Field(default="1.0.0", description="Tool version")
    author: str = Field(default="Eidolon", description="Tool author")
    
    # Risk and safety
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Default risk level")
    action_category: ActionCategory = Field(default=ActionCategory.UNKNOWN, description="Action category")
    requires_approval: bool = Field(default=False, description="Requires user approval")
    
    # Execution constraints
    timeout_seconds: float = Field(default=60.0, description="Default timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Input/output schema
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input parameter schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    conflicts: List[str] = Field(default_factory=list, description="Conflicting tools")


class BaseTool(ABC):
    """
    Base class for all tools in the Eidolon system.
    
    All tools must inherit from this class and implement the execute method.
    """
    
    def __init__(self, metadata: ToolMetadata):
        """Initialize the tool with metadata."""
        self.metadata = metadata
        self.status = ToolStatus.IDLE
        self.current_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._result: Optional[ToolResult] = None
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Input parameters for the tool
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If execution fails
            ToolTimeout: If execution times out
        """
        pass
    
    async def run(
        self, 
        parameters: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Run the tool with timeout and error handling.
        
        Args:
            parameters: Input parameters
            context: Execution context
            timeout: Override default timeout
            
        Returns:
            Tool execution result
        """
        if self.status == ToolStatus.RUNNING:
            raise ToolError("Tool is already running", self.metadata.name)
        
        self.status = ToolStatus.RUNNING
        self._start_time = time.time()
        
        try:
            # Use provided timeout or default
            timeout_seconds = timeout or self.metadata.timeout_seconds
            
            # Execute with timeout
            self._result = await asyncio.wait_for(
                self.execute(parameters, context),
                timeout=timeout_seconds
            )
            
            self._result.execution_time = time.time() - self._start_time
            self.status = ToolStatus.COMPLETED
            
            logger.info(f"Tool {self.metadata.name} completed in {self._result.execution_time:.2f}s")
            return self._result
            
        except asyncio.TimeoutError:
            self.status = ToolStatus.FAILED
            execution_time = time.time() - self._start_time
            
            error = ToolTimeout(timeout_seconds, self.metadata.name)
            logger.error(f"Tool {self.metadata.name} timed out after {execution_time:.2f}s")
            
            self._result = ToolResult(
                success=False,
                data={"error": str(error)},
                message=f"Tool execution timed out",
                execution_time=execution_time
            )
            
            raise error
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            execution_time = time.time() - self._start_time
            
            logger.error(f"Tool {self.metadata.name} failed: {e}")
            
            self._result = ToolResult(
                success=False,
                data={"error": str(e)},
                message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time
            )
            
            if isinstance(e, ToolError):
                raise
            else:
                raise ToolError(str(e), self.metadata.name)
        
        finally:
            if self._start_time:
                self._result = self._result or ToolResult(
                    success=False,
                    data={},
                    message="Tool execution interrupted",
                    execution_time=time.time() - self._start_time
                )
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input parameters against schema.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated parameters (with defaults applied)
            
        Raises:
            ToolError: If validation fails
        """
        # Basic validation - in production, would use jsonschema or pydantic
        validated = parameters.copy()
        
        if self.metadata.input_schema:
            required_fields = self.metadata.input_schema.get("required", [])
            
            for field in required_fields:
                if field not in validated:
                    raise ToolError(f"Missing required parameter: {field}", self.metadata.name)
        
        return validated
    
    def cancel(self) -> bool:
        """
        Cancel tool execution if running.
        
        Returns:
            True if cancellation was successful
        """
        if self.status != ToolStatus.RUNNING or not self.current_task:
            return False
        
        self.current_task.cancel()
        self.status = ToolStatus.CANCELLED
        
        logger.info(f"Tool {self.metadata.name} cancelled")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tool status and execution info."""
        status_info = {
            "name": self.metadata.name,
            "status": self.status,
            "execution_time": 0.0
        }
        
        if self._start_time:
            if self.status == ToolStatus.RUNNING:
                status_info["execution_time"] = time.time() - self._start_time
            elif self._result:
                status_info["execution_time"] = self._result.execution_time
        
        if self._result:
            status_info.update({
                "success": self._result.success,
                "message": self._result.message,
                "side_effects": len(self._result.side_effects)
            })
        
        return status_info
    
    def get_result(self) -> Optional[ToolResult]:
        """Get the last execution result."""
        return self._result
    
    def reset(self) -> None:
        """Reset tool state for reuse."""
        if self.status == ToolStatus.RUNNING:
            self.cancel()
        
        self.status = ToolStatus.IDLE
        self.current_task = None
        self._start_time = None
        self._result = None


class CompositeTool(BaseTool):
    """
    A tool that combines multiple sub-tools into a single operation.
    """
    
    def __init__(self, metadata: ToolMetadata, sub_tools: List[BaseTool]):
        """Initialize composite tool with sub-tools."""
        super().__init__(metadata)
        self.sub_tools = {tool.metadata.name: tool for tool in sub_tools}
        self.execution_order: List[str] = []
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute all sub-tools in sequence."""
        results = {}
        all_side_effects = []
        total_time = 0.0
        
        try:
            for tool_name in self.execution_order:
                if tool_name not in self.sub_tools:
                    continue
                
                tool = self.sub_tools[tool_name]
                tool_params = parameters.get(tool_name, {})
                
                logger.debug(f"Executing sub-tool: {tool_name}")
                result = await tool.run(tool_params, context)
                
                results[tool_name] = result
                all_side_effects.extend(result.side_effects)
                total_time += result.execution_time
                
                # Stop on failure if configured
                if not result.success and not parameters.get("continue_on_failure", False):
                    return ToolResult(
                        success=False,
                        data={"results": results, "failed_at": tool_name},
                        message=f"Composite tool failed at: {tool_name}",
                        execution_time=total_time,
                        side_effects=all_side_effects
                    )
            
            return ToolResult(
                success=True,
                data={"results": results},
                message=f"Composite tool completed {len(results)} sub-tools",
                execution_time=total_time,
                side_effects=all_side_effects
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={"results": results, "error": str(e)},
                message=f"Composite tool execution failed: {str(e)}",
                execution_time=total_time,
                side_effects=all_side_effects
            )
    
    def set_execution_order(self, order: List[str]) -> None:
        """Set the order in which sub-tools are executed."""
        self.execution_order = order
    
    def add_sub_tool(self, tool: BaseTool) -> None:
        """Add a sub-tool to the composite."""
        self.sub_tools[tool.metadata.name] = tool
        if tool.metadata.name not in self.execution_order:
            self.execution_order.append(tool.metadata.name)


class ParameterizedTool(BaseTool):
    """
    A tool that can be parameterized with different configurations.
    """
    
    def __init__(self, metadata: ToolMetadata, base_tool: BaseTool):
        """Initialize with a base tool to parameterize."""
        super().__init__(metadata)
        self.base_tool = base_tool
        self.parameter_overrides: Dict[str, Any] = {}
    
    def set_parameter_overrides(self, overrides: Dict[str, Any]) -> None:
        """Set parameter overrides for the base tool."""
        self.parameter_overrides = overrides
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute base tool with parameter overrides applied."""
        # Merge parameters with overrides
        merged_params = {**parameters, **self.parameter_overrides}
        
        # Execute base tool
        return await self.base_tool.execute(merged_params, context)