"""
Tool Registry for Eidolon Tool Orchestration Framework

Manages tool registration, discovery, and lifecycle.
"""

import asyncio
import inspect
from typing import Dict, List, Optional, Type, Callable, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.util

from ..utils.logging import get_component_logger
from ..core.safety import SafetyManager, RiskLevel
from .base import BaseTool, ToolMetadata, ToolResult, ToolError

logger = get_component_logger("tools.registry")


@dataclass
class ToolRegistration:
    """Information about a registered tool."""
    tool_class: Type[BaseTool]
    metadata: ToolMetadata
    instance: Optional[BaseTool] = None
    enabled: bool = True
    load_on_demand: bool = True
    dependencies_met: bool = True
    last_used: Optional[float] = None


class ToolRegistry:
    """
    Central registry for all tools in the Eidolon system.
    
    Manages tool registration, discovery, lifecycle, and provides
    a unified interface for tool execution.
    """
    
    def __init__(self, safety_manager: Optional[SafetyManager] = None):
        """Initialize the tool registry."""
        self.safety_manager = safety_manager or SafetyManager()
        
        # Registry storage
        self.tools: Dict[str, ToolRegistration] = {}
        self.categories: Dict[str, List[str]] = {}
        self.aliases: Dict[str, str] = {}
        
        # Execution tracking
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self.auto_load_tools = True
        self.max_concurrent_tools = 10
        
        logger.info("Tool registry initialized")
    
    def register_tool(
        self, 
        tool_class: Type[BaseTool], 
        metadata: Optional[ToolMetadata] = None,
        enabled: bool = True,
        load_on_demand: bool = True
    ) -> bool:
        """
        Register a tool class with the registry.
        
        Args:
            tool_class: Tool class to register
            metadata: Tool metadata (auto-detected if not provided)
            enabled: Whether tool is enabled by default
            load_on_demand: Whether to create instances on demand
            
        Returns:
            True if registration was successful
        """
        try:
            # Auto-detect metadata if not provided
            if metadata is None:
                metadata = self._extract_metadata(tool_class)
            
            tool_name = metadata.name
            
            # Check for conflicts
            if tool_name in self.tools:
                logger.warning(f"Tool {tool_name} is already registered, overriding")
            
            # Validate dependencies
            dependencies_met = self._check_dependencies(metadata.dependencies)
            
            # Create registration
            registration = ToolRegistration(
                tool_class=tool_class,
                metadata=metadata,
                enabled=enabled,
                load_on_demand=load_on_demand,
                dependencies_met=dependencies_met
            )
            
            self.tools[tool_name] = registration
            
            # Update category index
            category = metadata.category
            if category not in self.categories:
                self.categories[category] = []
            if tool_name not in self.categories[category]:
                self.categories[category].append(tool_name)
            
            # Initialize execution stats
            self.execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
            
            logger.info(f"Registered tool: {tool_name} (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_class.__name__}: {e}")
            return False
    
    def register_tool_instance(self, tool_instance: BaseTool, enabled: bool = True) -> bool:
        """
        Register an already instantiated tool.
        
        Args:
            tool_instance: Tool instance to register
            enabled: Whether tool is enabled
            
        Returns:
            True if registration was successful
        """
        try:
            metadata = tool_instance.metadata
            tool_name = metadata.name
            
            # Create registration with instance
            registration = ToolRegistration(
                tool_class=type(tool_instance),
                metadata=metadata,
                instance=tool_instance,
                enabled=enabled,
                load_on_demand=False,
                dependencies_met=True
            )
            
            self.tools[tool_name] = registration
            
            # Update category index
            category = metadata.category
            if category not in self.categories:
                self.categories[category] = []
            if tool_name not in self.categories[category]:
                self.categories[category].append(tool_name)
            
            # Initialize execution stats
            self.execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
            
            logger.info(f"Registered tool instance: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool instance: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            True if tool was unregistered
        """
        try:
            if tool_name not in self.tools:
                return False
            
            registration = self.tools[tool_name]
            
            # Cancel any active executions
            if tool_name in self.active_executions:
                self.active_executions[tool_name].cancel()
                del self.active_executions[tool_name]
            
            # Remove from category index
            category = registration.metadata.category
            if category in self.categories and tool_name in self.categories[category]:
                self.categories[category].remove(tool_name)
                if not self.categories[category]:
                    del self.categories[category]
            
            # Clean up
            del self.tools[tool_name]
            if tool_name in self.execution_stats:
                del self.execution_stats[tool_name]
            
            logger.info(f"Unregistered tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_name}: {e}")
            return False
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        validate_safety: bool = True
    ) -> ToolResult:
        """
        Execute a tool with safety validation.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            context: Execution context
            timeout: Override timeout
            validate_safety: Whether to perform safety validation
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If tool execution fails
        """
        if tool_name not in self.tools:
            raise ToolError(f"Tool not found: {tool_name}")
        
        registration = self.tools[tool_name]
        
        # Check if tool is enabled
        if not registration.enabled:
            raise ToolError(f"Tool is disabled: {tool_name}")
        
        # Check dependencies
        if not registration.dependencies_met:
            raise ToolError(f"Tool dependencies not met: {tool_name}")
        
        # Check concurrent execution limit
        if len(self.active_executions) >= self.max_concurrent_tools:
            raise ToolError("Maximum concurrent tool executions reached")
        
        # Safety validation
        if validate_safety:
            action = {
                "type": "tool_execution",
                "params": {
                    "tool_name": tool_name,
                    "parameters": parameters
                }
            }
            
            validation = await self.safety_manager.validate_action(action)
            if not validation.get("approved", False):
                raise ToolError(f"Tool execution not approved: {validation.get('reason', 'Unknown')}")
        
        # Get or create tool instance
        tool_instance = await self._get_tool_instance(tool_name)
        
        try:
            # Execute tool
            logger.info(f"Executing tool: {tool_name}")
            
            # Track execution
            execution_task = asyncio.create_task(
                tool_instance.run(parameters, context, timeout)
            )
            self.active_executions[tool_name] = execution_task
            
            # Wait for completion
            result = await execution_task
            
            # Update statistics
            self._update_execution_stats(tool_name, result)
            
            return result
            
        except Exception as e:
            # Update failure statistics
            self._update_execution_stats(tool_name, None, failed=True)
            
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise
            
        finally:
            # Clean up
            if tool_name in self.active_executions:
                del self.active_executions[tool_name]
    
    async def execute_tool_chain(
        self,
        tool_chain: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        stop_on_failure: bool = True
    ) -> List[ToolResult]:
        """
        Execute a chain of tools in sequence.
        
        Args:
            tool_chain: List of tool execution specs
            context: Shared execution context
            stop_on_failure: Whether to stop on first failure
            
        Returns:
            List of tool results
        """
        results = []
        chain_context = context or {}
        
        for i, tool_spec in enumerate(tool_chain):
            tool_name = tool_spec.get("tool")
            parameters = tool_spec.get("parameters", {})
            timeout = tool_spec.get("timeout")
            
            if not tool_name:
                error_result = ToolResult(
                    success=False,
                    data={"error": f"Missing tool name in chain step {i}"},
                    message="Invalid tool chain specification"
                )
                results.append(error_result)
                
                if stop_on_failure:
                    break
                continue
            
            try:
                # Add previous results to context
                chain_context["previous_results"] = results
                chain_context["step_index"] = i
                
                result = await self.execute_tool(
                    tool_name=tool_name,
                    parameters=parameters,
                    context=chain_context,
                    timeout=timeout
                )
                
                results.append(result)
                
                # Stop on failure if configured
                if not result.success and stop_on_failure:
                    logger.warning(f"Tool chain stopped at step {i} due to failure")
                    break
                
            except Exception as e:
                error_result = ToolResult(
                    success=False,
                    data={"error": str(e)},
                    message=f"Tool chain execution failed at step {i}: {tool_name}"
                )
                results.append(error_result)
                
                if stop_on_failure:
                    break
        
        return results
    
    def get_tool(self, tool_name: str) -> Optional[ToolRegistration]:
        """Get tool registration by name."""
        return self.tools.get(tool_name)
    
    def list_tools(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True,
        dependencies_met_only: bool = True
    ) -> List[str]:
        """
        List available tools.
        
        Args:
            category: Filter by category
            enabled_only: Only list enabled tools
            dependencies_met_only: Only list tools with met dependencies
            
        Returns:
            List of tool names
        """
        tools = []
        
        for name, registration in self.tools.items():
            # Apply filters
            if category and registration.metadata.category != category:
                continue
            if enabled_only and not registration.enabled:
                continue
            if dependencies_met_only and not registration.dependencies_met:
                continue
            
            tools.append(name)
        
        return sorted(tools)
    
    def get_categories(self) -> List[str]:
        """Get all tool categories."""
        return sorted(self.categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category."""
        return self.categories.get(category, []).copy()
    
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a tool."""
        if tool_name in self.tools:
            self.tools[tool_name].enabled = True
            logger.info(f"Enabled tool: {tool_name}")
            return True
        return False
    
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a tool."""
        if tool_name in self.tools:
            self.tools[tool_name].enabled = False
            logger.info(f"Disabled tool: {tool_name}")
            return True
        return False
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool."""
        registration = self.tools.get(tool_name)
        return registration.metadata if registration else None
    
    def get_execution_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if tool_name:
            return self.execution_stats.get(tool_name, {}).copy()
        else:
            return self.execution_stats.copy()
    
    def get_active_executions(self) -> List[str]:
        """Get list of currently executing tools."""
        return list(self.active_executions.keys())
    
    async def cancel_execution(self, tool_name: str) -> bool:
        """Cancel an active tool execution."""
        if tool_name in self.active_executions:
            self.active_executions[tool_name].cancel()
            return True
        return False
    
    async def discover_tools(self, package_path: str) -> int:
        """
        Discover and auto-register tools from a package.
        
        Args:
            package_path: Path to package containing tools
            
        Returns:
            Number of tools discovered and registered
        """
        discovered = 0
        
        try:
            package_dir = Path(package_path)
            if not package_dir.exists():
                logger.warning(f"Tool package path does not exist: {package_path}")
                return 0
            
            # Scan for Python files
            for py_file in package_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    # Import module
                    spec = importlib.util.spec_from_file_location(
                        py_file.stem, 
                        py_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find tool classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseTool) and 
                            obj != BaseTool):
                            
                            if self.register_tool(obj):
                                discovered += 1
                                logger.debug(f"Discovered tool: {name} from {py_file}")
                
                except Exception as e:
                    logger.warning(f"Failed to import tool module {py_file}: {e}")
            
            logger.info(f"Discovered {discovered} tools from {package_path}")
            return discovered
            
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return 0
    
    async def _get_tool_instance(self, tool_name: str) -> BaseTool:
        """Get or create tool instance."""
        registration = self.tools[tool_name]
        
        # Use existing instance if available
        if registration.instance:
            return registration.instance
        
        # Create new instance
        try:
            tool_instance = registration.tool_class(registration.metadata)
            
            # Cache instance if not load-on-demand
            if not registration.load_on_demand:
                registration.instance = tool_instance
            
            return tool_instance
            
        except Exception as e:
            raise ToolError(f"Failed to create tool instance: {e}", tool_name)
    
    def _extract_metadata(self, tool_class: Type[BaseTool]) -> ToolMetadata:
        """Extract metadata from tool class."""
        # Check if class has metadata attribute
        if hasattr(tool_class, 'METADATA'):
            return tool_class.METADATA
        
        # Auto-generate basic metadata
        return ToolMetadata(
            name=tool_class.__name__.lower().replace("tool", ""),
            description=tool_class.__doc__ or f"Tool: {tool_class.__name__}",
            category="general"
        )
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if tool dependencies are available."""
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"Tool dependency not available: {dep}")
                return False
        return True
    
    def _update_execution_stats(
        self, 
        tool_name: str, 
        result: Optional[ToolResult], 
        failed: bool = False
    ) -> None:
        """Update execution statistics for a tool."""
        if tool_name not in self.execution_stats:
            return
        
        stats = self.execution_stats[tool_name]
        stats["total_executions"] += 1
        
        if failed or (result and not result.success):
            stats["failed_executions"] += 1
        else:
            stats["successful_executions"] += 1
        
        if result and result.execution_time > 0:
            stats["total_execution_time"] += result.execution_time
            stats["average_execution_time"] = (
                stats["total_execution_time"] / stats["total_executions"]
            )


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool_class: Type[BaseTool], **kwargs) -> bool:
    """Register a tool with the global registry."""
    return get_global_registry().register_tool(tool_class, **kwargs)


async def execute_tool(tool_name: str, parameters: Dict[str, Any], **kwargs) -> ToolResult:
    """Execute a tool using the global registry."""
    return await get_global_registry().execute_tool(tool_name, parameters, **kwargs)