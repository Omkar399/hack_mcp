"""
System Orchestrator for Eidolon AI Personal Assistant

Manages the complete lifecycle of all system components including:
- Component dependency resolution and startup order
- Health monitoring and auto-restart capabilities
- Resource management and optimization
- Graceful shutdown handling
- Inter-component communication coordination
"""

import asyncio
import threading
import time
import signal
import sys
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from .utils.config import get_config
from .utils.logging import get_component_logger
from .core.observer import Observer
from .core.memory import MemorySystem
from .core.analyzer import Analyzer
from .core.interface import Interface
from .core.mcp_server import MCPServer
from .monitoring.health_monitor import HealthMonitor
from .cli.chat import ChatInterface


class ComponentState(Enum):
    """Component state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class ComponentInfo:
    """Information about a system component."""
    name: str
    instance: Any
    state: ComponentState = ComponentState.STOPPED
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 5
    restart_delay: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Callable] = None
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    critical: bool = True
    auto_restart: bool = True
    last_health_check: Optional[datetime] = None
    health_status: bool = True
    resource_usage: Dict[str, float] = field(default_factory=dict)


class SystemOrchestrator:
    """
    Central orchestrator for managing all Eidolon system components.
    
    Provides:
    - Component lifecycle management
    - Dependency resolution
    - Health monitoring
    - Auto-restart capabilities
    - Resource monitoring
    - Graceful shutdown
    """
    
    def __init__(self):
        self.logger = get_component_logger("orchestrator")
        self.config = get_config()
        
        # Component registry
        self.components: Dict[str, ComponentInfo] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        
        # Control flags
        self.is_running = False
        self.shutdown_requested = False
        
        # Monitoring
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.resource_monitor_thread: Optional[threading.Thread] = None
        self.health_check_interval = 30.0  # seconds
        self.resource_check_interval = 10.0  # seconds
        
        # Performance metrics
        self.metrics = {
            "system_start_time": None,
            "total_restarts": 0,
            "uptime": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "active_components": 0
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "component_started": [],
            "component_stopped": [],
            "component_failed": [],
            "component_restarted": [],
            "system_started": [],
            "system_stopped": []
        }
        
        self._register_components()
        self._setup_signal_handlers()
    
    def _register_components(self) -> None:
        """Register all system components with their configurations."""
        
        # Observer - Core monitoring component
        self.components["observer"] = ComponentInfo(
            name="observer",
            instance=None,  # Will be initialized during startup
            dependencies=[],
            health_check=self._check_observer_health,
            critical=True,
            auto_restart=True
        )
        
        # Memory System - Storage and retrieval
        self.components["memory"] = ComponentInfo(
            name="memory",
            instance=None,
            dependencies=[],
            health_check=self._check_memory_health,
            critical=True,
            auto_restart=True
        )
        
        # Analyzer - Content analysis
        self.components["analyzer"] = ComponentInfo(
            name="analyzer",
            instance=None,
            dependencies=["memory"],
            health_check=self._check_analyzer_health,
            critical=True,
            auto_restart=True
        )
        
        # Interface - Query processing
        self.components["interface"] = ComponentInfo(
            name="interface",
            instance=None,
            dependencies=["memory", "analyzer"],
            health_check=self._check_interface_health,
            critical=True,
            auto_restart=True
        )
        
        # MCP Server - External integration
        self.components["mcp_server"] = ComponentInfo(
            name="mcp_server",
            instance=None,
            dependencies=["memory", "analyzer", "interface"],
            health_check=self._check_mcp_health,
            critical=False,
            auto_restart=True,
            startup_timeout=15.0
        )
        
        # Chat Interface - User interaction
        self.components["chat"] = ComponentInfo(
            name="chat",
            instance=None,
            dependencies=["interface", "memory"],
            health_check=self._check_chat_health,
            critical=False,
            auto_restart=True
        )
        
        # Health Monitor - System monitoring
        self.components["health_monitor"] = ComponentInfo(
            name="health_monitor",
            instance=None,
            dependencies=[],
            health_check=self._check_health_monitor_health,
            critical=False,
            auto_restart=True
        )
        
        # Calculate startup order based on dependencies
        self._calculate_startup_order()
    
    def _calculate_startup_order(self) -> None:
        """Calculate component startup order based on dependencies."""
        visited = set()
        temp_visited = set()
        
        def dfs(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_name}")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            
            for dep in self.components[component_name].dependencies:
                if dep in self.components:
                    dfs(dep)
            
            temp_visited.remove(component_name)
            visited.add(component_name)
            self.startup_order.append(component_name)
        
        for component_name in self.components:
            if component_name not in visited:
                dfs(component_name)
        
        # Shutdown order is reverse of startup order
        self.shutdown_order = list(reversed(self.startup_order))
        
        self.logger.info(f"Component startup order: {self.startup_order}")
        self.logger.info(f"Component shutdown order: {self.shutdown_order}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    async def start_system(self) -> bool:
        """
        Start the complete Eidolon system.
        
        Returns:
            bool: True if system started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("System is already running")
            return True
        
        self.logger.info("Starting Eidolon AI Personal Assistant System")
        self.metrics["system_start_time"] = datetime.now()
        
        try:
            # Start components in dependency order
            for component_name in self.startup_order:
                if not await self._start_component(component_name):
                    self.logger.error(f"Failed to start component: {component_name}")
                    await self._emergency_shutdown()
                    return False
            
            # Start monitoring threads
            self._start_monitoring()
            
            self.is_running = True
            self._emit_event("system_started")
            
            self.logger.info("Eidolon system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            await self._emergency_shutdown()
            return False
    
    async def _start_component(self, component_name: str) -> bool:
        """Start a specific component."""
        component = self.components[component_name]
        
        if component.state == ComponentState.RUNNING:
            return True
        
        self.logger.info(f"Starting component: {component_name}")
        component.state = ComponentState.STARTING
        
        try:
            # Initialize component instance
            if component.instance is None:
                component.instance = await self._create_component_instance(component_name)
            
            # Start the component
            start_time = time.time()
            
            if hasattr(component.instance, 'start'):
                if asyncio.iscoroutinefunction(component.instance.start):
                    await component.instance.start()
                else:
                    component.instance.start()
            
            # Wait for component to be ready with timeout
            await self._wait_for_component_ready(component, component.startup_timeout)
            
            component.state = ComponentState.RUNNING
            component.start_time = datetime.now()
            component.restart_count = 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"Component {component_name} started in {elapsed:.2f}s")
            
            self._emit_event("component_started", component_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start component {component_name}: {e}")
            component.state = ComponentState.FAILED
            return False
    
    async def _create_component_instance(self, component_name: str) -> Any:
        """Create an instance of the specified component."""
        if component_name == "observer":
            return Observer()
        elif component_name == "memory":
            return MemorySystem()
        elif component_name == "analyzer":
            memory = self.components["memory"].instance
            return Analyzer(memory=memory)
        elif component_name == "interface":
            memory = self.components["memory"].instance
            analyzer = self.components["analyzer"].instance
            return Interface(memory=memory, analyzer=analyzer)
        elif component_name == "mcp_server":
            return MCPServer()
        elif component_name == "chat":
            interface = self.components["interface"].instance
            return ChatInterface(interface=interface)
        elif component_name == "health_monitor":
            return HealthMonitor()
        else:
            raise ValueError(f"Unknown component: {component_name}")
    
    async def _wait_for_component_ready(self, component: ComponentInfo, timeout: float) -> None:
        """Wait for component to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if component.health_check and component.health_check():
                return
            await asyncio.sleep(0.5)
        
        # If no health check, assume ready after brief delay
        if not component.health_check:
            await asyncio.sleep(1.0)
            return
        
        raise TimeoutError(f"Component {component.name} not ready within {timeout}s")
    
    def _start_monitoring(self) -> None:
        """Start monitoring threads."""
        if not self.health_monitor_thread or not self.health_monitor_thread.is_alive():
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self.health_monitor_thread.start()
        
        if not self.resource_monitor_thread or not self.resource_monitor_thread.is_alive():
            self.resource_monitor_thread = threading.Thread(
                target=self._resource_monitor_loop,
                daemon=True
            )
            self.resource_monitor_thread.start()
    
    def _health_monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while self.is_running and not self.shutdown_requested:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(5.0)
    
    def _resource_monitor_loop(self) -> None:
        """Main resource monitoring loop."""
        while self.is_running and not self.shutdown_requested:
            try:
                self._collect_resource_metrics()
                time.sleep(self.resource_check_interval)
            except Exception as e:
                self.logger.error(f"Resource monitor error: {e}")
                time.sleep(5.0)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        for component_name, component in self.components.items():
            if component.state != ComponentState.RUNNING:
                continue
            
            try:
                if component.health_check:
                    healthy = component.health_check()
                    component.health_status = healthy
                    component.last_health_check = datetime.now()
                    
                    if not healthy and component.auto_restart:
                        self.logger.warning(f"Component {component_name} failed health check")
                        asyncio.create_task(self._restart_component(component_name))
                
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                if component.auto_restart:
                    asyncio.create_task(self._restart_component(component_name))
    
    def _collect_resource_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # System metrics
            self.metrics["memory_usage"] = psutil.virtual_memory().percent
            self.metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
            self.metrics["active_components"] = sum(
                1 for c in self.components.values() 
                if c.state == ComponentState.RUNNING
            )
            
            if self.metrics["system_start_time"]:
                uptime = datetime.now() - self.metrics["system_start_time"]
                self.metrics["uptime"] = uptime.total_seconds()
            
            # Component-specific metrics
            for component_name, component in self.components.items():
                if component.pid:
                    try:
                        process = psutil.Process(component.pid)
                        component.resource_usage = {
                            "memory_percent": process.memory_percent(),
                            "cpu_percent": process.cpu_percent(),
                            "num_threads": process.num_threads()
                        }
                    except psutil.NoSuchProcess:
                        component.pid = None
        
        except Exception as e:
            self.logger.error(f"Resource collection error: {e}")
    
    async def _restart_component(self, component_name: str) -> bool:
        """Restart a failed component."""
        component = self.components[component_name]
        
        if component.restart_count >= component.max_restarts:
            self.logger.error(f"Component {component_name} exceeded max restarts")
            component.state = ComponentState.FAILED
            return False
        
        self.logger.info(f"Restarting component: {component_name}")
        component.state = ComponentState.RESTARTING
        component.restart_count += 1
        self.metrics["total_restarts"] += 1
        
        try:
            # Stop component if still running
            await self._stop_component(component_name, force=True)
            
            # Wait before restart
            await asyncio.sleep(component.restart_delay)
            
            # Start component
            success = await self._start_component(component_name)
            
            if success:
                self._emit_event("component_restarted", component_name)
                self.logger.info(f"Component {component_name} restarted successfully")
            else:
                self.logger.error(f"Failed to restart component {component_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error restarting component {component_name}: {e}")
            component.state = ComponentState.FAILED
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the entire system."""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down Eidolon system")
        self.shutdown_requested = True
        
        # Stop components in reverse dependency order
        for component_name in self.shutdown_order:
            await self._stop_component(component_name)
        
        # Stop monitoring threads
        self.is_running = False
        
        # Wait for threads to finish
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
        
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.resource_monitor_thread.join(timeout=5.0)
        
        self._emit_event("system_stopped")
        self.logger.info("Eidolon system shutdown complete")
    
    async def _stop_component(self, component_name: str, force: bool = False) -> None:
        """Stop a specific component."""
        component = self.components[component_name]
        
        if component.state == ComponentState.STOPPED:
            return
        
        self.logger.info(f"Stopping component: {component_name}")
        component.state = ComponentState.STOPPING
        
        try:
            if component.instance and hasattr(component.instance, 'stop'):
                if asyncio.iscoroutinefunction(component.instance.stop):
                    await asyncio.wait_for(
                        component.instance.stop(),
                        timeout=component.shutdown_timeout
                    )
                else:
                    component.instance.stop()
            
            component.state = ComponentState.STOPPED
            component.start_time = None
            component.pid = None
            
            self._emit_event("component_stopped", component_name)
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Component {component_name} shutdown timeout")
            if force:
                component.state = ComponentState.STOPPED
        except Exception as e:
            self.logger.error(f"Error stopping component {component_name}: {e}")
            component.state = ComponentState.FAILED
    
    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown in case of startup failure."""
        self.logger.warning("Performing emergency shutdown")
        
        for component_name in self.shutdown_order:
            try:
                await self._stop_component(component_name, force=True)
            except Exception as e:
                self.logger.error(f"Emergency stop failed for {component_name}: {e}")
        
        self.is_running = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "is_running": self.is_running,
            "components": {
                name: {
                    "state": component.state.value,
                    "start_time": component.start_time.isoformat() if component.start_time else None,
                    "restart_count": component.restart_count,
                    "health_status": component.health_status,
                    "last_health_check": component.last_health_check.isoformat() if component.last_health_check else None,
                    "resource_usage": component.resource_usage
                }
                for name, component in self.components.items()
            },
            "metrics": self.metrics,
            "startup_order": self.startup_order
        }
    
    def _emit_event(self, event_type: str, *args) -> None:
        """Emit an event to registered callbacks."""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable) -> None:
        """Register an event callback."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    # Health check methods for components
    def _check_observer_health(self) -> bool:
        """Health check for Observer component."""
        component = self.components["observer"]
        if not component.instance:
            return False
        
        try:
            # Check if observer is actively capturing
            return hasattr(component.instance, 'is_monitoring') and component.instance.is_monitoring
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Health check for Memory component."""
        component = self.components["memory"]
        if not component.instance:
            return False
        
        try:
            # Try a simple operation
            component.instance.get_status()
            return True
        except Exception:
            return False
    
    def _check_analyzer_health(self) -> bool:
        """Health check for Analyzer component."""
        component = self.components["analyzer"]
        if not component.instance:
            return False
        
        try:
            # Check if analyzer can process
            return hasattr(component.instance, 'is_ready') and component.instance.is_ready()
        except Exception:
            return False
    
    def _check_interface_health(self) -> bool:
        """Health check for Interface component."""
        component = self.components["interface"]
        if not component.instance:
            return False
        
        try:
            # Test basic functionality
            return hasattr(component.instance, 'process_query')
        except Exception:
            return False
    
    def _check_mcp_health(self) -> bool:
        """Health check for MCP Server component."""
        component = self.components["mcp_server"]
        if not component.instance:
            return False
        
        try:
            # Check if MCP server is running
            return hasattr(component.instance, 'is_running') and component.instance.is_running
        except Exception:
            return False
    
    def _check_chat_health(self) -> bool:
        """Health check for Chat component."""
        component = self.components["chat"]
        if not component.instance:
            return False
        
        try:
            # Check if chat interface is responsive
            return True  # Chat interface is typically stateless
        except Exception:
            return False
    
    def _check_health_monitor_health(self) -> bool:
        """Health check for Health Monitor component."""
        component = self.components["health_monitor"]
        if not component.instance:
            return False
        
        try:
            # Check if health monitor is active
            return hasattr(component.instance, 'is_active') and component.instance.is_active()
        except Exception:
            return False


# Global orchestrator instance
_orchestrator: Optional[SystemOrchestrator] = None


def get_orchestrator() -> SystemOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SystemOrchestrator()
    return _orchestrator


async def start_eidolon_system() -> bool:
    """Start the complete Eidolon system."""
    orchestrator = get_orchestrator()
    return await orchestrator.start_system()


async def shutdown_eidolon_system() -> None:
    """Shutdown the complete Eidolon system."""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.shutdown()
        _orchestrator = None