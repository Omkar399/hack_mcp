"""
System Operations Tool for Eidolon Tool Orchestration Framework

Provides safe system command execution and process management.
"""

import asyncio
import subprocess
import psutil
import signal
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import shlex

from .base import BaseTool, ToolMetadata, ToolResult, ToolError
from ..core.safety import RiskLevel, ActionCategory
from ..utils.logging import get_component_logger

logger = get_component_logger("tools.system_ops")


class SystemOperationsTool(BaseTool):
    """Tool for safe system operations and command execution."""
    
    METADATA = ToolMetadata(
        name="system_operations",
        description="Execute system commands and manage processes with safety controls",
        category="system",
        risk_level=RiskLevel.HIGH,
        action_category=ActionCategory.SYSTEM_COMMAND,
        requires_approval=True,
        timeout_seconds=60.0,
        input_schema={
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["execute", "ps", "kill", "env", "which", "ping", "disk_usage", "memory_info"]
                },
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "env": {"type": "object"},
                "timeout": {"type": "number"},
                "capture_output": {"type": "boolean", "default": True},
                "shell": {"type": "boolean", "default": False},
                "pid": {"type": "integer"},
                "signal": {"type": "string"},
                "host": {"type": "string"}
            }
        }
    )
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """Initialize system operations tool."""
        super().__init__(metadata or self.METADATA)
        
        # Safety controls
        self.allowed_commands = {
            # File operations
            'ls', 'cat', 'head', 'tail', 'find', 'grep', 'sort', 'uniq', 'wc',
            # Text processing
            'awk', 'sed', 'cut', 'tr', 'diff', 'comm',
            # Archive operations
            'tar', 'gzip', 'gunzip', 'zip', 'unzip',
            # Development tools
            'git', 'python', 'python3', 'node', 'npm', 'pip', 'pip3',
            # System info
            'ps', 'top', 'df', 'du', 'free', 'uptime', 'whoami', 'id',
            # Network (safe commands only)
            'ping', 'curl', 'wget', 'nslookup', 'dig',
            # Package managers (read-only operations)
            'which', 'whereis', 'type'
        }
        
        self.forbidden_commands = {
            # Destructive operations
            'rm', 'rmdir', 'dd', 'shred', 'format',
            # System modification
            'sudo', 'su', 'chmod', 'chown', 'chgrp', 'mount', 'umount',
            # User management
            'useradd', 'userdel', 'usermod', 'passwd', 'chpasswd',
            # Service management
            'systemctl', 'service', 'init', 'shutdown', 'reboot', 'halt',
            # Network configuration
            'iptables', 'netfilter', 'route', 'ifconfig', 'ip',
            # Package installation
            'apt-get', 'yum', 'dnf', 'brew', 'pip install', 'npm install -g'
        }
        
        self.dangerous_patterns = [
            r'>\s*/dev/',  # Writing to device files
            r'\|\s*sh',    # Piping to shell
            r'\|\s*bash',  # Piping to bash
            r'&\s*$',      # Background execution
            r';\s*rm',     # Command chaining with rm
            r'`.*`',       # Command substitution
            r'\$\(',       # Command substitution
        ]
        
        # Resource limits
        self.max_execution_time = 300  # 5 minutes
        self.max_output_size = 10 * 1024 * 1024  # 10MB
        
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute system operation."""
        try:
            # Validate parameters
            validated_params = await self.validate_parameters(parameters)
            operation = validated_params["operation"]
            
            # Route to specific operation
            if operation == "execute":
                return await self._execute_command(validated_params)
            elif operation == "ps":
                return await self._list_processes(validated_params)
            elif operation == "kill":
                return await self._kill_process(validated_params)
            elif operation == "env":
                return await self._get_environment(validated_params)
            elif operation == "which":
                return await self._which_command(validated_params)
            elif operation == "ping":
                return await self._ping_host(validated_params)
            elif operation == "disk_usage":
                return await self._get_disk_usage(validated_params)
            elif operation == "memory_info":
                return await self._get_memory_info(validated_params)
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"System operation failed: {e}")
            return ToolResult(
                success=False,
                data={"error": str(e)},
                message=f"System operation failed: {str(e)}"
            )
    
    async def _execute_command(self, params: Dict[str, Any]) -> ToolResult:
        """Execute a system command safely."""
        command = params.get("command", "")
        args = params.get("args", [])
        cwd = params.get("cwd")
        env = params.get("env")
        timeout = params.get("timeout", 30)
        capture_output = params.get("capture_output", True)
        use_shell = params.get("shell", False)
        
        # Build full command
        if args:
            full_command = [command] + args
        else:
            full_command = command if use_shell else shlex.split(command)
        
        # Safety checks
        if not self._is_command_safe(command):
            raise ToolError(f"Command not allowed: {command}")
        
        if self._contains_dangerous_patterns(command):
            raise ToolError(f"Command contains dangerous patterns: {command}")
        
        # Limit timeout
        timeout = min(timeout, self.max_execution_time)
        
        try:
            # Prepare environment
            proc_env = dict(os.environ)
            if env:
                proc_env.update(env)
            
            # Prepare working directory
            if cwd:
                cwd_path = Path(cwd)
                if not cwd_path.exists() or not cwd_path.is_dir():
                    raise ToolError(f"Working directory not found: {cwd}")
                cwd = str(cwd_path.resolve())
            
            logger.info(f"Executing command: {command}")
            
            # Execute command
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    *full_command if not use_shell else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=proc_env,
                    shell=use_shell if use_shell else False
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')
                
                # Limit output size
                if len(stdout_text) > self.max_output_size:
                    stdout_text = stdout_text[:self.max_output_size] + "\n... (output truncated)"
                
                if len(stderr_text) > self.max_output_size:
                    stderr_text = stderr_text[:self.max_output_size] + "\n... (output truncated)"
                
                return ToolResult(
                    success=process.returncode == 0,
                    data={
                        "command": command,
                        "args": args,
                        "returncode": process.returncode,
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "cwd": cwd
                    },
                    message=f"Command executed with exit code {process.returncode}",
                    side_effects=[f"Executed: {command}"]
                )
            else:
                # Execute without capturing output
                process = await asyncio.create_subprocess_exec(
                    *full_command if not use_shell else None,
                    cwd=cwd,
                    env=proc_env,
                    shell=use_shell if use_shell else False
                )
                
                returncode = await asyncio.wait_for(
                    process.wait(),
                    timeout=timeout
                )
                
                return ToolResult(
                    success=returncode == 0,
                    data={
                        "command": command,
                        "args": args,
                        "returncode": returncode,
                        "cwd": cwd
                    },
                    message=f"Command executed with exit code {returncode}",
                    side_effects=[f"Executed: {command}"]
                )
                
        except asyncio.TimeoutError:
            raise ToolError(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            raise ToolError(f"Command not found: {command}")
        except PermissionError:
            raise ToolError(f"Permission denied: {command}")
        except Exception as e:
            raise ToolError(f"Command execution failed: {e}")
    
    async def _list_processes(self, params: Dict[str, Any]) -> ToolResult:
        """List running processes."""
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            return ToolResult(
                success=True,
                data={
                    "processes": processes[:100],  # Limit to top 100
                    "total_count": len(processes)
                },
                message=f"Listed {len(processes)} processes"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to list processes: {e}")
    
    async def _kill_process(self, params: Dict[str, Any]) -> ToolResult:
        """Kill a process by PID."""
        pid = params.get("pid")
        signal_name = params.get("signal", "TERM")
        
        if not pid:
            raise ToolError("PID is required")
        
        try:
            # Get process info first
            proc = psutil.Process(pid)
            proc_info = {
                "pid": proc.pid,
                "name": proc.name(),
                "status": proc.status()
            }
            
            # Check if it's a system process (basic protection)
            if proc.pid == 1 or proc.name() in ['init', 'kernel', 'kthreadd']:
                raise ToolError(f"Cannot kill system process: {proc.name()}")
            
            # Send signal
            if hasattr(signal, signal_name):
                sig = getattr(signal, signal_name)
                proc.send_signal(sig)
            else:
                raise ToolError(f"Unknown signal: {signal_name}")
            
            return ToolResult(
                success=True,
                data={
                    "process": proc_info,
                    "signal": signal_name
                },
                message=f"Sent {signal_name} signal to process {pid}",
                side_effects=[f"Killed process: {pid}"]
            )
            
        except psutil.NoSuchProcess:
            raise ToolError(f"Process not found: {pid}")
        except psutil.AccessDenied:
            raise ToolError(f"Permission denied to kill process: {pid}")
        except Exception as e:
            raise ToolError(f"Failed to kill process: {e}")
    
    async def _get_environment(self, params: Dict[str, Any]) -> ToolResult:
        """Get environment variables."""
        try:
            # Filter out sensitive environment variables
            sensitive_vars = {'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'AUTH'}
            
            env_vars = {}
            for key, value in os.environ.items():
                # Check if variable name contains sensitive keywords
                if any(sensitive in key.upper() for sensitive in sensitive_vars):
                    env_vars[key] = "<redacted>"
                else:
                    env_vars[key] = value
            
            return ToolResult(
                success=True,
                data={
                    "environment": env_vars,
                    "count": len(env_vars)
                },
                message=f"Retrieved {len(env_vars)} environment variables"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to get environment: {e}")
    
    async def _which_command(self, params: Dict[str, Any]) -> ToolResult:
        """Find the location of a command."""
        command = params.get("command", "")
        
        if not command:
            raise ToolError("Command is required")
        
        try:
            # Use shutil.which to find command
            import shutil
            command_path = shutil.which(command)
            
            if command_path:
                return ToolResult(
                    success=True,
                    data={
                        "command": command,
                        "path": command_path,
                        "exists": True
                    },
                    message=f"Found {command} at {command_path}"
                )
            else:
                return ToolResult(
                    success=False,
                    data={
                        "command": command,
                        "path": None,
                        "exists": False
                    },
                    message=f"Command not found: {command}"
                )
                
        except Exception as e:
            raise ToolError(f"Failed to locate command: {e}")
    
    async def _ping_host(self, params: Dict[str, Any]) -> ToolResult:
        """Ping a host to check connectivity."""
        host = params.get("host", "")
        count = params.get("count", 3)
        
        if not host:
            raise ToolError("Host is required")
        
        # Basic hostname validation
        if not self._is_valid_hostname(host):
            raise ToolError(f"Invalid hostname: {host}")
        
        try:
            # Use ping command
            ping_cmd = ["ping", "-c", str(count), host]
            
            process = await asyncio.create_subprocess_exec(
                *ping_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )
            
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "host": host,
                    "count": count,
                    "returncode": process.returncode,
                    "output": stdout_text,
                    "error": stderr_text
                },
                message=f"Ping to {host} {'successful' if process.returncode == 0 else 'failed'}"
            )
            
        except asyncio.TimeoutError:
            raise ToolError(f"Ping to {host} timed out")
        except Exception as e:
            raise ToolError(f"Failed to ping host: {e}")
    
    async def _get_disk_usage(self, params: Dict[str, Any]) -> ToolResult:
        """Get disk usage information."""
        path = params.get("path", "/")
        
        try:
            # Get disk usage
            usage = psutil.disk_usage(path)
            
            # Get disk partitions
            partitions = []
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total": partition_usage.total,
                        "used": partition_usage.used,
                        "free": partition_usage.free,
                        "percent": round((partition_usage.used / partition_usage.total) * 100, 2)
                    })
                except PermissionError:
                    continue
            
            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "usage": {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": round((usage.used / usage.total) * 100, 2)
                    },
                    "partitions": partitions
                },
                message=f"Disk usage for {path}: {round((usage.used / usage.total) * 100, 2)}% used"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to get disk usage: {e}")
    
    async def _get_memory_info(self, params: Dict[str, Any]) -> ToolResult:
        """Get memory usage information."""
        try:
            # Virtual memory
            virtual_mem = psutil.virtual_memory()
            
            # Swap memory
            swap_mem = psutil.swap_memory()
            
            return ToolResult(
                success=True,
                data={
                    "virtual_memory": {
                        "total": virtual_mem.total,
                        "available": virtual_mem.available,
                        "used": virtual_mem.used,
                        "free": virtual_mem.free,
                        "percent": virtual_mem.percent
                    },
                    "swap_memory": {
                        "total": swap_mem.total,
                        "used": swap_mem.used,
                        "free": swap_mem.free,
                        "percent": swap_mem.percent
                    }
                },
                message=f"Memory usage: {virtual_mem.percent}%, Swap: {swap_mem.percent}%"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to get memory info: {e}")
    
    def _is_command_safe(self, command: str) -> bool:
        """Check if command is in allowed list."""
        # Extract base command
        base_command = command.split()[0] if command else ""
        
        # Check against forbidden commands
        if base_command in self.forbidden_commands:
            return False
        
        # Check against allowed commands (if allowlist is used)
        if self.allowed_commands and base_command not in self.allowed_commands:
            return False
        
        return True
    
    def _contains_dangerous_patterns(self, command: str) -> bool:
        """Check if command contains dangerous patterns."""
        import re
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command):
                return True
        
        return False
    
    def _is_valid_hostname(self, hostname: str) -> bool:
        """Basic hostname validation."""
        import re
        
        # Basic hostname pattern
        hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        
        # IP address pattern
        ip_pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        
        return bool(re.match(hostname_pattern, hostname) or re.match(ip_pattern, hostname))