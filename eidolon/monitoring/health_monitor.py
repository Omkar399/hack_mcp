"""
Health Monitor for Eidolon AI Personal Assistant

Comprehensive health monitoring with automatic recovery and alerting.
"""

import os
import psutil
import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import threading

from ..utils.logging import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: Optional[datetime] = None
    details: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}


@dataclass
class SystemHealth:
    """Overall system health summary."""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    uptime_seconds: float
    issues_count: int
    warnings_count: int


class HealthMonitor:
    """Monitors system health and performance metrics."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.eidolon_home = Path(os.getenv('EIDOLON_HOME', '~/.eidolon')).expanduser()
        
        # Monitoring configuration
        monitor_config = self.config.get('monitoring', {})
        self.enabled = monitor_config.get('enabled', True)
        self.check_interval = monitor_config.get('metrics_collection_interval', 60)
        
        # Thresholds
        thresholds = monitor_config.get('alert_thresholds', {})
        self.cpu_threshold = thresholds.get('cpu_percent', 75.0)
        self.memory_threshold_mb = thresholds.get('memory_mb', 8192)
        self.disk_threshold = thresholds.get('disk_usage_percent', 85)
        self.response_time_threshold = thresholds.get('response_time_ms', 5000)
        
        # Health database
        self.db_path = self.eidolon_home / 'data' / 'health.db'
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Monitoring state
        self.start_time = datetime.now()
        self.last_check_time = None
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Initialize database
        self._init_database()
        
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
            
        if not self.enabled:
            logger.info("Health monitoring is disabled")
            return
            
        logger.info("Starting health monitoring...")
        
        self.monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
            
        logger.info("Stopping health monitoring...")
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            
    def get_current_health(self) -> SystemHealth:
        """
        Get current system health status.
        
        Returns:
            SystemHealth object with current status
        """
        try:
            checks = []
            
            # CPU usage check
            cpu_check = self._check_cpu_usage()
            checks.append(cpu_check)
            
            # Memory usage check
            memory_check = self._check_memory_usage()
            checks.append(memory_check)
            
            # Disk usage check
            disk_check = self._check_disk_usage()
            checks.append(disk_check)
            
            # Process check
            process_check = self._check_eidolon_process()
            checks.append(process_check)
            
            # Database check
            db_check = self._check_database_health()
            checks.append(db_check)
            
            # Log check
            log_check = self._check_log_health()
            checks.append(log_check)
            
            # Response time check
            response_check = self._check_response_time()
            checks.append(response_check)
            
            # Determine overall status
            critical_checks = [c for c in checks if c.status == HealthStatus.CRITICAL]
            warning_checks = [c for c in checks if c.status == HealthStatus.WARNING]
            
            if critical_checks:
                overall_status = HealthStatus.CRITICAL
            elif warning_checks:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
                
            # Calculate uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            health = SystemHealth(
                overall_status=overall_status,
                checks=checks,
                timestamp=datetime.now(),
                uptime_seconds=uptime,
                issues_count=len(critical_checks),
                warnings_count=len(warning_checks)
            )
            
            # Store health data
            self._store_health_data(health)
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting current health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                checks=[HealthCheck("system", HealthStatus.CRITICAL, f"Health check failed: {e}")],
                timestamp=datetime.now(),
                uptime_seconds=0,
                issues_count=1,
                warnings_count=0
            )
            
    def get_health_history(self, hours: int = 24) -> List[Dict]:
        """
        Get health history for the specified time period.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of health records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                since = datetime.now() - timedelta(hours=hours)
                
                cursor.execute("""
                    SELECT * FROM health_records 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, (since.isoformat(),))
                
                records = []
                for row in cursor.fetchall():
                    record = dict(row)
                    if record['checks_json']:
                        record['checks'] = json.loads(record['checks_json'])
                    records.append(record)
                    
                return records
                
        except Exception as e:
            logger.error(f"Error getting health history: {e}")
            return []
            
    def get_performance_stats(self, hours: int = 24) -> Dict:
        """
        Get performance statistics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with performance statistics
        """
        try:
            history = self.get_health_history(hours)
            if not history:
                return {}
                
            cpu_values = []
            memory_values = []
            disk_values = []
            response_times = []
            
            for record in history:
                checks = record.get('checks', [])
                for check in checks:
                    if check['name'] == 'cpu_usage' and check['value'] is not None:
                        cpu_values.append(check['value'])
                    elif check['name'] == 'memory_usage' and check['value'] is not None:
                        memory_values.append(check['value'])
                    elif check['name'] == 'disk_usage' and check['value'] is not None:
                        disk_values.append(check['value'])
                    elif check['name'] == 'response_time' and check['value'] is not None:
                        response_times.append(check['value'])
                        
            def calc_stats(values):
                if not values:
                    return {'min': 0, 'max': 0, 'avg': 0, 'current': 0}
                return {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'current': values[0] if values else 0
                }
                
            return {
                'cpu_percent': calc_stats(cpu_values),
                'memory_mb': calc_stats(memory_values),
                'disk_percent': calc_stats(disk_values),
                'response_time_ms': calc_stats(response_times),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'data_points': len(history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {}
            
    def run_health_check(self) -> SystemHealth:
        """
        Run a complete health check and return results.
        
        Returns:
            SystemHealth object
        """
        health = self.get_current_health()
        self.last_check_time = datetime.now()
        
        # Log health status
        status_msg = f"Health check: {health.overall_status.value}"
        if health.issues_count > 0:
            status_msg += f" ({health.issues_count} critical, {health.warnings_count} warnings)"
            
        if health.overall_status == HealthStatus.CRITICAL:
            logger.error(status_msg)
        elif health.overall_status == HealthStatus.WARNING:
            logger.warning(status_msg)
        else:
            logger.info(status_msg)
            
        return health
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(f"Health monitoring started (interval: {self.check_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                self.run_health_check()
                
                # Wait for next check
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(10)  # Wait a bit before retrying
                
        logger.info("Health monitoring stopped")
        
    def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.CRITICAL
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > self.cpu_threshold * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
                
            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                value=cpu_percent,
                threshold=self.cpu_threshold
            )
            
        except Exception as e:
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.CRITICAL,
                message=f"CPU check failed: {e}"
            )
            
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            if memory_mb > self.memory_threshold_mb:
                status = HealthStatus.CRITICAL
                message = f"High memory usage: {memory_mb:.0f}MB"
            elif memory_mb > self.memory_threshold_mb * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated memory usage: {memory_mb:.0f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.0f}MB"
                
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                value=memory_mb,
                threshold=self.memory_threshold_mb,
                details={
                    'total_mb': memory.total / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'percent': memory.percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {e}"
            )
            
    def _check_disk_usage(self) -> HealthCheck:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage(str(self.eidolon_home))
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > self.disk_threshold:
                status = HealthStatus.CRITICAL
                message = f"High disk usage: {disk_percent:.1f}%"
            elif disk_percent > self.disk_threshold * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
                
            return HealthCheck(
                name="disk_usage",
                status=status,
                message=message,
                value=disk_percent,
                threshold=self.disk_threshold,
                details={
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_usage",
                status=HealthStatus.CRITICAL,
                message=f"Disk check failed: {e}"
            )
            
    def _check_eidolon_process(self) -> HealthCheck:
        """Check if Eidolon process is running."""
        try:
            eidolon_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if 'eidolon' in cmdline.lower() and 'capture' in cmdline:
                            eidolon_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            if eidolon_processes:
                status = HealthStatus.HEALTHY
                message = f"Eidolon process running (PID: {eidolon_processes[0].pid})"
                details = {
                    'pid': eidolon_processes[0].pid,
                    'process_count': len(eidolon_processes)
                }
            else:
                status = HealthStatus.CRITICAL
                message = "Eidolon process not found"
                details = {}
                
            return HealthCheck(
                name="eidolon_process",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="eidolon_process",
                status=HealthStatus.CRITICAL,
                message=f"Process check failed: {e}"
            )
            
    def _check_database_health(self) -> HealthCheck:
        """Check database health."""
        try:
            db_path = self.eidolon_home / 'data' / 'eidolon.db'
            
            if not db_path.exists():
                return HealthCheck(
                    name="database",
                    status=HealthStatus.CRITICAL,
                    message="Database file not found"
                )
                
            # Try to connect and run a simple query
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
            # Check database size
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message=f"Database healthy ({table_count} tables, {db_size_mb:.1f}MB)",
                details={
                    'table_count': table_count,
                    'size_mb': db_size_mb,
                    'path': str(db_path)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {e}"
            )
            
    def _check_log_health(self) -> HealthCheck:
        """Check log file health."""
        try:
            log_path = self.eidolon_home / 'logs' / 'eidolon.log'
            
            if not log_path.exists():
                return HealthCheck(
                    name="logs",
                    status=HealthStatus.WARNING,
                    message="Log file not found"
                )
                
            # Check log file size
            log_size_mb = log_path.stat().st_size / (1024 * 1024)
            
            # Count recent errors
            error_count = 0
            try:
                with open(log_path, 'r') as f:
                    # Read last 1000 lines
                    lines = f.readlines()[-1000:]
                    error_count = sum(1 for line in lines if 'ERROR' in line)
            except Exception:
                pass
                
            if error_count > 50:
                status = HealthStatus.CRITICAL
                message = f"High error count in logs: {error_count}"
            elif error_count > 10:
                status = HealthStatus.WARNING
                message = f"Some errors in logs: {error_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Logs healthy ({error_count} recent errors)"
                
            return HealthCheck(
                name="logs",
                status=status,
                message=message,
                details={
                    'size_mb': log_size_mb,
                    'error_count': error_count,
                    'path': str(log_path)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="logs",
                status=HealthStatus.CRITICAL,
                message=f"Log check failed: {e}"
            )
            
    def _check_response_time(self) -> HealthCheck:
        """Check system response time."""
        try:
            start_time = time.time()
            
            # Try to run a simple Eidolon command
            try:
                venv_path = self.eidolon_home / 'venv'
                python_path = venv_path / 'bin' / 'python'
                
                if not python_path.exists():
                    python_path = venv_path / 'Scripts' / 'python.exe'  # Windows
                    
                result = subprocess.run(
                    [str(python_path), '-m', 'eidolon', 'status'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                response_time_ms = (time.time() - start_time) * 1000
                
                if response_time_ms > self.response_time_threshold:
                    status = HealthStatus.WARNING
                    message = f"Slow response time: {response_time_ms:.0f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Response time good: {response_time_ms:.0f}ms"
                    
            except subprocess.TimeoutExpired:
                response_time_ms = self.response_time_threshold * 2
                status = HealthStatus.CRITICAL
                message = "Command timeout"
            except Exception:
                response_time_ms = (time.time() - start_time) * 1000
                status = HealthStatus.WARNING
                message = f"Command failed but responsive: {response_time_ms:.0f}ms"
                
            return HealthCheck(
                name="response_time",
                status=status,
                message=message,
                value=response_time_ms,
                threshold=self.response_time_threshold
            )
            
        except Exception as e:
            return HealthCheck(
                name="response_time",
                status=HealthStatus.CRITICAL,
                message=f"Response time check failed: {e}"
            )
            
    def _init_database(self) -> None:
        """Initialize health monitoring database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        overall_status TEXT NOT NULL,
                        uptime_seconds REAL NOT NULL,
                        issues_count INTEGER NOT NULL,
                        warnings_count INTEGER NOT NULL,
                        checks_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_health_timestamp 
                    ON health_records(timestamp)
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing health database: {e}")
            
    def _store_health_data(self, health: SystemHealth) -> None:
        """Store health data in database."""
        try:
            checks_json = json.dumps([
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'value': check.value,
                    'threshold': check.threshold,
                    'timestamp': check.timestamp.isoformat() if check.timestamp else None,
                    'details': check.details
                }
                for check in health.checks
            ])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO health_records 
                    (timestamp, overall_status, uptime_seconds, issues_count, warnings_count, checks_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    health.timestamp.isoformat(),
                    health.overall_status.value,
                    health.uptime_seconds,
                    health.issues_count,
                    health.warnings_count,
                    checks_json
                ))
                
                # Clean up old records (keep last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                conn.execute("DELETE FROM health_records WHERE timestamp < ?", (cutoff.isoformat(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing health data: {e}")