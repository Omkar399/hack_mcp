"""
Production monitoring and alerting system for Eidolon
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from pathlib import Path
import json

from .logging import get_component_logger
from .config import get_config


@dataclass
class AlertRule:
    """Defines an alert rule for monitoring."""
    name: str
    metric: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte'
    duration_minutes: int = 5  # How long condition must persist
    callback: Optional[Callable] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_usage_percent: float
    active_processes: int
    eidolon_memory_mb: float
    eidolon_cpu_percent: float


class ProductionMonitor:
    """Production monitoring system with alerting."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("production_monitor")
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.metrics_history: List[SystemMetrics] = []
        self.alert_states: Dict[str, Dict] = {}
        
        # Configuration
        self.monitoring_config = self.config.monitoring
        self.collection_interval = getattr(self.monitoring_config, "metrics_collection_interval", 60)
        
        # Get alert thresholds
        alert_thresholds = getattr(self.monitoring_config, "alert_thresholds", {})
        
        # Default alert rules
        self.alert_rules = [
            AlertRule(
                name="high_cpu",
                metric="cpu_percent",
                threshold=getattr(alert_thresholds, "cpu_percent", 80),
                operator="gt",
                duration_minutes=5
            ),
            AlertRule(
                name="high_memory",
                metric="memory_mb",
                threshold=getattr(alert_thresholds, "memory_mb", 8192),
                operator="gt",
                duration_minutes=3
            ),
            AlertRule(
                name="high_disk",
                metric="disk_usage_percent",
                threshold=getattr(alert_thresholds, "disk_usage_percent", 90),
                operator="gt",
                duration_minutes=10
            )
        ]
        
        self.logger.info("Production monitor initialized")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            self.logger.warning("Monitor already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Production monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history (keep last 24 hours)
                max_history = int(24 * 60 * 60 / self.collection_interval)
                if len(self.metrics_history) > max_history:
                    self.metrics_history = self.metrics_history[-max_history:]
                
                # Check alert rules
                self._check_alerts(metrics)
                
                # Log metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 collections
                    self._log_metrics_summary(metrics)
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Brief pause on error
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # System-wide metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Eidolon process metrics
        eidolon_memory = 0
        eidolon_cpu = 0
        
        try:
            # Find Eidolon processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any('eidolon' in str(cmd).lower() for cmd in proc.info['cmdline']):
                        proc_obj = psutil.Process(proc.info['pid'])
                        eidolon_memory += proc_obj.memory_info().rss / 1024 / 1024  # MB
                        eidolon_cpu += proc_obj.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.debug(f"Error collecting Eidolon process metrics: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory.used / 1024 / 1024,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            active_processes=len(psutil.pids()),
            eidolon_memory_mb=eidolon_memory,
            eidolon_cpu_percent=eidolon_cpu
        )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check alert rules against current metrics."""
        for rule in self.alert_rules:
            metric_value = getattr(metrics, rule.metric, 0)
            
            # Check threshold
            alert_condition = False
            if rule.operator == "gt":
                alert_condition = metric_value > rule.threshold
            elif rule.operator == "lt":
                alert_condition = metric_value < rule.threshold
            elif rule.operator == "gte":
                alert_condition = metric_value >= rule.threshold
            elif rule.operator == "lte":
                alert_condition = metric_value <= rule.threshold
            
            # Track alert state
            if rule.name not in self.alert_states:
                self.alert_states[rule.name] = {
                    "active": False,
                    "start_time": None,
                    "last_triggered": None
                }
            
            state = self.alert_states[rule.name]
            
            if alert_condition:
                if not state["active"]:
                    # Start tracking this alert
                    state["start_time"] = metrics.timestamp
                    state["active"] = True
                else:
                    # Check if duration threshold met
                    duration = (metrics.timestamp - state["start_time"]).total_seconds() / 60
                    if duration >= rule.duration_minutes:
                        # Trigger alert (but not repeatedly)
                        if (not state["last_triggered"] or 
                            (metrics.timestamp - state["last_triggered"]).total_seconds() > 300):  # 5 min cooldown
                            self._trigger_alert(rule, metric_value, metrics)
                            state["last_triggered"] = metrics.timestamp
            else:
                if state["active"]:
                    # Clear alert
                    state["active"] = False
                    state["start_time"] = None
                    self.logger.info(f"Alert cleared: {rule.name}")
    
    def _trigger_alert(self, rule: AlertRule, value: float, metrics: SystemMetrics):
        """Trigger an alert."""
        message = f"ALERT: {rule.name} - {rule.metric} is {value:.2f} (threshold: {rule.threshold})"
        self.logger.warning(message)
        
        # Call custom callback if provided
        if rule.callback:
            try:
                rule.callback(rule, value, metrics)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Store alert in logs
        self._log_alert(rule, value, metrics)
    
    def _log_alert(self, rule: AlertRule, value: float, metrics: SystemMetrics):
        """Log alert to file for tracking."""
        try:
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            
            alert_log = log_dir / "alerts.json"
            
            alert_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "alert_name": rule.name,
                "metric": rule.metric,
                "value": value,
                "threshold": rule.threshold,
                "system_state": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_mb": metrics.memory_mb,
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "eidolon_memory_mb": metrics.eidolon_memory_mb
                }
            }
            
            # Append to alerts log
            alerts = []
            if alert_log.exists():
                try:
                    with open(alert_log, 'r') as f:
                        alerts = json.load(f)
                except json.JSONDecodeError:
                    alerts = []
            
            alerts.append(alert_data)
            
            # Keep only last 100 alerts
            alerts = alerts[-100:]
            
            with open(alert_log, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
    
    def _log_metrics_summary(self, metrics: SystemMetrics):
        """Log periodic metrics summary."""
        self.logger.info(
            f"System metrics - CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_mb:.0f}MB ({metrics.memory_percent:.1f}%), "
            f"Disk: {metrics.disk_usage_percent:.1f}%, "
            f"Eidolon Memory: {metrics.eidolon_memory_mb:.1f}MB"
        )
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
        
        # Filter recent metrics
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No metrics found for last {hours} hours"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_mb for m in recent_metrics]
        eidolon_memory_values = [m.eidolon_memory_mb for m in recent_metrics]
        
        return {
            "period": f"Last {hours} hours",
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "avg_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "min_mb": min(memory_values)
            },
            "eidolon_memory": {
                "avg_mb": sum(eidolon_memory_values) / len(eidolon_memory_values),
                "max_mb": max(eidolon_memory_values),
                "current_mb": recent_metrics[-1].eidolon_memory_mb
            },
            "active_alerts": [name for name, state in self.alert_states.items() if state["active"]]
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.metrics_history:
            return {"status": "unknown", "reason": "No metrics available"}
        
        latest = self.metrics_history[-1]
        active_alerts = [name for name, state in self.alert_states.items() if state["active"]]
        
        # Determine health status
        if active_alerts:
            status = "unhealthy"
            reason = f"Active alerts: {', '.join(active_alerts)}"
        elif latest.cpu_percent > 90 or latest.memory_percent > 95:
            status = "degraded"
            reason = "High resource usage"
        else:
            status = "healthy"
            reason = "All systems normal"
        
        return {
            "status": status,
            "reason": reason,
            "timestamp": latest.timestamp.isoformat(),
            "active_alerts": active_alerts,
            "uptime_minutes": len(self.metrics_history) * (self.collection_interval / 60),
            "resource_usage": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_usage_percent
            }
        }


# Global monitor instance
_monitor_instance = None

def get_monitor() -> ProductionMonitor:
    """Get the global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProductionMonitor()
    return _monitor_instance

def start_monitoring():
    """Start production monitoring."""
    monitor = get_monitor()
    monitor.start_monitoring()

def stop_monitoring():
    """Stop production monitoring."""
    monitor = get_monitor()
    monitor.stop_monitoring()