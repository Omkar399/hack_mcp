"""
Performance Monitoring for Eidolon

Handles system performance monitoring, resource usage tracking,
and alerting for resource limit violations.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import psutil

from .logging import get_component_logger


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.cpu_percent = 0.0
        self.memory_mb = 0.0
        self.disk_usage_percent = 0.0
        self.network_bytes_sent = 0
        self.network_bytes_recv = 0
        self.capture_rate = 0.0
        self.error_count = 0


class PerformanceMonitor:
    """
    Performance monitoring system for tracking resource usage and system health.
    """
    
    def __init__(self, collection_interval: int = 60):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: Seconds between metric collections.
        """
        self.collection_interval = collection_interval
        self.logger = get_component_logger("monitoring")
        
        # State management
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history = 1440  # 24 hours worth at 1-minute intervals
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, PerformanceMetrics], None]] = []
        
        # Alert thresholds (will be loaded from config)
        self._thresholds = {
            "cpu_percent": 10.0,
            "memory_mb": 1000.0,
            "disk_usage_percent": 90.0
        }
        
        self.logger.info("Performance monitor initialized")
    
    def add_alert_callback(self, callback: Callable[[str, PerformanceMetrics], None]) -> None:
        """Add callback for performance alerts."""
        self._alert_callbacks.append(callback)
        self.logger.debug(f"Added alert callback: {callback.__name__}")
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update alert thresholds."""
        self._thresholds.update(thresholds)
        self.logger.info(f"Updated alert thresholds: {thresholds}")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._running:
            self.logger.warning("Performance monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="eidolon-performance-monitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._running:
            self.logger.warning("Performance monitor not running")
            return
        
        self._running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                self.logger.warning("Monitor thread did not stop gracefully")
        
        self._monitor_thread = None
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.debug("Performance monitor loop started")
        
        while self._running:
            try:
                # Check if we should stop before doing work
                if not self._running:
                    break
                
                # Collect metrics
                metrics = self._collect_metrics()
                
                if not self._running:
                    break
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next collection, checking for stop signal
                sleep_intervals = max(1, int(self.collection_interval))
                sleep_duration = self.collection_interval / sleep_intervals
                
                for _ in range(sleep_intervals):
                    if not self._running:
                        break
                    time.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                # Sleep with interrupt capability
                for _ in range(int(self.collection_interval)):
                    if not self._running:
                        break
                    time.sleep(1)
        
        self.logger.debug("Performance monitor loop ended")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics()
        
        try:
            # Get current process
            process = psutil.Process()
            
            # CPU usage (non-blocking for faster shutdown)
            metrics.cpu_percent = process.cpu_percent(interval=None)
            
            # Memory usage
            memory_info = process.memory_info()
            metrics.memory_mb = memory_info.rss / 1024 / 1024
            
            # Disk usage (for the data directory)
            disk_usage = psutil.disk_usage('.')
            metrics.disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.network_bytes_sent = net_io.bytes_sent
                metrics.network_bytes_recv = net_io.bytes_recv
            
            self.logger.debug(
                f"Metrics collected - CPU: {metrics.cpu_percent:.1f}%, "
                f"Memory: {metrics.memory_mb:.1f}MB, "
                f"Disk: {metrics.disk_usage_percent:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in history."""
        self._metrics_history.append(metrics)
        
        # Trim history to maximum size
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        # Check CPU usage (raised threshold for testing)
        if metrics.cpu_percent > self._thresholds.get("cpu_percent", 80.0):
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Check memory usage
        if metrics.memory_mb > self._thresholds.get("memory_mb", 1000.0):
            alerts.append(f"High memory usage: {metrics.memory_mb:.1f}MB")
        
        # Check disk usage
        if metrics.disk_usage_percent > self._thresholds.get("disk_usage_percent", 90.0):
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # Send alerts
        for alert_message in alerts:
            self.logger.warning(f"Performance alert: {alert_message}")
            
            for callback in self._alert_callbacks:
                try:
                    callback(alert_message, metrics)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[PerformanceMetrics]:
        """
        Get metrics history for the specified time period.
        
        Args:
            hours: Number of hours of history to return.
            
        Returns:
            List[PerformanceMetrics]: Metrics within the time period.
        """
        if not self._metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self._metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_summary_stats(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get summary statistics for the specified time period.
        
        Args:
            hours: Number of hours to summarize.
            
        Returns:
            Dict: Summary statistics.
        """
        history = self.get_metrics_history(hours)
        
        if not history:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_mb for m in history]
        disk_values = [m.disk_usage_percent for m in history]
        
        return {
            "period_hours": hours,
            "sample_count": len(history),
            "cpu_percent": {
                "average": sum(cpu_values) / len(cpu_values),
                "peak": max(cpu_values),
                "minimum": min(cpu_values)
            },
            "memory_mb": {
                "average": sum(memory_values) / len(memory_values),
                "peak": max(memory_values),
                "minimum": min(memory_values)
            },
            "disk_usage_percent": {
                "average": sum(disk_values) / len(disk_values),
                "peak": max(disk_values),
                "minimum": min(disk_values)
            }
        }
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform a health check of the system.
        
        Returns:
            Dict: Health check results.
        """
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return {
                "status": "unknown",
                "message": "No metrics available"
            }
        
        # Determine overall health status
        issues = []
        
        if current_metrics.cpu_percent > self._thresholds.get("cpu_percent", 10.0):
            issues.append(f"High CPU usage: {current_metrics.cpu_percent:.1f}%")
        
        if current_metrics.memory_mb > self._thresholds.get("memory_mb", 1000.0):
            issues.append(f"High memory usage: {current_metrics.memory_mb:.1f}MB")
        
        if current_metrics.disk_usage_percent > self._thresholds.get("disk_usage_percent", 90.0):
            issues.append(f"High disk usage: {current_metrics.disk_usage_percent:.1f}%")
        
        if issues:
            status = "warning" if len(issues) <= 2 else "critical"
            message = "; ".join(issues)
        else:
            status = "healthy"
            message = "All metrics within normal ranges"
        
        return {
            "status": status,
            "message": message,
            "metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_mb": current_metrics.memory_mb,
                "disk_usage_percent": current_metrics.disk_usage_percent
            },
            "thresholds": self._thresholds.copy()
        }