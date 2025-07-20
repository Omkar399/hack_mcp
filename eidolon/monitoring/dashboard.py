"""
Monitoring Dashboard for Eidolon AI Personal Assistant

Provides web-based monitoring dashboard with real-time metrics.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.config import get_config
from .health_monitor import HealthMonitor, HealthStatus

logger = get_logger(__name__)


class MonitoringDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.eidolon_home = Path(os.getenv('EIDOLON_HOME', '~/.eidolon')).expanduser()
        
        # Dashboard configuration
        self.enabled = self.config.get('monitoring', {}).get('dashboard_enabled', True)
        self.host = self.config.get('monitoring', {}).get('dashboard_host', 'localhost')
        self.port = self.config.get('monitoring', {}).get('dashboard_port', 8080)
        
        # Health monitor
        self.health_monitor = HealthMonitor(config_path)
        
        # Dashboard state
        self.app = None
        self.server_thread = None
        self.running = False
        self.websocket_connections = set()
        
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available - dashboard functionality disabled")
            self.enabled = False
            
    def start_dashboard(self) -> bool:
        """
        Start the monitoring dashboard.
        
        Returns:
            True if started successfully
        """
        if not self.enabled:
            logger.info("Monitoring dashboard is disabled")
            return False
            
        if self.running:
            logger.warning("Dashboard already running")
            return True
            
        try:
            logger.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
            
            # Create FastAPI app
            self.app = self._create_app()
            
            # Start server in separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return False
            
    def stop_dashboard(self) -> None:
        """Stop the monitoring dashboard."""
        if not self.running:
            return
            
        logger.info("Stopping monitoring dashboard...")
        self.running = False
        
        # Close websocket connections
        for websocket in self.websocket_connections.copy():
            try:
                asyncio.create_task(websocket.close())
            except Exception:
                pass
                
        self.websocket_connections.clear()
        
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.host}:{self.port}"
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Eidolon Monitoring Dashboard",
            description="Real-time monitoring dashboard for Eidolon AI Personal Assistant",
            version="1.0.0"
        )
        
        # Setup templates
        templates_dir = Path(__file__).parent / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        # Create basic template if it doesn't exist
        template_file = templates_dir / 'dashboard.html'
        if not template_file.exists():
            self._create_dashboard_template(template_file)
            
        templates = Jinja2Templates(directory=str(templates_dir))
        
        # Routes
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            return templates.TemplateResponse("dashboard.html", {"request": request})
            
        @app.get("/api/health")
        async def get_health():
            health = self.health_monitor.get_current_health()
            return {
                "status": health.overall_status.value,
                "timestamp": health.timestamp.isoformat(),
                "uptime_seconds": health.uptime_seconds,
                "issues_count": health.issues_count,
                "warnings_count": health.warnings_count,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "value": check.value,
                        "threshold": check.threshold,
                        "details": check.details
                    }
                    for check in health.checks
                ]
            }
            
        @app.get("/api/health/history")
        async def get_health_history(hours: int = 24):
            history = self.health_monitor.get_health_history(hours)
            return {"history": history}
            
        @app.get("/api/performance")
        async def get_performance():
            stats = self.health_monitor.get_performance_stats()
            return stats
            
        @app.get("/api/system")
        async def get_system_info():
            return {
                "eidolon_home": str(self.eidolon_home),
                "dashboard_url": self.get_dashboard_url(),
                "monitoring_enabled": self.health_monitor.enabled,
                "start_time": self.health_monitor.start_time.isoformat()
            }
            
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                # Send initial data
                health = self.health_monitor.get_current_health()
                await websocket.send_json({
                    "type": "health_update",
                    "data": {
                        "status": health.overall_status.value,
                        "timestamp": health.timestamp.isoformat(),
                        "uptime_seconds": health.uptime_seconds,
                        "checks": [
                            {
                                "name": check.name,
                                "status": check.status.value,
                                "message": check.message,
                                "value": check.value
                            }
                            for check in health.checks
                        ]
                    }
                })
                
                # Keep connection alive and send updates
                while True:
                    await asyncio.sleep(30)  # Send updates every 30 seconds
                    
                    if websocket.client_state.value == 3:  # DISCONNECTED
                        break
                        
                    health = self.health_monitor.get_current_health()
                    await websocket.send_json({
                        "type": "health_update",
                        "data": {
                            "status": health.overall_status.value,
                            "timestamp": health.timestamp.isoformat(),
                            "uptime_seconds": health.uptime_seconds,
                            "checks": [
                                {
                                    "name": check.name,
                                    "status": check.status.value,
                                    "message": check.message,
                                    "value": check.value
                                }
                                for check in health.checks
                            ]
                        }
                    })
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
                
        return app
        
    def _run_server(self) -> None:
        """Run the FastAPI server."""
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="error"  # Reduce uvicorn logging
            )
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            self.running = False
            
    def _create_dashboard_template(self, template_file: Path) -> None:
        """Create basic dashboard HTML template."""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eidolon Monitoring Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-healthy { border-left: 4px solid #28a745; }
        .status-warning { border-left: 4px solid #ffc107; }
        .status-critical { border-left: 4px solid #dc3545; }
        .status-unknown { border-left: 4px solid #6c757d; }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .health-checks {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .check-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .check-item:last-child {
            border-bottom: none;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-badge.healthy { background: #d4edda; color: #155724; }
        .status-badge.warning { background: #fff3cd; color: #856404; }
        .status-badge.critical { background: #f8d7da; color: #721c24; }
        .status-badge.unknown { background: #e2e3e5; color: #383d41; }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 4px;
            font-weight: bold;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .auto-refresh {
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Disconnected</div>
    
    <div class="container">
        <div class="header">
            <h1>Eidolon Monitoring Dashboard</h1>
            <p>Real-time system health and performance monitoring</p>
            <div id="lastUpdate">Last update: Never</div>
        </div>
        
        <div class="status-grid">
            <div class="status-card" id="overallStatus">
                <div class="metric-label">Overall Status</div>
                <div class="metric-value" id="statusValue">Unknown</div>
                <div id="uptimeValue">Uptime: Unknown</div>
            </div>
            
            <div class="status-card">
                <div class="metric-label">Issues</div>
                <div class="metric-value" id="issuesValue">0</div>
                <div id="warningsValue">0 warnings</div>
            </div>
        </div>
        
        <div class="health-checks">
            <h2>Health Checks</h2>
            <div id="healthChecks">
                <div class="check-item">
                    <span>Loading...</span>
                    <span class="status-badge unknown">Unknown</span>
                </div>
            </div>
        </div>
        
        <div class="auto-refresh">
            Dashboard updates automatically every 30 seconds
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectInterval = null;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('connectionStatus').textContent = 'Connected';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'health_update') {
                    updateHealthData(message.data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                
                // Attempt to reconnect
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connect, 5000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateHealthData(data) {
            // Update overall status
            const statusCard = document.getElementById('overallStatus');
            statusCard.className = `status-card status-${data.status}`;
            document.getElementById('statusValue').textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            
            // Update uptime
            const uptimeHours = Math.floor(data.uptime_seconds / 3600);
            const uptimeMinutes = Math.floor((data.uptime_seconds % 3600) / 60);
            document.getElementById('uptimeValue').textContent = `Uptime: ${uptimeHours}h ${uptimeMinutes}m`;
            
            // Update issues and warnings
            const issuesCount = data.checks.filter(c => c.status === 'critical').length;
            const warningsCount = data.checks.filter(c => c.status === 'warning').length;
            document.getElementById('issuesValue').textContent = issuesCount;
            document.getElementById('warningsValue').textContent = `${warningsCount} warnings`;
            
            // Update health checks
            const checksContainer = document.getElementById('healthChecks');
            checksContainer.innerHTML = '';
            
            data.checks.forEach(check => {
                const checkItem = document.createElement('div');
                checkItem.className = 'check-item';
                
                let displayMessage = check.message;
                if (check.value !== null) {
                    displayMessage += ` (${check.value})`;
                }
                
                checkItem.innerHTML = `
                    <span>${check.name}: ${displayMessage}</span>
                    <span class="status-badge ${check.status}">${check.status}</span>
                `;
                
                checksContainer.appendChild(checkItem);
            });
            
            // Update last update time
            document.getElementById('lastUpdate').textContent = `Last update: ${new Date(data.timestamp).toLocaleString()}`;
        }
        
        // Initialize connection
        connect();
        
        // Fallback: refresh every 60 seconds if WebSocket fails
        setInterval(async function() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    updateHealthData(data);
                } catch (error) {
                    console.error('Failed to fetch health data:', error);
                }
            }
        }, 60000);
    </script>
</body>
</html>'''
        
        template_file.write_text(html_content)
        logger.info(f"Created dashboard template: {template_file}")


def create_monitoring_dashboard(config_path: Optional[str] = None) -> Optional[MonitoringDashboard]:
    """
    Create and start a monitoring dashboard.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MonitoringDashboard instance or None if not available
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - cannot create dashboard")
        return None
        
    dashboard = MonitoringDashboard(config_path)
    
    if dashboard.start_dashboard():
        return dashboard
    else:
        return None