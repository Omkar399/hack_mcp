#!/bin/bash

#
# Eidolon AI Personal Assistant - Background Monitoring Service
# Continuous screenshot capture with integrated AI processing
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/eidolon_env/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/data/pids"
MONITOR_LOG="${LOG_DIR}/monitoring.log"
PID_FILE="${PID_DIR}/eidolon-monitor.pid"

# Service configuration
SERVICE_NAME="eidolon-monitor"
CAPTURE_INTERVAL=10
MAX_RETRIES=5
RETRY_DELAY=30
HEALTH_CHECK_INTERVAL=60

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$MONITOR_LOG"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
print_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if Python environment exists
    if [[ ! -f "$PYTHON_BIN" ]]; then
        log_error "Python environment not found at $PYTHON_BIN"
        print_error "Please run: python -m venv eidolon_env && source eidolon_env/bin/activate && pip install -e ."
        exit 1
    fi
    
    # Check if eidolon module can be imported
    if ! "$PYTHON_BIN" -c "import eidolon" 2>/dev/null; then
        log_error "Eidolon module not installed properly"
        print_error "Please run: pip install -e ."
        exit 1
    fi
    
    # Check required directories
    mkdir -p "$LOG_DIR" "$PID_DIR"
    
    log_success "Dependencies check passed"
}

# Check if service is already running
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            # Stale PID file
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Get service status
get_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        local uptime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo "unknown")
        print_success "Service is running (PID: $pid, Uptime: $uptime)"
        
        # Show recent activity
        if [[ -f "$MONITOR_LOG" ]]; then
            echo
            echo "Recent activity:"
            tail -n 5 "$MONITOR_LOG" | while read -r line; do
                echo "  $line"
            done
        fi
        
        # Show resource usage
        local mem_usage=$(ps -o pid,rss,vsz -p "$pid" 2>/dev/null | tail -n 1 | awk '{print $2/1024 " MB"}' || echo "unknown")
        local cpu_usage=$(ps -o pid,pcpu -p "$pid" 2>/dev/null | tail -n 1 | awk '{print $2 "%"}' || echo "unknown")
        echo "Resource usage: CPU: $cpu_usage, Memory: $mem_usage"
        
        return 0
    else
        print_warn "Service is not running"
        return 1
    fi
}

# Start the monitoring service
start_service() {
    log_info "Starting $SERVICE_NAME..."
    
    if is_running; then
        print_warn "Service is already running"
        get_status
        return 0
    fi
    
    check_dependencies
    
    # Create monitoring command
    local monitor_cmd="$PYTHON_BIN -c \"
import sys
import os
import time
import signal
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '$PROJECT_ROOT')

from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer
from eidolon.storage.metadata_db import MetadataDatabase
from eidolon.storage.vector_db import VectorDatabase
from eidolon.utils.logging import setup_logging
from eidolon.utils.config import get_config

# Setup logging
setup_logging()
logger = logging.getLogger('eidolon.monitor')

# Global state
observer = None
running = True

def signal_handler(signum, frame):
    global running, observer
    logger.info(f'Received signal {signum}, shutting down gracefully...')
    running = False
    if observer:
        observer.stop_monitoring()
    sys.exit(0)

def main():
    global observer, running
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info('Starting Eidolon background monitoring service')
        
        # Initialize components
        config = get_config()
        observer = Observer()
        analyzer = Analyzer()
        metadata_db = MetadataDatabase()
        vector_db = VectorDatabase()
        
        # Setup automatic processing callback
        def process_screenshot(screenshot):
            try:
                # This is already handled in observer._save_and_process_screenshot
                # but we can add additional processing here if needed
                logger.debug(f'Processing screenshot: {screenshot.hash[:8]}')
            except Exception as e:
                logger.error(f'Error processing screenshot: {e}')
        
        observer.add_activity_callback(process_screenshot)
        
        # Start monitoring
        observer.start_monitoring()
        logger.info('Background monitoring started successfully')
        
        retry_count = 0
        max_retries = $MAX_RETRIES
        
        # Main monitoring loop
        while running:
            try:
                # Health check
                if not observer._running:
                    logger.warning('Observer stopped unexpectedly, attempting restart...')
                    observer.start_monitoring()
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.error(f'Max retries ({max_retries}) exceeded, shutting down')
                        break
                else:
                    retry_count = 0  # Reset on successful operation
                
                # Sleep for health check interval
                time.sleep($HEALTH_CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info('Keyboard interrupt received')
                break
            except Exception as e:
                logger.error(f'Unexpected error in monitoring loop: {e}')
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f'Max retries ({max_retries}) exceeded, shutting down')
                    break
                time.sleep($RETRY_DELAY)
        
    except Exception as e:
        logger.error(f'Failed to start monitoring service: {e}')
        sys.exit(1)
    finally:
        if observer:
            observer.stop_monitoring()
        logger.info('Eidolon background monitoring service stopped')

if __name__ == '__main__':
    main()
\""
    
    # Start the service in background
    log_info "Launching background monitoring process..."
    nohup bash -c "$monitor_cmd" > "$MONITOR_LOG" 2>&1 &
    local pid=$!
    
    # Wait a moment to see if it starts successfully
    sleep 3
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid" > "$PID_FILE"
        log_success "Service started successfully (PID: $pid)"
        print_success "Background monitoring is now active"
        print_info "Log file: $MONITOR_LOG"
        print_info "PID file: $PID_FILE"
        
        # Show initial status
        echo
        get_status
    else
        log_error "Service failed to start"
        print_error "Check log file for details: $MONITOR_LOG"
        exit 1
    fi
}

# Stop the monitoring service
stop_service() {
    log_info "Stopping $SERVICE_NAME..."
    
    if ! is_running; then
        print_warn "Service is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    log_info "Sending TERM signal to process $pid"
    
    # Try graceful shutdown first
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [[ $count -lt 30 ]]; do
            sleep 1
            ((count++))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "Graceful shutdown failed, forcing termination"
            kill -KILL "$pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    if ! kill -0 "$pid" 2>/dev/null; then
        log_success "Service stopped successfully"
        print_success "Background monitoring stopped"
    else
        log_error "Failed to stop service"
        print_error "Process may still be running: $pid"
        exit 1
    fi
}

# Restart the monitoring service
restart_service() {
    log_info "Restarting $SERVICE_NAME..."
    stop_service
    sleep 2
    start_service
}

# Show service logs
show_logs() {
    local lines=${1:-50}
    
    if [[ -f "$MONITOR_LOG" ]]; then
        print_info "Showing last $lines lines of $MONITOR_LOG"
        echo
        tail -n "$lines" "$MONITOR_LOG"
    else
        print_warn "Log file not found: $MONITOR_LOG"
    fi
}

# Follow service logs
follow_logs() {
    if [[ -f "$MONITOR_LOG" ]]; then
        print_info "Following $MONITOR_LOG (Press Ctrl+C to exit)"
        echo
        tail -f "$MONITOR_LOG"
    else
        print_warn "Log file not found: $MONITOR_LOG"
        exit 1
    fi
}

# Clean up old log files
cleanup_logs() {
    local days=${1:-7}
    
    log_info "Cleaning up log files older than $days days"
    
    if [[ -d "$LOG_DIR" ]]; then
        find "$LOG_DIR" -name "*.log*" -mtime "+$days" -delete
        log_success "Log cleanup completed"
    fi
}

# Show performance metrics
show_metrics() {
    if ! is_running; then
        print_error "Service is not running"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    
    print_info "Performance metrics for $SERVICE_NAME (PID: $pid)"
    echo
    
    # System resources
    echo "System Resources:"
    ps -o pid,pcpu,pmem,rss,vsz,etime,comm -p "$pid" | head -n 2
    echo
    
    # Screenshot statistics
    local screenshot_count=$(find "${PROJECT_ROOT}/data/screenshots" -name "*.png" -mtime -1 | wc -l | tr -d ' ')
    echo "Screenshots captured (last 24h): $screenshot_count"
    
    # Database statistics
    if [[ -f "${PROJECT_ROOT}/data/eidolon.db" ]]; then
        local db_size=$(du -h "${PROJECT_ROOT}/data/eidolon.db" | cut -f1)
        echo "Database size: $db_size"
    fi
    
    # Storage usage
    local storage_usage=$(du -sh "${PROJECT_ROOT}/data" | cut -f1)
    echo "Total data storage: $storage_usage"
    
    # Recent activity from logs
    if [[ -f "$MONITOR_LOG" ]]; then
        echo
        echo "Recent activity (last 10 entries):"
        grep -E "(Screenshot|Analysis|Error)" "$MONITOR_LOG" | tail -n 10 | while read -r line; do
            echo "  $line"
        done
    fi
}

# Health check
health_check() {
    local exit_code=0
    
    print_info "Performing health check..."
    echo
    
    # Check if service is running
    if is_running; then
        print_success "✓ Service is running"
    else
        print_error "✗ Service is not running"
        exit_code=1
    fi
    
    # Check dependencies
    if [[ -f "$PYTHON_BIN" ]]; then
        print_success "✓ Python environment exists"
    else
        print_error "✗ Python environment missing"
        exit_code=1
    fi
    
    # Check log file
    if [[ -f "$MONITOR_LOG" ]]; then
        print_success "✓ Log file exists"
        
        # Check for recent activity
        if find "$MONITOR_LOG" -mmin -5 | grep -q .; then
            print_success "✓ Recent log activity detected"
        else
            print_warn "! No recent log activity (last 5 minutes)"
        fi
    else
        print_warn "! Log file missing"
    fi
    
    # Check screenshot directory
    if [[ -d "${PROJECT_ROOT}/data/screenshots" ]]; then
        print_success "✓ Screenshot directory exists"
        
        # Check for recent screenshots
        local recent_count=$(find "${PROJECT_ROOT}/data/screenshots" -name "*.png" -mmin -10 | wc -l | tr -d ' ')
        if [[ $recent_count -gt 0 ]]; then
            print_success "✓ Recent screenshots detected ($recent_count in last 10 min)"
        else
            print_warn "! No recent screenshots (last 10 minutes)"
        fi
    else
        print_error "✗ Screenshot directory missing"
        exit_code=1
    fi
    
    # Check database
    if [[ -f "${PROJECT_ROOT}/data/eidolon.db" ]]; then
        print_success "✓ Database file exists"
    else
        print_warn "! Database file missing"
    fi
    
    echo
    if [[ $exit_code -eq 0 ]]; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
    fi
    
    return $exit_code
}

# Show usage information
show_usage() {
    cat << EOF
Eidolon Background Monitoring Service

Usage: $(basename "$0") [COMMAND] [OPTIONS]

Commands:
    start               Start the background monitoring service
    stop                Stop the background monitoring service
    restart             Restart the background monitoring service
    status              Show service status and statistics
    logs [LINES]        Show recent log entries (default: 50 lines)
    follow              Follow log output in real-time
    metrics             Show performance metrics
    health              Perform health check
    cleanup [DAYS]      Clean up old log files (default: 7 days)

Options:
    -h, --help          Show this help message

Examples:
    $(basename "$0") start              # Start monitoring
    $(basename "$0") status             # Check status
    $(basename "$0") logs 100           # Show last 100 log lines
    $(basename "$0") follow             # Follow logs in real-time
    $(basename "$0") cleanup 14         # Clean logs older than 14 days

Configuration:
    Capture interval:   $CAPTURE_INTERVAL seconds
    Max retries:        $MAX_RETRIES
    Retry delay:        $RETRY_DELAY seconds
    Health check:       $HEALTH_CHECK_INTERVAL seconds

Files:
    Log file:           $MONITOR_LOG
    PID file:           $PID_FILE
    Config file:        $PROJECT_ROOT/eidolon/config/settings.yaml

For more information, see: $PROJECT_ROOT/README.md
EOF
}

# Main command dispatcher
main() {
    local command="${1:-}"
    
    case "$command" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            get_status
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        follow)
            follow_logs
            ;;
        metrics)
            show_metrics
            ;;
        health)
            health_check
            ;;
        cleanup)
            cleanup_logs "${2:-7}"
            ;;
        -h|--help|help)
            show_usage
            ;;
        "")
            print_error "No command specified"
            echo
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"