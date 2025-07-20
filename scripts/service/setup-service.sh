#!/bin/bash

# Eidolon Service Setup Script
# Configures Eidolon as a system service with auto-start capabilities

set -euo pipefail

# Configuration
EIDOLON_HOME="${HOME}/.eidolon"
SERVICE_NAME="eidolon"
SERVICE_USER="${USER}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS="macos" ;;
        Linux*)     OS="linux" ;;
        *)          OS="unknown" ;;
    esac
    log "Detected OS: $OS"
}

# Function to create systemd service (Linux)
create_systemd_service() {
    log "Creating systemd service for Eidolon..."
    
    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"
    local user_service_dir="${HOME}/.config/systemd/user"
    local user_service_file="${user_service_dir}/${SERVICE_NAME}.service"
    
    # Create user service directory
    mkdir -p "${user_service_dir}"
    
    # Create user service file
    cat > "${user_service_file}" <<EOF
[Unit]
Description=Eidolon AI Personal Assistant
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
Environment=DISPLAY=:0
Environment=HOME=${HOME}
Environment=USER=${USER}
Environment=EIDOLON_HOME=${EIDOLON_HOME}
EnvironmentFile=${EIDOLON_HOME}/.env
ExecStart=${EIDOLON_HOME}/venv/bin/python -m eidolon capture --service
ExecStop=${EIDOLON_HOME}/venv/bin/python -m eidolon stop
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
WorkingDirectory=${EIDOLON_HOME}

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=false
ReadWritePaths=${EIDOLON_HOME}

# Resource limits
MemoryMax=2G
CPUQuota=50%

[Install]
WantedBy=default.target
EOF

    # Enable and start user service
    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}.service"
    
    log "Systemd user service created and enabled"
    
    # Enable lingering for user to start services without login
    if command -v loginctl &> /dev/null; then
        sudo loginctl enable-linger "${USER}" || warn "Failed to enable user lingering"
    fi
}

# Function to create launchd service (macOS)
create_launchd_service() {
    log "Creating launchd service for Eidolon..."
    
    local plist_dir="${HOME}/Library/LaunchAgents"
    local plist_file="${plist_dir}/ai.eidolon.agent.plist"
    
    # Create LaunchAgents directory
    mkdir -p "${plist_dir}"
    
    # Create plist file
    cat > "${plist_file}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.eidolon.agent</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>${EIDOLON_HOME}/venv/bin/python</string>
        <string>-m</string>
        <string>eidolon</string>
        <string>capture</string>
        <string>--service</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>${EIDOLON_HOME}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>USER</key>
        <string>${USER}</string>
        <key>EIDOLON_HOME</key>
        <string>${EIDOLON_HOME}</string>
        <key>PATH</key>
        <string>${EIDOLON_HOME}/venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>${EIDOLON_HOME}/logs/service.log</string>
    
    <key>StandardErrorPath</key>
    <string>${EIDOLON_HOME}/logs/service-error.log</string>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <key>ProcessType</key>
    <string>Interactive</string>
    
    <key>LimitLoadToSessionType</key>
    <string>Aqua</string>
</dict>
</plist>
EOF

    # Load the service
    launchctl load "${plist_file}"
    
    log "Launchd service created and loaded"
}

# Function to create service management script
create_service_management() {
    log "Creating service management script..."
    
    cat > "${EIDOLON_HOME}/manage-service.sh" <<'EOF'
#!/bin/bash

# Eidolon Service Management Script

EIDOLON_HOME="${HOME}/.eidolon"
SERVICE_NAME="eidolon"

# Detect OS
case "$(uname -s)" in
    Darwin*)    OS="macos" ;;
    Linux*)     OS="linux" ;;
    *)          OS="unknown" ;;
esac

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}$1${NC}"
}

error() {
    echo -e "${RED}$1${NC}"
}

warn() {
    echo -e "${YELLOW}$1${NC}"
}

# Service control functions
start_service() {
    log "Starting Eidolon service..."
    case "$OS" in
        "macos")
            launchctl load "${HOME}/Library/LaunchAgents/ai.eidolon.agent.plist" 2>/dev/null || true
            ;;
        "linux")
            systemctl --user start "${SERVICE_NAME}.service"
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
    log "Service started"
}

stop_service() {
    log "Stopping Eidolon service..."
    case "$OS" in
        "macos")
            launchctl unload "${HOME}/Library/LaunchAgents/ai.eidolon.agent.plist" 2>/dev/null || true
            ;;
        "linux")
            systemctl --user stop "${SERVICE_NAME}.service"
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
    log "Service stopped"
}

restart_service() {
    log "Restarting Eidolon service..."
    stop_service
    sleep 2
    start_service
}

status_service() {
    log "Checking Eidolon service status..."
    case "$OS" in
        "macos")
            if launchctl list | grep -q "ai.eidolon.agent"; then
                log "Service is running"
                launchctl list ai.eidolon.agent
            else
                warn "Service is not running"
            fi
            ;;
        "linux")
            systemctl --user status "${SERVICE_NAME}.service"
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
}

enable_service() {
    log "Enabling Eidolon service..."
    case "$OS" in
        "macos")
            log "Service is automatically enabled on macOS"
            ;;
        "linux")
            systemctl --user enable "${SERVICE_NAME}.service"
            log "Service enabled"
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
}

disable_service() {
    log "Disabling Eidolon service..."
    case "$OS" in
        "macos")
            launchctl unload "${HOME}/Library/LaunchAgents/ai.eidolon.agent.plist" 2>/dev/null || true
            log "Service disabled"
            ;;
        "linux")
            systemctl --user disable "${SERVICE_NAME}.service"
            log "Service disabled"
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
}

logs_service() {
    log "Showing Eidolon service logs..."
    case "$OS" in
        "macos")
            echo "Service logs:"
            tail -f "${EIDOLON_HOME}/logs/service.log" 2>/dev/null || echo "No service logs found"
            ;;
        "linux")
            journalctl --user -u "${SERVICE_NAME}.service" -f
            ;;
        *)
            error "Unsupported OS: $OS"
            return 1
            ;;
    esac
}

# Main function
case "${1:-}" in
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
        status_service
        ;;
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    logs)
        logs_service
        ;;
    *)
        echo "Eidolon Service Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|enable|disable|logs}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the Eidolon service"
        echo "  stop      - Stop the Eidolon service"
        echo "  restart   - Restart the Eidolon service"
        echo "  status    - Show service status"
        echo "  enable    - Enable service autostart"
        echo "  disable   - Disable service autostart"
        echo "  logs      - Show service logs"
        echo ""
        exit 1
        ;;
esac
EOF

    chmod +x "${EIDOLON_HOME}/manage-service.sh"
    
    # Create convenience symlink
    mkdir -p "${EIDOLON_HOME}/bin"
    ln -sf "${EIDOLON_HOME}/manage-service.sh" "${EIDOLON_HOME}/bin/eidolon-service"
    
    log "Service management script created"
}

# Function to create health monitoring
create_health_monitoring() {
    log "Creating health monitoring..."
    
    cat > "${EIDOLON_HOME}/scripts/health-check.sh" <<'EOF'
#!/bin/bash

# Eidolon Health Check Script

EIDOLON_HOME="${HOME}/.eidolon"
source "${EIDOLON_HOME}/.env" 2>/dev/null || true

# Health check function
check_health() {
    local exit_code=0
    
    echo "Eidolon Health Check - $(date)"
    echo "================================"
    
    # Check if process is running
    if pgrep -f "eidolon capture" > /dev/null; then
        echo "✓ Eidolon process is running"
    else
        echo "✗ Eidolon process is not running"
        exit_code=1
    fi
    
    # Check disk space
    local disk_usage=$(df "${EIDOLON_HOME}" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 90 ]]; then
        echo "✓ Disk usage is healthy ($disk_usage%)"
    else
        echo "✗ Disk usage is high ($disk_usage%)"
        exit_code=1
    fi
    
    # Check memory usage
    local memory_usage=$(python3 -c "
import psutil
import sys
try:
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        if 'eidolon' in proc.info['name'].lower():
            mem_mb = proc.info['memory_info'].rss / 1024 / 1024
            print(f'{mem_mb:.1f}')
            break
    else:
        print('0')
except:
    print('0')
" 2>/dev/null)
    
    if (( $(echo "$memory_usage < 2048" | bc -l) )); then
        echo "✓ Memory usage is healthy (${memory_usage}MB)"
    else
        echo "✗ Memory usage is high (${memory_usage}MB)"
        exit_code=1
    fi
    
    # Check log files
    if [[ -f "${EIDOLON_HOME}/logs/eidolon.log" ]]; then
        local log_errors=$(grep -c "ERROR" "${EIDOLON_HOME}/logs/eidolon.log" | tail -1000 || echo "0")
        if [[ $log_errors -lt 10 ]]; then
            echo "✓ Log errors are acceptable ($log_errors recent errors)"
        else
            echo "✗ High number of log errors ($log_errors recent errors)"
            exit_code=1
        fi
    else
        echo "⚠ Log file not found"
    fi
    
    # Check database
    if [[ -f "${EIDOLON_HOME}/data/eidolon.db" ]]; then
        echo "✓ Database file exists"
    else
        echo "✗ Database file missing"
        exit_code=1
    fi
    
    echo "================================"
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ Overall health: GOOD"
    else
        echo "✗ Overall health: ISSUES DETECTED"
    fi
    
    return $exit_code
}

# Run health check
check_health

# If called with --restart-on-fail, restart service on failure
if [[ "${1:-}" == "--restart-on-fail" ]] && [[ $? -ne 0 ]]; then
    echo "Restarting Eidolon service due to health check failure..."
    "${EIDOLON_HOME}/manage-service.sh" restart
fi
EOF

    chmod +x "${EIDOLON_HOME}/scripts/health-check.sh"
    
    log "Health monitoring script created"
}

# Function to create auto-update monitoring
create_auto_update() {
    log "Creating auto-update configuration..."
    
    # Create update check script
    cat > "${EIDOLON_HOME}/scripts/check-updates.sh" <<'EOF'
#!/bin/bash

# Eidolon Update Check Script

EIDOLON_HOME="${HOME}/.eidolon"
CURRENT_VERSION_FILE="${EIDOLON_HOME}/.version"
UPDATE_CHECK_FILE="${EIDOLON_HOME}/.last_update_check"

# Get current version
get_current_version() {
    if [[ -f "$CURRENT_VERSION_FILE" ]]; then
        cat "$CURRENT_VERSION_FILE"
    else
        echo "unknown"
    fi
}

# Check for updates
check_updates() {
    local current_version=$(get_current_version)
    local now=$(date +%s)
    
    # Only check once per day
    if [[ -f "$UPDATE_CHECK_FILE" ]]; then
        local last_check=$(cat "$UPDATE_CHECK_FILE")
        local time_diff=$((now - last_check))
        if [[ $time_diff -lt 86400 ]]; then  # 24 hours
            return 0
        fi
    fi
    
    echo "$now" > "$UPDATE_CHECK_FILE"
    
    # Check PyPI for latest version (simplified)
    local latest_version=$(curl -s https://pypi.org/pypi/eidolon/json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['info']['version'])
except:
    print('unknown')
" 2>/dev/null)
    
    if [[ "$latest_version" != "unknown" ]] && [[ "$latest_version" != "$current_version" ]]; then
        echo "Update available: $current_version -> $latest_version"
        echo "Run 'eidolon update' to update"
        
        # Log update notification
        echo "$(date): Update available $current_version -> $latest_version" >> "${EIDOLON_HOME}/logs/updates.log"
    fi
}

check_updates
EOF

    chmod +x "${EIDOLON_HOME}/scripts/check-updates.sh"
    
    log "Auto-update monitoring created"
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     Eidolon Service Setup                       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    detect_os
    
    # Check if Eidolon is installed
    if [[ ! -d "$EIDOLON_HOME" ]]; then
        error "Eidolon not found. Please run the installation script first."
        exit 1
    fi
    
    # Create scripts directory
    mkdir -p "${EIDOLON_HOME}/scripts"
    
    # Setup service based on OS
    case "$OS" in
        "macos")
            create_launchd_service
            ;;
        "linux")
            create_systemd_service
            ;;
        *)
            error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    # Create management scripts
    create_service_management
    create_health_monitoring
    create_auto_update
    
    log "Service setup completed!"
    echo ""
    echo -e "${GREEN}Eidolon service has been configured${NC}"
    echo ""
    echo "Service management commands:"
    echo "  ${EIDOLON_HOME}/manage-service.sh start     - Start service"
    echo "  ${EIDOLON_HOME}/manage-service.sh stop      - Stop service"
    echo "  ${EIDOLON_HOME}/manage-service.sh status    - Check status"
    echo "  ${EIDOLON_HOME}/manage-service.sh logs      - View logs"
    echo ""
    echo "Health monitoring:"
    echo "  ${EIDOLON_HOME}/scripts/health-check.sh     - Manual health check"
    echo ""
    echo -e "${YELLOW}Note:${NC} The service will start automatically on system boot"
}

# Parse arguments
case "${1:-setup}" in
    setup)
        main
        ;;
    --help)
        echo "Eidolon Service Setup"
        echo ""
        echo "Usage: $0 [setup|--help]"
        echo ""
        echo "This script configures Eidolon as a system service with auto-start capabilities."
        exit 0
        ;;
    *)
        error "Unknown option: $1"
        exit 1
        ;;
esac