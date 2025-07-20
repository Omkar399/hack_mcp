#!/bin/bash

# Eidolon User Onboarding Wizard
# Interactive setup and configuration guide for new users

set -euo pipefail

# Configuration
EIDOLON_HOME="${HOME}/.eidolon"
WIZARD_VERSION="1.0.0"

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Unicode symbols
CHECK_MARK="✓"
CROSS_MARK="✗"
ARROW="→"
STAR="★"

# State tracking
WIZARD_STATE_FILE="${EIDOLON_HOME}/.wizard_state"
ONBOARDING_LOG="${EIDOLON_HOME}/logs/onboarding.log"

# Initialize logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${ONBOARDING_LOG}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "${ONBOARDING_LOG}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "${ONBOARDING_LOG}"
}

info() {
    echo -e "${CYAN}$1${NC}"
}

success() {
    echo -e "${GREEN}${CHECK_MARK} $1${NC}"
}

# Function to display welcome screen
show_welcome() {
    clear
    echo -e "${BLUE}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    Welcome to Eidolon AI Personal Assistant                 ║
║                                                                              ║
║                           🤖 Setup Wizard v1.0.0                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}Thank you for choosing Eidolon! This wizard will help you:${NC}"
    echo ""
    echo -e "  ${ARROW} Configure your personal AI assistant"
    echo -e "  ${ARROW} Set up privacy and security preferences"
    echo -e "  ${ARROW} Connect to AI services"
    echo -e "  ${ARROW} Customize monitoring and analysis"
    echo -e "  ${ARROW} Test your setup"
    echo ""
    echo -e "${YELLOW}Estimated setup time: 5-10 minutes${NC}"
    echo ""
    echo -n "Press Enter to begin setup..."
    read -r
}

# Function to check system requirements
check_requirements() {
    clear
    echo -e "${BLUE}${BOLD}Step 1: System Requirements Check${NC}"
    echo "======================================="
    echo ""
    
    local all_checks_passed=true
    
    # Check Python version
    echo -n "Checking Python version... "
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            success "Python $python_version"
        else
            error "Python $python_version (requires 3.9+)"
            all_checks_passed=false
        fi
    else
        error "Python not found"
        all_checks_passed=false
    fi
    
    # Check available memory
    echo -n "Checking available memory... "
    local memory_gb=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "unknown")
    if [[ "$memory_gb" != "unknown" ]] && (( $(echo "$memory_gb >= 4.0" | bc -l) )); then
        success "${memory_gb}GB available"
    else
        warn "Low memory detected (${memory_gb}GB), 8GB+ recommended"
    fi
    
    # Check disk space
    echo -n "Checking disk space... "
    local disk_space=$(df -h "$HOME" | awk 'NR==2 {print $4}')
    local disk_space_gb=$(df "$HOME" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
    if (( $(echo "$disk_space_gb >= 10.0" | bc -l) )); then
        success "${disk_space} available"
    else
        warn "Low disk space (${disk_space}), 100GB+ recommended"
    fi
    
    # Check tesseract
    echo -n "Checking tesseract OCR... "
    if command -v tesseract &> /dev/null; then
        success "Found tesseract"
    else
        warn "tesseract not found (OCR functionality limited)"
    fi
    
    # Check network connectivity
    echo -n "Checking internet connectivity... "
    if curl -s --connect-timeout 5 google.com > /dev/null; then
        success "Internet connection active"
    else
        warn "Internet connection issues detected"
    fi
    
    echo ""
    if [[ "$all_checks_passed" == true ]]; then
        success "All critical requirements met!"
    else
        warn "Some requirements not met. Continue anyway? (y/N)"
        read -r continue_anyway
        if [[ "$continue_anyway" != "y" && "$continue_anyway" != "Y" ]]; then
            echo "Please address the requirements and run the wizard again."
            exit 1
        fi
    fi
    
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to configure privacy settings
configure_privacy() {
    clear
    echo -e "${BLUE}${BOLD}Step 2: Privacy & Security Configuration${NC}"
    echo "========================================="
    echo ""
    
    echo -e "${CYAN}Eidolon respects your privacy. Let's configure your preferences:${NC}"
    echo ""
    
    # Privacy mode selection
    echo "1. Privacy Mode:"
    echo "   a) Balanced - Local processing + cloud AI for complex tasks"
    echo "   b) Private - Local processing only (no cloud services)"
    echo "   c) Performance - Cloud AI for best results (with privacy controls)"
    echo ""
    echo -n "Choose your privacy mode (a/b/c) [a]: "
    read -r privacy_mode
    privacy_mode=${privacy_mode:-a}
    
    local local_only="false"
    local cloud_threshold="0.7"
    
    case "$privacy_mode" in
        "b")
            local_only="true"
            info "Selected: Private mode - All processing will be done locally"
            ;;
        "c")
            cloud_threshold="0.5"
            info "Selected: Performance mode - Optimized for best AI results"
            ;;
        *)
            info "Selected: Balanced mode - Good balance of privacy and performance"
            ;;
    esac
    
    echo ""
    
    # Data retention
    echo "2. Data Retention:"
    echo "   How long should Eidolon keep your data?"
    echo "   a) 30 days"
    echo "   b) 90 days" 
    echo "   c) 1 year"
    echo "   d) Until manually deleted"
    echo ""
    echo -n "Choose retention period (a/b/c/d) [c]: "
    read -r retention_choice
    retention_choice=${retention_choice:-c}
    
    local retention_days=365
    case "$retention_choice" in
        "a") retention_days=30 ;;
        "b") retention_days=90 ;;
        "d") retention_days=9999 ;;
    esac
    
    info "Selected: Data will be kept for $retention_days days"
    echo ""
    
    # Sensitive content handling
    echo "3. Sensitive Content Protection:"
    echo -n "Enable automatic detection and redaction of sensitive information? (Y/n): "
    read -r enable_redaction
    enable_redaction=${enable_redaction:-Y}
    
    local auto_redaction="true"
    if [[ "$enable_redaction" == "n" || "$enable_redaction" == "N" ]]; then
        auto_redaction="false"
        warn "Sensitive content protection disabled"
    else
        success "Sensitive content protection enabled"
    fi
    
    echo ""
    
    # Excluded applications
    echo "4. Application Exclusions:"
    echo "   Which types of applications should Eidolon avoid monitoring?"
    echo "   (You can modify this later)"
    echo ""
    echo -n "Exclude password managers? (Y/n): "
    read -r exclude_passwords
    exclude_passwords=${exclude_passwords:-Y}
    
    echo -n "Exclude banking/financial apps? (Y/n): "
    read -r exclude_banking  
    exclude_banking=${exclude_banking:-Y}
    
    echo -n "Exclude private browsing/incognito windows? (Y/n): "
    read -r exclude_private
    exclude_private=${exclude_private:-Y}
    
    # Save privacy configuration
    save_wizard_state "privacy" "{
        \"local_only\": $local_only,
        \"cloud_threshold\": $cloud_threshold,
        \"retention_days\": $retention_days,
        \"auto_redaction\": $auto_redaction,
        \"exclude_passwords\": \"$exclude_passwords\",
        \"exclude_banking\": \"$exclude_banking\",
        \"exclude_private\": \"$exclude_private\"
    }"
    
    success "Privacy settings configured!"
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to configure AI services
configure_ai_services() {
    clear
    echo -e "${BLUE}${BOLD}Step 3: AI Services Configuration${NC}"
    echo "=================================="
    echo ""
    
    # Check privacy mode
    local privacy_config=$(load_wizard_state "privacy")
    local local_only=$(echo "$privacy_config" | jq -r '.local_only // false')
    
    if [[ "$local_only" == "true" ]]; then
        info "Local-only mode selected. Skipping cloud AI configuration."
        echo ""
        echo -n "Press Enter to continue..."
        read -r
        return 0
    fi
    
    echo -e "${CYAN}To provide the best AI assistance, Eidolon can connect to cloud AI services.${NC}"
    echo "You can configure one or more services for redundancy."
    echo ""
    
    local api_keys="{}"
    
    # Configure Gemini API
    echo "1. Google Gemini (Recommended for performance)"
    echo -n "Do you have a Gemini API key? (y/N): "
    read -r has_gemini
    
    if [[ "$has_gemini" == "y" || "$has_gemini" == "Y" ]]; then
        echo -n "Enter your Gemini API key: "
        read -rs gemini_key
        echo ""
        if [[ -n "$gemini_key" ]]; then
            api_keys=$(echo "$api_keys" | jq ". + {\"gemini\": \"$gemini_key\"}")
            success "Gemini API key configured"
        fi
    else
        info "You can get a free Gemini API key at: https://makersuite.google.com/app/apikey"
    fi
    
    echo ""
    
    # Configure Claude API
    echo "2. Anthropic Claude (Recommended for analysis)"
    echo -n "Do you have a Claude API key? (y/N): "
    read -r has_claude
    
    if [[ "$has_claude" == "y" || "$has_claude" == "Y" ]]; then
        echo -n "Enter your Claude API key: "
        read -rs claude_key
        echo ""
        if [[ -n "$claude_key" ]]; then
            api_keys=$(echo "$api_keys" | jq ". + {\"claude\": \"$claude_key\"}")
            success "Claude API key configured"
        fi
    else
        info "You can get a Claude API key at: https://console.anthropic.com/"
    fi
    
    echo ""
    
    # Configure OpenAI API
    echo "3. OpenAI (Alternative option)"
    echo -n "Do you have an OpenAI API key? (y/N): "
    read -r has_openai
    
    if [[ "$has_openai" == "y" || "$has_openai" == "Y" ]]; then
        echo -n "Enter your OpenAI API key: "
        read -rs openai_key
        echo ""
        if [[ -n "$openai_key" ]]; then
            api_keys=$(echo "$api_keys" | jq ". + {\"openai\": \"$openai_key\"}")
            success "OpenAI API key configured"
        fi
    else
        info "You can get an OpenAI API key at: https://platform.openai.com/api-keys"
    fi
    
    echo ""
    
    # Check if any keys were configured
    local key_count=$(echo "$api_keys" | jq 'keys | length')
    if [[ "$key_count" -eq 0 ]]; then
        warn "No API keys configured. Eidolon will use local AI only."
        warn "This may limit functionality but protects privacy."
        echo ""
        echo -n "Continue with local-only mode? (Y/n): "
        read -r continue_local
        if [[ "$continue_local" == "n" || "$continue_local" == "N" ]]; then
            configure_ai_services  # Restart this step
            return
        fi
        # Update privacy settings to local-only
        local privacy_config_updated=$(echo "$privacy_config" | jq '.local_only = true')
        save_wizard_state "privacy" "$privacy_config_updated"
    else
        success "$key_count AI service(s) configured!"
    fi
    
    # Save AI configuration
    save_wizard_state "ai_services" "$api_keys"
    
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to configure monitoring preferences
configure_monitoring() {
    clear
    echo -e "${BLUE}${BOLD}Step 4: Monitoring & Analysis Configuration${NC}"
    echo "==========================================="
    echo ""
    
    echo -e "${CYAN}Configure how Eidolon monitors and analyzes your activity:${NC}"
    echo ""
    
    # Capture frequency
    echo "1. Screenshot Capture Frequency:"
    echo "   a) Conservative (every 30 seconds) - Lower resource usage"
    echo "   b) Balanced (every 15 seconds) - Good balance"
    echo "   c) Detailed (every 5 seconds) - More complete monitoring"
    echo ""
    echo -n "Choose capture frequency (a/b/c) [b]: "
    read -r capture_frequency
    capture_frequency=${capture_frequency:-b}
    
    local capture_interval=15
    case "$capture_frequency" in
        "a") capture_interval=30 ;;
        "c") capture_interval=5 ;;
    esac
    
    info "Selected: Screenshot every $capture_interval seconds"
    echo ""
    
    # Analysis depth
    echo "2. Analysis Depth:"
    echo "   a) Basic - Text extraction and simple categorization"
    echo "   b) Standard - Text analysis + UI understanding"
    echo "   c) Advanced - Full AI analysis with insights"
    echo ""
    echo -n "Choose analysis depth (a/b/c) [b]: "
    read -r analysis_depth
    analysis_depth=${analysis_depth:-b}
    
    local importance_threshold="0.5"
    case "$analysis_depth" in
        "a") importance_threshold="0.8" ;;
        "c") importance_threshold="0.3" ;;
    esac
    
    info "Selected: $analysis_depth analysis depth"
    echo ""
    
    # Resource limits
    echo "3. Resource Usage:"
    echo "   a) Low impact - Minimal CPU/memory usage"
    echo "   b) Balanced - Good performance without interference"
    echo "   c) High performance - Maximum capabilities"
    echo ""
    echo -n "Choose resource profile (a/b/c) [b]: "
    read -r resource_profile
    resource_profile=${resource_profile:-b}
    
    local max_cpu=15
    local max_memory=4096
    case "$resource_profile" in
        "a") 
            max_cpu=10
            max_memory=2048
            ;;
        "c")
            max_cpu=25
            max_memory=8192
            ;;
    esac
    
    info "Selected: Max ${max_cpu}% CPU, ${max_memory}MB memory"
    echo ""
    
    # Dashboard access
    echo "4. Monitoring Dashboard:"
    echo -n "Enable web dashboard for monitoring? (Y/n): "
    read -r enable_dashboard
    enable_dashboard=${enable_dashboard:-Y}
    
    local dashboard_enabled="true"
    if [[ "$enable_dashboard" == "n" || "$enable_dashboard" == "N" ]]; then
        dashboard_enabled="false"
    else
        success "Dashboard will be available at http://localhost:8080"
    fi
    
    # Save monitoring configuration
    save_wizard_state "monitoring" "{
        \"capture_interval\": $capture_interval,
        \"importance_threshold\": $importance_threshold,
        \"max_cpu_percent\": $max_cpu,
        \"max_memory_mb\": $max_memory,
        \"dashboard_enabled\": $dashboard_enabled
    }"
    
    success "Monitoring settings configured!"
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to configure autostart
configure_autostart() {
    clear
    echo -e "${BLUE}${BOLD}Step 5: Service & Autostart Configuration${NC}"
    echo "=========================================="
    echo ""
    
    echo -e "${CYAN}Configure how Eidolon runs on your system:${NC}"
    echo ""
    
    # Autostart preference
    echo "1. Automatic Startup:"
    echo -n "Start Eidolon automatically when you log in? (Y/n): "
    read -r enable_autostart
    enable_autostart=${enable_autostart:-Y}
    
    local autostart_enabled="true"
    if [[ "$enable_autostart" == "n" || "$enable_autostart" == "N" ]]; then
        autostart_enabled="false"
        info "Manual startup mode selected"
    else
        success "Autostart enabled - Eidolon will start with your system"
    fi
    
    echo ""
    
    # Service user preference
    echo "2. Service Configuration:"
    echo "   a) User service - Runs when you're logged in"
    echo "   b) System service - Always running (requires admin)"
    echo ""
    echo -n "Choose service type (a/b) [a]: "
    read -r service_type
    service_type=${service_type:-a}
    
    local system_service="false"
    if [[ "$service_type" == "b" ]]; then
        system_service="true"
        warn "System service requires administrator privileges"
    else
        info "User service selected"
    fi
    
    echo ""
    
    # Resource monitoring
    echo "3. Performance Monitoring:"
    echo -n "Enable performance monitoring and alerts? (Y/n): "
    read -r enable_monitoring
    enable_monitoring=${enable_monitoring:-Y}
    
    local monitoring_enabled="true"
    if [[ "$enable_monitoring" == "n" || "$enable_monitoring" == "N" ]]; then
        monitoring_enabled="false"
    fi
    
    # Save service configuration
    save_wizard_state "service" "{
        \"autostart_enabled\": $autostart_enabled,
        \"system_service\": $system_service,
        \"monitoring_enabled\": $monitoring_enabled
    }"
    
    success "Service settings configured!"
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to generate final configuration
generate_configuration() {
    clear
    echo -e "${BLUE}${BOLD}Step 6: Generating Configuration${NC}"
    echo "================================="
    echo ""
    
    echo "Generating your personalized Eidolon configuration..."
    echo ""
    
    # Load all wizard states
    local privacy_config=$(load_wizard_state "privacy")
    local ai_services=$(load_wizard_state "ai_services")
    local monitoring_config=$(load_wizard_state "monitoring")
    local service_config=$(load_wizard_state "service")
    
    # Create configuration directory
    mkdir -p "${EIDOLON_HOME}/config"
    
    # Generate main configuration file
    cat > "${EIDOLON_HOME}/config/settings.yaml" <<EOF
# Eidolon AI Personal Assistant Configuration
# Generated by Setup Wizard on $(date)

# Observer settings
observer:
  capture_interval: $(echo "$monitoring_config" | jq -r '.capture_interval')
  activity_threshold: 0.05
  storage_path: "./data/screenshots"
  max_storage_gb: 50
  
  max_cpu_percent: $(echo "$monitoring_config" | jq -r '.max_cpu_percent').0
  max_memory_mb: $(echo "$monitoring_config" | jq -r '.max_memory_mb')
  
  sensitive_patterns:
    - "password"
    - "api_key"
    - "secret"
    - "token"
    - "ssn"
    - "credit_card"
  
  excluded_apps:
    - "1Password"
    - "Keychain Access"
    - "LastPass"
    - "Bitwarden"

# Analysis settings
analysis:
  local_models:
    vision: "microsoft/florence-2-base"
    clip: "openai/clip-vit-base-patch32"
    embedding: "sentence-transformers/all-MiniLM-L6-v2"
  
  routing:
    importance_threshold: $(echo "$monitoring_config" | jq -r '.importance_threshold')
    cost_limit_daily: 10.0
    local_first: true

# Memory system
memory:
  vector_db: "chromadb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  overlap: 50
  db_path: "./data/eidolon.db"

# Privacy settings
privacy:
  local_only_mode: $(echo "$privacy_config" | jq -r '.local_only')
  auto_redaction: $(echo "$privacy_config" | jq -r '.auto_redaction')
  data_retention_days: $(echo "$privacy_config" | jq -r '.retention_days')
  encrypt_at_rest: true

# Monitoring
monitoring:
  enabled: $(echo "$service_config" | jq -r '.monitoring_enabled')
  metrics_collection_interval: 60
  dashboard_enabled: $(echo "$monitoring_config" | jq -r '.dashboard_enabled')

# Logging
logging:
  level: "INFO"
  file_path: "./logs/eidolon.log"
  max_file_size_mb: 10
  backup_count: 5

# MCP Server
mcp_server:
  enabled: true
  transport: "stdio"
  title: "eidolon-personal-assistant"

# Chat
chat:
  enabled: true
  default_model: "gemini-2.0-flash-exp"
  max_context_events: 10
EOF

    # Generate environment file with API keys
    cat > "${EIDOLON_HOME}/.env" <<EOF
# Eidolon Environment Configuration
# Generated by Setup Wizard on $(date)

EIDOLON_HOME=${EIDOLON_HOME}
EIDOLON_CONFIG=${EIDOLON_HOME}/config

# API Keys
EOF

    # Add API keys if configured
    if [[ "$(echo "$ai_services" | jq 'has("gemini")')" == "true" ]]; then
        echo "GEMINI_API_KEY=$(echo "$ai_services" | jq -r '.gemini')" >> "${EIDOLON_HOME}/.env"
    fi
    
    if [[ "$(echo "$ai_services" | jq 'has("claude")')" == "true" ]]; then
        echo "CLAUDE_API_KEY=$(echo "$ai_services" | jq -r '.claude')" >> "${EIDOLON_HOME}/.env"
    fi
    
    if [[ "$(echo "$ai_services" | jq 'has("openai")')" == "true" ]]; then
        echo "OPENAI_API_KEY=$(echo "$ai_services" | jq -r '.openai')" >> "${EIDOLON_HOME}/.env"
    fi
    
    # Set secure permissions
    chmod 600 "${EIDOLON_HOME}/.env"
    chmod 644 "${EIDOLON_HOME}/config/settings.yaml"
    
    success "Configuration files generated!"
    echo ""
    echo "Generated files:"
    echo "  - ${EIDOLON_HOME}/config/settings.yaml"
    echo "  - ${EIDOLON_HOME}/.env"
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to setup service
setup_service() {
    clear
    echo -e "${BLUE}${BOLD}Step 7: Service Setup${NC}"
    echo "====================="
    echo ""
    
    local service_config=$(load_wizard_state "service")
    local autostart_enabled=$(echo "$service_config" | jq -r '.autostart_enabled')
    
    if [[ "$autostart_enabled" == "true" ]]; then
        echo "Setting up Eidolon service..."
        
        # Run service setup
        if [[ -f "${EIDOLON_HOME}/scripts/service/setup-service.sh" ]]; then
            echo "Running service installation..."
            "${EIDOLON_HOME}/scripts/service/setup-service.sh" setup
            success "Service installed successfully!"
        else
            warn "Service setup script not found. You may need to set up autostart manually."
        fi
    else
        info "Autostart disabled. You can start Eidolon manually with 'eidolon capture'"
    fi
    
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to run initial test
run_initial_test() {
    clear
    echo -e "${BLUE}${BOLD}Step 8: Initial Test${NC}"
    echo "===================="
    echo ""
    
    echo "Running initial system test..."
    echo ""
    
    # Test configuration
    echo -n "Testing configuration... "
    if eidolon status > /dev/null 2>&1; then
        success "Configuration valid"
    else
        error "Configuration test failed"
        return 1
    fi
    
    # Test database initialization
    echo -n "Initializing database... "
    if eidolon init-db > /dev/null 2>&1; then
        success "Database initialized"
    else
        warn "Database initialization had issues (may already exist)"
    fi
    
    # Test AI services (if configured)
    local ai_services=$(load_wizard_state "ai_services")
    local key_count=$(echo "$ai_services" | jq 'keys | length')
    
    if [[ "$key_count" -gt 0 ]]; then
        echo -n "Testing AI service connections... "
        if eidolon test ai-connections > /dev/null 2>&1; then
            success "AI services responding"
        else
            warn "Some AI services may have issues"
        fi
    fi
    
    # Test capture system
    echo -n "Testing screenshot capture... "
    if eidolon capture --test > /dev/null 2>&1; then
        success "Capture system ready"
    else
        warn "Capture system needs attention"
    fi
    
    echo ""
    success "Initial test completed!"
    echo ""
    echo -n "Press Enter to continue..."
    read -r
}

# Function to show completion summary
show_completion() {
    clear
    echo -e "${GREEN}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                     🎉 Setup Complete! 🎉                                   ║
║                                                                              ║
║                   Eidolon is ready to assist you!                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo ""
    
    # Load configurations for summary
    local privacy_config=$(load_wizard_state "privacy")
    local monitoring_config=$(load_wizard_state "monitoring")
    local service_config=$(load_wizard_state "service")
    
    echo -e "${CYAN}${BOLD}Your Configuration Summary:${NC}"
    echo "============================="
    echo ""
    
    # Privacy settings
    echo -e "${YELLOW}Privacy & Security:${NC}"
    local local_only=$(echo "$privacy_config" | jq -r '.local_only')
    if [[ "$local_only" == "true" ]]; then
        echo "  • Local-only processing (maximum privacy)"
    else
        echo "  • Balanced privacy with cloud AI assistance"
    fi
    echo "  • Data retention: $(echo "$privacy_config" | jq -r '.retention_days') days"
    echo "  • Automatic sensitive data redaction: $(echo "$privacy_config" | jq -r '.auto_redaction')"
    echo ""
    
    # Monitoring settings
    echo -e "${YELLOW}Monitoring & Analysis:${NC}"
    echo "  • Screenshot interval: $(echo "$monitoring_config" | jq -r '.capture_interval') seconds"
    echo "  • Resource limits: $(echo "$monitoring_config" | jq -r '.max_cpu_percent')% CPU, $(echo "$monitoring_config" | jq -r '.max_memory_mb')MB RAM"
    
    local dashboard_enabled=$(echo "$monitoring_config" | jq -r '.dashboard_enabled')
    if [[ "$dashboard_enabled" == "true" ]]; then
        echo "  • Dashboard: http://localhost:8080"
    fi
    echo ""
    
    # Service status
    echo -e "${YELLOW}Service Configuration:${NC}"
    local autostart_enabled=$(echo "$service_config" | jq -r '.autostart_enabled')
    if [[ "$autostart_enabled" == "true" ]]; then
        echo "  • Autostart: Enabled (starts with system)"
    else
        echo "  • Autostart: Disabled (manual start required)"
    fi
    echo ""
    
    echo -e "${CYAN}${BOLD}Quick Start Commands:${NC}"
    echo "====================="
    echo ""
    echo -e "${GREEN}Start Eidolon:${NC}"
    echo "  eidolon capture                    # Start monitoring"
    echo ""
    echo -e "${GREEN}Check Status:${NC}"
    echo "  eidolon status                     # System status"
    echo "  eidolon health-check               # Detailed health check"
    echo ""
    echo -e "${GREEN}Search & Query:${NC}"
    echo "  eidolon search \"meeting notes\"     # Search captured content"
    echo "  eidolon chat                       # Interactive AI chat"
    echo ""
    echo -e "${GREEN}Management:${NC}"
    echo "  eidolon dashboard                  # Open monitoring dashboard"
    echo "  ${EIDOLON_HOME}/manage-service.sh status    # Service management"
    echo ""
    
    echo -e "${CYAN}${BOLD}Additional Resources:${NC}"
    echo "====================="
    echo ""
    echo "• Documentation: ${EIDOLON_HOME}/docs/"
    echo "• Configuration: ${EIDOLON_HOME}/config/settings.yaml"
    echo "• Logs: ${EIDOLON_HOME}/logs/eidolon.log"
    echo "• Privacy controls: eidolon privacy --help"
    echo "• Backup management: ${EIDOLON_HOME}/scripts/backup/backup-manager.sh"
    echo ""
    
    echo -e "${YELLOW}Need Help?${NC}"
    echo "• Run 'eidolon --help' for command reference"
    echo "• Visit the documentation for detailed guides"
    echo "• Check the logs if you encounter issues"
    echo ""
    
    # Offer to start Eidolon
    local autostart_enabled=$(echo "$service_config" | jq -r '.autostart_enabled')
    if [[ "$autostart_enabled" == "false" ]]; then
        echo -n "Would you like to start Eidolon now? (Y/n): "
        read -r start_now
        start_now=${start_now:-Y}
        
        if [[ "$start_now" == "y" || "$start_now" == "Y" ]]; then
            echo ""
            echo "Starting Eidolon..."
            if eidolon capture --daemon > /dev/null 2>&1; then
                success "Eidolon started successfully!"
                echo "Monitor progress with: eidolon status"
            else
                error "Failed to start Eidolon. Check logs for details."
            fi
        fi
    fi
    
    echo ""
    echo -e "${GREEN}${BOLD}Thank you for choosing Eidolon! 🤖${NC}"
    echo ""
    
    # Clean up wizard state
    rm -f "$WIZARD_STATE_FILE"
}

# Utility functions for state management
save_wizard_state() {
    local key="$1"
    local value="$2"
    
    mkdir -p "$(dirname "$WIZARD_STATE_FILE")"
    
    local state="{}"
    if [[ -f "$WIZARD_STATE_FILE" ]]; then
        state=$(cat "$WIZARD_STATE_FILE")
    fi
    
    state=$(echo "$state" | jq ". + {\"$key\": $value}")
    echo "$state" > "$WIZARD_STATE_FILE"
}

load_wizard_state() {
    local key="$1"
    
    if [[ -f "$WIZARD_STATE_FILE" ]]; then
        jq -r ".$key // {}" "$WIZARD_STATE_FILE"
    else
        echo "{}"
    fi
}

# Function to check dependencies
check_dependencies() {
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo "Installing jq for configuration management..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y jq
        elif command -v brew &> /dev/null; then
            brew install jq
        elif command -v yum &> /dev/null; then
            sudo yum install -y jq
        else
            error "Please install jq manually: https://stedolan.github.io/jq/download/"
            exit 1
        fi
    fi
}

# Main execution
main() {
    # Ensure logs directory exists
    mkdir -p "${EIDOLON_HOME}/logs"
    
    # Start logging
    log "Starting Eidolon Setup Wizard v${WIZARD_VERSION}"
    
    # Check dependencies
    check_dependencies
    
    # Check if already configured
    if [[ -f "${EIDOLON_HOME}/config/settings.yaml" ]]; then
        echo -e "${YELLOW}Eidolon appears to already be configured.${NC}"
        echo -n "Run setup wizard anyway? (y/N): "
        read -r run_anyway
        if [[ "$run_anyway" != "y" && "$run_anyway" != "Y" ]]; then
            echo "Setup cancelled. Use 'eidolon config' to modify settings."
            exit 0
        fi
    fi
    
    # Run wizard steps
    show_welcome
    check_requirements
    configure_privacy
    configure_ai_services
    configure_monitoring
    configure_autostart
    generate_configuration
    setup_service
    run_initial_test
    show_completion
    
    log "Setup wizard completed successfully"
}

# Parse command line arguments
case "${1:-setup}" in
    "setup"|"")
        main
        ;;
    "--help"|"-h")
        echo "Eidolon Setup Wizard"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup    - Run interactive setup wizard (default)"
        echo "  --help   - Show this help message"
        echo ""
        exit 0
        ;;
    *)
        error "Unknown command: $1"
        exit 1
        ;;
esac