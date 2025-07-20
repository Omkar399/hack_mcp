#!/bin/bash

# Eidolon AI Personal Assistant - Production Installation Script
# This script provides one-click installation with comprehensive dependency management

set -euo pipefail

# Configuration
EIDOLON_VERSION="0.2.0"
EIDOLON_HOME="${HOME}/.eidolon"
EIDOLON_DATA="${EIDOLON_HOME}/data"
EIDOLON_LOGS="${EIDOLON_HOME}/logs"
EIDOLON_CONFIG="${EIDOLON_HOME}/config"
EIDOLON_SERVICE_USER="${USER}"
PYTHON_MIN_VERSION="3.9"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
INSTALL_LOG="${EIDOLON_HOME}/install.log"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${INSTALL_LOG}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "${INSTALL_LOG}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "${INSTALL_LOG}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS="macos" ;;
        Linux*)     OS="linux" ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*) OS="windows" ;;
        *)          OS="unknown" ;;
    esac
    log "Detected OS: $OS"
}

# Function to check Python version
check_python() {
    log "Checking Python installation..."
    
    # Try different Python commands
    for cmd in python3 python; do
        if command -v "$cmd" &> /dev/null; then
            PYTHON_CMD="$cmd"
            PYTHON_VERSION=$($cmd --version 2>&1 | cut -d' ' -f2)
            log "Found Python: $cmd (version $PYTHON_VERSION)"
            
            # Check if version meets minimum requirement
            if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
                log "Python version meets requirements (>= $PYTHON_MIN_VERSION)"
                return 0
            else
                warn "Python version $PYTHON_VERSION does not meet minimum requirement ($PYTHON_MIN_VERSION)"
            fi
        fi
    done
    
    error "Python $PYTHON_MIN_VERSION or higher is required but not found"
    return 1
}

# Function to install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    case "$OS" in
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                log "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            # Install dependencies
            log "Installing macOS dependencies..."
            brew update
            brew install tesseract python@3.11 ffmpeg
            
            # Install Python if needed
            if ! check_python; then
                brew install python@3.11
                export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
            fi
            ;;
            
        "linux")
            # Detect Linux distribution
            if command -v apt-get &> /dev/null; then
                log "Installing Ubuntu/Debian dependencies..."
                sudo apt-get update
                sudo apt-get install -y \
                    python3 python3-pip python3-venv python3-dev \
                    tesseract-ocr tesseract-ocr-eng \
                    libgl1-mesa-glx libglib2.0-0 \
                    libxrandr2 libxss1 libxcursor1 libxcomposite1 \
                    libasound2 libxi6 libxtst6 \
                    build-essential libffi-dev libssl-dev \
                    ffmpeg
                    
            elif command -v yum &> /dev/null; then
                log "Installing CentOS/RHEL dependencies..."
                sudo yum update -y
                sudo yum install -y \
                    python3 python3-pip python3-devel \
                    tesseract tesseract-langpack-eng \
                    mesa-libGL glib2 \
                    libXrandr libXScrnSaver libXcursor libXcomposite \
                    alsa-lib libXi libXtst \
                    gcc gcc-c++ libffi-devel openssl-devel \
                    ffmpeg
                    
            elif command -v pacman &> /dev/null; then
                log "Installing Arch Linux dependencies..."
                sudo pacman -Syu --noconfirm
                sudo pacman -S --noconfirm \
                    python python-pip \
                    tesseract tesseract-data-eng \
                    mesa glib2 \
                    libxrandr libxss libxcursor libxcomposite \
                    alsa-lib libxi libxtst \
                    base-devel libffi openssl \
                    ffmpeg
            else
                warn "Unsupported Linux distribution. Please install dependencies manually."
            fi
            ;;
            
        "windows")
            error "Windows installation not yet supported. Please use WSL2 with Ubuntu."
            exit 1
            ;;
            
        *)
            error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
}

# Function to create directories
create_directories() {
    log "Creating Eidolon directories..."
    
    mkdir -p "${EIDOLON_HOME}"
    mkdir -p "${EIDOLON_DATA}/screenshots"
    mkdir -p "${EIDOLON_DATA}/vector_db"
    mkdir -p "${EIDOLON_LOGS}"
    mkdir -p "${EIDOLON_CONFIG}"
    mkdir -p "${EIDOLON_HOME}/models"
    mkdir -p "${EIDOLON_HOME}/backup"
    mkdir -p "${EIDOLON_HOME}/temp"
    
    # Set appropriate permissions
    chmod 700 "${EIDOLON_HOME}"
    chmod 755 "${EIDOLON_DATA}"
    chmod 755 "${EIDOLON_LOGS}"
    
    log "Created directory structure in ${EIDOLON_HOME}"
}

# Function to create Python virtual environment
create_venv() {
    log "Creating Python virtual environment..."
    
    local venv_path="${EIDOLON_HOME}/venv"
    
    if [[ -d "$venv_path" ]]; then
        warn "Virtual environment already exists. Removing and recreating..."
        rm -rf "$venv_path"
    fi
    
    "$PYTHON_CMD" -m venv "$venv_path"
    
    # Activate virtual environment
    source "$venv_path/bin/activate"
    
    # Upgrade pip and install wheel
    pip install --upgrade pip wheel setuptools
    
    log "Created virtual environment at $venv_path"
}

# Function to install Eidolon
install_eidolon() {
    log "Installing Eidolon..."
    
    local venv_path="${EIDOLON_HOME}/venv"
    source "$venv_path/bin/activate"
    
    # Install from PyPI or local development
    if [[ -f "pyproject.toml" ]]; then
        log "Installing from local source..."
        pip install -e .
    else
        log "Installing from PyPI..."
        pip install eidolon=="${EIDOLON_VERSION}"
    fi
    
    # Install optional dependencies based on system capabilities
    log "Installing optional dependencies..."
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected, installing GPU acceleration packages..."
        pip install "eidolon[gpu]"
    fi
    
    # Install enterprise features if requested
    if [[ "${INSTALL_ENTERPRISE:-false}" == "true" ]]; then
        log "Installing enterprise features..."
        pip install "eidolon[enterprise]"
    fi
    
    log "Eidolon installation completed"
}

# Function to configure Eidolon
configure_eidolon() {
    log "Configuring Eidolon..."
    
    # Copy default configuration
    local config_source="./eidolon/config"
    local config_dest="${EIDOLON_CONFIG}"
    
    if [[ -d "$config_source" ]]; then
        cp -r "$config_source"/* "$config_dest/"
    else
        # Create basic configuration
        cat > "${config_dest}/settings.yaml" <<EOF
# Eidolon Production Configuration
observer:
  capture_interval: 30
  storage_path: "${EIDOLON_DATA}/screenshots"
  max_storage_gb: 100

analysis:
  cloud_apis:
    gemini_key: "\${GEMINI_API_KEY}"
    claude_key: "\${CLAUDE_API_KEY}"
    openai_key: "\${OPENAI_API_KEY}"
  routing:
    local_first: true
    importance_threshold: 0.8

memory:
  db_path: "${EIDOLON_DATA}/eidolon.db"
  vector_db: "chromadb"

logging:
  level: "INFO"
  file_path: "${EIDOLON_LOGS}/eidolon.log"

privacy:
  local_only_mode: false
  auto_redaction: true
  encrypt_at_rest: true

monitoring:
  enabled: true
  metrics_collection_interval: 60
EOF
    fi
    
    # Create environment file
    cat > "${EIDOLON_HOME}/.env" <<EOF
# Eidolon Environment Configuration
EIDOLON_HOME=${EIDOLON_HOME}
EIDOLON_CONFIG=${EIDOLON_CONFIG}
EIDOLON_DATA=${EIDOLON_DATA}
EIDOLON_LOGS=${EIDOLON_LOGS}

# API Keys (configure these with your actual keys)
# GEMINI_API_KEY=your_gemini_key_here
# CLAUDE_API_KEY=your_claude_key_here
# OPENAI_API_KEY=your_openai_key_here
EOF
    
    chmod 600 "${EIDOLON_HOME}/.env"
    
    log "Configuration completed"
}

# Function to create shell integration
create_shell_integration() {
    log "Setting up shell integration..."
    
    local shell_config=""
    case "$SHELL" in
        */bash) shell_config="$HOME/.bashrc" ;;
        */zsh) shell_config="$HOME/.zshrc" ;;
        */fish) shell_config="$HOME/.config/fish/config.fish" ;;
        *) warn "Unsupported shell: $SHELL" ;;
    esac
    
    if [[ -n "$shell_config" ]]; then
        # Add Eidolon to PATH
        local eidolon_bin="${EIDOLON_HOME}/venv/bin"
        
        if ! grep -q "EIDOLON_HOME" "$shell_config" 2>/dev/null; then
            echo "" >> "$shell_config"
            echo "# Eidolon AI Personal Assistant" >> "$shell_config"
            echo "export EIDOLON_HOME=\"${EIDOLON_HOME}\"" >> "$shell_config"
            echo "export PATH=\"${eidolon_bin}:\$PATH\"" >> "$shell_config"
            echo "source \"${EIDOLON_HOME}/.env\"" >> "$shell_config"
            
            log "Added Eidolon to $shell_config"
            warn "Please run 'source $shell_config' or restart your terminal to activate"
        fi
    fi
    
    # Create activation script
    cat > "${EIDOLON_HOME}/activate.sh" <<EOF
#!/bin/bash
# Eidolon Environment Activation Script
export EIDOLON_HOME="${EIDOLON_HOME}"
source "${EIDOLON_HOME}/.env"
source "${EIDOLON_HOME}/venv/bin/activate"
export PATH="${EIDOLON_HOME}/venv/bin:\$PATH"
echo "Eidolon environment activated"
EOF
    
    chmod +x "${EIDOLON_HOME}/activate.sh"
}

# Function to run initial setup
run_initial_setup() {
    log "Running initial setup..."
    
    local venv_path="${EIDOLON_HOME}/venv"
    source "$venv_path/bin/activate"
    source "${EIDOLON_HOME}/.env"
    
    # Initialize database
    log "Initializing database..."
    python -m eidolon init-db || warn "Database initialization failed - will retry on first run"
    
    # Download initial models (optional)
    if [[ "${DOWNLOAD_MODELS:-true}" == "true" ]]; then
        log "Downloading AI models (this may take a while)..."
        python -m eidolon download-models --quiet || warn "Model download failed - will download on first use"
    fi
    
    # Run health check
    log "Running health check..."
    python -m eidolon status || warn "Health check failed - please check configuration"
    
    log "Initial setup completed"
}

# Function to show post-install instructions
show_post_install() {
    log "Installation completed successfully!"
    echo ""
    echo -e "${GREEN}Eidolon AI Personal Assistant v${EIDOLON_VERSION} has been installed${NC}"
    echo ""
    echo "Installation Directory: ${EIDOLON_HOME}"
    echo "Configuration: ${EIDOLON_CONFIG}/settings.yaml"
    echo "Data Directory: ${EIDOLON_DATA}"
    echo "Logs: ${EIDOLON_LOGS}/eidolon.log"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Configure your API keys in ${EIDOLON_HOME}/.env"
    echo "2. Review configuration in ${EIDOLON_CONFIG}/settings.yaml"
    echo "3. Start Eidolon: 'eidolon capture' or 'eidolon status'"
    echo "4. Set up autostart: run 'bash ${EIDOLON_HOME}/scripts/setup-service.sh'"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  eidolon status          - Check system status"
    echo "  eidolon capture         - Start screenshot capture"
    echo "  eidolon search 'query'  - Search captured content"
    echo "  eidolon chat            - Start interactive chat"
    echo ""
    echo -e "${YELLOW}Note:${NC} Please restart your terminal or run 'source ~/.bashrc' to activate"
    echo ""
}

# Function to handle errors
cleanup_on_error() {
    error "Installation failed. Check the log at ${INSTALL_LOG}"
    error "You may need to run the installation again or install dependencies manually"
    exit 1
}

# Main installation function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Eidolon AI Personal Assistant                             ║"
    echo "║                         Production Installer                                 ║"
    echo "║                              v${EIDOLON_VERSION}                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Set trap for cleanup on error
    trap cleanup_on_error ERR
    
    # Create initial directories for logging
    mkdir -p "${EIDOLON_HOME}"
    touch "${INSTALL_LOG}"
    
    log "Starting Eidolon installation..."
    
    # Pre-installation checks
    check_root
    detect_os
    
    # Installation steps
    install_system_deps
    check_python || { error "Python check failed"; exit 1; }
    create_directories
    create_venv
    install_eidolon
    configure_eidolon
    create_shell_integration
    run_initial_setup
    
    # Show completion message
    show_post_install
    
    log "Installation completed successfully"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --enterprise)
            INSTALL_ENTERPRISE=true
            shift
            ;;
        --no-models)
            DOWNLOAD_MODELS=false
            shift
            ;;
        --help)
            echo "Eidolon Installation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --enterprise    Install enterprise features"
            echo "  --no-models     Skip downloading AI models"
            echo "  --help          Show this help message"
            echo ""
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main "$@"