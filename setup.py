"""
Setup script for Eidolon AI Personal Assistant

This script helps with installation, dependency management, and system permissions.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_system_permissions():
    """Check and request system permissions for screenshot capture."""
    system = platform.system().lower()
    
    print(f"\n🔍 Checking system permissions for {system}...")
    
    if system == "darwin":  # macOS
        print("\n📱 macOS System Permissions Required:")
        print("1. Screen Recording permission")
        print("2. Accessibility permission (for input monitoring)")
        print("\nTo grant permissions:")
        print("• Go to System Preferences > Security & Privacy > Privacy")
        print("• Select 'Screen Recording' and add Python/Terminal")
        print("• Select 'Accessibility' and add Python/Terminal")
        print("• You may need to restart the terminal after granting permissions")
        
        # Test screenshot capability
        try:
            import mss
            with mss.mss() as sct:
                sct.grab(sct.monitors[0])
            print("✅ Screenshot permissions appear to be working")
        except Exception as e:
            print(f"❌ Screenshot test failed: {e}")
            print("Please grant Screen Recording permissions and try again")
            return False
            
    elif system == "windows":  # Windows
        print("\n🪟 Windows System Information:")
        print("• Windows may show a security warning when first running")
        print("• Click 'Allow' when prompted for screen capture")
        print("• Some antivirus software may flag screen monitoring")
        print("• Add Eidolon to your antivirus whitelist if needed")
        
    elif system == "linux":  # Linux
        print("\n🐧 Linux System Information:")
        print("• X11 or Wayland display server required")
        print("• Some distributions may require additional packages:")
        print("  - python3-tk (for screenshot capture)")
        print("  - python3-dev (for building dependencies)")
        print("• Running in a virtual environment is recommended")
    
    return True


def install_tesseract():
    """Install or check Tesseract OCR."""
    print("\n🔤 Checking Tesseract OCR...")
    
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR not found")
    system = platform.system().lower()
    
    if system == "darwin":
        print("Install with Homebrew: brew install tesseract")
    elif system == "windows":
        print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Or install with chocolatey: choco install tesseract")
    elif system == "linux":
        print("Install with package manager:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  CentOS/RHEL: sudo yum install tesseract")
        print("  Arch: sudo pacman -S tesseract")
    
    return False


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data",
        "data/screenshots", 
        "data/extracted",
        "data/models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def setup_config():
    """Set up configuration files."""
    print("\n⚙️ Setting up configuration...")
    
    # Copy example config if main config doesn't exist
    config_path = Path("eidolon/config/settings.yaml")
    if not config_path.exists():
        print("✅ Configuration file already exists at eidolon/config/settings.yaml")
    else:
        print("ℹ️ Using existing configuration file")
    
    # Copy .env example
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        import shutil
        shutil.copy(env_example_path, env_path)
        print("✅ Created .env file from example")
        print("💡 Edit .env file to add your API keys")
    elif env_path.exists():
        print("ℹ️ .env file already exists")


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install main package
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)
        print("✅ Main package installed")
        
        # Ask about development dependencies
        response = input("\nInstall development dependencies? (y/N): ").lower()
        if response in ['y', 'yes']:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", ".[dev]"
            ], check=True)
            print("✅ Development dependencies installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_tests():
    """Run basic tests to verify installation."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        import eidolon
        from eidolon.core.observer import Observer
        from eidolon.utils.config import load_config
        print("✅ Core modules import successfully")
        
        # Test configuration loading
        print("Testing configuration...")
        config = load_config()
        print("✅ Configuration loads successfully")
        
        # Test observer creation (but don't start monitoring)
        print("Testing observer creation...")
        observer = Observer()
        print("✅ Observer creates successfully")
        
        print("\n🎉 Basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🤖 Eidolon AI Personal Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system permissions
    if not check_system_permissions():
        print("\n⚠️ Please grant necessary permissions and re-run setup")
        sys.exit(1)
    
    # Check Tesseract
    if not install_tesseract():
        print("\n⚠️ Please install Tesseract OCR and re-run setup")
        print("Installation can continue without Tesseract, but OCR won't work")
        response = input("Continue anyway? (y/N): ").lower()
        if response not in ['y', 'yes']:
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Setup configuration
    setup_config()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("\n⚠️ Setup completed but tests failed")
        print("Try running 'eidolon status' to check system state")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file to add API keys (optional)")
    print("2. Review eidolon/config/settings.yaml for customization")
    print("3. Run 'eidolon capture' to start monitoring")
    print("4. Run 'eidolon --help' to see all commands")
    
    print("\n📚 For more information:")
    print("• README.md - Full documentation")
    print("• CLAUDE.md - Technical specifications")
    print("• PROGRESS_PLAN.md - Development roadmap")


if __name__ == "__main__":
    main()