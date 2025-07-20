# Eidolon Repository Structure

This document describes the organized structure of the Eidolon AI Personal Assistant repository after cleanup and reorganization.

## 📁 Directory Structure

```
eidolon/
├── 📄 README.md                          # Main project documentation
├── 📄 CLAUDE.md                          # Claude Code configuration
├── 📄 .gitignore                         # Git ignore patterns
├── 📄 pyproject.toml                     # Python project configuration
├── 📄 setup.py                           # Python package setup
├── 📄 requirements.txt                   # Production dependencies
├── 📄 requirements-dev.txt               # Development dependencies
│
├── 🧪 test_phase1.py                     # Phase 1 validation script
├── 🧪 test_phase2.py                     # Phase 2 validation script
├── 🧪 test_phase3.py                     # Phase 3 validation script
│
├── 📂 src/eidolon/                       # Main source code
│   ├── 📄 __init__.py                    # Package initialization
│   │
│   ├── 📂 cli/                           # Command-line interface
│   │   ├── 📄 __init__.py
│   │   └── 📄 main.py                    # CLI entry point
│   │
│   ├── 📂 core/                          # Core system components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 observer.py                # Screenshot capture & monitoring
│   │   ├── 📄 analyzer.py                # OCR & AI content analysis
│   │   ├── 📄 memory.py                  # Memory management system
│   │   └── 📄 interface.py               # User interface components
│   │
│   ├── 📂 models/                        # AI model integrations
│   │   ├── 📄 __init__.py
│   │   ├── 📄 local_vision.py            # Local vision models (Florence-2)
│   │   ├── 📄 cloud_api.py               # Cloud AI API integrations
│   │   └── 📄 decision_engine.py         # AI routing & decision logic
│   │
│   ├── 📂 storage/                       # Data storage & management
│   │   ├── 📄 __init__.py
│   │   ├── 📄 metadata_db.py             # SQLite database management
│   │   ├── 📄 vector_db.py               # Vector database (ChromaDB)
│   │   └── 📄 file_manager.py            # File system operations
│   │
│   └── 📂 utils/                         # Shared utilities
│       ├── 📄 __init__.py
│       ├── 📄 config.py                  # Configuration management
│       ├── 📄 logging.py                 # Logging utilities
│       └── 📄 monitoring.py              # Performance monitoring
│
├── 📂 config/                            # Configuration files
│   ├── 📄 settings.yaml                  # Main configuration (8GB memory)
│   ├── 📄 settings-high-performance.yaml # High-performance template (16GB)
│   └── 📄 logging.yaml                   # Logging configuration
│
├── 📂 docs/                              # Documentation
│   ├── 📂 user-guide/                    # User documentation
│   │   └── 📄 INSTALL.md                 # Installation instructions
│   │
│   ├── 📂 development/                   # Development documentation
│   │   ├── 📄 CONTRIBUTING.md            # Contribution guidelines
│   │   ├── 📄 PROGRESS_PLAN.md           # Development roadmap
│   │   └── 📄 REPOSITORY_STRUCTURE.md   # This file
│   │
│   ├── 📂 api/                           # API documentation (Phase 4)
│   └── 📂 examples/                      # Usage examples
│       └── 📄 USAGE_EXAMPLES.md          # Practical usage examples
│
├── 📂 tests/                             # Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_config.py                 # Configuration tests
│   └── 📄 test_observer.py               # Observer component tests
│
├── 📂 eidolon_env/                       # Virtual environment (gitignored)
├── 📂 data/                              # Generated data (gitignored)
│   ├── 📄 eidolon.db                     # SQLite database
│   └── 📂 screenshots/                   # Captured screenshots
└── 📂 logs/                              # Log files (gitignored)
    └── 📄 eidolon.log                    # Application logs
```

## 🧹 Cleanup Summary

### Removed Items
- ✅ **Empty directories**: `docs/` (empty), `data/extracted/`, `data/models/`
- ✅ **Temporary files**: `htmlcov/` (coverage reports), `__pycache__/`, `*.pyc`
- ✅ **Build artifacts**: Various temporary build files

### Reorganized Items
- ✅ **Documentation**: Moved to structured `docs/` hierarchy
  - `INSTALL.md` → `docs/user-guide/INSTALL.md`
  - `PROGRESS_PLAN.md` → `docs/development/PROGRESS_PLAN.md`
- ✅ **New documentation**: Added comprehensive guides
  - `docs/development/CONTRIBUTING.md`
  - `docs/examples/USAGE_EXAMPLES.md`
  - `docs/development/REPOSITORY_STRUCTURE.md`

### Updated Files
- ✅ **README.md**: Updated with new structure and documentation links
- ✅ **.gitignore**: Enhanced with Eidolon-specific patterns
- ✅ **Documentation**: All `.md` files updated with proper cross-references

## 🏗️ Architecture Overview

### Core Components
1. **Observer** (`src/eidolon/core/observer.py`)
   - Screenshot capture with intelligent change detection
   - Activity monitoring (keyboard, mouse, window changes)
   - Resource management and performance optimization

2. **Analyzer** (`src/eidolon/core/analyzer.py`)
   - OCR text extraction (Tesseract + EasyOCR)
   - Florence-2 vision model integration
   - Content classification and scene analysis
   - AI-enhanced understanding with fallbacks

3. **Storage** (`src/eidolon/storage/`)
   - SQLite database with FTS5 search
   - Vector database preparation (ChromaDB)
   - File system management

4. **Interface** (`src/eidolon/cli/main.py`)
   - Command-line interface
   - Search and query functionality
   - System status and management

## 🎯 Phase Status

### ✅ Completed Phases
- **Phase 1: Foundation** - Basic capture and monitoring
- **Phase 2: Intelligent Capture** - OCR and content analysis
- **Phase 3: Local AI Integration** - Florence-2 vision model

### 🚧 Current Status
All completed phases validated and working after cleanup:
- ✅ Phase 1: 7/7 tests passed
- ✅ Phase 2: 7/7 tests passed  
- ✅ Phase 3: All components operational

### 📋 Next Phase
**Phase 4: Cloud AI & Semantic Memory**
- Vector database implementation
- Cloud AI API integration
- Natural language query processing
- Semantic search capabilities

## 🔧 Development Workflow

### Setup
```bash
git clone <repository>
cd eidolon
python3 -m venv eidolon_env
source eidolon_env/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### Testing
```bash
# Phase validation
python test_phase1.py
python test_phase2.py
python test_phase3.py

# Unit tests
pytest

# Coverage
pytest --cov=src/eidolon
```

### Documentation
- User guides in `docs/user-guide/`
- Development docs in `docs/development/`
- Examples in `docs/examples/`
- API docs in `docs/api/` (Phase 4)

## 📝 Configuration

### Performance Settings
- **Standard**: 8GB memory, 20% CPU (for 8GB+ systems)
- **High-Performance**: 16GB memory, 30% CPU (for 16GB+ systems)
- **Memory-Optimized**: Configured for AI model loading

### Security
- Local-first architecture
- Privacy-preserving design
- Configurable data retention
- Sensitive data auto-redaction

## 🔍 File Naming Conventions

### Screenshots
- Format: `screenshot_YYYYMMDD_HHMMSS_mmm_hash.png`
- Metadata: `screenshot_YYYYMMDD_HHMMSS_mmm_hash.json`

### Tests
- Phase tests: `test_phase{N}.py`
- Unit tests: `test_{component}.py`

### Documentation
- User guides: Descriptive names in `docs/user-guide/`
- Dev docs: Uppercase names in `docs/development/`
- Examples: Descriptive names in `docs/examples/`

---

This structure provides a clean, maintainable, and scalable foundation for the Eidolon AI Personal Assistant project. 🚀