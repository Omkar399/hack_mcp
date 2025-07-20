"""
Main entry point for Eidolon when run as a module.

Allows the package to be executed directly with:
    python -m eidolon [commands...]
"""

from .cli.main import main

if __name__ == "__main__":
    main()