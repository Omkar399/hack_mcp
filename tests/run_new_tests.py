#!/usr/bin/env python3
"""
Test runner for new Eidolon functionality.

This script runs all the newly created tests for EnrichMCP and chat functionality
to validate the test suite implementation.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr:
                print("Error:")
                print(result.stderr[-500:])
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False


def main():
    """Run all new test categories."""
    print("🧪 Eidolon New Functionality Test Runner")
    print("Testing EnrichMCP and Chat Integration")
    
    test_commands = [
        # Unit tests for new components
        (
            ["python", "-m", "pytest", "tests/unit/test_mcp_server.py", "-v", "--no-cov"],
            "MCP Server Unit Tests"
        ),
        (
            ["python", "-m", "pytest", "tests/unit/test_enhanced_interface.py", "-v", "--no-cov"],
            "Enhanced Interface Unit Tests"
        ),
        
        # Integration tests
        (
            ["python", "-m", "pytest", "tests/integration/test_mcp_integration.py", "-v", "--no-cov"],
            "MCP Integration Tests"
        ),
        (
            ["python", "-m", "pytest", "tests/integration/test_chat_integration.py", "-v", "--no-cov"],
            "Chat Integration Tests"
        ),
        (
            ["python", "-m", "pytest", "tests/integration/test_e2e_workflow.py", "-v", "--no-cov"],
            "End-to-End Workflow Tests"
        ),
        
        # Test collection verification
        (
            ["python", "-m", "pytest", "--collect-only", "tests/unit/test_mcp_server.py", "tests/unit/test_enhanced_interface.py"],
            "Test Collection Verification"
        ),
        
        # Quick smoke test of existing functionality
        (
            ["python", "-m", "pytest", "tests/unit/test_config.py", "-v", "--no-cov"],
            "Existing Functionality Smoke Test"
        ),
    ]
    
    results = []
    
    for cmd, description in test_commands:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("🏆 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
    
    print(f"\nOverall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Test suite is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())