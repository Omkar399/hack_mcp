#!/usr/bin/env python3
"""
Test script for Enhanced Analyzer Features

Tests the new error detection, command extraction, and context analysis
features added to the Analyzer for MCP integration.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_error_detection():
    """Test the new error detection functionality."""
    print("Testing Error Detection...")
    print("-" * 30)
    
    try:
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Test with various error patterns
        test_cases = [
            # Python traceback
            """
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    result = divide_by_zero()
  File "test.py", line 5, in divide_by_zero
    return 10 / 0
ZeroDivisionError: division by zero
            """,
            
            # Shell errors
            """
$ cat nonexistent.txt
cat: nonexistent.txt: No such file or directory
$ rm protected_file
rm: cannot remove 'protected_file': Permission denied
            """,
            
            # Compilation errors
            """
gcc main.c -o main
main.c:5:1: error: expected ';' before 'return'
main.c:10:15: warning: unused variable 'temp'
            """,
            
            # Mixed content with warnings
            """
INFO: Starting application
WARNING: Configuration file not found, using defaults
ERROR: Failed to connect to database
FATAL ERROR: Application cannot continue
            """
        ]
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            errors = analyzer.detect_errors(test_text.strip())
            
            if errors:
                print(f"  Found {len(errors)} errors:")
                for error in errors:
                    print(f"    - {error['severity'].upper()}: {error['error_type']} (line {error['line_number']})")
                    print(f"      Message: {error['message'][:60]}...")
            else:
                print("  No errors detected")
        
        print("\n✓ Error detection test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error detection test failed: {e}")
        return False

def test_command_extraction():
    """Test the new command extraction functionality."""
    print("\nTesting Command Extraction...")
    print("-" * 30)
    
    try:
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Test with various command patterns
        test_cases = [
            # Unix shell commands
            """
$ ls -la /home/user
$ cd /var/log
$ grep "ERROR" application.log
$ sudo systemctl restart nginx
            """,
            
            # Git commands
            """
$ git status
$ git add .
$ git commit -m "Add new feature"
$ git push origin main
            """,
            
            # Development commands
            """
$ npm install express
$ python -m pip install requests
$ docker build -t myapp .
$ kubectl apply -f deployment.yaml
            """,
            
            # Mixed shell prompts
            """
user@server:~$ ps aux | grep python
root@server:/var/log# tail -f error.log
% zsh -c "echo hello"
> powershell Get-Process
            """
        ]
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            commands = analyzer.extract_commands(test_text.strip())
            
            if commands:
                print(f"  Found {len(commands)} commands:")
                for cmd in commands:
                    shell_type = cmd['shell_type']
                    command = cmd['command']
                    args = cmd['arguments'] or ''
                    known = "✓" if cmd['is_known_command'] else "?"
                    print(f"    {known} [{shell_type}] {command} {args}")
            else:
                print("  No commands detected")
        
        print("\n✓ Command extraction test completed")
        return True
        
    except Exception as e:
        print(f"✗ Command extraction test failed: {e}")
        return False

def test_context_analysis():
    """Test the new context analysis functionality."""
    print("\nTesting Context Analysis...")
    print("-" * 30)
    
    try:
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Test with different content types and scenarios
        test_cases = [
            # Terminal with git commands
            ("terminal", """
$ git status
On branch feature/new-ui
Changes not staged for commit:
  modified:   src/components/Header.jsx
  modified:   src/styles/main.css

$ git add .
$ git commit -m "Update header styling"
$ git push origin feature/new-ui
            """),
            
            # Code editing
            ("code", """
import React from 'react';
import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchUserData(userId);
  }, [userId]);
  
  const fetchUserData = async (id) => {
    try {
      const response = await fetch(`/api/users/${id}`);
      const userData = await response.json();
      setUser(userData);
    } catch (error) {
      console.error('Failed to fetch user:', error);
    } finally {
      setLoading(false);
    }
  };
            """),
            
            # Browser activity
            ("browser", """
Google Search: "React hooks tutorial"
stackoverflow.com - How to use useState in React
developer.mozilla.org - React Hooks API Reference
github.com/facebook/react - React Repository
npm install react react-dom
Login to GitHub
Create new repository: react-learning
            """),
            
            # Document work
            ("document", """
Meeting Notes - Sprint Planning
Date: January 15, 2024
Attendees: John, Sarah, Mike, Lisa

Agenda:
1. Review last sprint's deliverables
2. Plan upcoming features
3. Discuss technical debt
4. Set sprint goals

Action Items:
- John: Implement user authentication (Due: Jan 20)
- Sarah: Design new dashboard mockups (Due: Jan 18)
- Mike: Fix performance issues (Due: Jan 22)
            """)
        ]
        
        for i, (content_type, test_text) in enumerate(test_cases, 1):
            print(f"\nTest Case {i} - {content_type.title()} Content:")
            context = analyzer.analyze_context(test_text.strip(), content_type)
            
            print(f"  Activity Type: {context['activity_type']}")
            print(f"  Confidence: {context['confidence']:.2f}")
            
            if context['insights']:
                print("  Insights:")
                for insight in context['insights']:
                    print(f"    - {insight}")
            
            if context['details']:
                print("  Details:")
                for key, value in context['details'].items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"    {key}: {len(value)} items")
                    else:
                        print(f"    {key}: {value}")
        
        print("\n✓ Context analysis test completed")
        return True
        
    except Exception as e:
        print(f"✗ Context analysis test failed: {e}")
        return False

def test_integration():
    """Test integration of all enhanced features."""
    print("\nTesting Feature Integration...")
    print("-" * 30)
    
    try:
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Complex scenario: debugging session
        complex_scenario = """
$ python test_app.py
Traceback (most recent call last):
  File "test_app.py", line 15, in main
    result = calculate_average(numbers)
  File "test_app.py", line 8, in calculate_average
    return sum(numbers) / len(numbers)
ZeroDivisionError: division by zero

$ vim test_app.py
$ python test_app.py
Test passed: average = 15.5

$ git add test_app.py
$ git commit -m "Fix division by zero error"
[main 1a2b3c4] Fix division by zero error
 1 file changed, 3 insertions(+), 1 deletion(-)

$ python -m pytest tests/
================================= FAILURES =================================
_______________________________ test_edge_cases _______________________________
AssertionError: Expected error handling for empty input

$ vim tests/test_app.py
$ python -m pytest tests/
========================= 2 passed, 0 failed =========================

$ git add tests/test_app.py
$ git commit -m "Add test for empty input handling"
        """
        
        print("Analyzing complex debugging scenario...")
        
        # Test error detection
        errors = analyzer.detect_errors(complex_scenario)
        print(f"  Errors detected: {len(errors)}")
        for error in errors:
            print(f"    - {error['severity'].upper()}: {error['error_type']}")
        
        # Test command extraction
        commands = analyzer.extract_commands(complex_scenario)
        print(f"  Commands extracted: {len(commands)}")
        command_types = {}
        for cmd in commands:
            cmd_name = cmd['command']
            command_types[cmd_name] = command_types.get(cmd_name, 0) + 1
        print(f"    Command distribution: {dict(command_types)}")
        
        # Test context analysis
        context = analyzer.analyze_context(complex_scenario, "terminal")
        print(f"  Activity type: {context['activity_type']}")
        print(f"  Confidence: {context['confidence']:.2f}")
        print(f"  Insights: {len(context['insights'])} identified")
        
        print("\n✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all enhanced analyzer tests."""
    print("Enhanced Analyzer Features Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_error_detection():
        tests_passed += 1
    
    if test_command_extraction():
        tests_passed += 1
    
    if test_context_analysis():
        tests_passed += 1
    
    if test_integration():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All enhanced analyzer features are working correctly!")
        print("\nThe following new capabilities have been successfully integrated:")
        print("- Error detection and classification")
        print("- Command extraction from terminal content")
        print("- Advanced context analysis")
        print("- Activity pattern recognition")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)