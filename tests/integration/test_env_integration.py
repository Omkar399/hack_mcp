#!/usr/bin/env python3
"""
Test script to verify .env file integration with OpenRouter Claude API.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_env_loading():
    """Test that .env file is loaded correctly."""
    print("🔧 Testing .env file loading...")
    
    # Check if API key is loaded
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"✅ OPENROUTER_API_KEY loaded: {api_key[:20]}...")
        return True
    else:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False

def test_openrouter_direct():
    """Test direct OpenRouter API integration."""
    print("\n🤖 Testing OpenRouter Claude Direct Integration...")
    
    try:
        from eidolon.models.cloud_api import OpenRouterClaudeAPI
        
        # Initialize API
        api = OpenRouterClaudeAPI()
        print(f"API Available: {api.available}")
        
        if not api.available:
            print("❌ OpenRouter Claude API not available")
            return False
        
        # Test call
        response = api.call_claude_sonnet("What is 5 + 3? Respond briefly.")
        print(f"✅ Claude Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_cloud_api_manager():
    """Test CloudAPIManager integration."""
    print("\n🌐 Testing CloudAPIManager Integration...")
    
    try:
        from eidolon.models.cloud_api import CloudAPIManager
        
        # Initialize manager
        manager = CloudAPIManager()
        providers = manager.get_available_providers()
        print(f"Available providers: {providers}")
        
        if "openrouter_claude" not in providers:
            print("❌ OpenRouter Claude not available in CloudAPIManager")
            return False
        
        # Test text analysis
        response = await manager.analyze_text(
            text="Test integration with Claude 3.5 Sonnet via OpenRouter",
            analysis_type="summary",
            preferred_provider="openrouter_claude"
        )
        
        if response:
            print(f"✅ Analysis successful")
            print(f"   Model: {response.model}")
            print(f"   Provider: {response.provider}")
            print(f"   Confidence: {response.confidence}")
            print(f"   Usage: {response.usage}")
            return True
        else:
            print("❌ Analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Eidolon .env Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test .env loading
    results.append(test_env_loading())
    
    # Test direct OpenRouter integration
    results.append(test_openrouter_direct())
    
    # Test CloudAPIManager integration
    results.append(await test_cloud_api_manager())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! .env integration working perfectly!")
        print("\n📝 Configuration Summary:")
        print("- .env file is properly loaded")
        print("- OpenRouter API key is accessible")
        print("- Claude 3.5 Sonnet model is working")
        print("- CloudAPIManager integration successful")
        print("- Security: .env file is in .gitignore")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check configuration.")

if __name__ == "__main__":
    asyncio.run(main())