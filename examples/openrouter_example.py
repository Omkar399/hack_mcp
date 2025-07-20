#!/usr/bin/env python3
"""
OpenRouter.ai Claude Integration Example

This example demonstrates how to use Eidolon with OpenRouter.ai for cost-effective
Claude API access. Based on the user's provided example code.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def example_openrouter_direct():
    """
    Direct OpenRouter Claude integration example.
    This matches the user's provided example code.
    """
    print("🔗 OpenRouter.ai Direct Integration Example")
    print("=" * 50)
    
    try:
        from eidolon.models.cloud_api import OpenRouterClaudeAPI
        
        # Check if API key is available
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠️  OPENROUTER_API_KEY environment variable not set")
            print("   Set it with: export OPENROUTER_API_KEY='sk-or-v1-your-key-here'")
            return
        
        # Initialize OpenRouter Claude API
        openrouter_claude = OpenRouterClaudeAPI(api_key)
        
        if not openrouter_claude.available:
            print("❌ OpenRouter Claude API not available")
            return
        
        print("✅ OpenRouter Claude API initialized successfully")
        
        # Example using the direct call_claude_sonnet method
        user_prompt = "What is the capital of France?"
        print(f"\n📝 Prompt: {user_prompt}")
        
        response = openrouter_claude.call_claude_sonnet(user_prompt)
        print(f"🤖 Claude Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_eidolon_integration():
    """
    Example using OpenRouter Claude through Eidolon's cloud API manager.
    """
    print("\n🎯 Eidolon Cloud API Manager Integration")
    print("=" * 50)
    
    try:
        from eidolon.models.cloud_api import CloudAPIManager
        
        # Initialize cloud API manager
        api_manager = CloudAPIManager()
        
        # Check available providers
        providers = api_manager.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        if "openrouter_claude" not in providers:
            print("⚠️  OpenRouter Claude not available (check API key)")
            return
        
        # Example text analysis
        test_text = "Artificial intelligence is transforming how we interact with computers."
        print(f"\n📝 Analyzing text: {test_text}")
        
        response = await api_manager.analyze_text(
            text=test_text,
            analysis_type="general",
            preferred_provider="openrouter_claude"
        )
        
        if response:
            print(f"✅ Analysis completed")
            print(f"   - Model: {response.model}")
            print(f"   - Provider: {response.provider}")
            print(f"   - Confidence: {response.confidence}")
            print(f"   - Usage: {response.usage}")
            print(f"   - Response: {response.content[:200]}...")
        else:
            print("❌ Analysis failed")
        
        # Show usage statistics
        stats = api_manager.get_usage_stats()
        print(f"\n📊 Usage Statistics:")
        print(f"   - Total requests: {stats['total_requests']}")
        print(f"   - Total cost: ${stats['total_cost_usd']}")
        print(f"   - OpenRouter requests: {stats['by_provider']['openrouter_claude']['requests']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_image_analysis():
    """
    Example of image analysis using OpenRouter Claude.
    """
    print("\n🖼️  OpenRouter Claude Image Analysis")
    print("=" * 50)
    
    try:
        from eidolon.models.cloud_api import CloudAPIManager
        
        # Look for screenshot files to analyze
        screenshots_dir = Path("data/screenshots")
        if not screenshots_dir.exists():
            print("⚠️  No screenshots directory found")
            print("   Run 'python -m eidolon capture' first to capture some screenshots")
            return
        
        # Find a recent screenshot
        screenshots = list(screenshots_dir.glob("*.png"))
        if not screenshots:
            print("⚠️  No screenshots found to analyze")
            return
        
        # Use the most recent screenshot
        latest_screenshot = max(screenshots, key=lambda p: p.stat().st_mtime)
        print(f"📸 Analyzing screenshot: {latest_screenshot.name}")
        
        # Initialize API manager
        api_manager = CloudAPIManager()
        
        if "openrouter_claude" not in api_manager.get_available_providers():
            print("⚠️  OpenRouter Claude not available")
            return
        
        # Analyze the image
        response = await api_manager.analyze_image(
            image_path=latest_screenshot,
            prompt="Describe what you see in this screenshot. What application or activity does it show?",
            preferred_provider="openrouter_claude"
        )
        
        if response:
            print(f"✅ Image analysis completed")
            print(f"   - Model: {response.model}")
            print(f"   - Analysis: {response.content}")
        else:
            print("❌ Image analysis failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all OpenRouter integration examples."""
    print("🎯 Eidolon OpenRouter.ai Integration Examples")
    print("=" * 60)
    
    # Check if OpenRouter API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not found in environment variables")
        print("\n💡 To use OpenRouter.ai with Eidolon:")
        print("1. Sign up at https://openrouter.ai/")
        print("2. Get your API key from the dashboard")
        print("3. Export it: export OPENROUTER_API_KEY='sk-or-v1-your-key-here'")
        print("4. Re-run this example")
        return
    
    # Run direct integration example
    example_openrouter_direct()
    
    # Run async examples
    asyncio.run(example_eidolon_integration())
    asyncio.run(example_image_analysis())
    
    print("\n✨ OpenRouter integration examples completed!")
    print("\n📚 Next steps:")
    print("- Explore Eidolon's automatic AI analysis with 'python -m eidolon capture'")
    print("- Use semantic search with 'python -m eidolon search \"your query\"'")
    print("- Check system status with 'python -m eidolon status'")


if __name__ == "__main__":
    main()