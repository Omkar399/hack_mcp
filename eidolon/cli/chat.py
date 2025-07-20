"""
Chat CLI module for Eidolon AI Personal Assistant

Provides interactive chat interface with LLM-powered responses
based on screen capture history.
"""

import asyncio
import sys
from typing import Optional
import click
from datetime import datetime

from ..core.interface import Interface
from ..utils.logging import get_logger


class ChatInterface:
    """Interactive chat interface for Eidolon."""
    
    def __init__(self, interface: Interface):
        self.interface = interface
        self.logger = get_logger(__name__)
        self.running = False
    
    async def start_chat_loop(self, interactive: bool = True):
        """Start the interactive chat loop."""
        self.running = True
        
        # Check if chat is available
        status = await self.interface.get_chat_status()
        if not status["available"]:
            click.echo("⚠️  No AI providers available. Please configure API keys in your environment:")
            click.echo("   - GEMINI_API_KEY for Google Gemini")
            click.echo("   - CLAUDE_API_KEY for Anthropic Claude")
            click.echo("   - OPENAI_API_KEY for OpenAI GPT")
            return
        
        # Show welcome message
        click.echo("\n🧠 Eidolon AI Assistant - Screen Memory Chat")
        click.echo(f"   Using {status['provider']} for responses")
        click.echo(f"   {status['memory_stats']['total_searchable_content']} searchable items in memory")
        click.echo("\nAsk me anything about your screen activity!")
        click.echo("Commands: /help, /clear, /suggestions, /status, /quit\n")
        
        while self.running and interactive:
            try:
                # Get user input
                user_input = click.prompt("You", type=str, prompt_suffix=": ")
                
                # Handle special commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue
                
                if not user_input.strip():
                    continue
                
                # Show thinking indicator
                click.echo("🤔 Thinking...", nl=False)
                
                # Process query
                result = await self.interface.process_query(user_input)
                
                # Clear thinking indicator and show response
                click.echo("\r" + " " * 20 + "\r", nl=False)  # Clear the line
                click.echo(f"🧠 Eidolon: {result.response}")
                
                # Show context sources if available
                if result.context_used and len(result.context_used) > 0:
                    click.echo(f"\n   📎 Based on {len(result.context_used)} relevant captures", dim=True)
                
            except KeyboardInterrupt:
                click.echo("\n\n👋 Chat interrupted. Type /quit to exit.")
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                click.echo(f"\n❌ Error: {e}")
    
    async def _handle_command(self, command: str):
        """Handle special chat commands."""
        cmd = command.lower().strip()
        
        if cmd in ["/quit", "/exit", "/q"]:
            self.running = False
            click.echo("👋 Goodbye!")
        
        elif cmd == "/help":
            click.echo("\n📚 Available commands:")
            click.echo("  /help        - Show this help message")
            click.echo("  /clear       - Clear conversation history")
            click.echo("  /suggestions - Show query suggestions")
            click.echo("  /status      - Show chat system status")
            click.echo("  /quit        - Exit chat\n")
        
        elif cmd == "/clear":
            self.interface.clear_conversation()
            click.echo("🧹 Conversation history cleared.")
        
        elif cmd == "/suggestions":
            suggestions = self.interface.get_suggestions()
            click.echo("\n💡 Try asking:")
            for i, suggestion in enumerate(suggestions, 1):
                click.echo(f"   {i}. {suggestion}")
            click.echo()
        
        elif cmd == "/status":
            status = await self.interface.get_chat_status()
            click.echo("\n📊 Chat System Status:")
            click.echo(f"   Provider: {status['provider']}")
            click.echo(f"   Available providers: {', '.join(status['available_providers'])}")
            click.echo(f"   Conversation length: {status['conversation_length']} turns")
            click.echo(f"   Searchable content: {status['memory_stats']['total_searchable_content']} items")
            
            if status['memory_stats'].get('cloud_usage'):
                usage = status['memory_stats']['cloud_usage']
                click.echo(f"   API requests today: {usage['total_requests']}")
                click.echo(f"   Estimated cost: ${usage['total_cost_usd']:.4f}")
            click.echo()
        
        else:
            click.echo(f"❓ Unknown command: {command}. Type /help for available commands.")
    
    async def process_single_query(self, query: str):
        """Process a single query without entering interactive mode."""
        # Check if chat is available
        status = await self.interface.get_chat_status()
        if not status["available"]:
            click.echo("⚠️  No AI providers available. Please configure API keys.", err=True)
            sys.exit(1)
        
        # Process query
        click.echo("🤔 Processing query...", nl=False)
        result = await self.interface.process_query(query)
        
        # Clear processing indicator and show response
        click.echo("\r" + " " * 20 + "\r", nl=False)
        click.echo(f"🧠 Eidolon: {result.response}")
        
        # Show context sources if available
        if result.context_used and len(result.context_used) > 0:
            click.echo(f"\n📎 Based on {len(result.context_used)} relevant captures", dim=True)
            
            if click.confirm("\nShow context details?", default=False):
                click.echo("\n📄 Context used:")
                for i, ctx in enumerate(result.context_used, 1):
                    timestamp_str = ctx.timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(ctx.timestamp, datetime) else str(ctx.timestamp)
                    click.echo(f"\n{i}. {timestamp_str} (relevance: {ctx.similarity_score:.2f})")
                    
                    # Show metadata
                    if ctx.metadata:
                        app = ctx.metadata.get("app_name", "Unknown")
                        window = ctx.metadata.get("window_title", "")
                        if window:
                            click.echo(f"   App: {app} - {window}")
                        else:
                            click.echo(f"   App: {app}")
                    
                    # Show content preview
                    content_preview = ctx.content[:200]
                    if len(ctx.content) > 200:
                        content_preview += "..."
                    click.echo(f"   Content: {content_preview}")


async def run_chat(interface: Interface, query: Optional[str] = None):
    """Run the chat interface."""
    chat = ChatInterface(interface)
    
    if query:
        # Single query mode
        await chat.process_single_query(query)
    else:
        # Interactive mode
        await chat.start_chat_loop()


# CLI command integration
@click.command()
@click.argument("query", required=False)
@click.option(
    "--interactive/--no-interactive",
    "-i/-n",
    default=None,
    help="Force interactive mode on/off"
)
@click.pass_context
def chat(ctx, query, interactive):
    """
    Start an AI-powered chat about your screen activity.
    
    Examples:
        eidolon chat                    # Start interactive chat
        eidolon chat "What was I doing?" # Single query
    """
    logger = get_logger(__name__)
    
    try:
        # Initialize interface
        interface = Interface()
        
        # Determine mode
        if interactive is None:
            # Auto-detect: interactive if no query provided
            interactive = query is None
        
        # Run chat
        if interactive and query:
            # Both interactive flag and query: process query then enter interactive
            asyncio.run(run_chat(interface, query))
            asyncio.run(run_chat(interface, None))
        else:
            asyncio.run(run_chat(interface, query))
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    chat()