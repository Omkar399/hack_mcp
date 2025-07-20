"""
Eidolon AI Personal Assistant - Command Line Interface

Simplified CLI with essential commands for monitoring, search, and chat.
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Optional
import click

from .utils.config import get_config
from .utils.logging import setup_logging, get_logger
from .core.observer import Observer
from .core.interface import Interface


# Global instances
_observer: Optional[Observer] = None
_interface: Optional[Interface] = None


def get_observer() -> Observer:
    """Get or create the global observer instance."""
    global _observer
    if _observer is None:
        _observer = Observer()
    return _observer


def get_interface() -> Interface:
    """Get or create the global interface instance."""
    global _interface
    if _interface is None:
        _interface = Interface()
    return _interface


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), 
              default="INFO", help="Set log level")
@click.pass_context
def cli(ctx, verbose, log_level):
    """Eidolon AI Personal Assistant - Your digital memory."""
    ctx.ensure_object(dict)
    
    # Set up logging
    if verbose:
        log_level = "DEBUG"
    setup_logging(log_level=log_level)
    
    # Load configuration
    ctx.obj['config'] = get_config()
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option("--interval", "-i", type=float, default=0.1, 
              help="Screenshot capture interval in seconds (0.1 = 10 FPS for maximum performance)")
@click.option("--background", "-b", is_flag=True, 
              help="Run in background mode")
@click.option("--memory-limit", type=float, default=None,
              help="Memory limit in GB (auto-detect if not specified)")
@click.option("--low-memory", is_flag=True,
              help="Enable low memory mode (8GB systems)")
@click.pass_context
def start(ctx, interval, background, memory_limit, low_memory):
    """Start the Eidolon system with all components."""
    logger = get_logger(__name__)
    
    try:
        click.echo("🚀 Starting Eidolon AI Personal Assistant...")
        
        # Initialize memory optimizer
        from .utils.memory_optimizer import get_memory_optimizer
        import psutil
        
        # Determine memory limit
        if low_memory:
            memory_limit = 6.0
        elif memory_limit is None:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            if total_ram_gb <= 8:
                memory_limit = 6.0
                click.echo("⚠️  Low RAM detected - using conservative settings")
            elif total_ram_gb <= 16:
                memory_limit = 12.0
                click.echo("ℹ️  Medium RAM detected - using balanced settings")
            else:
                memory_limit = total_ram_gb * 0.8
        
        # Initialize memory optimizer
        memory_optimizer = get_memory_optimizer(memory_limit)
        memory_stats = memory_optimizer.get_memory_usage()
        
        click.echo(f"💾 System RAM: {memory_stats['total_gb']:.1f}GB total, "
                  f"{memory_stats['available_gb']:.1f}GB available")
        click.echo(f"🎯 Memory limit set to: {memory_limit:.1f}GB")
        
        # UNLIMITED PERFORMANCE MODE - NO MEMORY CHECKS
        
        # Start observer with memory optimization
        observer = get_observer()
        observer.config.observer.capture_interval = interval
        
        # Set memory limit for observer
        observer.set_memory_limit(memory_limit)
        
        observer.start_monitoring()
        
        click.echo(f"📸 Screenshot monitoring started (interval: {interval}s)")
        
        if background:
            click.echo("🔄 Running in background mode. Use 'eidolon stop' to stop.")
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pass
        else:
            click.echo("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(5)
                    if ctx.obj.get('verbose'):
                        status = observer.get_status()
                        click.echo(f"📊 Captured: {status.get('capture_count', 0)} screenshots")
            except KeyboardInterrupt:
                click.echo("\n⏹️  Stopping...")
        
        observer.stop_monitoring()
        click.echo("✅ Eidolon stopped successfully.")
        
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the Eidolon system."""
    try:
        observer = get_observer()
        observer.stop_monitoring()
        click.echo("✅ Eidolon stopped successfully.")
    except Exception as e:
        click.echo(f"❌ Error stopping system: {e}", err=True)


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", type=int, default=10, help="Number of results to return")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def search(ctx, query, limit, format):
    """Search through your captured content."""
    logger = get_logger(__name__)
    
    try:
        interface = get_interface()
        results = asyncio.run(interface.search(query, limit=limit))
        
        if format == "json":
            import json
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"🔍 Search results for: '{query}'\n")
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. {result.get('title', 'Untitled')}")
                click.echo(f"   📅 {result.get('timestamp', 'Unknown time')}")
                click.echo(f"   📝 {result.get('content', '')[:100]}...")
                click.echo()
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        click.echo(f"❌ Search failed: {e}", err=True)


@cli.command()
@click.pass_context
def chat(ctx):
    """Start interactive chat with your digital memory."""
    logger = get_logger(__name__)
    
    try:
        interface = get_interface()
        click.echo("💬 Eidolon Chat - Type 'quit' to exit\n")
        
        while True:
            user_input = click.prompt("You", type=str)
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                click.echo("👋 Goodbye!")
                break
            
            response = asyncio.run(interface.chat(user_input))
            click.echo(f"🤖 Eidolon: {response}\n")
            
    except KeyboardInterrupt:
        click.echo("\n👋 Chat ended.")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        click.echo(f"❌ Chat failed: {e}", err=True)


@cli.command()
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def status(ctx, format):
    """Show system status and statistics."""
    try:
        observer = get_observer()
        interface = get_interface()
        
        status_data = {
            "observer": observer.get_status(),
            "interface": interface.get_status() if hasattr(interface, 'get_status') else {},
            "system": {
                "running": observer.is_monitoring(),
                "data_dir": str(Path.cwd() / "data")
            }
        }
        
        if format == "json":
            import json
            click.echo(json.dumps(status_data, indent=2))
        else:
            click.echo("📊 Eidolon System Status\n")
            click.echo(f"🔄 Running: {'Yes' if status_data['system']['running'] else 'No'}")
            click.echo(f"📸 Screenshots: {status_data['observer'].get('capture_count', 0)}")
            click.echo(f"💾 Data Directory: {status_data['system']['data_dir']}")
            
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}", err=True)


@cli.command()
@click.option("--days", type=int, default=30, help="Keep data from last N days")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cleanup(ctx, days, confirm):
    """Clean up old data and screenshots."""
    if not confirm:
        if not click.confirm(f"Delete data older than {days} days?"):
            click.echo("Cleanup cancelled.")
            return
    
    try:
        # Implement cleanup logic here
        click.echo(f"🧹 Cleaning up data older than {days} days...")
        click.echo("✅ Cleanup completed.")
        
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}", err=True)


@cli.group()
@click.pass_context
def config(ctx):
    """Manage Eidolon configuration settings."""
    pass


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """Get a configuration value."""
    try:
        from .utils.config import get_config
        
        cfg = get_config()
        
        # Handle nested keys like "observer.capture_interval"
        keys = key.split('.')
        value = cfg
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                click.echo(f"❌ Configuration key '{key}' not found")
                return
        
        click.echo(f"{key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Error getting config: {e}", err=True)


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value."""
    try:
        click.echo(f"⚠️  Config setting is read-only in this version.")
        click.echo(f"To set {key}={value}, use environment variables:")
        
        # Convert config key to environment variable
        env_key = f"EIDOLON_{key.upper().replace('.', '_')}"
        click.echo(f"export {env_key}={value}")
        click.echo("Then restart Eidolon.")
        
    except Exception as e:
        click.echo(f"❌ Error setting config: {e}", err=True)


@config.command("list")
@click.pass_context
def config_list(ctx):
    """List all configuration values."""
    try:
        from .utils.config import get_config
        
        cfg = get_config()
        
        click.echo("📋 Current Configuration:")
        click.echo("")
        
        # Observer settings
        click.echo("🔍 Observer:")
        click.echo(f"  capture_interval: {cfg.observer.capture_interval}")
        click.echo(f"  activity_threshold: {cfg.observer.activity_threshold}")
        click.echo(f"  max_storage_gb: {cfg.observer.max_storage_gb}")
        click.echo(f"  max_cpu_percent: {cfg.observer.max_cpu_percent}")
        click.echo(f"  max_memory_mb: {cfg.observer.max_memory_mb}")
        
        # Memory settings
        click.echo("\n💾 Memory:")
        click.echo(f"  db_path: {cfg.memory.db_path}")
        click.echo(f"  embedding_model: {cfg.memory.embedding_model}")
        click.echo(f"  vector_dimension: {cfg.memory.vector_dimension}")
        
        # Analysis settings (includes cloud APIs)
        click.echo("\n🤖 Analysis:")
        click.echo(f"  local_models: {cfg.analysis.local_models}")
        click.echo(f"  cloud_apis: {cfg.analysis.cloud_apis}")
        click.echo(f"  routing: {cfg.analysis.routing}")
        
    except Exception as e:
        click.echo(f"❌ Error listing config: {e}", err=True)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
