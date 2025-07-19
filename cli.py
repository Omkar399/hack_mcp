#!/usr/bin/env python3
"""
CLI tool for Screen Memory Assistant
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Rich console for beautiful output
console = Console()

# Default API base URL
DEFAULT_API_BASE = "http://localhost:5003"


class ScreenMemoryClient:
    """Client for Screen Memory Assistant API"""
    
    def __init__(self, base_url: str = DEFAULT_API_BASE):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def capture_now(self, save_image: bool = True, force_vision: bool = False):
        """Take a screenshot now"""
        response = await self.client.post(f"{self.base_url}/capture_now", json={
            "save_image": save_image,
            "force_vision": force_vision
        })
        response.raise_for_status()
        return response.json()
    
    async def search(self, query: str, limit: int = 10, since_minutes: Optional[int] = None, app_name: Optional[str] = None):
        """Search screen events"""
        response = await self.client.post(f"{self.base_url}/find", json={
            "query": query,
            "limit": limit,
            "since_minutes": since_minutes,
            "app_name": app_name
        })
        response.raise_for_status()
        return response.json()
    
    async def semantic_search(self, query: str, k: int = 5, threshold: float = 0.7):
        """Semantic search using CLIP embeddings"""
        response = await self.client.post(f"{self.base_url}/search_semantic", json={
            "query": query,
            "k": k,
            "threshold": threshold
        })
        response.raise_for_status()
        return response.json()
    
    async def recent_events(self, limit: int = 20, hours: int = 24):
        """Get recent events"""
        response = await self.client.get(f"{self.base_url}/recent", params={
            "limit": limit,
            "hours": hours
        })
        response.raise_for_status()
        return response.json()
    
    async def recent_errors(self, window_min: int = 30):
        """Get recent errors"""
        response = await self.client.get(f"{self.base_url}/recent_errors", params={
            "window_min": window_min
        })
        response.raise_for_status()
        return response.json()
    
    async def last_docker(self):
        """Get last Docker command"""
        response = await self.client.get(f"{self.base_url}/last_docker")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self):
        """Check system health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def stats(self):
        """Get system stats"""
        response = await self.client.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


def format_timestamp(ts_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        
        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = int(diff.total_seconds() / 86400)
            return f"{days}d ago"
    except:
        return ts_str


def display_events(events: list, title: str = "Screen Events"):
    """Display events in a nice table"""
    if not events:
        console.print(f"No {title.lower()} found", style="yellow")
        return
    
    table = Table(title=title, show_header=True)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("App", style="green", width=15)
    table.add_column("Window", style="blue", width=25)
    table.add_column("Text Preview", style="white", width=50)
    table.add_column("OCR", style="magenta", width=6)
    
    for event in events:
        text_preview = event.get('full_text', '')[:100]
        if len(text_preview) == 100:
            text_preview += "..."
        
        ocr_conf = event.get('ocr_conf')
        ocr_display = f"{ocr_conf}%" if ocr_conf else "N/A"
        
        table.add_row(
            format_timestamp(event['ts']),
            event.get('app_name', 'Unknown')[:15],
            event.get('window_title', 'Unknown')[:25],
            text_preview,
            ocr_display
        )
    
    console.print(table)


# CLI Commands
@click.group()
@click.option('--api-url', default=DEFAULT_API_BASE, envvar='SCREEN_MEMORY_API', help='API base URL')
@click.pass_context
def cli(ctx, api_url):
    """Screen Memory Assistant - Local screen capture and search"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url


@cli.command()
@click.option('--save/--no-save', default=True, help='Save screenshot image')
@click.option('--force-vision', is_flag=True, help='Force GPT-4o Vision even if OCR works')
@click.pass_context
def capture(ctx, save, force_vision):
    """Take a screenshot now and process it"""
    
    async def do_capture():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            with console.status("Taking screenshot and processing..."):
                result = await client.capture_now(save_image=save, force_vision=force_vision)
            
            console.print("✅ Capture successful!", style="green")
            
            # Display result
            panel_content = f"""
📸 Image: {result.get('image_path', 'Not saved')}
🪟 Window: {result.get('window_title', 'Unknown')}
📱 App: {result.get('app_name', 'Unknown')}
🔍 OCR Confidence: {result.get('ocr_conf', 'N/A')}%
📄 Text Length: {len(result.get('full_text', ''))} chars
            """
            
            console.print(Panel(panel_content.strip(), title="Capture Result", border_style="green"))
            
            # Show text preview
            if result.get('full_text'):
                text_preview = result['full_text'][:200]
                if len(result['full_text']) > 200:
                    text_preview += "..."
                console.print(f"\n📝 Text Preview:\n{text_preview}", style="dim")
            
        except httpx.HTTPError as e:
            console.print(f"❌ Capture failed: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_capture())


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--since', '-s', type=int, help='Search only within last N minutes')
@click.option('--app', '-a', help='Filter by app name')
@click.pass_context
def search(ctx, query, limit, since, app):
    """Search screen events by text content"""
    
    async def do_search():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            with console.status(f"Searching for '{query}'..."):
                results = await client.search(
                    query=query,
                    limit=limit,
                    since_minutes=since,
                    app_name=app
                )
            
            display_events(results, f"Search Results for '{query}'")
            
        except httpx.HTTPError as e:
            console.print(f"❌ Search failed: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_search())


@cli.command()
@click.argument('query')
@click.option('--k', default=5, help='Number of similar results')
@click.option('--threshold', default=0.7, help='Similarity threshold (0-1)')
@click.pass_context
def semantic(ctx, query, k, threshold):
    """Semantic search using CLIP embeddings"""
    
    async def do_semantic_search():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            with console.status(f"Semantic search for '{query}'..."):
                results = await client.semantic_search(
                    query=query,
                    k=k,
                    threshold=threshold
                )
            
            display_events(results, f"Semantic Search Results for '{query}'")
            
        except httpx.HTTPError as e:
            if "CLIP not available" in str(e):
                console.print("❌ CLIP not available - semantic search requires CLIP embeddings", style="red")
            else:
                console.print(f"❌ Semantic search failed: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_semantic_search())


@cli.command()
@click.option('--limit', '-l', default=20, help='Number of events to show')
@click.option('--hours', '-h', default=24, help='Hours to look back')
@click.pass_context
def recent(ctx, limit, hours):
    """Show recent screen events"""
    
    async def do_recent():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            with console.status("Fetching recent events..."):
                results = await client.recent_events(limit=limit, hours=hours)
            
            display_events(results, f"Recent Events (last {hours}h)")
            
        except httpx.HTTPError as e:
            console.print(f"❌ Failed to get recent events: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_recent())


@cli.command()
@click.option('--minutes', '-m', default=30, help='Minutes to look back')
@click.pass_context
def errors(ctx, minutes):
    """Show recent error events"""
    
    async def do_errors():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            with console.status("Fetching recent errors..."):
                results = await client.recent_errors(window_min=minutes)
            
            if not results:
                console.print(f"✅ No errors found in the last {minutes} minutes", style="green")
                return
            
            table = Table(title=f"Recent Errors (last {minutes}m)", show_header=True)
            table.add_column("Time", style="cyan")
            table.add_column("App", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Message", style="red")
            table.add_column("Severity", style="magenta")
            
            for error in results:
                table.add_row(
                    format_timestamp(error['ts']),
                    error.get('app_name', 'Unknown'),
                    error.get('error_type', 'Unknown'),
                    error.get('error_msg', 'N/A')[:50],
                    error.get('severity', 'medium')
                )
            
            console.print(table)
            
        except httpx.HTTPError as e:
            console.print(f"❌ Failed to get recent errors: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_errors())


@cli.command()
@click.pass_context
def docker(ctx):
    """Get the last Docker command"""
    
    async def do_docker():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            result = await client.last_docker()
            
            if not result:
                console.print("No Docker commands found", style="yellow")
                return
            
            panel_content = f"""
🐳 Command: {result['cmd']}
📅 Time: {format_timestamp(result['ts'])}
🏠 Working Dir: {result.get('working_dir', 'Unknown')}
🐚 Shell: {result.get('shell', 'Unknown')}
⚡ Exit Code: {result.get('exit_code', 'Unknown')}
            """
            
            console.print(Panel(panel_content.strip(), title="Last Docker Command", border_style="blue"))
            
        except httpx.HTTPError as e:
            console.print(f"❌ Failed to get Docker command: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_docker())


@cli.command()
@click.pass_context
def health(ctx):
    """Check system health status"""
    
    async def do_health():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            result = await client.health_check()
            
            # Overall status
            if result['status'] == 'healthy':
                console.print("✅ System is healthy", style="green")
            else:
                console.print("❌ System has issues", style="red")
            
            # Database status
            db_status = result.get('database', 'unknown')
            if db_status == 'connected':
                console.print("✅ Database connected", style="green")
            else:
                console.print("❌ Database disconnected", style="red")
            
            # Capture system details
            capture_info = result.get('capture_system', {})
            if capture_info:
                console.print("\n📸 Capture System:")
                console.print(f"  OCR Engines: {', '.join(capture_info.get('ocr_engines', []))}")
                console.print(f"  OCR Failures: {capture_info.get('ocr_failures', 0)}")
                console.print(f"  CLIP Available: {capture_info.get('clip_available', False)}")
                console.print(f"  Vision Available: {capture_info.get('vision_available', False)}")
            
        except httpx.ConnectError:
            console.print("❌ Cannot connect to API server - is it running?", style="red")
        except httpx.HTTPError as e:
            console.print(f"❌ Health check failed: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_health())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics"""
    
    async def do_stats():
        client = ScreenMemoryClient(ctx.obj['api_url'])
        try:
            result = await client.stats()
            
            db_stats = result.get('database', {})
            capture_stats = result.get('capture_system', {})
            
            # Database statistics
            console.print(Panel.fit(f"""
📊 Database Statistics

Total Events: {db_stats.get('total_events', 0)}
Last Hour: {db_stats.get('events_last_hour', 0)}
Last Day: {db_stats.get('events_last_day', 0)}
Last Week: {db_stats.get('events_last_week', 0)}

Average OCR Confidence: {db_stats.get('avg_ocr_confidence', 'N/A')}%
Events with Embeddings: {db_stats.get('events_with_embeddings', 0)}
            """.strip(), title="System Statistics", border_style="blue"))
            
        except httpx.HTTPError as e:
            console.print(f"❌ Failed to get stats: {e}", style="red")
        finally:
            await client.close()
    
    asyncio.run(do_stats())


if __name__ == '__main__':
    cli() 