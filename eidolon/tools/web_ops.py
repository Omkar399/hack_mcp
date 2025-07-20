"""
Web Operations Tool for Eidolon Tool Orchestration Framework

Provides web browsing, API calls, and web automation capabilities.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin
import re
from pathlib import Path

from .base import BaseTool, ToolMetadata, ToolResult, ToolError
from ..core.safety import RiskLevel, ActionCategory
from ..utils.logging import get_component_logger

logger = get_component_logger("tools.web_ops")


class WebOperationsTool(BaseTool):
    """Tool for web operations including HTTP requests and basic automation."""
    
    METADATA = ToolMetadata(
        name="web_operations",
        description="Perform web operations including HTTP requests and basic automation",
        category="web",
        risk_level=RiskLevel.MEDIUM,
        action_category=ActionCategory.NETWORK_REQUEST,
        requires_approval=True,
        timeout_seconds=60.0,
        input_schema={
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get", "post", "put", "delete", "download", "scrape", "health_check"]
                },
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "data": {"type": "object"},
                "json": {"type": "object"},
                "params": {"type": "object"},
                "timeout": {"type": "number"},
                "follow_redirects": {"type": "boolean", "default": True},
                "verify_ssl": {"type": "boolean", "default": True},
                "save_to": {"type": "string"},
                "selectors": {"type": "object"}
            }
        }
    )
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """Initialize web operations tool."""
        super().__init__(metadata or self.METADATA)
        
        # Safety controls
        self.allowed_domains = set()  # Empty means all allowed
        self.blocked_domains = {
            'localhost', '127.0.0.1', '0.0.0.0',
            '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'  # Private networks
        }
        
        # Request limits
        self.max_response_size = 50 * 1024 * 1024  # 50MB
        self.max_redirects = 10
        self.default_timeout = 30
        
        # User agent
        self.user_agent = "Eidolon-Agent/1.0 (Autonomous Assistant)"
        
        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute web operation."""
        try:
            # Validate parameters
            validated_params = await self.validate_parameters(parameters)
            operation = validated_params["operation"]
            
            # Initialize session if needed
            if not self._session:
                await self._init_session()
            
            # Route to specific operation
            if operation == "get":
                return await self._http_get(validated_params)
            elif operation == "post":
                return await self._http_post(validated_params)
            elif operation == "put":
                return await self._http_put(validated_params)
            elif operation == "delete":
                return await self._http_delete(validated_params)
            elif operation == "download":
                return await self._download_file(validated_params)
            elif operation == "scrape":
                return await self._scrape_page(validated_params)
            elif operation == "health_check":
                return await self._health_check(validated_params)
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Web operation failed: {e}")
            return ToolResult(
                success=False,
                data={"error": str(e)},
                message=f"Web operation failed: {str(e)}"
            )
        finally:
            # Clean up session if needed
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    async def _init_session(self) -> None:
        """Initialize HTTP session."""
        timeout = aiohttp.ClientTimeout(total=self.default_timeout)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            enable_cleanup_closed=True
        )
        
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.user_agent}
        )
    
    async def _http_get(self, params: Dict[str, Any]) -> ToolResult:
        """Perform HTTP GET request."""
        url = params["url"]
        headers = params.get("headers", {})
        params_dict = params.get("params", {})
        timeout = params.get("timeout", self.default_timeout)
        follow_redirects = params.get("follow_redirects", True)
        verify_ssl = params.get("verify_ssl", True)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            async with self._session.get(
                url,
                headers=headers,
                params=params_dict,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=follow_redirects,
                ssl=verify_ssl
            ) as response:
                
                # Check response size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_response_size:
                    raise ToolError(f"Response too large: {content_length} bytes")
                
                # Read response
                content = await response.read()
                
                if len(content) > self.max_response_size:
                    raise ToolError(f"Response too large: {len(content)} bytes")
                
                # Try to decode as text
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('utf-8', errors='replace')
                
                # Parse JSON if content type indicates it
                json_data = None
                if 'application/json' in response.headers.get('content-type', ''):
                    try:
                        json_data = json.loads(text_content)
                    except json.JSONDecodeError:
                        pass
                
                return ToolResult(
                    success=response.status < 400,
                    data={
                        "url": str(response.url),
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": text_content,
                        "json": json_data,
                        "size": len(content)
                    },
                    message=f"GET {url} -> {response.status}"
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"HTTP request failed: {e}")
        except asyncio.TimeoutError:
            raise ToolError(f"Request timed out after {timeout} seconds")
    
    async def _http_post(self, params: Dict[str, Any]) -> ToolResult:
        """Perform HTTP POST request."""
        url = params["url"]
        headers = params.get("headers", {})
        data = params.get("data")
        json_data = params.get("json")
        timeout = params.get("timeout", self.default_timeout)
        verify_ssl = params.get("verify_ssl", True)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            kwargs = {
                "headers": headers,
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "ssl": verify_ssl
            }
            
            if json_data:
                kwargs["json"] = json_data
            elif data:
                kwargs["data"] = data
            
            async with self._session.post(url, **kwargs) as response:
                
                # Read response
                content = await response.read()
                
                if len(content) > self.max_response_size:
                    raise ToolError(f"Response too large: {len(content)} bytes")
                
                # Try to decode as text
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('utf-8', errors='replace')
                
                # Parse JSON if possible
                json_response = None
                if 'application/json' in response.headers.get('content-type', ''):
                    try:
                        json_response = json.loads(text_content)
                    except json.JSONDecodeError:
                        pass
                
                return ToolResult(
                    success=response.status < 400,
                    data={
                        "url": str(response.url),
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": text_content,
                        "json": json_response,
                        "size": len(content)
                    },
                    message=f"POST {url} -> {response.status}",
                    side_effects=[f"Posted data to: {url}"]
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"HTTP POST failed: {e}")
        except asyncio.TimeoutError:
            raise ToolError(f"Request timed out after {timeout} seconds")
    
    async def _http_put(self, params: Dict[str, Any]) -> ToolResult:
        """Perform HTTP PUT request."""
        url = params["url"]
        headers = params.get("headers", {})
        data = params.get("data")
        json_data = params.get("json")
        timeout = params.get("timeout", self.default_timeout)
        verify_ssl = params.get("verify_ssl", True)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            kwargs = {
                "headers": headers,
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "ssl": verify_ssl
            }
            
            if json_data:
                kwargs["json"] = json_data
            elif data:
                kwargs["data"] = data
            
            async with self._session.put(url, **kwargs) as response:
                content = await response.read()
                
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('utf-8', errors='replace')
                
                return ToolResult(
                    success=response.status < 400,
                    data={
                        "url": str(response.url),
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": text_content,
                        "size": len(content)
                    },
                    message=f"PUT {url} -> {response.status}",
                    side_effects=[f"Put data to: {url}"]
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"HTTP PUT failed: {e}")
    
    async def _http_delete(self, params: Dict[str, Any]) -> ToolResult:
        """Perform HTTP DELETE request."""
        url = params["url"]
        headers = params.get("headers", {})
        timeout = params.get("timeout", self.default_timeout)
        verify_ssl = params.get("verify_ssl", True)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            async with self._session.delete(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                ssl=verify_ssl
            ) as response:
                content = await response.read()
                
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('utf-8', errors='replace')
                
                return ToolResult(
                    success=response.status < 400,
                    data={
                        "url": str(response.url),
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": text_content,
                        "size": len(content)
                    },
                    message=f"DELETE {url} -> {response.status}",
                    side_effects=[f"Deleted resource: {url}"]
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"HTTP DELETE failed: {e}")
    
    async def _download_file(self, params: Dict[str, Any]) -> ToolResult:
        """Download a file from URL."""
        url = params["url"]
        save_to = params.get("save_to")
        headers = params.get("headers", {})
        timeout = params.get("timeout", 60)  # Longer timeout for downloads
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        if not save_to:
            raise ToolError("save_to parameter is required for download")
        
        save_path = Path(save_to)
        
        # Validate save path
        if not self._is_safe_file_path(save_path):
            raise ToolError(f"Save path not allowed: {save_path}")
        
        try:
            async with self._session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status >= 400:
                    raise ToolError(f"Download failed with status {response.status}")
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_response_size:
                    raise ToolError(f"File too large: {content_length} bytes")
                
                # Create parent directories
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                total_size = 0
                with open(save_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        total_size += len(chunk)
                        
                        if total_size > self.max_response_size:
                            # Clean up partial file
                            f.close()
                            save_path.unlink(missing_ok=True)
                            raise ToolError(f"File too large: {total_size} bytes")
                        
                        f.write(chunk)
                
                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "saved_to": str(save_path),
                        "size": total_size,
                        "content_type": response.headers.get('content-type')
                    },
                    message=f"Downloaded {total_size} bytes to {save_path.name}",
                    side_effects=[f"Created file: {save_path}"]
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"Download failed: {e}")
    
    async def _scrape_page(self, params: Dict[str, Any]) -> ToolResult:
        """Scrape data from a web page."""
        url = params["url"]
        selectors = params.get("selectors", {})
        headers = params.get("headers", {})
        timeout = params.get("timeout", self.default_timeout)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            # Get page content
            async with self._session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status >= 400:
                    raise ToolError(f"Failed to fetch page: {response.status}")
                
                content = await response.text()
                
                # Basic HTML parsing (would use BeautifulSoup in production)
                scraped_data = {
                    "url": url,
                    "title": self._extract_title(content),
                    "content_length": len(content),
                    "status": response.status
                }
                
                # Extract meta description
                meta_desc = self._extract_meta_description(content)
                if meta_desc:
                    scraped_data["description"] = meta_desc
                
                # Extract links
                links = self._extract_links(content, url)
                scraped_data["links"] = links[:50]  # Limit to 50 links
                
                # Apply custom selectors (basic implementation)
                if selectors:
                    scraped_data["custom"] = {}
                    for name, selector in selectors.items():
                        # This is a very basic implementation
                        # In production, would use proper HTML parsing
                        scraped_data["custom"][name] = self._extract_by_selector(content, selector)
                
                return ToolResult(
                    success=True,
                    data=scraped_data,
                    message=f"Scraped {url}: {scraped_data.get('title', 'No title')}"
                )
                
        except aiohttp.ClientError as e:
            raise ToolError(f"Scraping failed: {e}")
    
    async def _health_check(self, params: Dict[str, Any]) -> ToolResult:
        """Check if a URL is accessible."""
        url = params["url"]
        timeout = params.get("timeout", 10)
        
        # Validate URL
        if not self._is_url_safe(url):
            raise ToolError(f"URL not allowed: {url}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self._session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True
            ) as response:
                
                end_time = asyncio.get_event_loop().time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                is_healthy = response.status < 400
                
                return ToolResult(
                    success=is_healthy,
                    data={
                        "url": str(response.url),
                        "status": response.status,
                        "response_time_ms": round(response_time, 2),
                        "headers": dict(response.headers),
                        "healthy": is_healthy
                    },
                    message=f"Health check: {url} -> {response.status} ({response_time:.0f}ms)"
                )
                
        except aiohttp.ClientError as e:
            return ToolResult(
                success=False,
                data={
                    "url": url,
                    "error": str(e),
                    "healthy": False
                },
                message=f"Health check failed: {e}"
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                data={
                    "url": url,
                    "error": "Timeout",
                    "healthy": False
                },
                message=f"Health check timed out after {timeout}s"
            )
    
    def _is_url_safe(self, url: str) -> bool:
        """Check if URL is safe to access."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check if domain is blocked
            hostname = parsed.hostname
            if hostname in self.blocked_domains:
                return False
            
            # Check against allowed domains (if specified)
            if self.allowed_domains and hostname not in self.allowed_domains:
                return False
            
            # Check for private IP ranges (basic check)
            if hostname and self._is_private_ip(hostname):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname is a private IP address."""
        import ipaddress
        
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            # Not an IP address
            return False
    
    def _is_safe_file_path(self, path: Path) -> bool:
        """Check if file path is safe for writing."""
        try:
            # Resolve to absolute path
            abs_path = path.resolve()
            
            # Don't allow writing to system directories
            forbidden_roots = ['/etc', '/usr', '/var', '/System']
            for root in forbidden_roots:
                if str(abs_path).startswith(root):
                    return False
            
            # Allow writing to user directory, temp, and current working directory
            allowed_roots = [
                Path.home(),
                Path.cwd(),
                Path('/tmp'),
                Path('/var/tmp')
            ]
            
            for root in allowed_roots:
                try:
                    abs_path.relative_to(root.resolve())
                    return True
                except ValueError:
                    continue
            
            return False
            
        except Exception:
            return False
    
    def _extract_title(self, html_content: str) -> Optional[str]:
        """Extract title from HTML content."""
        import re
        
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return None
    
    def _extract_meta_description(self, html_content: str) -> Optional[str]:
        """Extract meta description from HTML content."""
        import re
        
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
            html_content,
            re.IGNORECASE
        )
        if desc_match:
            return desc_match.group(1).strip()
        return None
    
    def _extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML content."""
        import re
        
        link_pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>'
        links = []
        
        for match in re.finditer(link_pattern, html_content, re.IGNORECASE | re.DOTALL):
            href = match.group(1).strip()
            text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            
            # Resolve relative URLs
            if href.startswith(('http://', 'https://')):
                absolute_url = href
            else:
                absolute_url = urljoin(base_url, href)
            
            links.append({
                "url": absolute_url,
                "text": text[:100],  # Limit text length
                "href": href
            })
        
        return links
    
    def _extract_by_selector(self, html_content: str, selector: str) -> str:
        """Basic CSS selector extraction (very limited implementation)."""
        # This is a very basic implementation
        # In production, would use proper HTML parsing libraries
        
        if selector.startswith('#'):
            # ID selector
            element_id = selector[1:]
            pattern = rf'<[^>]*id=["\']({element_id})["\'][^>]*>(.*?)</[^>]*>'
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                return re.sub(r'<[^>]+>', '', match.group(2)).strip()
        
        elif selector.startswith('.'):
            # Class selector
            class_name = selector[1:]
            pattern = rf'<[^>]*class=["\'][^"\']*{class_name}[^"\']*["\'][^>]*>(.*?)</[^>]*>'
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                return re.sub(r'<[^>]+>', '', match.group(1)).strip()
        
        else:
            # Tag selector
            pattern = rf'<{selector}[^>]*>(.*?)</{selector}>'
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                return re.sub(r'<[^>]+>', '', match.group(1)).strip()
        
        return ""