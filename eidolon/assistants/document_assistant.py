"""
Document Assistant for Eidolon AI Personal Assistant

Provides intelligent document analysis, generation, and management capabilities.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import mimetypes

from ..utils.logging import get_component_logger
from ..utils.config import get_config
from ..models.cloud_api import CloudAPIManager
from ..tools.file_ops import FileOperationsTool
from ..core.safety import SafetyManager

logger = get_component_logger("assistants.document")


@dataclass
class DocumentAnalysis:
    """Analysis result for a document."""
    document_type: str  # pdf, word, text, code, etc.
    language: str      # programming language or natural language
    summary: str
    key_topics: List[str]
    word_count: int
    page_count: Optional[int]
    readability_score: float
    contains_sensitive_data: bool
    suggested_tags: List[str]
    confidence: float


@dataclass
class DocumentTemplate:
    """Document template structure."""
    name: str
    content_template: str
    category: str
    description: str
    variables: List[str]
    file_extension: str


class DocumentAssistant:
    """
    Intelligent document assistant for analysis, generation, and management.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize document assistant."""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.cloud_api = CloudAPIManager()
        self.file_tool = FileOperationsTool()
        self.safety_manager = SafetyManager()
        
        # Load document templates
        self.templates: Dict[str, DocumentTemplate] = {}
        self._load_default_templates()
        
        # Document analysis patterns
        self.code_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
            '.rb': 'ruby', '.php': 'php', '.go': 'go', '.rs': 'rust',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala'
        }
        
        self.document_extensions = {
            '.txt': 'text', '.md': 'markdown', '.rst': 'restructuredtext',
            '.doc': 'word', '.docx': 'word', '.pdf': 'pdf',
            '.rtf': 'rtf', '.odt': 'opendocument'
        }
        
        # Common document patterns
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[.-]?\d{3}[.-]?\d{4}\b',  # Phone number
        ]
        
        logger.info("Document assistant initialized")
    
    async def analyze_document(
        self,
        file_path: str,
        content: Optional[str] = None,
        analysis_type: str = "comprehensive"
    ) -> DocumentAnalysis:
        """
        Analyze a document for content, structure, and metadata.
        
        Args:
            file_path: Path to document file
            content: Document content (if already loaded)
            analysis_type: Type of analysis (basic, comprehensive, security)
            
        Returns:
            Document analysis result
        """
        try:
            logger.info(f"Analyzing document: {file_path}")
            
            # Load content if not provided
            if content is None:
                content = await self._load_document_content(file_path)
            
            # Determine document type
            doc_type = self._determine_document_type(file_path, content)
            language = self._detect_language(file_path, content, doc_type)
            
            # Basic analysis
            word_count = len(content.split())
            page_count = self._estimate_page_count(content, doc_type)
            readability = self._calculate_readability(content)
            
            # Check for sensitive data
            sensitive_data = self._contains_sensitive_data(content)
            
            # Extract key information
            summary = ""
            key_topics = []
            suggested_tags = []
            confidence = 0.7
            
            # Use AI for advanced analysis if available
            if self.cloud_api and analysis_type == "comprehensive":
                ai_analysis = await self._ai_analyze_document(content, doc_type)
                
                summary = ai_analysis.get("summary", self._extract_summary(content))
                key_topics = ai_analysis.get("key_topics", self._extract_topics(content))
                suggested_tags = ai_analysis.get("suggested_tags", self._suggest_tags(content, doc_type))
                confidence = ai_analysis.get("confidence", 0.8)
            else:
                # Fallback to basic analysis
                summary = self._extract_summary(content)
                key_topics = self._extract_topics(content)
                suggested_tags = self._suggest_tags(content, doc_type)
            
            return DocumentAnalysis(
                document_type=doc_type,
                language=language,
                summary=summary,
                key_topics=key_topics,
                word_count=word_count,
                page_count=page_count,
                readability_score=readability,
                contains_sensitive_data=sensitive_data,
                suggested_tags=suggested_tags,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return DocumentAnalysis(
                document_type="unknown",
                language="unknown",
                summary="Analysis failed",
                key_topics=[],
                word_count=0,
                page_count=None,
                readability_score=0.0,
                contains_sensitive_data=False,
                suggested_tags=[],
                confidence=0.0
            )
    
    async def generate_document(
        self,
        request: str,
        document_type: str = "text",
        template_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a document based on user request.
        
        Args:
            request: User's request for document content
            document_type: Type of document to generate
            template_name: Optional template to use
            context: Additional context information
            output_path: Where to save the document
            
        Returns:
            Generation result
        """
        try:
            logger.info(f"Generating {document_type} document for: {request[:100]}...")
            
            # Use template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                return await self._generate_from_template(template, request, context, output_path)
            
            # AI-powered generation
            return await self._generate_with_ai(request, document_type, context, output_path)
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate document"
            }
    
    async def summarize_document(
        self,
        file_path: str,
        content: Optional[str] = None,
        summary_length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate a summary of a document.
        
        Args:
            file_path: Path to document
            content: Document content (if already loaded)
            summary_length: Length of summary (short, medium, long)
            
        Returns:
            Summary result
        """
        try:
            # Load content if not provided
            if content is None:
                content = await self._load_document_content(file_path)
            
            # Determine target length
            length_mapping = {
                "short": 100,
                "medium": 300,
                "long": 500
            }
            target_words = length_mapping.get(summary_length, 300)
            
            # Use AI for summarization if available
            if self.cloud_api:
                summary = await self._ai_summarize(content, target_words)
            else:
                summary = self._basic_summarize(content, target_words)
            
            return {
                "success": True,
                "summary": summary,
                "original_length": len(content.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": len(summary) / len(content) if content else 0
            }
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to summarize document"
            }
    
    async def extract_information(
        self,
        file_path: str,
        extraction_type: str,
        query: Optional[str] = None,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract specific information from a document.
        
        Args:
            file_path: Path to document
            extraction_type: Type of extraction (entities, keywords, dates, etc.)
            query: Specific query for extraction
            content: Document content (if already loaded)
            
        Returns:
            Extraction result
        """
        try:
            # Load content if not provided
            if content is None:
                content = await self._load_document_content(file_path)
            
            extracted_data = {}
            
            if extraction_type == "entities":
                extracted_data = await self._extract_entities(content)
            elif extraction_type == "keywords":
                extracted_data = await self._extract_keywords(content)
            elif extraction_type == "dates":
                extracted_data = await self._extract_dates(content)
            elif extraction_type == "contacts":
                extracted_data = await self._extract_contacts(content)
            elif extraction_type == "custom" and query:
                extracted_data = await self._extract_custom(content, query)
            else:
                raise ValueError(f"Unknown extraction type: {extraction_type}")
            
            return {
                "success": True,
                "extraction_type": extraction_type,
                "data": extracted_data,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Information extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to extract {extraction_type}"
            }
    
    async def transform_document(
        self,
        file_path: str,
        transformation_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transform a document (convert format, style, etc.).
        
        Args:
            file_path: Path to source document
            transformation_type: Type of transformation
            parameters: Transformation parameters
            output_path: Where to save transformed document
            
        Returns:
            Transformation result
        """
        try:
            params = parameters or {}
            
            if transformation_type == "format_convert":
                return await self._convert_format(file_path, params, output_path)
            elif transformation_type == "style_change":
                return await self._change_style(file_path, params, output_path)
            elif transformation_type == "language_translate":
                return await self._translate_document(file_path, params, output_path)
            elif transformation_type == "redact_sensitive":
                return await self._redact_sensitive_info(file_path, output_path)
            else:
                raise ValueError(f"Unknown transformation type: {transformation_type}")
                
        except Exception as e:
            logger.error(f"Document transformation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to transform document"
            }
    
    def add_template(self, template: DocumentTemplate) -> bool:
        """Add a custom document template."""
        try:
            self.templates[template.name] = template
            logger.info(f"Added document template: {template.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add template: {e}")
            return False
    
    def get_templates(self) -> List[DocumentTemplate]:
        """Get all available document templates."""
        return list(self.templates.values())
    
    async def _load_document_content(self, file_path: str) -> str:
        """Load document content from file."""
        try:
            result = await self.file_tool.execute({
                "operation": "read",
                "path": file_path
            })
            
            if result.success:
                return result.data.get("content", "")
            else:
                raise Exception(f"Failed to read file: {result.message}")
                
        except Exception as e:
            logger.error(f"Failed to load document content: {e}")
            raise
    
    def _determine_document_type(self, file_path: str, content: str) -> str:
        """Determine document type from path and content."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Check by extension first
        if extension in self.code_extensions:
            return "code"
        elif extension in self.document_extensions:
            return self.document_extensions[extension]
        
        # Check by content patterns
        if self._looks_like_code(content):
            return "code"
        elif self._looks_like_markdown(content):
            return "markdown"
        elif self._looks_like_json(content):
            return "json"
        elif self._looks_like_yaml(content):
            return "yaml"
        else:
            return "text"
    
    def _detect_language(self, file_path: str, content: str, doc_type: str) -> str:
        """Detect programming language or natural language."""
        if doc_type == "code":
            extension = Path(file_path).suffix.lower()
            return self.code_extensions.get(extension, "unknown")
        else:
            # Basic natural language detection
            return "english"  # Simplified for now
    
    def _looks_like_code(self, content: str) -> bool:
        """Check if content looks like code."""
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'include ',
            '{', '}', ';', '==', '!=', '&&', '||'
        ]
        
        lines = content.split('\n')
        code_line_count = 0
        
        for line in lines[:50]:  # Check first 50 lines
            if any(indicator in line for indicator in code_indicators):
                code_line_count += 1
        
        return code_line_count > len(lines) * 0.2  # 20% threshold
    
    def _looks_like_markdown(self, content: str) -> bool:
        """Check if content looks like Markdown."""
        md_indicators = ['#', '##', '###', '*', '**', '-', '+', '```', '[', ']']
        return sum(1 for indicator in md_indicators if indicator in content) >= 3
    
    def _looks_like_json(self, content: str) -> bool:
        """Check if content looks like JSON."""
        try:
            json.loads(content.strip())
            return True
        except json.JSONDecodeError:
            return False
    
    def _looks_like_yaml(self, content: str) -> bool:
        """Check if content looks like YAML."""
        yaml_indicators = ['---', ':', '- ', '  ']
        lines = content.split('\n')
        
        # Check for YAML structure patterns
        yaml_lines = sum(1 for line in lines if any(ind in line for ind in yaml_indicators))
        return yaml_lines > len(lines) * 0.3
    
    def _estimate_page_count(self, content: str, doc_type: str) -> Optional[int]:
        """Estimate page count based on content length."""
        if doc_type == "code":
            # Code typically has fewer words per page
            words_per_page = 200
        else:
            # Standard document
            words_per_page = 500
        
        word_count = len(content.split())
        return max(1, word_count // words_per_page)
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate basic readability score (simplified Flesch-Kincaid)."""
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        syllables = self._count_syllables(content)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score)) / 100.0  # Normalize to 0-1
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified method)."""
        # Very basic syllable counting
        vowels = 'aeiouy'
        syllable_count = 0
        
        for word in text.lower().split():
            word = re.sub(r'[^a-z]', '', word)
            if word:
                syllables_in_word = 0
                for i, char in enumerate(word):
                    if char in vowels:
                        if i == 0 or word[i-1] not in vowels:
                            syllables_in_word += 1
                
                # Minimum one syllable per word
                syllable_count += max(1, syllables_in_word)
        
        return syllable_count
    
    def _contains_sensitive_data(self, content: str) -> bool:
        """Check if document contains sensitive data."""
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content):
                return True
        
        # Check with safety manager
        return self.safety_manager._contains_sensitive_data(content)
    
    def _extract_summary(self, content: str) -> str:
        """Extract basic summary from content."""
        # Simple extraction - first paragraph or sentences
        paragraphs = content.split('\n\n')
        
        if paragraphs:
            first_paragraph = paragraphs[0].strip()
            if len(first_paragraph) > 50:
                return first_paragraph[:300] + "..." if len(first_paragraph) > 300 else first_paragraph
        
        # Fallback to first few sentences
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = sentences[:3]
        return '. '.join(s.strip() for s in summary_sentences if s.strip()) + "."
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content."""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{4,}\b', content.lower())
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words (excluding common words)
        common_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
            'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there'
        }
        
        topics = []
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if word not in common_words and len(topics) < 10:
                topics.append(word)
        
        return topics[:5]
    
    def _suggest_tags(self, content: str, doc_type: str) -> List[str]:
        """Suggest tags for the document."""
        tags = [doc_type]
        
        # Add tags based on content
        if 'meeting' in content.lower():
            tags.append('meeting')
        if 'project' in content.lower():
            tags.append('project')
        if 'report' in content.lower():
            tags.append('report')
        if 'analysis' in content.lower():
            tags.append('analysis')
        if 'proposal' in content.lower():
            tags.append('proposal')
        
        # Add date-based tags
        today = datetime.now()
        tags.append(f"year-{today.year}")
        tags.append(f"month-{today.month:02d}")
        
        return tags[:5]
    
    async def _ai_analyze_document(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Use AI to analyze document content."""
        try:
            analysis_prompt = f"""
            Analyze this {doc_type} document and provide:
            
            Content: {content[:2000]}...
            
            Return JSON with:
            - summary: brief summary (2-3 sentences)
            - key_topics: list of main topics/themes
            - suggested_tags: list of relevant tags
            - confidence: analysis confidence (0.0-1.0)
            """
            
            response = await self.cloud_api.analyze_text(
                analysis_prompt,
                analysis_type="document_analysis"
            )
            
            if response and response.content:
                return json.loads(response.content)
                
        except Exception as e:
            logger.warning(f"AI document analysis failed: {e}")
        
        return {}
    
    async def _generate_from_template(
        self,
        template: DocumentTemplate,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate document from template."""
        try:
            # Extract variables from request using AI
            variables = {}
            if self.cloud_api:
                extraction_prompt = f"""
                Extract variable values for this document template:
                Template: {template.name}
                Variables needed: {template.variables}
                User request: {request}
                Context: {context or {}}
                
                Return a JSON object with variable names and values.
                """
                
                response = await self.cloud_api.analyze_text(
                    extraction_prompt,
                    analysis_type="variable_extraction"
                )
                
                if response and response.content:
                    try:
                        variables = json.loads(response.content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse AI variable extraction")
            
            # Apply template
            content = template.content_template
            
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                content = content.replace(placeholder, str(var_value))
            
            # Save if output path provided
            if output_path:
                await self._save_document(content, output_path)
            
            return {
                "success": True,
                "content": content,
                "template_used": template.name,
                "variables": variables,
                "output_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate from template"
            }
    
    async def _generate_with_ai(
        self,
        request: str,
        document_type: str,
        context: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate document using AI."""
        try:
            if not self.cloud_api:
                return await self._generate_basic(request, document_type, output_path)
            
            generation_prompt = f"""
            Generate a {document_type} document based on this request:
            
            Request: {request}
            Document type: {document_type}
            Context: {context or 'None provided'}
            
            Guidelines:
            - Create well-structured content
            - Use appropriate formatting for {document_type}
            - Be comprehensive but concise
            - Include relevant sections/headings
            
            Return the document content only.
            """
            
            response = await self.cloud_api.analyze_text(
                generation_prompt,
                analysis_type="document_generation"
            )
            
            if response and response.content:
                content = response.content
                
                # Save if output path provided
                if output_path:
                    await self._save_document(content, output_path)
                
                return {
                    "success": True,
                    "content": content,
                    "document_type": document_type,
                    "ai_generated": True,
                    "output_path": output_path
                }
            
            # Fallback to basic generation
            return await self._generate_basic(request, document_type, output_path)
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return await self._generate_basic(request, document_type, output_path)
    
    async def _generate_basic(
        self,
        request: str,
        document_type: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Basic document generation without AI."""
        # Simple template-based generation
        if document_type == "markdown":
            content = f"""# Document

## Overview
{request}

## Details
[Add detailed content here]

## Conclusion
[Add conclusion here]
"""
        else:
            content = f"""Document

Overview:
{request}

Details:
[Add detailed content here]

Conclusion:
[Add conclusion here]
"""
        
        # Save if output path provided
        if output_path:
            await self._save_document(content, output_path)
        
        return {
            "success": True,
            "content": content,
            "document_type": document_type,
            "ai_generated": False,
            "output_path": output_path,
            "note": "Basic generation - consider using AI for better results"
        }
    
    async def _save_document(self, content: str, output_path: str) -> None:
        """Save document content to file."""
        result = await self.file_tool.execute({
            "operation": "write",
            "path": output_path,
            "content": content
        })
        
        if not result.success:
            raise Exception(f"Failed to save document: {result.message}")
    
    async def _ai_summarize(self, content: str, target_words: int) -> str:
        """Use AI to summarize document."""
        try:
            summarization_prompt = f"""
            Summarize this document in approximately {target_words} words:
            
            {content[:3000]}...
            
            Provide a clear, concise summary that captures the main points.
            """
            
            response = await self.cloud_api.analyze_text(
                summarization_prompt,
                analysis_type="summarization"
            )
            
            if response and response.content:
                return response.content
                
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
        
        return self._basic_summarize(content, target_words)
    
    def _basic_summarize(self, content: str, target_words: int) -> str:
        """Basic summarization without AI."""
        sentences = re.split(r'[.!?]+', content)
        
        # Take first few sentences to approximate target length
        summary_sentences = []
        current_words = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence_words = len(sentence.split())
                if current_words + sentence_words <= target_words:
                    summary_sentences.append(sentence)
                    current_words += sentence_words
                else:
                    break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else content[:target_words * 5]
    
    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content."""
        # Basic entity extraction using patterns
        entities = {
            "names": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "emails": [],
            "phones": []
        }
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, content)
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[.-]?\d{3}[.-]?\d{4}\b'
        entities["phones"] = re.findall(phone_pattern, content)
        
        # Extract dates (basic patterns)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, content, re.IGNORECASE))
        
        return entities
    
    async def _extract_keywords(self, content: str) -> Dict[str, Any]:
        """Extract keywords from content."""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', content.lower())
        
        # Filter common words
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'men', 'oil', 'put', 'say', 'she', 'too', 'use'
        }
        
        filtered_words = [word for word in words if word not in common_words]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "keywords": [word for word, freq in top_keywords],
            "frequencies": dict(top_keywords)
        }
    
    async def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content."""
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        
        return list(set(dates))  # Remove duplicates
    
    async def _extract_contacts(self, content: str) -> Dict[str, List[str]]:
        """Extract contact information from content."""
        contacts = {
            "emails": [],
            "phones": [],
            "addresses": []
        }
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contacts["emails"] = re.findall(email_pattern, content)
        
        # Extract phone numbers
        phone_patterns = [
            r'\b\d{3}[.-]?\d{3}[.-]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[.-]?\d{4}\b',
            r'\+\d{1,3}\s*\d{3}[.-]?\d{3}[.-]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            contacts["phones"].extend(re.findall(pattern, content))
        
        # Basic address extraction (simplified)
        address_pattern = r'\b\d+\s+[A-Za-z\s]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane)\b'
        contacts["addresses"] = re.findall(address_pattern, content, re.IGNORECASE)
        
        return contacts
    
    async def _extract_custom(self, content: str, query: str) -> Dict[str, Any]:
        """Extract custom information based on query."""
        # Use AI if available for custom extraction
        if self.cloud_api:
            try:
                extraction_prompt = f"""
                Extract information from this document based on the query:
                
                Query: {query}
                Document: {content[:2000]}...
                
                Return relevant information that answers the query.
                """
                
                response = await self.cloud_api.analyze_text(
                    extraction_prompt,
                    analysis_type="custom_extraction"
                )
                
                if response and response.content:
                    return {"extracted_info": response.content}
                    
            except Exception as e:
                logger.warning(f"AI custom extraction failed: {e}")
        
        # Basic pattern matching fallback
        query_words = query.lower().split()
        relevant_sentences = []
        
        for sentence in content.split('.'):
            if any(word in sentence.lower() for word in query_words):
                relevant_sentences.append(sentence.strip())
        
        return {"extracted_info": relevant_sentences[:5]}
    
    def _load_default_templates(self) -> None:
        """Load default document templates."""
        default_templates = [
            DocumentTemplate(
                name="meeting_notes",
                content_template="""# Meeting Notes

**Date:** {date}
**Attendees:** {attendees}
**Subject:** {subject}

## Agenda
{agenda}

## Discussion Points
{discussion}

## Action Items
{action_items}

## Next Steps
{next_steps}
""",
                category="business",
                description="Meeting notes template",
                variables=["date", "attendees", "subject", "agenda", "discussion", "action_items", "next_steps"],
                file_extension=".md"
            ),
            
            DocumentTemplate(
                name="project_report",
                content_template="""# Project Report: {project_name}

## Executive Summary
{executive_summary}

## Project Overview
**Start Date:** {start_date}
**End Date:** {end_date}
**Team Members:** {team_members}

## Objectives
{objectives}

## Methodology
{methodology}

## Results
{results}

## Conclusions
{conclusions}

## Recommendations
{recommendations}
""",
                category="business",
                description="Project report template",
                variables=["project_name", "executive_summary", "start_date", "end_date", "team_members", "objectives", "methodology", "results", "conclusions", "recommendations"],
                file_extension=".md"
            ),
            
            DocumentTemplate(
                name="technical_spec",
                content_template="""# Technical Specification: {feature_name}

## Overview
{overview}

## Requirements
### Functional Requirements
{functional_requirements}

### Non-Functional Requirements
{non_functional_requirements}

## Architecture
{architecture}

## Implementation Plan
{implementation_plan}

## Testing Strategy
{testing_strategy}

## Deployment
{deployment}

## Maintenance
{maintenance}
""",
                category="technical",
                description="Technical specification template",
                variables=["feature_name", "overview", "functional_requirements", "non_functional_requirements", "architecture", "implementation_plan", "testing_strategy", "deployment", "maintenance"],
                file_extension=".md"
            )
        ]
        
        for template in default_templates:
            self.templates[template.name] = template
        
        logger.info(f"Loaded {len(default_templates)} default document templates")