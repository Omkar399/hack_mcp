"""
Analyzer component for Eidolon AI Personal Assistant

Handles AI-powered content analysis, OCR text extraction, and intelligent
content understanding using both local and cloud AI models.
"""

import os
import re
import warnings

# Set tokenizer parallelism to avoid warnings in multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optimize for Apple Silicon M3 GPU performance
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # Use maximum GPU memory
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Enable fallback for unsupported ops

# Suppress transformer warnings for cleaner output
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Also suppress warnings in transformers logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from PIL import Image
import numpy as np

# Suppress known warnings from external AI libraries
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# OCR imports
import pytesseract
import easyocr

# AI Model imports for Phase 3
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    FLORENCE_AVAILABLE = True
except ImportError:
    FLORENCE_AVAILABLE = False

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config


class TextRegion:
    """Represents a region of text with coordinates."""
    
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        self.text = text
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence


class ExtractedText:
    """Represents text extracted from a screenshot."""
    
    def __init__(
        self,
        text: str,
        confidence: float,
        language: str = "en",
        regions: Optional[List[TextRegion]] = None,
        method: str = "tesseract"
    ):
        self.text = text
        self.confidence = confidence
        self.language = language
        self.regions = regions or []
        self.method = method
        self.timestamp = datetime.now()
        self.word_count = len(self.text.split()) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "method": self.method,
            "word_count": self.word_count,
            "timestamp": self.timestamp.isoformat(),
            "regions": [
                {
                    "text": region.text,
                    "bbox": region.bbox,
                    "confidence": region.confidence
                } for region in self.regions
            ]
        }


class VisionAnalysis:
    """Represents advanced vision analysis results using AI models."""
    
    def __init__(
        self,
        description: str,
        objects: List[Dict[str, Any]] = None,
        scene_type: str = "",
        confidence: float = 0.0,
        ui_elements: List[Dict[str, Any]] = None,
        model_used: str = "florence-2"
    ):
        self.description = description
        self.objects = objects or []
        self.scene_type = scene_type
        self.confidence = confidence
        self.ui_elements = ui_elements or []
        self.model_used = model_used
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "description": self.description,
            "objects": self.objects,
            "scene_type": self.scene_type,
            "confidence": self.confidence,
            "ui_elements": self.ui_elements,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat()
        }


class ContentAnalysis:
    """Represents the result of AI content analysis."""
    
    def __init__(
        self,
        content_type: str,
        description: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ui_elements: Optional[List[Dict[str, Any]]] = None,
        vision_analysis: Optional[VisionAnalysis] = None
    ):
        self.content_type = content_type
        self.description = description
        self.confidence = confidence
        self.tags = tags or []
        self.metadata = metadata or {}
        self.ui_elements = ui_elements or []
        self.vision_analysis = vision_analysis
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "content_type": self.content_type,
            "description": self.description,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
            "ui_elements": self.ui_elements,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.vision_analysis:
            result["vision_analysis"] = self.vision_analysis.to_dict()
        
        return result


class Analyzer:
    """
    Content analysis component for understanding screenshots and extracted text.
    
    Provides OCR text extraction, content classification, and basic AI analysis.
    """
    
    # Compiled regex patterns for performance (Phase 5 optimization)
    _COMPILED_PATTERNS = {
        'code_patterns': re.compile(r'(def |class |import |function|var |let |const |if |for |while |return |print\()', re.IGNORECASE),
        'browser_patterns': re.compile(r'(http[s]?://|www\.|\.com|\.org|\.edu|google|search|browser)', re.IGNORECASE),
        'file_extensions': re.compile(r'\.(py|js|ts|html|css|java|cpp|c|go|rs|rb|php)(?:\s|$)', re.IGNORECASE),
        'git_commands': re.compile(r'\bgit\s+(add|commit|push|pull|checkout|merge|branch|status)', re.IGNORECASE),
        'error_patterns': re.compile(r'(error|exception|traceback|failed|fatal|warning)', re.IGNORECASE),
        'ui_elements': re.compile(r'(button|click|menu|dialog|window|tab|icon|dropdown)', re.IGNORECASE)
    }
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("analyzer")
        
        # Performance optimization caches (Phase 5)
        self._classification_cache = {}
        self._tag_cache = {}
        self._analysis_cache = {}
        
        # Initialize OCR engines
        self._tesseract_available = self._check_tesseract()
        self._easyocr_reader = None
        self._easyocr_available = False
        
        # Initialize AI models (Phase 3) - Multi-instance for maximum performance
        self._florence_model_pool = []
        self._florence_processor = None
        self._florence_available = False
        # Configurable model pool size from environment (default 4)
        self._model_pool_size = int(os.environ.get('EIDOLON_MODEL_INSTANCES', '4'))
        self._current_model_index = 0
        
        if FLORENCE_AVAILABLE:
            self._florence_available = self._load_florence_model_pool()
        
        # Content classification patterns
        self._content_patterns = self._load_content_patterns()
        
        self.logger.info(f"Analyzer initialized - Tesseract: {self._tesseract_available}, EasyOCR: pending, Florence-2: {self._florence_available}")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            # Test tesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _load_florence_model_pool(self) -> bool:
        """Load multiple Florence-2 model instances using parallel loading for maximum performance."""
        if not FLORENCE_AVAILABLE:
            self.logger.warning("Florence-2 dependencies not available")
            return False
        
        try:
            # Use parallel loader for ultra-fast startup
            from ..utils.parallel_loader import load_florence_models_parallel, ModelLoadTimer
            
            with ModelLoadTimer(f"Parallel loading of {self._model_pool_size} Florence-2 instances"):
                self._florence_model_pool, self._florence_processor, self._device = load_florence_models_parallel(
                    self._model_pool_size
                )
            self.logger.info(f"Florence-2 model pool ready: {self._model_pool_size} instances on {self._device.upper()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Florence-2 model pool: {e}")
            self._florence_model_pool = []
            self._florence_processor = None
            return False
    
    def _get_available_model(self):
        """Get next available model from pool for parallel processing."""
        if not self._florence_model_pool:
            return None
        
        # Round-robin model selection for load balancing
        model = self._florence_model_pool[self._current_model_index]
        self._current_model_index = (self._current_model_index + 1) % len(self._florence_model_pool)
        return model
    
    def _load_florence_model(self) -> bool:
        """Load Florence-2 vision model for advanced image understanding."""
        if not FLORENCE_AVAILABLE:
            self.logger.warning("Florence-2 dependencies not available")
            return False
        
        try:
            self.logger.info("Loading Florence-2 vision model...")
            model_name = "microsoft/Florence-2-base"
            
            # Determine best device with M3 GPU optimization
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
                torch_dtype = torch.float32  # MPS optimized for M1/M2/M3
                # Enable MPS optimizations for M3
                torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                self.logger.info("M3 GPU optimizations enabled")
            else:
                torch_dtype = torch.float32
            
            # Load model and processor
            self._florence_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            # Move to best available device
            self._florence_model = self._florence_model.to(device)
            self._device = device
            
            self.logger.info(f"Florence-2 model loaded on {device.upper()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Florence-2 model: {e}")
            self._florence_model = None
            self._florence_processor = None
            return False
    
    def _get_easyocr_reader(self):
        """Initialize EasyOCR reader lazily."""
        if self._easyocr_reader is None and not self._easyocr_available:
            try:
                languages = self.config.analysis.ocr.get("languages", ["en"])
                self._easyocr_reader = easyocr.Reader(languages)
                self._easyocr_available = True
                self.logger.info("EasyOCR reader initialized")
            except Exception as e:
                self.logger.warning(f"EasyOCR not available: {e}")
                self._easyocr_available = False
        return self._easyocr_reader
    
    def _load_content_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for content type classification."""
        return {
            "code": [
                r"def\s+\w+\s*\(",  # Python functions
                r"function\s+\w+\s*\(",  # JavaScript functions
                r"class\s+\w+",  # Class definitions
                r"import\s+\w+",  # Import statements
                r"from\s+\w+\s+import",  # Python imports
                r"#include\s*<",  # C/C++ includes
                r"package\s+\w+",  # Package declarations
                r"public\s+class",  # Java classes
                r"console\.log",  # JavaScript console
                r"print\s*\(",  # Print statements
                r"if\s*\(.+\)\s*{",  # Conditional blocks
                r"for\s*\(.+\)\s*{",  # For loops
                r"while\s*\(.+\)\s*{",  # While loops
                r"\/\*[\s\S]*?\*\/",  # Block comments
                r"\/\/.*",  # Line comments
                r"[a-zA-Z_]\w*\s*=\s*.+",  # Variable assignments
            ],
            "terminal": [
                r"^\$\s+",  # Shell prompt
                r"^>\s+",  # Command prompt
                r"^\w+@\w+:",  # SSH/terminal prompt
                r"npm\s+install",  # npm commands
                r"pip\s+install",  # pip commands
                r"git\s+\w+",  # git commands
                r"sudo\s+\w+",  # sudo commands
                r"ls\s+",  # ls command
                r"cd\s+",  # cd command
                r"mkdir\s+",  # mkdir command
                r"chmod\s+",  # chmod command
                r"curl\s+",  # curl command
            ],
            "browser": [
                r"https?://",  # URLs
                r"www\.\w+\.",  # www domains
                r"@\w+\.\w+",  # Email addresses
                r"Search|Google|Yahoo|Bing",  # Search engines
                r"Sign in|Log in|Login",  # Login pages
                r"Add to cart|Buy now|Checkout",  # E-commerce
                r"Like|Share|Comment|Follow",  # Social media
            ],
            "document": [
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Names
                r"Page\s+\d+",  # Page numbers
                r"Chapter\s+\d+",  # Chapter numbers
                r"\b\w+@\w+\.\w+\b",  # Email addresses
                r"\(\d{3}\)\s*\d{3}-\d{4}",  # Phone numbers
            ],
            "chat": [
                r"\d{1,2}:\d{2}\s*(AM|PM)",  # Timestamps
                r"^[A-Za-z\s]+:",  # Speaker names
                r"You:",  # User messages
                r"Online|Offline|Active",  # Status indicators
                r"Type a message",  # Input prompts
                r"Send|Reply|Forward",  # Action buttons
            ],
            "email": [
                r"From:|To:|Subject:|Date:",  # Email headers
                r"Inbox|Sent|Drafts|Trash",  # Email folders
                r"Reply|Forward|Delete",  # Email actions
                r"Compose|New Message",  # Compose actions
                r"Attachment|Download",  # Attachments
            ],
            "editor": [
                r"File|Edit|View|Tools|Help",  # Menu items
                r"Save|Open|New|Close",  # File operations
                r"Copy|Paste|Cut|Undo|Redo",  # Edit operations
                r"Find|Replace|Search",  # Search operations
                r"Line\s+\d+",  # Line numbers
            ]
        }
    
    @log_performance
    @log_exceptions("eidolon.analyzer")
    def extract_text(self, image_path: Union[str, Path, Image.Image]) -> ExtractedText:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to image file, Path object, or PIL Image.
            
        Returns:
            ExtractedText: Extracted text with confidence and metadata.
        """
        try:
            # Load image
            if isinstance(image_path, Image.Image):
                image = image_path
                image_path_str = "PIL_Image"
            else:
                image_path = Path(image_path)
                image = Image.open(image_path)
                image_path_str = str(image_path)
            
            self.logger.debug(f"Extracting text from: {image_path_str}")
            
            # Get OCR engine preference
            ocr_engine = self.config.analysis.ocr.get("engine", "tesseract")
            confidence_threshold = self.config.analysis.ocr.get("confidence_threshold", 0.6)
            
            # Try primary OCR engine
            if ocr_engine == "tesseract" and self._tesseract_available:
                result = self._extract_with_tesseract(image, confidence_threshold)
            elif ocr_engine == "easyocr":
                result = self._extract_with_easyocr(image, confidence_threshold)
            else:
                # Fallback to available engine
                if self._tesseract_available:
                    result = self._extract_with_tesseract(image, confidence_threshold)
                else:
                    result = self._extract_with_easyocr(image, confidence_threshold)
            
            if result:
                self.logger.debug(f"OCR extracted {result.word_count} words with {result.confidence:.2f} confidence")
                return result
            else:
                # Return empty result if both fail
                return ExtractedText(
                    text="",
                    confidence=0.0,
                    language="en",
                    method="none"
                )
                
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return ExtractedText(
                text="",
                confidence=0.0,
                language="en",
                method="error"
            )
    
    def _extract_with_tesseract(self, image: Image.Image, confidence_threshold: float) -> Optional[ExtractedText]:
        """Extract text using Tesseract OCR."""
        try:
            # Configure tesseract
            config = '--oem 3 --psm 6'  # Use LSTM OCR Engine Mode with uniform text block
            
            # Extract text with confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            
            # Filter by confidence and extract regions
            regions = []
            all_text = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > (confidence_threshold * 100):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    regions.append(TextRegion(
                        text=text,
                        bbox=(x, y, x + w, y + h),
                        confidence=conf / 100.0
                    ))
                    all_text.append(text)
                    confidences.append(conf / 100.0)
            
            # Calculate overall confidence
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Join text
            full_text = ' '.join(all_text)
            
            return ExtractedText(
                text=full_text,
                confidence=overall_confidence,
                language="en",
                regions=regions,
                method="tesseract"
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return None
    
    def _extract_with_easyocr(self, image: Image.Image, confidence_threshold: float) -> Optional[ExtractedText]:
        """Extract text using EasyOCR."""
        try:
            reader = self._get_easyocr_reader()
            if not reader:
                return None
            
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Extract text
            results = reader.readtext(image_array)
            
            # Process results
            regions = []
            all_text = []
            confidences = []
            
            for bbox, text, confidence in results:
                if confidence > confidence_threshold:
                    # Convert bbox to (x1, y1, x2, y2)
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    
                    regions.append(TextRegion(
                        text=text,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence
                    ))
                    all_text.append(text)
                    confidences.append(confidence)
            
            # Calculate overall confidence
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Join text
            full_text = ' '.join(all_text)
            
            return ExtractedText(
                text=full_text,
                confidence=overall_confidence,
                language="en",
                regions=regions,
                method="easyocr"
            )
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return None
    
    @functools.lru_cache(maxsize=256)  # Performance optimization (Phase 5)
    def _classify_text_cached(self, text_hash: str, text: str) -> str:
        """Cached version of content classification for performance."""
        return self._classify_text_internal(text)
    
    def classify_content_type(self, text: str, image_path: Optional[str] = None) -> str:
        """
        Classify content type based on extracted text.
        
        Args:
            text: Extracted text from image.
            image_path: Optional path to image for additional context.
            
        Returns:
            str: Content type classification.
        """
        if not text:
            return "app"  # Default to app for empty content
        
        # Use caching for performance (Phase 5)
        text_hash = str(hash(text))
        return self._classify_text_cached(text_hash, text)
    
    def _classify_text_internal(self, text: str) -> str:
        """Internal classification logic with optimized patterns."""
        text_lower = text.lower()
        scores = {}
        
        # Use compiled patterns for better performance (Phase 5)
        # Check for code content
        if self._COMPILED_PATTERNS['code_patterns'].search(text) or self._COMPILED_PATTERNS['file_extensions'].search(text):
            scores['code'] = 2.0
        
        # Check for browser content
        if self._COMPILED_PATTERNS['browser_patterns'].search(text):
            scores['browser'] = 1.5
        
        # Check for git/development content
        if self._COMPILED_PATTERNS['git_commands'].search(text):
            scores['code'] = scores.get('code', 0) + 1.0
        
        # Check for error content
        if self._COMPILED_PATTERNS['error_patterns'].search(text):
            scores['terminal'] = 1.0
        
        # Check for UI elements
        if self._COMPILED_PATTERNS['ui_elements'].search(text):
            scores['app'] = 1.0
        
        # Fallback to pattern matching for other types
        for content_type, patterns in self._content_patterns.items():
            if content_type not in scores:
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
                    score += matches
                
                # Normalize by pattern count
                if patterns:
                    scores[content_type] = score / len(patterns)
        
        # Find highest scoring type
        if scores:
            max_type = max(scores, key=scores.get)
            if scores[max_type] > 0.1:  # Minimum threshold
                return max_type
        
        # Additional heuristics - map to valid types only
        if any(keyword in text_lower for keyword in ["error", "exception", "traceback", "warning"]):
            return "terminal"
        elif any(keyword in text_lower for keyword in ["menu", "settings", "preferences", "options"]):
            return "app"
        elif len(text.split()) < 5:
            return "app"  # Short text likely UI elements
        elif len(text.split()) > 100:
            return "document"
        else:
            return "document"  # Default to document for general content
    
    @log_performance
    def analyze_content(self, image_path: Union[str, Path], text: str = "") -> ContentAnalysis:
        """
        Analyze content using basic heuristics and text analysis.
        
        Args:
            image_path: Path to the image file.
            text: Optional extracted text for context.
            
        Returns:
            ContentAnalysis: Analysis results with content type and description.
        """
        try:
            self.logger.debug(f"Analyzing content: {image_path}")
            
            # Extract text if not provided
            if not text:
                extracted = self.extract_text(image_path)
                text = extracted.text
            
            # Classify content type
            content_type = self.classify_content_type(text, str(image_path))
            
            # Generate description
            description = self._generate_description(content_type, text)
            
            # Extract tags
            tags = self._extract_tags(content_type, text)
            
            # Detect UI elements
            ui_elements = self._detect_ui_elements(text)
            
            # Perform vision analysis with Florence-2 if available
            vision_analysis = None
            if self._florence_available:
                try:
                    vision_analysis = self._analyze_with_florence(image_path)
                    self.logger.debug("Florence-2 vision analysis completed")
                except Exception as e:
                    self.logger.warning(f"Florence-2 analysis failed: {e}")
            
            # Calculate confidence based on text quality and length
            confidence = self._calculate_analysis_confidence(text, content_type)
            
            # Enhance confidence if we have vision analysis
            if vision_analysis and vision_analysis.confidence > 0:
                confidence = (confidence + vision_analysis.confidence) / 2
            
            return ContentAnalysis(
                content_type=content_type,
                description=description,
                confidence=confidence,
                tags=tags,
                ui_elements=ui_elements,
                vision_analysis=vision_analysis,
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split()) if text else 0,
                    "has_text": bool(text.strip()),
                    "image_path": str(image_path),
                    "vision_model_used": vision_analysis.model_used if vision_analysis else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return ContentAnalysis(
                content_type="error",
                description=f"Analysis failed: {str(e)}",
                confidence=0.0,
                tags=["error"]
            )
    
    def _generate_description(self, content_type: str, text: str) -> str:
        """Generate a description based on content type and text."""
        descriptions = {
            "code": "Programming code or development environment",
            "terminal": "Command line interface or terminal session",
            "browser": "Web browser or internet content",
            "document": "Document or text content",
            "chat": "Chat or messaging interface",
            "email": "Email application or message",
            "editor": "Text or code editor interface",
            "interface": "User interface or settings panel",
            "error": "Error message or warning dialog",
            "empty": "No readable text content",
            "short_text": "Brief text or labels",
            "general": "General content or mixed elements"
        }
        
        base_description = descriptions.get(content_type, "Unknown content type")
        
        if text:
            word_count = len(text.split())
            if word_count > 0:
                base_description += f" ({word_count} words)"
        
        return base_description
    
    def _extract_tags(self, content_type: str, text: str) -> List[str]:
        """Extract relevant tags from content."""
        tags = [content_type]
        
        if not text:
            tags.append("no_text")
            return tags
        
        text_lower = text.lower()
        
        # Programming language detection
        if content_type == "code":
            if any(keyword in text_lower for keyword in ["python", "def ", "import ", ".py"]):
                tags.append("python")
            if any(keyword in text_lower for keyword in ["javascript", "function", "var ", "const ", ".js"]):
                tags.append("javascript")
            if any(keyword in text_lower for keyword in ["java", "public class", "private ", ".java"]):
                tags.append("java")
            if any(keyword in text_lower for keyword in ["cpp", "c++", "#include", "std::"]):
                tags.append("cpp")
        
        # Application detection
        if any(keyword in text_lower for keyword in ["vscode", "visual studio code"]):
            tags.append("vscode")
        if any(keyword in text_lower for keyword in ["terminal", "bash", "zsh"]):
            tags.append("terminal")
        if any(keyword in text_lower for keyword in ["chrome", "firefox", "safari", "browser"]):
            tags.append("browser")
        
        # Content characteristics
        if len(text.split()) > 50:
            tags.append("long_text")
        if any(char.isdigit() for char in text):
            tags.append("contains_numbers")
        if re.search(r'https?://', text):
            tags.append("contains_urls")
        if re.search(r'\b\w+@\w+\.\w+\b', text):
            tags.append("contains_email")
        
        return tags
    
    def _detect_ui_elements(self, text: str) -> List[Dict[str, Any]]:
        """Detect UI elements from text."""
        ui_elements = []
        
        if not text:
            return ui_elements
        
        # Common UI element patterns
        button_patterns = [
            r"\b(Save|Cancel|OK|Apply|Submit|Send|Delete|Edit|Create|Update)\b",
            r"\b(Yes|No|Continue|Back|Next|Finish|Close)\b",
            r"\b(Login|Sign In|Sign Up|Log Out|Register)\b"
        ]
        
        menu_patterns = [
            r"\b(File|Edit|View|Tools|Help|Settings|Options|Preferences)\b",
            r"\b(Home|About|Contact|Search|Profile|Account)\b"
        ]
        
        for pattern in button_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ui_elements.append({
                    "type": "button",
                    "text": match.group(),
                    "position": match.span()
                })
        
        for pattern in menu_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ui_elements.append({
                    "type": "menu",
                    "text": match.group(),
                    "position": match.span()
                })
        
        return ui_elements
    
    def _calculate_analysis_confidence(self, text: str, content_type: str) -> float:
        """Calculate confidence score for content analysis."""
        if not text:
            return 0.1
        
        confidence = 0.5  # Base confidence
        
        # Text quality factors
        word_count = len(text.split())
        if word_count > 5:
            confidence += 0.2
        if word_count > 20:
            confidence += 0.1
        
        # Content type specific confidence boosts
        if content_type != "general":
            confidence += 0.2
        
        # Penalize very short or very long text
        if word_count < 3:
            confidence -= 0.2
        elif word_count > 500:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _analyze_with_florence(self, image_path: Union[str, Path]) -> Optional[VisionAnalysis]:
        """
        Analyze image using AI-enhanced analysis (Florence-2 or fallback methods).
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            VisionAnalysis: Vision analysis results or None if failed.
        """
        if not self._florence_available or not self._florence_model_pool:
            # Fallback to basic AI-enhanced analysis
            return self._analyze_with_basic_ai(image_path)
        
        try:
            # Try Florence-2 model pool analysis for maximum performance
            return self._analyze_with_florence_model_pool(image_path)
        except Exception as e:
            self.logger.warning(f"Florence-2 model pool analysis failed, using fallback: {e}")
            return self._analyze_with_basic_ai(image_path)
    
    def _analyze_with_florence_model_pool(self, image_path: Union[str, Path]) -> Optional[VisionAnalysis]:
        """Analyze using Florence-2 model pool for maximum performance."""
        # Get available model from pool for parallel processing
        model = self._get_available_model()
        if not model:
            return None
        
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Ultra-fast caption task with optimized parameters
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self._florence_processor(text=prompt, images=image, return_tensors="pt")
        
        # Create proper attention mask for maximum performance
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # Move inputs to GPU device for maximum speed
        device = getattr(self, '_device', 'cpu')
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with maximum performance parameters
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,  # Reduced for maximum speed
                do_sample=False,
                num_beams=1,  # Fastest generation
                pad_token_id=self._florence_processor.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for speed
            )
        
        # Decode results
        generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self._florence_processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        # Extract description
        description = parsed_answer.get(prompt, "")
        if isinstance(description, dict):
            description = description.get("text", str(description))
        
        # Create VisionAnalysis with fast processing
        return VisionAnalysis(
            description=description,
            objects=[],
            scene_type="unknown",
            confidence=0.85  # High confidence for fast processing
        )
    
    def _analyze_with_florence_model(self, image_path: Union[str, Path]) -> Optional[VisionAnalysis]:
        """Analyze using Florence-2 model."""
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Simple caption task first
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self._florence_processor(text=prompt, images=image, return_tensors="pt")
        
        # Create proper attention mask to avoid warnings
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # Move inputs to same device as model
        device = getattr(self, '_device', 'cpu')
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with optimized parameters for M3 GPU
        with torch.no_grad():
            generated_ids = self._florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,  # Reduced for faster processing
                do_sample=False,
                num_beams=1,  # Faster than beam search
                pad_token_id=self._florence_processor.tokenizer.eos_token_id
            )
        
        # Decode results
        generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self._florence_processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        # Extract description
        description = ""
        if isinstance(parsed_answer, dict) and prompt in parsed_answer:
            description = parsed_answer[prompt]
        
        # Determine scene type
        scene_type = self._classify_scene_type(description, [])
        
        return VisionAnalysis(
            description=description,
            objects=[],
            scene_type=scene_type,
            confidence=0.8 if description else 0.1,
            ui_elements=[],
            model_used="florence-2"
        )
    
    def _analyze_with_basic_ai(self, image_path: Union[str, Path]) -> VisionAnalysis:
        """
        Analyze image using basic AI-enhanced heuristics.
        
        This provides a fallback when Florence-2 is not available.
        """
        try:
            # Load image for basic analysis
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # Basic image analysis
            description = f"Screenshot image ({width}x{height})"
            objects = []
            ui_elements = []
            confidence = 0.6
            
            # Analyze image properties
            if width > height:
                description += " in landscape orientation"
            else:
                description += " in portrait orientation"
            
            # Estimate scene type based on image characteristics
            aspect_ratio = width / height
            if 1.5 <= aspect_ratio <= 2.0:
                scene_type = "desktop"
                description += " showing desktop content"
            elif aspect_ratio > 2.0:
                scene_type = "web_browsing" 
                description += " possibly showing web browser"
            else:
                scene_type = "application"
                description += " showing application window"
            
            # Basic object detection simulation
            objects.append({
                "label": "window",
                "bbox": [0, 0, width, height],
                "confidence": 0.9
            })
            
            # Basic UI element detection
            ui_elements.append({
                "type": "window",
                "text": "Main Window",
                "bbox": [0, 0, width, min(50, height)],
                "confidence": 0.8
            })
            
            return VisionAnalysis(
                description=description,
                objects=objects,
                scene_type=scene_type,
                confidence=confidence,
                ui_elements=ui_elements,
                model_used="basic-ai"
            )
            
        except Exception as e:
            self.logger.error(f"Basic AI analysis failed: {e}")
            return VisionAnalysis(
                description="AI analysis unavailable",
                objects=[],
                scene_type="unknown", 
                confidence=0.1,
                ui_elements=[],
                model_used="fallback"
            )
    
    def _classify_scene_type(self, description: str, objects: List[Dict[str, Any]]) -> str:
        """Classify the scene type based on description and detected objects."""
        if not description and not objects:
            return "unknown"
        
        text_to_analyze = description.lower()
        object_labels = [obj.get("label", "").lower() for obj in objects]
        all_text = text_to_analyze + " " + " ".join(object_labels)
        
        # Scene classification patterns
        if any(word in all_text for word in ["code", "editor", "terminal", "console", "programming"]):
            return "development"
        elif any(word in all_text for word in ["browser", "website", "webpage", "chrome", "firefox"]):
            return "web_browsing"
        elif any(word in all_text for word in ["document", "text", "writing", "word", "pdf"]):
            return "document"
        elif any(word in all_text for word in ["email", "message", "chat", "conversation"]):
            return "communication"
        elif any(word in all_text for word in ["settings", "preferences", "configuration", "options"]):
            return "system_settings"
        elif any(word in all_text for word in ["file", "folder", "directory", "explorer", "finder"]):
            return "file_management"
        else:
            return "application"
    
    def detect_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect error messages and warnings in text content.
        
        Args:
            text: Text content to analyze for errors.
            
        Returns:
            List of detected errors with metadata.
        """
        if not text:
            return []
        
        errors = []
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Error patterns with severity levels
        error_patterns = [
            # Critical errors
            (r'traceback.*most recent call last', 'critical', 'python_traceback'),
            (r'fatal\s*error', 'critical', 'fatal'),
            (r'segmentation\s*fault', 'critical', 'segfault'),
            (r'core\s*dumped', 'critical', 'core_dump'),
            (r'panic:', 'critical', 'panic'),
            (r'assertion\s*failed', 'critical', 'assertion'),
            
            # High severity errors
            (r'error:\s*.*', 'high', 'general_error'),
            (r'exception\s*in', 'high', 'exception'),
            (r'uncaught\s*exception', 'high', 'uncaught_exception'),
            (r'nullpointerexception', 'high', 'null_pointer'),
            (r'syntax\s*error', 'high', 'syntax_error'),
            (r'compilation\s*error', 'high', 'compilation_error'),
            (r'connection\s*refused', 'high', 'connection_error'),
            (r'permission\s*denied', 'high', 'permission_error'),
            (r'file\s*not\s*found', 'high', 'file_not_found'),
            (r'command\s*not\s*found', 'high', 'command_not_found'),
            
            # Medium severity errors
            (r'failed\s*to', 'medium', 'failure'),
            (r'unable\s*to', 'medium', 'unable'),
            (r'could\s*not', 'medium', 'could_not'),
            (r'timeout', 'medium', 'timeout'),
            (r'invalid\s*.*', 'medium', 'invalid'),
            (r'unexpected\s*.*', 'medium', 'unexpected'),
            
            # Low severity warnings
            (r'warning:\s*.*', 'low', 'warning'),
            (r'deprecated', 'low', 'deprecated'),
            (r'caution', 'low', 'caution'),
            (r'notice', 'low', 'notice')
        ]
        
        # Detect errors in text
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            
            for pattern, severity, error_type in error_patterns:
                if re.search(pattern, line_lower):
                    # Extract context around the error
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context_lines = lines[context_start:context_end]
                    
                    # Clean up the error message
                    error_message = line.strip()
                    if len(error_message) > 200:
                        error_message = error_message[:200] + "..."
                    
                    errors.append({
                        'line_number': i + 1,
                        'error_type': error_type,
                        'severity': severity,
                        'message': error_message,
                        'context': '\n'.join(context_lines),
                        'pattern_matched': pattern
                    })
                    break  # Only match first pattern per line
        
        return errors
    
    def extract_commands(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract command-line commands from text content.
        
        Args:
            text: Text content to analyze for commands.
            
        Returns:
            List of detected commands with metadata.
        """
        if not text:
            return []
        
        commands = []
        lines = text.split('\n')
        
        # Command patterns for different shells
        command_patterns = [
            # Unix/Linux shell patterns
            (r'^\$\s+(.+)', 'bash', 'unix_shell'),
            (r'^%\s+(.+)', 'zsh', 'unix_shell'),
            (r'^>\s+(.+)', 'powershell', 'windows_shell'),
            (r'^[a-zA-Z_][\w]*@[a-zA-Z_][\w]*[~$#]\s+(.+)', 'ssh', 'remote_shell'),
            
            # Common command prefixes
            (r'^sudo\s+(.+)', 'sudo', 'elevated_command'),
            (r'^(\w+)\s+.*', 'command', 'general_command')
        ]
        
        # Well-known commands to identify
        known_commands = {
            'ls', 'cd', 'pwd', 'mkdir', 'rmdir', 'rm', 'cp', 'mv', 'chmod', 'chown',
            'cat', 'less', 'more', 'head', 'tail', 'grep', 'find', 'locate',
            'git', 'npm', 'pip', 'docker', 'kubectl', 'ssh', 'scp', 'rsync',
            'make', 'cmake', 'gcc', 'python', 'node', 'java', 'go', 'rust',
            'vim', 'nano', 'emacs', 'code', 'subl',
            'systemctl', 'service', 'ps', 'top', 'htop', 'kill', 'killall',
            'wget', 'curl', 'ping', 'netstat', 'lsof', 'df', 'du', 'mount'
        }
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            for pattern, shell_type, command_category in command_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    if len(match.groups()) > 0:
                        command_text = match.group(1).strip()
                    else:
                        command_text = line_stripped
                    
                    # Parse command and arguments
                    parts = command_text.split()
                    if not parts:
                        continue
                    
                    command_name = parts[0]
                    arguments = ' '.join(parts[1:]) if len(parts) > 1 else None
                    
                    # Check if it's a known command
                    is_known_command = command_name in known_commands
                    
                    commands.append({
                        'line_number': i + 1,
                        'command': command_name,
                        'arguments': arguments,
                        'full_command': command_text,
                        'shell_type': shell_type,
                        'category': command_category,
                        'is_known_command': is_known_command,
                        'raw_line': line_stripped
                    })
                    break  # Only match first pattern per line
        
        return commands
    
    def analyze_context(self, text: str, content_type: str) -> Dict[str, Any]:
        """
        Analyze the context and activities shown in screen content.
        
        Args:
            text: Extracted text content.
            content_type: Classified content type.
            
        Returns:
            Dictionary with context analysis results.
        """
        if not text:
            return {
                'activity_type': 'unknown',
                'confidence': 0.0,
                'details': {},
                'insights': []
            }
        
        insights = []
        details = {}
        activity_type = 'general'
        confidence = 0.5
        
        # Detect specific activities based on content type and text
        if content_type == 'terminal':
            commands = self.extract_commands(text)
            errors = self.detect_errors(text)
            
            if commands:
                activity_type = 'command_execution'
                details['commands'] = commands
                insights.append(f"Detected {len(commands)} commands")
                confidence = 0.8
                
                # Analyze command patterns
                command_types = [cmd['category'] for cmd in commands]
                if 'elevated_command' in command_types:
                    insights.append("Administrative/elevated commands detected")
                
                git_commands = [cmd for cmd in commands if cmd['command'] == 'git']
                if git_commands:
                    insights.append(f"Git version control activity ({len(git_commands)} commands)")
                
                docker_commands = [cmd for cmd in commands if cmd['command'] == 'docker']
                if docker_commands:
                    insights.append(f"Docker containerization activity ({len(docker_commands)} commands)")
            
            if errors:
                details['errors'] = errors
                error_severities = [err['severity'] for err in errors]
                critical_errors = sum(1 for s in error_severities if s == 'critical')
                high_errors = sum(1 for s in error_severities if s == 'high')
                
                if critical_errors > 0:
                    insights.append(f"Critical errors detected ({critical_errors})")
                elif high_errors > 0:
                    insights.append(f"High severity errors detected ({high_errors})")
                else:
                    insights.append(f"Warnings or minor errors detected ({len(errors)})")
        
        elif content_type == 'code':
            # Analyze programming activity
            activity_type = 'programming'
            confidence = 0.7
            
            # Detect programming languages
            language_indicators = {
                'python': ['def ', 'import ', 'from ', '__init__', 'self.'],
                'javascript': ['function', 'var ', 'let ', 'const ', 'console.log'],
                'java': ['public class', 'private ', 'public ', 'import java'],
                'cpp': ['#include', 'std::', 'namespace', 'class '],
                'go': ['func ', 'package ', 'import ', 'go '],
                'rust': ['fn ', 'use ', 'struct ', 'impl '],
                'typescript': ['interface ', 'type ', 'export ', 'import ']
            }
            
            detected_languages = []
            text_lower = text.lower()
            for lang, indicators in language_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    detected_languages.append(lang)
            
            if detected_languages:
                details['languages'] = detected_languages
                insights.append(f"Programming languages: {', '.join(detected_languages)}")
            
            # Look for specific development activities
            if any(word in text_lower for word in ['test', 'spec', 'assert']):
                insights.append("Testing activity detected")
            if any(word in text_lower for word in ['debug', 'breakpoint', 'console']):
                insights.append("Debugging activity detected")
            if any(word in text_lower for word in ['commit', 'merge', 'branch', 'pull request']):
                insights.append("Version control activity detected")
        
        elif content_type == 'browser':
            activity_type = 'web_browsing'
            confidence = 0.6
            
            # Detect URLs and domains
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, text)
            if urls:
                details['urls'] = urls[:5]  # Limit to first 5
                insights.append(f"Web browsing activity ({len(urls)} URLs detected)")
            
            # Detect common web activities
            if any(word in text.lower() for word in ['search', 'google', 'bing', 'duckduckgo']):
                insights.append("Search activity detected")
            if any(word in text.lower() for word in ['login', 'sign in', 'password', 'username']):
                insights.append("Authentication activity detected")
            if any(word in text.lower() for word in ['cart', 'checkout', 'payment', 'buy', 'purchase']):
                insights.append("E-commerce activity detected")
        
        elif content_type == 'document':
            activity_type = 'document_work'
            confidence = 0.6
            
            # Analyze document type and content
            word_count = len(text.split())
            details['word_count'] = word_count
            
            if word_count > 500:
                insights.append("Long-form document or article")
            elif word_count > 100:
                insights.append("Medium-length document")
            else:
                insights.append("Short document or notes")
            
            # Detect document types
            if any(word in text.lower() for word in ['agenda', 'meeting', 'notes', 'minutes']):
                insights.append("Meeting or note-taking activity")
            if any(word in text.lower() for word in ['email', 'subject:', 'from:', 'to:']):
                insights.append("Email communication")
            if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text):
                insights.append("Contains dates - possibly scheduling or planning")
        
        # Add general insights
        if len(text.split()) > 1000:
            insights.append("High text density - detailed content")
        
        errors = self.detect_errors(text)
        if errors:
            details['detected_errors'] = len(errors)
            severity_counts = {}
            for error in errors:
                severity = error['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            details['error_severities'] = severity_counts
        
        return {
            'activity_type': activity_type,
            'confidence': confidence,
            'details': details,
            'insights': insights,
            'text_stats': {
                'word_count': len(text.split()),
                'line_count': len(text.split('\n')),
                'char_count': len(text)
            }
        }