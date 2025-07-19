"""
Core screen capture functionality with ML pipeline and fallbacks
"""
import asyncio
import hashlib
import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import base64

import cv2
import numpy as np
import pyautogui
from PIL import Image
import pygetwindow as gw

# OCR options - multiple fallbacks for stability
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import tesserocr
    TESSEROCR_AVAILABLE = True
except ImportError:
    TESSEROCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# CLIP for semantic embeddings
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Vision API client
import httpx
from openai import OpenAI

# Image similarity
from skimage.metrics import structural_similarity as ssim

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenCapture:
    """Resilient screen capture with multiple ML backends"""
    
    def __init__(self, 
                 screenshots_dir: str = "screenshots",
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 clip_model: str = "ViT-B/32"):
        
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.openai_api_base = openai_api_base
        self.clip_model_name = clip_model
        
        # ML component status
        self.ocr_failures = 0
        self.clip_failures = 0
        self.vision_failures = 0
        
        # Circuit breaker thresholds
        self.max_failures = 3
        
        # Initialize ML components
        self._init_ocr()
        self._init_clip()
        self._init_vision_client()
        
        # For scene change detection
        self.last_image = None
        self.last_clip_time = 0
        self.clip_interval = 15  # seconds between CLIP embeddings
    
    def _init_ocr(self):
        """Initialize OCR engines with fallbacks"""
        self.ocr_engines = []
        
        if TESSEROCR_AVAILABLE:
            try:
                self.tesseract_api = tesserocr.PyTessBaseAPI()
                self.tesseract_api.SetPageSegMode(tesserocr.PSM.AUTO)
                self.ocr_engines.append("tesserocr")
                logger.info("TesserOCR initialized successfully")
            except Exception as e:
                logger.warning(f"TesserOCR init failed: {e}")
        
        if TESSERACT_AVAILABLE:
            try:
                # Test pytesseract
                pytesseract.get_tesseract_version()
                self.ocr_engines.append("pytesseract")
                logger.info("PyTesseract available")
            except Exception as e:
                logger.warning(f"PyTesseract not available: {e}")
        
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                self.ocr_engines.append("easyocr")
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
        
        if not self.ocr_engines:
            logger.warning("No OCR engines available! Install pytesseract, tesserocr, or easyocr")
    
    def _init_clip(self):
        """Initialize CLIP model for embeddings"""
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available - semantic search disabled")
            return
            
        try:
            self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name, 
                device=self.clip_device
            )
            logger.info(f"CLIP model loaded on {self.clip_device}")
        except Exception as e:
            logger.error(f"CLIP initialization failed: {e}")
            self.clip_failures = self.max_failures  # Disable CLIP
    
    def _init_vision_client(self):
        """Initialize OpenAI client for OpenRouter"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key or api_key == 'your-openrouter-key-here':
                logger.warning("No valid OpenRouter API key found - vision fallback disabled")
                self.vision_client = None
                return
                
            self.vision_client = OpenAI(
                api_key=api_key,
                base_url=self.openai_api_base
            )
            logger.info("Vision client initialized with OpenRouter")
        except Exception as e:
            logger.error(f"Vision client init failed: {e}")
            self.vision_client = None
    
    async def capture_screen(self, 
                           save_image: bool = True, 
                           force_vision: bool = False) -> Dict[str, Any]:
        """
        Capture screen and extract information
        
        Returns:
            Dict with keys: image_path, full_text, ocr_conf, clip_vec, 
                           window_title, app_name, scene_hash
        """
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            img_array = np.array(screenshot)
            
            # Get active window info
            window_info = self._get_active_window()
            
            # Generate scene hash for duplicate detection
            scene_hash = self._compute_scene_hash(img_array)
            
            # Save image if requested
            image_path = None
            if save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = self.screenshots_dir / f"screen_{timestamp}.png"
                screenshot.save(image_path)
                logger.info(f"Screenshot saved: {image_path}")
            
            # Extract text via OCR
            full_text, ocr_conf = await self._extract_text(img_array, force_vision)
            
            # Generate CLIP embedding if scene changed
            clip_vec = await self._generate_clip_embedding(img_array)
            
            # Update last image for change detection
            self.last_image = img_array
            
            return {
                "image_path": str(image_path) if image_path else None,
                "full_text": full_text,
                "ocr_conf": ocr_conf,
                "clip_vec": clip_vec.tolist() if clip_vec is not None else None,
                "window_title": window_info.get("title"),
                "app_name": window_info.get("app"),
                "scene_hash": scene_hash
            }
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            # Return minimal fallback data
            return {
                "image_path": None,
                "full_text": f"Capture error: {str(e)}",
                "ocr_conf": 0,
                "clip_vec": None,
                "window_title": None,
                "app_name": None,
                "scene_hash": None
            }
    
    async def _extract_text(self, img_array: np.ndarray, force_vision: bool = False) -> Tuple[str, int]:
        """Extract text using OCR with fallback to GPT-4o Vision"""
        
        if not force_vision and self.ocr_failures < self.max_failures:
            # Try OCR engines in order of preference
            for engine in self.ocr_engines:
                try:
                    text, confidence = await self._try_ocr_engine(img_array, engine)
                    
                    # If confidence is high enough, use OCR result
                    if confidence >= 80:
                        logger.info(f"OCR success with {engine}: conf={confidence}")
                        return text, confidence
                    
                    # Low confidence - might try vision fallback
                    if confidence < 80 and not force_vision:
                        logger.info(f"Low OCR confidence ({confidence}) - trying vision fallback")
                        vision_text = await self._try_vision_fallback(img_array)
                        if vision_text:
                            return vision_text, 95  # Assume high confidence for vision
                    
                    return text, confidence
                    
                except Exception as e:
                    logger.warning(f"OCR engine {engine} failed: {e}")
                    continue
            
            # All OCR engines failed
            self.ocr_failures += 1
            logger.warning(f"All OCR engines failed ({self.ocr_failures}/{self.max_failures})")
        
        # Fallback to vision
        if self.vision_failures < self.max_failures:
            try:
                vision_text = await self._try_vision_fallback(img_array)
                if vision_text:
                    return vision_text, 95
            except Exception as e:
                self.vision_failures += 1
                logger.error(f"Vision fallback failed: {e}")
        
        # Ultimate fallback
        return f"Text extraction failed at {datetime.now()}", 0
    
    async def _try_ocr_engine(self, img_array: np.ndarray, engine: str) -> Tuple[str, int]:
        """Try a specific OCR engine"""
        
        if engine == "tesserocr" and hasattr(self, 'tesseract_api'):
            pil_img = Image.fromarray(img_array)
            self.tesseract_api.SetImage(pil_img)
            text = self.tesseract_api.GetUTF8Text()
            conf = self.tesseract_api.MeanTextConf()
            return text.strip(), max(0, conf)
        
        elif engine == "pytesseract":
            pil_img = Image.fromarray(img_array)
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            text = " ".join([word for word in data['text'] if word.strip()])
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            conf = sum(confidences) // len(confidences) if confidences else 0
            return text, conf
        
        elif engine == "easyocr" and hasattr(self, 'easyocr_reader'):
            results = self.easyocr_reader.readtext(img_array)
            text = " ".join([result[1] for result in results])
            # EasyOCR confidence is 0-1, convert to 0-100
            confidences = [result[2] for result in results]
            conf = int((sum(confidences) / len(confidences)) * 100) if confidences else 0
            return text, conf
        
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")
    
    async def _try_vision_fallback(self, img_array: np.ndarray) -> Optional[str]:
        """Use GPT-4o Vision via OpenRouter as fallback"""
        if self.vision_client is None:
            logger.warning("Vision client not available")
            return None
            
        try:
            # Convert image to base64
            pil_img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Call vision API via OpenRouter
            response = self.vision_client.chat.completions.create(
                model="openai/gpt-4o-mini",  # OpenRouter model name
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Extract all visible text from this screenshot. Return only the text content, no descriptions."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            text = response.choices[0].message.content
            logger.info("Vision API successful via OpenRouter")
            return text
            
        except Exception as e:
            logger.error(f"Vision API failed: {e}")
            self.vision_failures += 1
            return None
    
    async def _generate_clip_embedding(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """Generate CLIP embedding if scene changed"""
        
        if self.clip_failures >= self.max_failures or not CLIP_AVAILABLE:
            return None
        
        # Check if scene changed significantly
        if self.last_image is not None:
            similarity = ssim(
                cv2.cvtColor(self.last_image, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY),
                full=False
            )
            
            # If scene didn't change much and we generated embedding recently, skip
            current_time = time.time()
            if (similarity > 0.95 and 
                current_time - self.last_clip_time < self.clip_interval):
                return None
        
        try:
            pil_img = Image.fromarray(img_array)
            image_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.clip_device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            self.last_clip_time = time.time()
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            self.clip_failures += 1
            return None
    
    def _get_active_window(self) -> Dict[str, Optional[str]]:
        """Get information about the active window"""
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                # Make sure we get string values, not method objects
                title = getattr(active_window, 'title', None)
                if callable(title):
                    title = title()
                elif hasattr(active_window, '_title'):
                    title = active_window._title
                
                return {
                    "title": str(title) if title else "Unknown Window",
                    "app": str(getattr(active_window, 'title', 'Unknown App'))[:50] if hasattr(active_window, 'title') else None
                }
        except Exception as e:
            logger.warning(f"Failed to get active window: {e}")
        
        return {"title": "Unknown Window", "app": "Unknown App"}
    
    def _compute_scene_hash(self, img_array: np.ndarray) -> str:
        """Compute hash of image for duplicate detection"""
        try:
            # Resize to small size for consistent hashing
            small_img = cv2.resize(img_array, (64, 64))
            img_bytes = small_img.tobytes()
            return hashlib.md5(img_bytes).hexdigest()
        except Exception:
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status of all components"""
        return {
            "ocr_engines": self.ocr_engines,
            "ocr_failures": self.ocr_failures,
            "clip_available": CLIP_AVAILABLE and self.clip_failures < self.max_failures,
            "clip_failures": self.clip_failures,
            "vision_available": self.vision_failures < self.max_failures,
            "vision_failures": self.vision_failures
        } 