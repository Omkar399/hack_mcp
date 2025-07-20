"""
Style Replication system for Eidolon AI Personal Assistant

Replicates user communication style, writing patterns, and response preferences
to enable authentic digital twin interactions.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..core.memory import MemorySystem
from ..models.cloud_api import CloudAPIManager


class StyleDimension(Enum):
    """Dimensions of communication style"""
    FORMALITY = "formality"  # formal, casual, mixed
    VERBOSITY = "verbosity"  # concise, moderate, verbose
    TONE = "tone"  # friendly, professional, neutral, enthusiastic
    STRUCTURE = "structure"  # structured, flowing, mixed
    TECHNICAL_LEVEL = "technical_level"  # basic, intermediate, advanced
    EMOTION = "emotion"  # expressive, controlled, neutral


class ResponseType(Enum):
    """Types of responses the system can generate"""
    EMAIL = "email"
    MESSAGE = "message"
    DOCUMENT = "document"
    CODE_COMMENT = "code_comment"
    SOCIAL_POST = "social_post"
    MEETING_NOTES = "meeting_notes"
    TASK_DESCRIPTION = "task_description"


@dataclass
class StyleFeature:
    """Individual style feature with measurements"""
    name: str
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    examples: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleModel:
    """Comprehensive model of user's communication style"""
    user_id: str = "default"
    
    # Style dimensions
    formality_score: float = 0.5  # 0=very casual, 1=very formal
    verbosity_score: float = 0.5  # 0=very concise, 1=very verbose
    technical_score: float = 0.5  # 0=basic, 1=highly technical
    emotion_score: float = 0.5   # 0=neutral, 1=very expressive
    structure_score: float = 0.5  # 0=loose, 1=highly structured
    
    # Writing patterns
    avg_sentence_length: float = 15.0
    avg_paragraph_length: float = 3.0
    vocabulary_complexity: float = 0.5
    punctuation_style: Dict[str, float] = field(default_factory=dict)
    
    # Common phrases and expressions
    common_openings: List[str] = field(default_factory=list)
    common_closings: List[str] = field(default_factory=list)
    frequent_phrases: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    
    # Response patterns by type
    response_templates: Dict[ResponseType, Dict[str, Any]] = field(default_factory=dict)
    
    # Confidence and metadata
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'formality_score': self.formality_score,
            'verbosity_score': self.verbosity_score,
            'technical_score': self.technical_score,
            'emotion_score': self.emotion_score,
            'structure_score': self.structure_score,
            'avg_sentence_length': self.avg_sentence_length,
            'avg_paragraph_length': self.avg_paragraph_length,
            'vocabulary_complexity': self.vocabulary_complexity,
            'punctuation_style': self.punctuation_style,
            'common_openings': self.common_openings,
            'common_closings': self.common_closings,
            'frequent_phrases': self.frequent_phrases,
            'technical_terms': self.technical_terms,
            'response_templates': {
                k.value: v for k, v in self.response_templates.items()
            },
            'confidence': self.confidence,
            'sample_count': self.sample_count,
            'last_updated': self.last_updated.isoformat(),
            'version': self.version
        }


@dataclass
class ResponseAdaptation:
    """Adaptation of a response to match user style"""
    original_response: str
    adapted_response: str
    adaptation_type: ResponseType
    style_adjustments: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_response': self.original_response,
            'adapted_response': self.adapted_response,
            'adaptation_type': self.adaptation_type.value,
            'style_adjustments': self.style_adjustments,
            'confidence': self.confidence
        }


class StyleReplicator:
    """Advanced system for replicating user communication style"""
    
    def __init__(self):
        self.logger = get_component_logger("style_replicator")
        self.config = get_config()
        self.memory = MemorySystem()
        self.cloud_api = CloudAPIManager()
        
        # Style models
        self.style_model: Optional[StyleModel] = None
        self.adaptation_history: List[ResponseAdaptation] = []
        
        # Analysis patterns
        self.formality_indicators = {
            'formal': ['dear', 'sincerely', 'regards', 'please find', 'i would like to', 'thank you for'],
            'casual': ['hey', 'hi', 'thanks', 'no problem', 'sure thing', 'catch you later']
        }
        
        self.emotion_indicators = {
            'high': ['!', 'amazing', 'fantastic', 'love', 'hate', 'excited', 'thrilled'],
            'low': ['.', 'okay', 'fine', 'alright', 'noted', 'understood']
        }
        
        # Load existing style model
        asyncio.create_task(self._load_style_model())
    
    @log_exceptions
    async def analyze_communication_samples(
        self,
        text_samples: List[Dict[str, Any]]
    ) -> StyleModel:
        """Analyze communication samples to build style model"""
        
        self.logger.info(f"Analyzing {len(text_samples)} communication samples")
        
        if not text_samples:
            return StyleModel()
        
        # Initialize or update style model
        if self.style_model is None:
            self.style_model = StyleModel()
        
        # Analyze different style dimensions
        formality_scores = []
        verbosity_scores = []
        technical_scores = []
        emotion_scores = []
        structure_scores = []
        
        sentence_lengths = []
        paragraph_lengths = []
        all_openings = []
        all_closings = []
        all_phrases = []
        all_technical_terms = []
        
        for sample in text_samples:
            text = sample.get('text', '')
            sample_type = sample.get('type', 'unknown')
            
            if not text or len(text.strip()) < 10:
                continue
            
            # Analyze style dimensions
            formality_scores.append(self._analyze_formality(text))
            verbosity_scores.append(self._analyze_verbosity(text))
            technical_scores.append(self._analyze_technical_level(text))
            emotion_scores.append(self._analyze_emotion_level(text))
            structure_scores.append(self._analyze_structure(text))
            
            # Analyze writing patterns
            sentences = self._split_sentences(text)
            sentence_lengths.extend([len(s.split()) for s in sentences])
            
            paragraphs = text.split('\n\n')
            paragraph_lengths.extend([len(p.split('\n')) for p in paragraphs if p.strip()])
            
            # Extract openings and closings
            opening = self._extract_opening(text)
            if opening:
                all_openings.append(opening.lower())
            
            closing = self._extract_closing(text)
            if closing:
                all_closings.append(closing.lower())
            
            # Extract frequent phrases
            phrases = self._extract_phrases(text)
            all_phrases.extend(phrases)
            
            # Extract technical terms
            tech_terms = self._extract_technical_terms(text)
            all_technical_terms.extend(tech_terms)
        
        # Update style model with aggregated analysis
        if formality_scores:
            self.style_model.formality_score = statistics.mean(formality_scores)
        if verbosity_scores:
            self.style_model.verbosity_score = statistics.mean(verbosity_scores)
        if technical_scores:
            self.style_model.technical_score = statistics.mean(technical_scores)
        if emotion_scores:
            self.style_model.emotion_score = statistics.mean(emotion_scores)
        if structure_scores:
            self.style_model.structure_score = statistics.mean(structure_scores)
        
        # Update writing patterns
        if sentence_lengths:
            self.style_model.avg_sentence_length = statistics.mean(sentence_lengths)
        if paragraph_lengths:
            self.style_model.avg_paragraph_length = statistics.mean(paragraph_lengths)
        
        # Update common patterns (keep most frequent)
        from collections import Counter
        
        if all_openings:
            opening_counts = Counter(all_openings)
            self.style_model.common_openings = [
                phrase for phrase, count in opening_counts.most_common(10)
                if count >= 2
            ]
        
        if all_closings:
            closing_counts = Counter(all_closings)
            self.style_model.common_closings = [
                phrase for phrase, count in closing_counts.most_common(10)
                if count >= 2
            ]
        
        if all_phrases:
            phrase_counts = Counter(all_phrases)
            self.style_model.frequent_phrases = [
                phrase for phrase, count in phrase_counts.most_common(20)
                if count >= 2 and len(phrase) > 5
            ]
        
        if all_technical_terms:
            tech_counts = Counter(all_technical_terms)
            self.style_model.technical_terms = [
                term for term, count in tech_counts.most_common(30)
                if count >= 2
            ]
        
        # Calculate overall confidence
        self.style_model.sample_count = len(text_samples)
        self.style_model.confidence = min(1.0, len(text_samples) / 50.0)  # Full confidence at 50+ samples
        self.style_model.last_updated = datetime.now()
        
        # Save updated model
        await self._save_style_model()
        
        self.logger.info(f"Updated style model with confidence {self.style_model.confidence:.2f}")
        return self.style_model
    
    def _analyze_formality(self, text: str) -> float:
        """Analyze formality level of text (0=casual, 1=formal)"""
        
        text_lower = text.lower()
        
        formal_score = 0
        casual_score = 0
        
        # Count formal indicators
        for indicator in self.formality_indicators['formal']:
            formal_score += text_lower.count(indicator)
        
        # Count casual indicators
        for indicator in self.formality_indicators['casual']:
            casual_score += text_lower.count(indicator)
        
        # Additional formal indicators
        if re.search(r'\b(would|could|may|might|shall)\b', text_lower):
            formal_score += 2
        
        if re.search(r'\b(gonna|wanna|gotta|yeah|yep)\b', text_lower):
            casual_score += 2
        
        # Sentence structure indicators
        complex_sentences = len(re.findall(r'[,;:]', text))
        if complex_sentences > len(text.split('.')) / 2:
            formal_score += 1
        
        # Contractions (casual indicator)
        contractions = len(re.findall(r"n't|'re|'ve|'ll|'d", text))
        casual_score += contractions
        
        total_score = formal_score + casual_score
        if total_score == 0:
            return 0.5  # Default neutral
        
        return formal_score / total_score
    
    def _analyze_verbosity(self, text: str) -> float:
        """Analyze verbosity level (0=concise, 1=verbose)"""
        
        words = text.split()
        sentences = self._split_sentences(text)
        
        if not sentences:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Normalize: 5-15 words = concise, 25+ words = verbose
        if avg_sentence_length <= 10:
            return 0.2
        elif avg_sentence_length <= 15:
            return 0.4
        elif avg_sentence_length <= 20:
            return 0.6
        elif avg_sentence_length <= 25:
            return 0.8
        else:
            return 1.0
    
    def _analyze_technical_level(self, text: str) -> float:
        """Analyze technical complexity (0=basic, 1=highly technical)"""
        
        text_lower = text.lower()
        
        # Technical indicators
        technical_terms = [
            'algorithm', 'implementation', 'architecture', 'framework', 'database',
            'api', 'function', 'class', 'method', 'variable', 'parameter',
            'optimization', 'performance', 'scalability', 'deployment',
            'configuration', 'integration', 'authentication', 'encryption'
        ]
        
        tech_count = sum(text_lower.count(term) for term in technical_terms)
        
        # Code-like patterns
        code_patterns = len(re.findall(r'[{}();]|def |class |import |from ', text))
        
        # Acronyms (often technical)
        acronyms = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        total_indicators = tech_count + code_patterns + acronyms
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        technical_density = total_indicators / word_count
        return min(1.0, technical_density * 10)  # Scale appropriately
    
    def _analyze_emotion_level(self, text: str) -> float:
        """Analyze emotional expressiveness (0=neutral, 1=very expressive)"""
        
        text_lower = text.lower()
        
        # Count emotional indicators
        high_emotion = 0
        for indicator in self.emotion_indicators['high']:
            if indicator == '!':
                high_emotion += text.count('!')
            else:
                high_emotion += text_lower.count(indicator)
        
        # Emotional punctuation
        exclamations = text.count('!')
        question_marks = text.count('?')
        ellipses = text.count('...')
        
        high_emotion += exclamations + question_marks + ellipses
        
        # Emotional words
        emotional_words = [
            'amazing', 'terrible', 'fantastic', 'awful', 'wonderful', 'horrible',
            'brilliant', 'stupid', 'incredible', 'ridiculous', 'perfect', 'disaster'
        ]
        
        emotion_word_count = sum(text_lower.count(word) for word in emotional_words)
        high_emotion += emotion_word_count
        
        # Normalize by text length
        sentences = len(self._split_sentences(text))
        if sentences == 0:
            return 0.0
        
        emotion_density = high_emotion / sentences
        return min(1.0, emotion_density / 2)  # Scale appropriately
    
    def _analyze_structure(self, text: str) -> float:
        """Analyze structural organization (0=loose, 1=highly structured)"""
        
        structure_score = 0
        
        # Numbered/bulleted lists
        if re.search(r'^\d+\.|\*|\-', text, re.MULTILINE):
            structure_score += 0.3
        
        # Clear paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 2:
            structure_score += 0.2
        
        # Headers or sections
        if re.search(r'^[A-Z][^a-z]*:$', text, re.MULTILINE):
            structure_score += 0.2
        
        # Consistent formatting
        if re.search(r'\n\n.*\n\n', text):
            structure_score += 0.2
        
        # Logical flow indicators
        flow_words = ['first', 'second', 'then', 'next', 'finally', 'however', 'therefore']
        flow_count = sum(text.lower().count(word) for word in flow_words)
        if flow_count > 0:
            structure_score += min(0.3, flow_count * 0.1)
        
        return min(1.0, structure_score)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_opening(self, text: str) -> Optional[str]:
        """Extract opening phrase from text"""
        sentences = self._split_sentences(text)
        if sentences:
            first_sentence = sentences[0]
            # Look for common opening patterns
            if len(first_sentence.split()) <= 5:
                return first_sentence
        return None
    
    def _extract_closing(self, text: str) -> Optional[str]:
        """Extract closing phrase from text"""
        sentences = self._split_sentences(text)
        if sentences:
            last_sentence = sentences[-1]
            # Look for common closing patterns
            closing_patterns = ['thanks', 'regards', 'best', 'sincerely', 'cheers', 'talk soon']
            if any(pattern in last_sentence.lower() for pattern in closing_patterns):
                return last_sentence
        return None
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract common phrases from text"""
        # Extract 2-4 word phrases
        words = text.lower().split()
        phrases = []
        
        for i in range(len(words) - 1):
            for length in [2, 3, 4]:
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    # Filter out common but meaningless phrases
                    if not any(stop in phrase for stop in ['the', 'and', 'or', 'but', 'in', 'on', 'at']):
                        phrases.append(phrase)
        
        return phrases
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
        # Look for programming-related terms, camelCase, snake_case, etc.
        technical_patterns = [
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # camelCase
            r'\b[a-z]+_[a-z]+\b',         # snake_case
            r'\b[A-Z]{2,}\b',             # ACRONYMS
            r'\b\w+\(\)\b',               # function calls
        ]
        
        terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)
        
        return terms
    
    @log_exceptions
    async def adapt_response(
        self,
        original_response: str,
        response_type: ResponseType,
        context: Optional[Dict[str, Any]] = None
    ) -> ResponseAdaptation:
        """Adapt a response to match user's communication style"""
        
        if not self.style_model or self.style_model.confidence < 0.3:
            self.logger.warning("Style model not available or low confidence, returning original response")
            return ResponseAdaptation(
                original_response=original_response,
                adapted_response=original_response,
                adaptation_type=response_type,
                confidence=0.0
            )
        
        self.logger.info(f"Adapting {response_type.value} response to user style")
        
        # Start with original response
        adapted_response = original_response
        adjustments = []
        
        # Apply style adaptations
        adapted_response, formality_adjustments = self._adapt_formality(
            adapted_response, self.style_model.formality_score
        )
        adjustments.extend(formality_adjustments)
        
        adapted_response, verbosity_adjustments = self._adapt_verbosity(
            adapted_response, self.style_model.verbosity_score
        )
        adjustments.extend(verbosity_adjustments)
        
        adapted_response, emotion_adjustments = self._adapt_emotion(
            adapted_response, self.style_model.emotion_score
        )
        adjustments.extend(emotion_adjustments)
        
        adapted_response, structure_adjustments = self._adapt_structure(
            adapted_response, self.style_model.structure_score, response_type
        )
        adjustments.extend(structure_adjustments)
        
        # Apply common phrases and expressions
        adapted_response, phrase_adjustments = self._apply_common_phrases(
            adapted_response, response_type
        )
        adjustments.extend(phrase_adjustments)
        
        # Calculate adaptation confidence
        confidence = self._calculate_adaptation_confidence(adjustments)
        
        adaptation = ResponseAdaptation(
            original_response=original_response,
            adapted_response=adapted_response,
            adaptation_type=response_type,
            style_adjustments=adjustments,
            confidence=confidence
        )
        
        # Store adaptation for learning
        self.adaptation_history.append(adaptation)
        
        return adaptation
    
    def _adapt_formality(self, text: str, target_formality: float) -> Tuple[str, List[str]]:
        """Adapt text formality level"""
        adjustments = []
        
        if target_formality > 0.7:  # Make more formal
            # Replace casual with formal
            replacements = {
                "hey": "hello",
                "hi": "hello",
                "thanks": "thank you",
                "no problem": "you're welcome",
                "sure thing": "certainly",
                "okay": "very well",
                "can't": "cannot",
                "won't": "will not",
                "don't": "do not"
            }
            
            for casual, formal in replacements.items():
                if casual in text.lower():
                    text = re.sub(r'\b' + casual + r'\b', formal, text, flags=re.IGNORECASE)
                    adjustments.append(f"Formalized: {casual} -> {formal}")
        
        elif target_formality < 0.3:  # Make more casual
            # Replace formal with casual
            replacements = {
                "hello": "hi",
                "thank you": "thanks",
                "you're welcome": "no problem",
                "certainly": "sure",
                "very well": "okay",
                "cannot": "can't",
                "will not": "won't",
                "do not": "don't"
            }
            
            for formal, casual in replacements.items():
                if formal in text.lower():
                    text = re.sub(r'\b' + formal + r'\b', casual, text, flags=re.IGNORECASE)
                    adjustments.append(f"Casualized: {formal} -> {casual}")
        
        return text, adjustments
    
    def _adapt_verbosity(self, text: str, target_verbosity: float) -> Tuple[str, List[str]]:
        """Adapt text verbosity level"""
        adjustments = []
        
        if target_verbosity < 0.3:  # Make more concise
            # Remove redundant words and phrases
            concise_replacements = {
                r'\bin order to\b': 'to',
                r'\bdue to the fact that\b': 'because',
                r'\bat this point in time\b': 'now',
                r'\bfor the purpose of\b': 'to',
                r'\bin the event that\b': 'if',
                r'\bvery\s+': '',  # Remove most "very"
                r'\breally\s+': ''  # Remove most "really"
            }
            
            for verbose, concise in concise_replacements.items():
                if re.search(verbose, text, re.IGNORECASE):
                    text = re.sub(verbose, concise, text, flags=re.IGNORECASE)
                    adjustments.append(f"Concisified: {verbose} -> {concise}")
        
        elif target_verbosity > 0.7:  # Make more verbose
            # Add descriptive words and phrases
            verbose_replacements = {
                r'\bto\b': 'in order to',
                r'\bbecause\b': 'due to the fact that',
                r'\bnow\b': 'at this point in time',
                r'\bif\b': 'in the event that'
            }
            
            for concise, verbose in verbose_replacements.items():
                if re.search(concise, text, re.IGNORECASE):
                    # Only replace some instances to avoid over-verbosity
                    text = re.sub(concise, verbose, text, count=1, flags=re.IGNORECASE)
                    adjustments.append(f"Enhanced verbosity: {concise} -> {verbose}")
        
        return text, adjustments
    
    def _adapt_emotion(self, text: str, target_emotion: float) -> Tuple[str, List[str]]:
        """Adapt emotional expressiveness"""
        adjustments = []
        
        if target_emotion > 0.6:  # Make more expressive
            # Add emotional words and punctuation
            if not text.endswith('!') and target_emotion > 0.8:
                # Sometimes add exclamation for high emotion users
                if any(positive in text.lower() for positive in ['great', 'good', 'excellent', 'perfect']):
                    text = text.rstrip('.') + '!'
                    adjustments.append("Added expressive punctuation")
            
            # Enhance positive words
            emotion_enhancements = {
                r'\bgood\b': 'great',
                r'\bnice\b': 'wonderful',
                r'\bokay\b': 'perfect',
                r'\bfine\b': 'excellent'
            }
            
            for neutral, emotional in emotion_enhancements.items():
                if re.search(neutral, text, re.IGNORECASE):
                    text = re.sub(neutral, emotional, text, count=1, flags=re.IGNORECASE)
                    adjustments.append(f"Enhanced emotion: {neutral} -> {emotional}")
        
        elif target_emotion < 0.3:  # Make more neutral
            # Tone down emotional language
            neutralizing_replacements = {
                r'\bamazing\b': 'good',
                r'\bfantastic\b': 'good',
                r'\bawesome\b': 'good',
                r'\bterrible\b': 'poor',
                r'\bawful\b': 'poor',
                r'!+': '.'  # Replace exclamations with periods
            }
            
            for emotional, neutral in neutralizing_replacements.items():
                if re.search(emotional, text, re.IGNORECASE):
                    text = re.sub(emotional, neutral, text, flags=re.IGNORECASE)
                    adjustments.append(f"Neutralized emotion: {emotional} -> {neutral}")
        
        return text, adjustments
    
    def _adapt_structure(
        self,
        text: str,
        target_structure: float,
        response_type: ResponseType
    ) -> Tuple[str, List[str]]:
        """Adapt structural organization"""
        adjustments = []
        
        if target_structure > 0.6 and response_type in [ResponseType.EMAIL, ResponseType.DOCUMENT]:
            # Add structure for emails and documents
            paragraphs = text.split('\n\n')
            
            if len(paragraphs) == 1 and len(text) > 200:
                # Break into paragraphs
                sentences = self._split_sentences(text)
                if len(sentences) > 3:
                    # Group sentences into paragraphs
                    new_paragraphs = []
                    current_paragraph = []
                    
                    for i, sentence in enumerate(sentences):
                        current_paragraph.append(sentence)
                        if len(current_paragraph) >= 2 or i == len(sentences) - 1:
                            new_paragraphs.append('. '.join(current_paragraph) + '.')
                            current_paragraph = []
                    
                    text = '\n\n'.join(new_paragraphs)
                    adjustments.append("Added paragraph structure")
        
        return text, adjustments
    
    def _apply_common_phrases(
        self,
        text: str,
        response_type: ResponseType
    ) -> Tuple[str, List[str]]:
        """Apply user's common phrases and expressions"""
        adjustments = []
        
        if not self.style_model:
            return text, adjustments
        
        # Apply openings for emails
        if response_type == ResponseType.EMAIL and self.style_model.common_openings:
            first_word = text.split()[0].lower() if text.split() else ""
            
            # Check if user typically uses specific openings
            user_openings = [o for o in self.style_model.common_openings if len(o.split()) <= 3]
            
            if user_openings and first_word in ['hello', 'hi', 'hey']:
                preferred_opening = user_openings[0]
                # Replace generic opening with user's preferred style
                words = text.split()
                words[0] = preferred_opening.capitalize()
                text = ' '.join(words)
                adjustments.append(f"Applied preferred opening: {preferred_opening}")
        
        # Apply closings for emails
        if response_type == ResponseType.EMAIL and self.style_model.common_closings:
            user_closings = [c for c in self.style_model.common_closings if len(c.split()) <= 4]
            
            if user_closings:
                # Check if text needs a closing
                last_sentence = text.split('.')[-1].strip()
                
                closing_indicators = ['thanks', 'regards', 'best', 'sincerely']
                has_closing = any(indicator in last_sentence.lower() for indicator in closing_indicators)
                
                if not has_closing:
                    preferred_closing = user_closings[0]
                    text += f"\n\n{preferred_closing.capitalize()}"
                    adjustments.append(f"Added preferred closing: {preferred_closing}")
        
        # Apply frequent phrases where appropriate
        if self.style_model.frequent_phrases:
            common_phrase_replacements = {
                "let me know": self.style_model.frequent_phrases[0] if "let me know" in self.style_model.frequent_phrases[0] else "let me know",
                "sounds good": next((p for p in self.style_model.frequent_phrases if "good" in p), "sounds good"),
                "no problem": next((p for p in self.style_model.frequent_phrases if "problem" in p), "no problem")
            }
            
            for generic, user_phrase in common_phrase_replacements.items():
                if generic in text.lower() and generic != user_phrase:
                    text = text.replace(generic, user_phrase)
                    adjustments.append(f"Applied user phrase: {generic} -> {user_phrase}")
        
        return text, adjustments
    
    def _calculate_adaptation_confidence(self, adjustments: List[str]) -> float:
        """Calculate confidence in the adaptation"""
        if not self.style_model:
            return 0.0
        
        # Base confidence on style model confidence
        base_confidence = self.style_model.confidence
        
        # Boost confidence based on number of adjustments made
        adjustment_boost = min(0.3, len(adjustments) * 0.1)
        
        return min(1.0, base_confidence + adjustment_boost)
    
    async def generate_styled_response(
        self,
        prompt: str,
        response_type: ResponseType,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response that matches user's style from scratch"""
        
        if not self.style_model or self.style_model.confidence < 0.5:
            # Use standard AI response if no style model
            try:
                response = await self.cloud_api.analyze_complex_request(
                    prompt,
                    context=f"generate_{response_type.value}"
                )
                return {
                    "response": response.get('content', prompt),
                    "styled": False,
                    "confidence": 0.0,
                    "adjustments": []
                }
            except Exception as e:
                return {
                    "response": "I'm sorry, I couldn't generate a response at this time.",
                    "styled": False,
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        # Build style-aware prompt
        style_prompt = self._build_style_aware_prompt(prompt, response_type)
        
        try:
            # Generate response with AI
            ai_response = await self.cloud_api.analyze_complex_request(
                style_prompt,
                context=f"styled_{response_type.value}"
            )
            
            base_response = ai_response.get('content', '')
            
            # Apply additional style adaptations
            adaptation = await self.adapt_response(base_response, response_type, context)
            
            return {
                "response": adaptation.adapted_response,
                "styled": True,
                "confidence": adaptation.confidence,
                "adjustments": adaptation.style_adjustments,
                "original_response": adaptation.original_response
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate styled response: {e}")
            return {
                "response": "I'm sorry, I couldn't generate a styled response at this time.",
                "styled": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _build_style_aware_prompt(self, original_prompt: str, response_type: ResponseType) -> str:
        """Build a prompt that incorporates user's style preferences"""
        
        if not self.style_model:
            return original_prompt
        
        style_description = self._get_style_description()
        
        style_prompt = f"""
Please respond to the following {response_type.value} request in a style that matches these characteristics:

USER STYLE PROFILE:
{style_description}

COMMON PHRASES TO USE:
{', '.join(self.style_model.frequent_phrases[:5]) if self.style_model.frequent_phrases else 'None specified'}

TYPICAL OPENINGS:
{', '.join(self.style_model.common_openings[:3]) if self.style_model.common_openings else 'None specified'}

TYPICAL CLOSINGS:
{', '.join(self.style_model.common_closings[:3]) if self.style_model.common_closings else 'None specified'}

REQUEST:
{original_prompt}

Please generate a response that naturally incorporates the user's communication style while being helpful and appropriate for the {response_type.value} format.
"""
        
        return style_prompt
    
    def _get_style_description(self) -> str:
        """Get a human-readable description of the user's style"""
        
        if not self.style_model:
            return "No style profile available"
        
        descriptions = []
        
        # Formality
        if self.style_model.formality_score > 0.7:
            descriptions.append("Formal and professional")
        elif self.style_model.formality_score < 0.3:
            descriptions.append("Casual and informal")
        else:
            descriptions.append("Moderately formal")
        
        # Verbosity
        if self.style_model.verbosity_score > 0.7:
            descriptions.append("Detailed and verbose")
        elif self.style_model.verbosity_score < 0.3:
            descriptions.append("Concise and to-the-point")
        else:
            descriptions.append("Moderately detailed")
        
        # Emotion
        if self.style_model.emotion_score > 0.6:
            descriptions.append("Expressive and enthusiastic")
        elif self.style_model.emotion_score < 0.3:
            descriptions.append("Neutral and controlled")
        else:
            descriptions.append("Moderately expressive")
        
        # Technical level
        if self.style_model.technical_score > 0.6:
            descriptions.append("Technical and specific")
        elif self.style_model.technical_score < 0.3:
            descriptions.append("Non-technical and accessible")
        else:
            descriptions.append("Moderately technical")
        
        # Structure
        if self.style_model.structure_score > 0.6:
            descriptions.append("Well-structured and organized")
        elif self.style_model.structure_score < 0.3:
            descriptions.append("Flowing and conversational")
        else:
            descriptions.append("Moderately structured")
        
        return "; ".join(descriptions)
    
    async def _save_style_model(self):
        """Save the current style model"""
        
        if self.style_model:
            try:
                await self.memory.store_content(
                    content_id="user_style_model",
                    content=json.dumps(self.style_model.to_dict()),
                    content_type="style_model",
                    metadata={
                        "confidence": self.style_model.confidence,
                        "sample_count": self.style_model.sample_count
                    }
                )
                self.logger.info("Saved style model to memory")
            except Exception as e:
                self.logger.error(f"Failed to save style model: {e}")
    
    async def _load_style_model(self):
        """Load existing style model"""
        
        try:
            results = await self.memory.search_content(
                "user_style_model",
                filters={"content_type": "style_model"}
            )
            
            if results:
                model_data = json.loads(results[0].content)
                
                # Convert response_templates back to enum keys
                if 'response_templates' in model_data:
                    response_templates = {}
                    for key, value in model_data['response_templates'].items():
                        try:
                            response_type = ResponseType(key)
                            response_templates[response_type] = value
                        except ValueError:
                            continue
                    model_data['response_templates'] = response_templates
                
                # Convert datetime strings back to datetime objects
                if 'last_updated' in model_data:
                    model_data['last_updated'] = datetime.fromisoformat(model_data['last_updated'])
                
                self.style_model = StyleModel(**model_data)
                self.logger.info(f"Loaded style model with confidence {self.style_model.confidence:.2f}")
                
        except Exception as e:
            self.logger.error(f"Failed to load style model: {e}")
            self.style_model = StyleModel()
    
    async def get_style_analytics(self) -> Dict[str, Any]:
        """Get analytics about the style model and adaptations"""
        
        analytics = {
            "style_model": {
                "available": self.style_model is not None,
                "confidence": self.style_model.confidence if self.style_model else 0.0,
                "sample_count": self.style_model.sample_count if self.style_model else 0,
                "last_updated": self.style_model.last_updated.isoformat() if self.style_model else None
            },
            "adaptations": {
                "total_adaptations": len(self.adaptation_history),
                "recent_adaptations": len([a for a in self.adaptation_history 
                                         if datetime.now() - datetime.fromisoformat(a.to_dict()['created_at'] if 'created_at' in a.to_dict() else datetime.now().isoformat()) < timedelta(days=7)]),
                "average_confidence": statistics.mean([a.confidence for a in self.adaptation_history]) if self.adaptation_history else 0.0
            },
            "style_profile": {}
        }
        
        if self.style_model:
            analytics["style_profile"] = {
                "formality": {
                    "score": self.style_model.formality_score,
                    "description": "Formal" if self.style_model.formality_score > 0.6 else "Casual" if self.style_model.formality_score < 0.4 else "Mixed"
                },
                "verbosity": {
                    "score": self.style_model.verbosity_score,
                    "description": "Verbose" if self.style_model.verbosity_score > 0.6 else "Concise" if self.style_model.verbosity_score < 0.4 else "Moderate"
                },
                "technical_level": {
                    "score": self.style_model.technical_score,
                    "description": "Technical" if self.style_model.technical_score > 0.6 else "Basic" if self.style_model.technical_score < 0.4 else "Intermediate"
                },
                "emotion": {
                    "score": self.style_model.emotion_score,
                    "description": "Expressive" if self.style_model.emotion_score > 0.6 else "Neutral" if self.style_model.emotion_score < 0.4 else "Moderate"
                },
                "avg_sentence_length": self.style_model.avg_sentence_length,
                "common_phrases_count": len(self.style_model.frequent_phrases)
            }
        
        return analytics
    
    async def update_style_from_feedback(
        self,
        adaptation_id: str,
        feedback: Dict[str, Any]
    ):
        """Update style model based on user feedback"""
        
        # Find the adaptation
        adaptation = next(
            (a for a in self.adaptation_history if hasattr(a, 'id') and a.id == adaptation_id),
            None
        )
        
        if not adaptation or not self.style_model:
            return
        
        # Adjust style model based on feedback
        if feedback.get('too_formal'):
            self.style_model.formality_score = max(0.0, self.style_model.formality_score - 0.1)
        elif feedback.get('too_casual'):
            self.style_model.formality_score = min(1.0, self.style_model.formality_score + 0.1)
        
        if feedback.get('too_verbose'):
            self.style_model.verbosity_score = max(0.0, self.style_model.verbosity_score - 0.1)
        elif feedback.get('too_concise'):
            self.style_model.verbosity_score = min(1.0, self.style_model.verbosity_score + 0.1)
        
        # Update model timestamp and save
        self.style_model.last_updated = datetime.now()
        await self._save_style_model()
        
        self.logger.info(f"Updated style model based on feedback for adaptation {adaptation_id}")
    
    def get_style_model(self) -> Optional[StyleModel]:
        """Get the current style model"""
        return self.style_model