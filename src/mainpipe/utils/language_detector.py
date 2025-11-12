"""
Language detection utilities.
"""
from typing import Dict, Optional, Tuple
import logging
from langdetect import detect, detect_langs, LangDetectException

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Handles language detection for text data."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize language detector.
        
        Args:
            confidence_threshold: Minimum confidence score to accept detection
        """
        self.confidence_threshold = confidence_threshold
        self.stats = {
            'total_detected': 0,
            'detection_failures': 0,
            'languages': {}
        }
    
    def detect_language(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language code, confidence score)
        """
        if not text or len(text.strip()) < 3:
            self.stats['detection_failures'] += 1
            return None, None
        
        try:
            langs = detect_langs(text)
            if langs:
                top_lang = langs[0]
                lang_code = top_lang.lang
                confidence = top_lang.prob
                
                self.stats['total_detected'] += 1
                self.stats['languages'][lang_code] = self.stats['languages'].get(lang_code, 0) + 1
                
                if confidence >= self.confidence_threshold:
                    return lang_code, confidence
                else:
                    return lang_code, confidence
            else:
                self.stats['detection_failures'] += 1
                return None, None
                
        except LangDetectException as e:
            logger.debug(f"Language detection failed: {e}")
            self.stats['detection_failures'] += 1
            return None, None
    
    def detect_language_simple(self, text: str) -> Optional[str]:
        """
        Detect language of text (simple version, returns only language code).
        
        Args:
            text: Input text
            
        Returns:
            Language code or None
        """
        try:
            return detect(text)
        except LangDetectException:
            return None
    
    def filter_by_language(
        self,
        text: str,
        allowed_languages: list
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if text is in allowed languages.
        
        Args:
            text: Input text
            allowed_languages: List of allowed language codes
            
        Returns:
            Tuple of (is_allowed, language, confidence)
        """
        lang, confidence = self.detect_language(text)
        
        if lang is None:
            return False, None, None
        
        is_allowed = lang in allowed_languages
        return is_allowed, lang, confidence
    
    def get_stats(self) -> Dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
