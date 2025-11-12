"""
Data cleaning and normalization module.
"""
import re
import regex
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and normalization."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data cleaner.
        
        Args:
            config: Configuration dictionary for cleaning parameters
        """
        self.config = config or {}
        self.stats = {
            'total_processed': 0,
            'total_dropped': 0,
            'drop_reasons': {},
            'length_before': [],
            'length_after': []
        }
    
    def clean_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (cleaned text, drop reason if dropped else None)
        """
        self.stats['total_processed'] += 1
        original_length = len(text)
        self.stats['length_before'].append(original_length)
        
        # Check minimum length
        min_length = self.config.get('min_length', 10)
        if len(text) < min_length:
            self.stats['total_dropped'] += 1
            reason = f'too_short_{min_length}'
            self.stats['drop_reasons'][reason] = self.stats['drop_reasons'].get(reason, 0) + 1
            return None, reason
        
        # Check maximum length
        max_length = self.config.get('max_length', 1000000)
        if len(text) > max_length:
            self.stats['total_dropped'] += 1
            reason = f'too_long_{max_length}'
            self.stats['drop_reasons'][reason] = self.stats['drop_reasons'].get(reason, 0) + 1
            return None, reason
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode
        text = text.strip()
        
        # Remove control characters except newlines and tabs
        text = regex.sub(r'[\p{Cc}&&[^\n\t]]', '', text)
        
        # Check if text is empty after cleaning
        if not text or len(text) < min_length:
            self.stats['total_dropped'] += 1
            reason = 'empty_after_cleaning'
            self.stats['drop_reasons'][reason] = self.stats['drop_reasons'].get(reason, 0) + 1
            return None, reason
        
        self.stats['length_after'].append(len(text))
        return text, None
    
    def clean_batch(self, texts: List[str]) -> List[Dict]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of dictionaries with 'text', 'dropped', and 'drop_reason' keys
        """
        results = []
        for text in texts:
            cleaned, drop_reason = self.clean_text(text)
            results.append({
                'text': cleaned,
                'dropped': cleaned is None,
                'drop_reason': drop_reason
            })
        return results
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with emails removed
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def get_stats(self) -> Dict:
        """
        Get cleaning statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total_processed': 0,
            'total_dropped': 0,
            'drop_reasons': {},
            'length_before': [],
            'length_after': []
        }
