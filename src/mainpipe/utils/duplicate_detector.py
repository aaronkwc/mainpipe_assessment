"""
Duplicate detection utilities.
"""
from typing import List, Set, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Handles duplicate detection using various methods."""
    
    def __init__(self, method: str = "exact"):
        """
        Initialize duplicate detector.
        
        Args:
            method: Detection method ('exact', 'fuzzy', 'hash')
        """
        self.method = method
        self.seen_hashes: Set[str] = set()
        self.stats = {
            'total_checked': 0,
            'duplicates_found': 0,
            'unique_items': 0
        }
    
    def compute_hash(self, text: str) -> str:
        """
        Compute hash of text.
        
        Args:
            text: Input text
            
        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate.
        
        Args:
            text: Input text
            
        Returns:
            True if duplicate, False otherwise
        """
        self.stats['total_checked'] += 1
        
        text_hash = self.compute_hash(text)
        
        if text_hash in self.seen_hashes:
            self.stats['duplicates_found'] += 1
            return True
        else:
            self.seen_hashes.add(text_hash)
            self.stats['unique_items'] += 1
            return False
    
    def mark_duplicates(self, texts: List[str]) -> List[bool]:
        """
        Mark duplicates in a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of boolean markers (True = duplicate)
        """
        markers = []
        for text in texts:
            is_dup = self.is_duplicate(text)
            markers.append(is_dup)
        return markers
    
    def filter_duplicates(self, texts: List[str]) -> List[str]:
        """
        Filter out duplicates from list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of unique texts
        """
        unique_texts = []
        for text in texts:
            if not self.is_duplicate(text):
                unique_texts.append(text)
        return unique_texts
    
    def reset(self):
        """Reset the seen hashes and statistics."""
        self.seen_hashes.clear()
        self.stats = {
            'total_checked': 0,
            'duplicates_found': 0,
            'unique_items': 0
        }
    
    def get_stats(self) -> Dict:
        """
        Get duplicate detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
