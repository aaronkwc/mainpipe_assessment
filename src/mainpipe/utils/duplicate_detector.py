"""
Duplicate detection utilities.
Implements methods from "Deduplicating Training Data Makes Language Models Better" (Lee et al., 2022)
"""
from typing import List, Set, Dict, Tuple
import hashlib
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Handles duplicate detection using various methods including MinHash LSH."""
    
    def __init__(self, method: str = "exact", num_perm: int = 128, threshold: float = 0.8, ngram_size: int = 5):
        """
        Initialize duplicate detector.
        
        Args:
            method: Detection method ('exact', 'minhash', 'hash')
            num_perm: Number of permutations for MinHash (default: 128)
            threshold: Jaccard similarity threshold for MinHash (default: 0.8)
            ngram_size: N-gram size for MinHash (default: 5 - character-level 5-grams)
        """
        self.method = method
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.seen_hashes: Set[str] = set()
        
        # MinHash LSH structures
        if method == "minhash":
            self.lsh_bands = defaultdict(set)  # band_id -> set of signature hashes
            self.doc_signatures = {}  # doc_hash -> MinHash signature
            self.num_bands = 16  # Number of LSH bands
            self.rows_per_band = num_perm // self.num_bands
        
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
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """
        Extract character-level n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            Set of n-grams
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower())
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def _compute_minhash_signature(self, ngrams: Set[str]) -> List[int]:
        """
        Compute MinHash signature using permutation hashing.
        
        Args:
            ngrams: Set of n-grams
            
        Returns:
            List of minimum hash values (signature)
        """
        if not ngrams:
            return [0] * self.num_perm
        
        signature = []
        for i in range(self.num_perm):
            min_hash = float('inf')
            for ngram in ngrams:
                # Create a hash with permutation i
                h = hashlib.sha1(f"{ngram}_{i}".encode('utf-8'))
                hash_val = int(h.hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash)
        return signature
    
    def _lsh_hash_signature(self, signature: List[int]) -> List[int]:
        """
        Hash signature into LSH bands.
        
        Args:
            signature: MinHash signature
            
        Returns:
            List of band hashes
        """
        band_hashes = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            # Hash the band
            band_hash = hash(band)
            band_hashes.append(band_hash)
        return band_hashes
    
    def _is_duplicate_minhash(self, text: str) -> bool:
        """
        Check if text is a near-duplicate using MinHash LSH.
        
        Args:
            text: Input text
            
        Returns:
            True if near-duplicate found, False otherwise
        """
        # Extract n-grams
        ngrams = self._get_ngrams(text, self.ngram_size)
        if not ngrams:
            return False
        
        # Compute MinHash signature
        signature = self._compute_minhash_signature(ngrams)
        
        # Hash into LSH bands
        band_hashes = self._lsh_hash_signature(signature)
        
        # Check if any band matches existing documents
        candidate_docs = set()
        for band_idx, band_hash in enumerate(band_hashes):
            if band_hash in self.lsh_bands[band_idx]:
                # Found potential duplicate, need to verify
                for doc_hash in self.lsh_bands[band_idx]:
                    if doc_hash != band_hash:  # Skip self
                        candidate_docs.add(doc_hash)
        
        # Verify candidates with actual Jaccard similarity
        doc_hash = self.compute_hash(text)
        if candidate_docs:
            for candidate_hash in candidate_docs:
                if candidate_hash in self.doc_signatures:
                    candidate_sig = self.doc_signatures[candidate_hash]
                    # Estimate Jaccard similarity from MinHash signatures
                    matches = sum(1 for i in range(self.num_perm) if signature[i] == candidate_sig[i])
                    similarity = matches / self.num_perm
                    
                    if similarity >= self.threshold:
                        return True
        
        # Not a duplicate, add to LSH bands
        self.doc_signatures[doc_hash] = signature
        for band_idx, band_hash in enumerate(band_hashes):
            self.lsh_bands[band_idx].add(band_hash)
        
        return False
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate.
        
        Args:
            text: Input text
            
        Returns:
            True if duplicate, False otherwise
        """
        self.stats['total_checked'] += 1
        
        if self.method == "minhash":
            is_dup = self._is_duplicate_minhash(text)
        else:
            # Exact hash-based deduplication
            text_hash = self.compute_hash(text)
            is_dup = text_hash in self.seen_hashes
            if not is_dup:
                self.seen_hashes.add(text_hash)
        
        if is_dup:
            self.stats['duplicates_found'] += 1
        else:
            self.stats['unique_items'] += 1
        
        return is_dup
    
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
