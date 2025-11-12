"""
PII detection utilities using Presidio.
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available, PII detection will be limited")


class PIIDetector:
    """Handles PII (Personally Identifiable Information) detection."""
    
    def __init__(self, language: str = "en", score_threshold: float = 0.5):
        """
        Initialize PII detector.
        
        Args:
            language: Language code for detection
            score_threshold: Minimum confidence score for PII detection
        """
        self.language = language
        self.score_threshold = score_threshold
        self.stats = {
            'total_checked': 0,
            'total_pii_found': 0,
            'pii_types': {}
        }
        
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None
    
    def detect_pii(self, text: str) -> List[Dict]:
        """
        Detect PII in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected PII entities
        """
        self.stats['total_checked'] += 1
        
        if not PRESIDIO_AVAILABLE or self.analyzer is None:
            # Fallback: simple pattern matching
            return self._simple_pii_detection(text)
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                score_threshold=self.score_threshold
            )
            
            pii_entities = []
            for result in results:
                entity = {
                    'type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'score': result.score,
                    'text': text[result.start:result.end]
                }
                pii_entities.append(entity)
                
                # Update stats
                self.stats['total_pii_found'] += 1
                pii_type = result.entity_type
                self.stats['pii_types'][pii_type] = self.stats['pii_types'].get(pii_type, 0) + 1
            
            return pii_entities
            
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            return []
    
    def _simple_pii_detection(self, text: str) -> List[Dict]:
        """
        Simple pattern-based PII detection fallback.
        
        Args:
            text: Input text
            
        Returns:
            List of detected PII entities
        """
        import re
        
        pii_entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            pii_entities.append({
                'type': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'score': 0.9,
                'text': match.group()
            })
            self.stats['pii_types']['EMAIL'] = self.stats['pii_types'].get('EMAIL', 0) + 1
        
        # Phone pattern (simple US format)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            pii_entities.append({
                'type': 'PHONE',
                'start': match.start(),
                'end': match.end(),
                'score': 0.8,
                'text': match.group()
            })
            self.stats['pii_types']['PHONE'] = self.stats['pii_types'].get('PHONE', 0) + 1
        
        # SSN pattern (simple)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            pii_entities.append({
                'type': 'SSN',
                'start': match.start(),
                'end': match.end(),
                'score': 0.9,
                'text': match.group()
            })
            self.stats['pii_types']['SSN'] = self.stats['pii_types'].get('SSN', 0) + 1
        
        if pii_entities:
            self.stats['total_pii_found'] += len(pii_entities)
        
        return pii_entities
    
    def anonymize_pii(self, text: str) -> str:
        """
        Anonymize PII in text.
        
        Args:
            text: Input text
            
        Returns:
            Anonymized text
        """
        if not PRESIDIO_AVAILABLE or self.analyzer is None or self.anonymizer is None:
            # Simple replacement
            pii_entities = self._simple_pii_detection(text)
            result_text = text
            # Replace from end to start to preserve indices
            for entity in sorted(pii_entities, key=lambda x: x['start'], reverse=True):
                result_text = (
                    result_text[:entity['start']] +
                    f"<{entity['type']}>" +
                    result_text[entity['end']:]
                )
            return result_text
        
        try:
            # Analyze
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                score_threshold=self.score_threshold
            )
            
            # Anonymize
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text
            
        except Exception as e:
            logger.error(f"PII anonymization failed: {e}")
            return text
    
    def has_pii(self, text: str) -> bool:
        """
        Check if text contains PII.
        
        Args:
            text: Input text
            
        Returns:
            True if PII found, False otherwise
        """
        pii_entities = self.detect_pii(text)
        return len(pii_entities) > 0
    
    def get_stats(self) -> Dict:
        """
        Get PII detection statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
