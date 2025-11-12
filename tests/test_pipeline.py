"""
Basic tests for the mainpipe package.
"""
import pytest
from pathlib import Path
import tempfile
import shutil


class TestDataAcquisition:
    """Tests for data acquisition module."""
    
    def test_data_acquisition_init(self):
        """Test data acquisition initialization."""
        from mainpipe.acquisition.downloader import DataAcquisition
        
        with tempfile.TemporaryDirectory() as tmpdir:
            acq = DataAcquisition(data_dir=tmpdir)
            assert acq.data_dir == Path(tmpdir)
            assert acq.data_dir.exists()


class TestDataCleaner:
    """Tests for data cleaning module."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        from mainpipe.cleaning.cleaner import DataCleaner
        
        cleaner = DataCleaner(config={'min_length': 10})
        text = "This is a sample text for cleaning and normalization."
        cleaned, drop_reason = cleaner.clean_text(text)
        
        assert cleaned is not None
        assert drop_reason is None
        assert len(cleaned) > 0
    
    def test_clean_text_too_short(self):
        """Test text that is too short."""
        from mainpipe.cleaning.cleaner import DataCleaner
        
        cleaner = DataCleaner(config={'min_length': 100})
        text = "Short"
        cleaned, drop_reason = cleaner.clean_text(text)
        
        assert cleaned is None
        assert 'too_short' in drop_reason


class TestLanguageDetector:
    """Tests for language detection module."""
    
    def test_detect_language_english(self):
        """Test English language detection."""
        from mainpipe.utils.language_detector import LanguageDetector
        
        detector = LanguageDetector()
        text = "This is a text in English language."
        lang, confidence = detector.detect_language(text)
        
        assert lang == 'en'
        assert confidence > 0.9


class TestDuplicateDetector:
    """Tests for duplicate detection module."""
    
    def test_duplicate_detection(self):
        """Test duplicate detection."""
        from mainpipe.utils.duplicate_detector import DuplicateDetector
        
        detector = DuplicateDetector()
        text1 = "This is a sample text."
        text2 = "This is a sample text."
        
        # First occurrence should not be duplicate
        assert not detector.is_duplicate(text1)
        # Second occurrence should be duplicate
        assert detector.is_duplicate(text2)


class TestTokenizer:
    """Tests for tokenization module."""
    
    def test_tokenizer_init(self):
        """Test tokenizer initialization."""
        from mainpipe.tokenization.tokenizer import TextTokenizer
        
        tokenizer = TextTokenizer(tokenizer_type='bpe', vocab_size=1000)
        assert tokenizer.tokenizer_type == 'bpe'
        assert tokenizer.vocab_size == 1000


class TestExporter:
    """Tests for export module."""
    
    def test_exporter_init(self):
        """Test exporter initialization."""
        from mainpipe.export.exporter import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter(output_dir=tmpdir)
            assert exporter.output_dir == Path(tmpdir)
            assert exporter.output_dir.exists()


class TestInspector:
    """Tests for inspectability module."""
    
    def test_inspector_init(self):
        """Test inspector initialization."""
        from mainpipe.inspectability.inspector import DataInspector
        
        with tempfile.TemporaryDirectory() as tmpdir:
            inspector = DataInspector(output_dir=tmpdir)
            assert inspector.output_dir == Path(tmpdir)
            assert inspector.output_dir.exists()
    
    def test_analyze_lengths(self):
        """Test length analysis."""
        from mainpipe.inspectability.inspector import DataInspector
        
        with tempfile.TemporaryDirectory() as tmpdir:
            inspector = DataInspector(output_dir=tmpdir)
            lengths = [10, 20, 30, 40, 50]
            stats = inspector.analyze_lengths(lengths, 'test')
            
            assert stats['test_count'] == 5
            assert stats['test_min'] == 10
            assert stats['test_max'] == 50
            assert stats['test_mean'] == 30.0
