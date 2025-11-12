"""
Main pipeline orchestrator for end-to-end data processing.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from tqdm import tqdm

from mainpipe.acquisition.downloader import DataAcquisition
from mainpipe.cleaning.cleaner import DataCleaner
from mainpipe.tokenization.tokenizer import TextTokenizer
from mainpipe.export.exporter import DataExporter
from mainpipe.inspectability.inspector import DataInspector
from mainpipe.utils.language_detector import LanguageDetector
from mainpipe.utils.pii_detector import PIIDetector
from mainpipe.utils.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)


class Pipeline:
    """Main data processing pipeline."""
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.acquisition = DataAcquisition(
            data_dir=config.get('data_dir', 'data/raw')
        )
        
        self.cleaner = DataCleaner(
            config=config.get('cleaning', {})
        )
        
        self.tokenizer = TextTokenizer(
            tokenizer_type=config.get('tokenizer_type', 'bpe'),
            vocab_size=config.get('vocab_size', 30000)
        )
        
        self.exporter = DataExporter(
            output_dir=config.get('output_dir', 'data/processed')
        )
        
        self.inspector = DataInspector(
            output_dir=config.get('reports_dir', 'data/reports')
        )
        
        self.lang_detector = LanguageDetector(
            confidence_threshold=config.get('lang_confidence', 0.7)
        )
        
        self.pii_detector = PIIDetector(
            language=config.get('language', 'en'),
            score_threshold=config.get('pii_threshold', 0.5)
        )
        
        self.dup_detector = DuplicateDetector(
            method=config.get('dup_method', 'exact')
        )
        
        self.processed_data = []
        self.pipeline_stats = {}
    
    def acquire_data(self) -> List[Path]:
        """
        Acquire data from configured sources.
        
        Returns:
            List of data file paths
        """
        logger.info("Starting data acquisition...")
        
        urls = self.config.get('data_urls', [])
        local_files = self.config.get('local_files', [])
        
        paths = []
        
        # Download from URLs
        if urls:
            paths.extend(self.acquisition.download_urls(urls))
        
        # Load local files
        for filepath in local_files:
            try:
                path = self.acquisition.load_local_file(filepath)
                paths.append(path)
            except FileNotFoundError as e:
                logger.error(f"Local file not found: {e}")
        
        logger.info(f"Acquired {len(paths)} data files")
        return paths
    
    def load_texts_from_files(self, filepaths: List[Path]) -> List[str]:
        """
        Load texts from files.
        
        Args:
            filepaths: List of file paths
            
        Returns:
            List of text strings
        """
        texts = []
        
        for filepath in filepaths:
            logger.info(f"Loading texts from {filepath}")
            
            try:
                if filepath.suffix == '.jsonl':
                    with open(filepath, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            if isinstance(data, dict) and 'text' in data:
                                texts.append(data['text'])
                            elif isinstance(data, str):
                                texts.append(data)
                
                elif filepath.suffix == '.json':
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    texts.append(item['text'])
                                elif isinstance(item, str):
                                    texts.append(item)
                        elif isinstance(data, dict) and 'text' in data:
                            texts.append(data['text'])
                
                elif filepath.suffix in ['.txt', '.text']:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                
                else:
                    logger.warning(f"Unsupported file format: {filepath.suffix}")
            
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f"Loaded {len(texts)} texts")
        return texts
    
    def process_data(self, texts: List[str]) -> List[Dict]:
        """
        Process texts through cleaning, normalization, and analysis.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed data dictionaries
        """
        logger.info(f"Processing {len(texts)} texts...")
        
        processed = []
        
        for text in tqdm(texts, desc="Processing texts"):
            # Clean text
            cleaned_text, drop_reason = self.cleaner.clean_text(text)
            
            if cleaned_text is None:
                processed.append({
                    'text': None,
                    'dropped': True,
                    'drop_reason': drop_reason
                })
                continue
            
            # Language detection
            lang, lang_score = self.lang_detector.detect_language(cleaned_text)
            
            # Check language filter
            allowed_langs = self.config.get('allowed_languages', None)
            if allowed_langs and lang not in allowed_langs:
                processed.append({
                    'text': None,
                    'dropped': True,
                    'drop_reason': f'language_filtered_{lang}',
                    'language': lang,
                    'language_score': lang_score
                })
                continue
            
            # Duplicate detection
            is_duplicate = self.dup_detector.is_duplicate(cleaned_text)
            
            if is_duplicate and self.config.get('remove_duplicates', True):
                processed.append({
                    'text': None,
                    'dropped': True,
                    'drop_reason': 'duplicate',
                    'language': lang,
                    'language_score': lang_score
                })
                continue
            
            # PII detection
            pii_entities = self.pii_detector.detect_pii(cleaned_text)
            has_pii = len(pii_entities) > 0
            
            if has_pii and self.config.get('remove_pii', False):
                processed.append({
                    'text': None,
                    'dropped': True,
                    'drop_reason': 'pii_found',
                    'language': lang,
                    'language_score': lang_score,
                    'pii_count': len(pii_entities)
                })
                continue
            
            # Add to processed data
            processed.append({
                'text': cleaned_text,
                'dropped': False,
                'language': lang,
                'language_score': lang_score,
                'is_duplicate': is_duplicate,
                'has_pii': has_pii,
                'pii_count': len(pii_entities),
                'length': len(cleaned_text)
            })
        
        logger.info(f"Processed {len(processed)} items")
        return processed
    
    def tokenize_data(self, processed_data: List[Dict]) -> List[Dict]:
        """
        Tokenize processed data.
        
        Args:
            processed_data: List of processed data dictionaries
            
        Returns:
            List of tokenized data dictionaries
        """
        logger.info("Tokenizing data...")
        
        # Filter out dropped items
        valid_data = [item for item in processed_data if not item['dropped']]
        
        if not valid_data:
            logger.warning("No valid data to tokenize")
            return []
        
        # Train or load tokenizer
        if self.config.get('train_tokenizer', False):
            texts = [item['text'] for item in valid_data]
            self.tokenizer.train_tokenizer(
                texts,
                output_path=Path(self.config.get('output_dir', 'data/processed')) / 'tokenizer.json'
            )
        elif self.config.get('pretrained_tokenizer'):
            self.tokenizer.load_pretrained(self.config['pretrained_tokenizer'])
        elif self.config.get('tokenizer_path'):
            self.tokenizer.load_from_file(Path(self.config['tokenizer_path']))
        
        # Tokenize each valid item
        tokenized_data = []
        for item in tqdm(valid_data, desc="Tokenizing"):
            try:
                tokenization = self.tokenizer.tokenize(item['text'])
                item['tokens'] = tokenization['tokens']
                item['token_ids'] = tokenization['token_ids']
                item['token_length'] = tokenization['length']
                tokenized_data.append(item)
            except Exception as e:
                logger.error(f"Tokenization failed: {e}")
        
        logger.info(f"Tokenized {len(tokenized_data)} items")
        return tokenized_data
    
    def generate_inspections(self, processed_data: List[Dict]):
        """
        Generate inspection reports and visualizations.
        
        Args:
            processed_data: List of processed data dictionaries
        """
        logger.info("Generating inspection reports...")
        
        # Collect statistics
        lengths_before = self.cleaner.get_stats().get('length_before', [])
        lengths_after = self.cleaner.get_stats().get('length_after', [])
        drop_reasons = self.cleaner.get_stats().get('drop_reasons', {})
        
        # Length analysis
        if lengths_after:
            self.inspector.analyze_lengths(lengths_after, "cleaned_text")
            self.inspector.plot_length_histogram(
                lengths_after,
                title="Text Length Distribution (After Cleaning)",
                filename="text_lengths.png"
            )
        
        # Token length analysis
        token_lengths = [item.get('token_length', 0) for item in processed_data if 'token_length' in item]
        if token_lengths:
            self.inspector.analyze_lengths(token_lengths, "token")
            self.inspector.plot_length_histogram(
                token_lengths,
                title="Token Length Distribution",
                filename="token_lengths.png"
            )
        
        # Drop reasons
        if drop_reasons:
            self.inspector.plot_drop_reasons(drop_reasons)
        
        # Language scores
        lang_scores = {}
        for item in processed_data:
            if 'language' in item and 'language_score' in item:
                lang = item['language']
                score = item['language_score']
                if lang and score:
                    if lang not in lang_scores:
                        lang_scores[lang] = []
                    lang_scores[lang].append(score)
        
        if lang_scores:
            self.inspector.analyze_language_scores(lang_scores)
        
        # Duplicate analysis
        dup_markers = [item.get('is_duplicate', False) for item in processed_data]
        if dup_markers:
            self.inspector.analyze_duplicates(dup_markers)
        
        # PII analysis
        pii_stats = self.pii_detector.get_stats()
        pii_hits = pii_stats.get('pii_types', {})
        if pii_hits:
            self.inspector.analyze_pii_hits(pii_hits)
        
        # Combine all stats
        self.pipeline_stats.update(self.cleaner.get_stats())
        self.pipeline_stats.update(self.lang_detector.get_stats())
        self.pipeline_stats.update(self.dup_detector.get_stats())
        self.pipeline_stats.update(self.pii_detector.get_stats())
        self.pipeline_stats.update(self.tokenizer.get_stats())
        self.pipeline_stats.update(self.inspector.get_stats())
        
        # Generate report
        self.inspector.stats = self.pipeline_stats
        self.inspector.generate_report()
        
        logger.info("Inspection reports generated")
    
    def export_data(self, tokenized_data: List[Dict]):
        """
        Export tokenized data in training-ready formats.
        
        Args:
            tokenized_data: List of tokenized data dictionaries
        """
        logger.info("Exporting data...")
        
        # Export as JSONL
        self.exporter.export_jsonl(tokenized_data, "tokenized_data.jsonl")
        
        # Create shards if configured
        shard_size = self.config.get('shard_size', 10000)
        if len(tokenized_data) > shard_size:
            self.exporter.create_shards(tokenized_data, shard_size=shard_size)
        
        # Create mixture if multiple datasets
        # (This is a placeholder - would need multiple dataset sources)
        
        logger.info("Data export complete")
    
    def run(self):
        """
        Run the complete pipeline end-to-end.
        """
        logger.info("Starting pipeline execution...")
        
        try:
            # Step 1: Acquire data
            filepaths = self.acquire_data()
            
            if not filepaths:
                logger.error("No data files to process")
                return
            
            # Step 2: Load texts
            texts = self.load_texts_from_files(filepaths)
            
            if not texts:
                logger.error("No texts loaded from files")
                return
            
            # Step 3: Process data
            processed_data = self.process_data(texts)
            
            # Step 4: Tokenize data
            tokenized_data = self.tokenize_data(processed_data)
            
            # Step 5: Generate inspections
            self.generate_inspections(processed_data)
            
            # Step 6: Export data
            if tokenized_data:
                self.export_data(tokenized_data)
            
            logger.info("Pipeline execution complete!")
            logger.info(f"Pipeline statistics: {self.pipeline_stats}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'Pipeline':
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Pipeline instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config)
