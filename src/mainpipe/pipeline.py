"""
Main pipeline orchestrator for end-to-end data processing.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from tqdm import tqdm
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from mainpipe.acquisition.downloader import DataAcquisition
from mainpipe.cleaning.cleaner import DataCleaner
from mainpipe.tokenization.tokenizer import TextTokenizer
from mainpipe.export.exporter import DataExporter
from mainpipe.inspectability.inspector import DataInspector
from mainpipe.utils.language_detector import LanguageDetector
from mainpipe.utils.pii_detector import PIIDetector
from mainpipe.utils.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)


# Globals used when using multiprocessing initializer
_GLOBAL_CLEANER = None
_GLOBAL_LANG = None
_GLOBAL_PII = None
_GLOBAL_DUP = None
_GLOBAL_CONFIG = None


def _worker_init(config: Dict):
    """Initializer for worker processes: instantiate heavy objects once per process."""
    global _GLOBAL_CLEANER, _GLOBAL_LANG, _GLOBAL_PII, _GLOBAL_DUP, _GLOBAL_CONFIG
    from mainpipe.cleaning.cleaner import DataCleaner
    from mainpipe.utils.language_detector import LanguageDetector
    from mainpipe.utils.pii_detector import PIIDetector
    from mainpipe.utils.duplicate_detector import DuplicateDetector

    _GLOBAL_CONFIG = config
    _GLOBAL_CLEANER = DataCleaner(config=config.get('cleaning', {}))
    _GLOBAL_LANG = LanguageDetector(confidence_threshold=config.get('lang_confidence', 0.7))
    _GLOBAL_PII = PIIDetector(
        language=config.get('language', 'en'),
        score_threshold=config.get('pii_threshold', 0.5),
        spacy_model=config.get('spacy_model')
    )
    _GLOBAL_DUP = DuplicateDetector(
        method=config.get('dup_method', 'exact'),
        num_perm=config.get('minhash_num_perm', 128),
        threshold=config.get('minhash_threshold', 0.8),
        ngram_size=config.get('minhash_ngram_size', 5)
    )


def _process_text_worker(text: str) -> Dict:
    """Process a single text using global worker instances. Returns processed dict."""
    global _GLOBAL_CLEANER, _GLOBAL_LANG, _GLOBAL_PII, _GLOBAL_DUP, _GLOBAL_CONFIG

    # Fallback to raising if not initialized
    if _GLOBAL_CLEANER is None:
        raise RuntimeError("Worker not initialized: call _worker_init in process pool initializer")

    cleaned_text, drop_reason = _GLOBAL_CLEANER.clean_text(text)

    if cleaned_text is None:
        return {'text': None, 'dropped': True, 'drop_reason': drop_reason}

    lang, lang_score = _GLOBAL_LANG.detect_language(cleaned_text)

    allowed_langs = _GLOBAL_CONFIG.get('allowed_languages', None)
    if allowed_langs and lang not in allowed_langs:
        return {
            'text': None,
            'dropped': True,
            'drop_reason': f'language_filtered_{lang}',
            'language': lang,
            'language_score': lang_score
        }

    is_duplicate = _GLOBAL_DUP.is_duplicate(cleaned_text)
    if is_duplicate and _GLOBAL_CONFIG.get('remove_duplicates', True):
        return {
            'text': None,
            'dropped': True,
            'drop_reason': 'duplicate',
            'is_duplicate': True,
            'language': lang,
            'language_score': lang_score
        }

    # Only run PII detection if we're redacting PII or need it for inspection
    # This is the slowest step, so skip if not needed
    pii_entities = []
    has_pii = False
    pii_redacted = False
    if _GLOBAL_CONFIG.get('redact_pii', False) or _GLOBAL_CONFIG.get('inspect_pii', True):
        pii_entities = _GLOBAL_PII.detect_pii(cleaned_text)
        has_pii = len(pii_entities) > 0
        if has_pii and _GLOBAL_CONFIG.get('redact_pii', False):
            # Redact PII by replacing entities with <TYPE> placeholders
            cleaned_text = _GLOBAL_PII.redact_pii(cleaned_text, pii_entities)
            pii_redacted = True

    return {
        'text': cleaned_text,
        'dropped': False,
        'language': lang,
        'language_score': lang_score,
        'is_duplicate': is_duplicate,
        'has_pii': has_pii,
        'pii_redacted': pii_redacted,
        'pii_count': len(pii_entities),
        'pii_entities': pii_entities,
        'length': len(cleaned_text)
    }


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
        
        # Parallelization options
        # workers: int (number of workers to use)
        # parallel_mode: 'process' | 'thread'
        # sample_size: optional int to limit number of texts for quick tests
        self.workers = int(config.get('workers', max(1, (os.cpu_count() or 1) - 1)))
        # If a spaCy model is configured (often used by Presidio), prefer threaded
        # execution by default to avoid re-loading heavy models per process.
        if config.get('parallel_mode') is not None:
            self.parallel_mode = config.get('parallel_mode')
        else:
            self.parallel_mode = 'thread' if config.get('spacy_model') else 'process'
        self.sample_size = config.get('sample_size', None)

        self.pii_detector = PIIDetector(
            language=config.get('language', 'en'),
            score_threshold=config.get('pii_threshold', 0.5),
            spacy_model=config.get('spacy_model')
        )
        
        self.dup_detector = DuplicateDetector(
            method=config.get('dup_method', 'exact'),
            num_perm=config.get('minhash_num_perm', 128),
            threshold=config.get('minhash_threshold', 0.8),
            ngram_size=config.get('minhash_ngram_size', 5)
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
        # If sample_size configured, trim for quick tests
        if self.sample_size:
            orig = len(texts)
            texts = texts[: int(self.sample_size) ]
            logger.info(f"Trimmed texts to sample_size={self.sample_size} (from {orig} to {len(texts)})")

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

        # If only single worker requested, fall back to sequential processing
        if not self.workers or int(self.workers) <= 1:
            processed = []
            for text in tqdm(texts, desc="Processing texts"):
                cleaned_text, drop_reason = self.cleaner.clean_text(text)
                if cleaned_text is None:
                    processed.append({'text': None, 'dropped': True, 'drop_reason': drop_reason})
                    continue

                lang, lang_score = self.lang_detector.detect_language(cleaned_text)
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

                is_duplicate = self.dup_detector.is_duplicate(cleaned_text)
                if is_duplicate and self.config.get('remove_duplicates', True):
                    processed.append({
                        'text': None,
                        'dropped': True,
                        'drop_reason': 'duplicate',
                        'is_duplicate': True,
                        'language': lang,
                        'language_score': lang_score
                    })
                    continue

                pii_entities = self.pii_detector.detect_pii(cleaned_text)
                has_pii = len(pii_entities) > 0
                pii_redacted = False
                if has_pii and self.config.get('redact_pii', False):
                    # Redact PII by replacing entities with <TYPE> placeholders
                    cleaned_text = self.pii_detector.redact_pii(cleaned_text, pii_entities)
                    pii_redacted = True

                processed.append({
                    'text': cleaned_text,
                    'dropped': False,
                    'language': lang,
                    'language_score': lang_score,
                    'is_duplicate': is_duplicate,
                    'has_pii': has_pii,
                    'pii_redacted': pii_redacted,
                    'pii_count': len(pii_entities),
                    'pii_entities': pii_entities,
                    'length': len(cleaned_text)
                })

            logger.info(f"Processed {len(processed)} items")
            return processed

        # Parallel processing
        processed = []
        mode = (self.parallel_mode or 'process').lower()

        if mode == 'process':
            logger.info(f"Processing using multiprocessing with {self.workers} workers")
            with multiprocessing.Pool(processes=self.workers, initializer=_worker_init, initargs=(self.config,)) as pool:
                # imap preserves order and is memory-efficient
                for result in tqdm(pool.imap(_process_text_worker, texts), total=len(texts), desc="Processing texts"):
                    processed.append(result)

        else:
            # Threaded execution (shares interpreter state, avoids reloading models per process)
            logger.info(f"Processing using threads with {self.workers} workers")
            def _thread_worker(text: str) -> Dict:
                cleaned_text, drop_reason = self.cleaner.clean_text(text)
                if cleaned_text is None:
                    return {'text': None, 'dropped': True, 'drop_reason': drop_reason}

                lang, lang_score = self.lang_detector.detect_language(cleaned_text)
                allowed_langs = self.config.get('allowed_languages', None)
                if allowed_langs and lang not in allowed_langs:
                    return {
                        'text': None,
                        'dropped': True,
                        'drop_reason': f'language_filtered_{lang}',
                        'language': lang,
                        'language_score': lang_score
                    }

                is_duplicate = self.dup_detector.is_duplicate(cleaned_text)
                if is_duplicate and self.config.get('remove_duplicates', True):
                    return {
                        'text': None,
                        'dropped': True,
                        'drop_reason': 'duplicate',
                        'is_duplicate': True,
                        'language': lang,
                        'language_score': lang_score
                    }

                # Only run PII detection if we're redacting PII or need it for inspection
                # This is the slowest step, so skip if not needed
                pii_entities = []
                has_pii = False
                pii_redacted = False
                if self.config.get('redact_pii', False) or self.config.get('inspect_pii', True):
                    pii_entities = self.pii_detector.detect_pii(cleaned_text)
                    has_pii = len(pii_entities) > 0
                    if has_pii and self.config.get('redact_pii', False):
                        # Redact PII by replacing entities with <TYPE> placeholders
                        cleaned_text = self.pii_detector.redact_pii(cleaned_text, pii_entities)
                        pii_redacted = True

                return {
                    'text': cleaned_text,
                    'dropped': False,
                    'language': lang,
                    'language_score': lang_score,
                    'is_duplicate': is_duplicate,
                    'has_pii': has_pii,
                    'pii_redacted': pii_redacted,
                    'pii_count': len(pii_entities),
                    'pii_entities': pii_entities,
                    'length': len(cleaned_text)
                }

            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = [ex.submit(_thread_worker, t) for t in texts]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing texts"):
                    try:
                        processed.append(fut.result())
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

        logger.info(f"Processed {len(processed)} items")
        
        # Post-process: Run PII detection in batch if needed (much faster than per-text)
        # Only run if inspect_pii is enabled and we haven't already detected during processing
        if self.config.get('inspect_pii', True) and not self.config.get('redact_pii', False):
            logger.info("Running batch PII detection for inspection...")
            valid_items = [item for item in processed if not item.get('dropped', False)]
            if valid_items:
                texts_for_pii = [item['text'] for item in valid_items]
                batch_size = self.config.get('pii_batch_size', 50)
                pii_results = self.pii_detector.detect_pii_batch(texts_for_pii, batch_size=batch_size)
                
                # Update items with PII information
                for item, pii_entities in zip(valid_items, pii_results):
                    item['has_pii'] = len(pii_entities) > 0
                    item['pii_count'] = len(pii_entities)
                    
                logger.info(f"Batch PII detection complete for {len(valid_items)} items")
        
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
        
        # Collect statistics from processed data directly (more reliable than component stats in chunked processing)
        lengths_after = [item['length'] for item in processed_data if not item.get('dropped', False) and 'length' in item]
        
        # Calculate drop reasons from processed data instead of cleaner stats (which may be empty in parallel processing)
        drop_reasons = {}
        for item in processed_data:
            if item.get('dropped', False) and 'drop_reason' in item:
                reason = item['drop_reason']
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
        
        lengths_before = self.cleaner.get_stats().get('length_before', [])
        
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
        
        # Duplicate analysis - only include items that made it to duplicate detection
        # (exclude items already dropped for language, length, etc.)
        dup_markers = [
            item.get('is_duplicate', False) 
            for item in processed_data 
            if item.get('language') is not None  # Only items that passed language detection
        ]
        if dup_markers:
            self.inspector.analyze_duplicates(dup_markers)
        
        # PII analysis - calculate from processed_data in parallel mode
        # Aggregate PII stats from processed data instead of detector stats (which may be empty in parallel)
        pii_type_counts = {}
        total_with_pii = 0
        for item in processed_data:
            if not item.get('dropped', False) and item.get('has_pii', False):
                total_with_pii += 1
                # If we have pii_entities in the item, count them by type
                if 'pii_entities' in item:
                    for entity in item['pii_entities']:
                        entity_type = entity.get('type', 'UNKNOWN')
                        pii_type_counts[entity_type] = pii_type_counts.get(entity_type, 0) + 1
        
        # If we have PII data, analyze it
        if pii_type_counts:
            self.inspector.analyze_pii_hits(pii_type_counts)
        
        # Combine all stats from components
        # Note: cleaner and pii_detector stats may be empty in parallel/chunked processing
        self.pipeline_stats.update(self.lang_detector.get_stats())
        self.pipeline_stats.update(self.dup_detector.get_stats())
        # Skip pii_detector.get_stats() as it's empty in parallel mode - inspector has the PII stats
        self.pipeline_stats.update(self.tokenizer.get_stats())
        self.pipeline_stats.update(self.inspector.get_stats())
        
        # Calculate accurate statistics from actual processed data
        # This is essential for chunked/parallel processing where component stats are incomplete
        total_texts = len(processed_data)
        dropped_texts = sum(1 for item in processed_data if item.get('dropped', False))
        retained_texts = total_texts - dropped_texts
        
        self.pipeline_stats['total_samples'] = total_texts
        self.pipeline_stats['total_processed'] = total_texts
        self.pipeline_stats['cleaned_text_count'] = retained_texts
        self.pipeline_stats['total_dropped'] = dropped_texts
        
        # Calculate drop reasons from processed data
        drop_reasons_from_data = {}
        for item in processed_data:
            if item.get('dropped', False) and 'drop_reason' in item:
                reason = item['drop_reason']
                drop_reasons_from_data[reason] = drop_reasons_from_data.get(reason, 0) + 1
        # Always set drop_reasons, even if empty
        self.pipeline_stats['drop_reasons'] = drop_reasons_from_data
        
        # For now, use cleaner stats for length_before if available
        cleaner_stats = self.cleaner.get_stats()
        if cleaner_stats.get('length_before'):
            self.pipeline_stats['length_before'] = cleaner_stats['length_before']
        else:
            self.pipeline_stats['length_before'] = []
        
        # Calculate length statistics from processed data
        lengths_from_data = [item.get('length', 0) for item in processed_data if not item.get('dropped', False) and item.get('length')]
        # Always set length_after, even if empty
        self.pipeline_stats['length_after'] = lengths_from_data
        
        if lengths_from_data:
            self.pipeline_stats['cleaned_text_count'] = len(lengths_from_data)
            self.pipeline_stats['cleaned_text_min'] = min(lengths_from_data)
            self.pipeline_stats['cleaned_text_max'] = max(lengths_from_data)
            self.pipeline_stats['cleaned_text_mean'] = sum(lengths_from_data) / len(lengths_from_data)
            self.pipeline_stats['cleaned_text_median'] = sorted(lengths_from_data)[len(lengths_from_data) // 2]
        else:
            # Set defaults if no lengths available
            self.pipeline_stats['cleaned_text_count'] = 0
            self.pipeline_stats['cleaned_text_min'] = 0
            self.pipeline_stats['cleaned_text_max'] = 0
            self.pipeline_stats['cleaned_text_mean'] = 0
            self.pipeline_stats['cleaned_text_median'] = 0
        
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
    
    def generate_html_report(self):
        """
        Generate an HTML report from pipeline statistics and charts.
        """
        logger.info("Generating HTML report...")
        
        try:
            import base64
            from datetime import datetime
            
            reports_dir = Path(self.config.get('reports_dir', 'data/reports'))
            report_path = reports_dir / 'pipeline_report.json'
            if not report_path.exists():
                logger.warning("No pipeline_report.json found, skipping HTML generation")
                return
            
            with open(report_path, 'r') as f:
                stats = json.load(f)
            
            # Read chart images and convert to base64
            charts = {}
            chart_files = [
                'text_lengths.png',
                'token_lengths.png', 
                'drop_reasons.png',
                'language_scores.png',
                'duplicates.png',
                'pii_hits.png'
            ]
            
            for chart_file in chart_files:
                chart_path = reports_dir / chart_file
                if chart_path.exists():
                    with open(chart_path, 'rb') as f:
                        encoded = base64.b64encode(f.read()).decode('utf-8')
                        charts[chart_file] = f"data:image/png;base64,{encoded}"
            
            # Generate HTML
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MainPipe Pipeline Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #444;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .summary-box {{
            background: #e8f4f8;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #888;
            padding: 20px;
            margin-top: 40px;
            border-top: 2px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 MainPipe Pipeline Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>📈 Overview Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Samples</div>
                <div class="stat-value">{stats.get('total_samples', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Texts Retained</div>
                <div class="stat-value">{stats.get('cleaned_text_count', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Texts Dropped</div>
                <div class="stat-value">{stats.get('total_dropped', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tokenized Sequences</div>
                <div class="stat-value">{stats.get('total_sequences', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-value">{stats.get('total_tokens', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Sequence Length</div>
                <div class="stat-value">{stats.get('avg_sequence_length', 0):.1f}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📝 Text Length Statistics</h2>
        <div class="summary-box">
            <strong>After Cleaning:</strong>
            Min: {stats.get('cleaned_text_min', 0):,} | 
            Max: {stats.get('cleaned_text_max', 0):,} | 
            Mean: {stats.get('cleaned_text_mean', 0):.1f} | 
            Median: {stats.get('cleaned_text_median', 0):.1f}
        </div>
        {'<div class="chart"><div class="chart-title">Text Length Distribution</div><img src="' + charts.get('text_lengths.png', '') + '" alt="Text Lengths"></div>' if 'text_lengths.png' in charts else ''}
    </div>

    <div class="section">
        <h2>🔤 Token Statistics</h2>
        <div class="summary-box">
            Min: {stats.get('token_min', 0):,} | 
            Max: {stats.get('token_max', 0):,} | 
            Mean: {stats.get('token_mean', 0):.1f} | 
            Median: {stats.get('token_median', 0):.1f}
        </div>
        {'<div class="chart"><div class="chart-title">Token Length Distribution</div><img src="' + charts.get('token_lengths.png', '') + '" alt="Token Lengths"></div>' if 'token_lengths.png' in charts else ''}
    </div>

    <div class="section">
        <h2>❌ Drop Reasons</h2>
        <table>
            <thead>
                <tr>
                    <th>Reason</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
"""
            
            drop_reasons = stats.get('drop_reasons', {})
            for reason, count in drop_reasons.items():
                html += f"""                <tr>
                    <td>{reason}</td>
                    <td>{count:,}</td>
                </tr>
"""
            
            html += """            </tbody>
        </table>
"""
            if 'drop_reasons.png' in charts:
                html += f"""        <div class="chart">
            <div class="chart-title">Drop Reasons Visualization</div>
            <img src="{charts['drop_reasons.png']}" alt="Drop Reasons">
        </div>
"""
            
            html += """    </div>

    <div class="section">
        <h2>🌍 Language Detection</h2>
        <table>
            <thead>
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                    <th>Mean Score</th>
                    <th>Median Score</th>
                </tr>
            </thead>
            <tbody>
"""
            
            languages = stats.get('languages', {})
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                mean_score = stats.get(f'{lang}_mean_score', 0)
                median_score = stats.get(f'{lang}_median_score', 0)
                html += f"""                <tr>
                    <td><strong>{lang.upper()}</strong></td>
                    <td>{count:,}</td>
                    <td>{mean_score:.4f}</td>
                    <td>{median_score:.4f}</td>
                </tr>
"""
            
            html += """            </tbody>
        </table>
"""
            if 'language_scores.png' in charts:
                html += f"""        <div class="chart">
            <div class="chart-title">Language Detection Scores</div>
            <img src="{charts['language_scores.png']}" alt="Language Scores">
        </div>
"""
            
            html += """    </div>

    <div class="section">
        <h2>🔍 PII Detection</h2>
        <div class="summary-box">
            <strong>Total PII Instances Found:</strong> """ + f"{stats.get('total_pii_hits', 0):,}" + """
        </div>
        <table>
            <thead>
                <tr>
                    <th>PII Type</th>
                    <th>Count</th>
                    <th>Hit Rate</th>
                </tr>
            </thead>
            <tbody>
"""
            
            pii_counts = stats.get('pii_type_counts', {})
            for pii_type, count in sorted(pii_counts.items(), key=lambda x: x[1], reverse=True):
                hit_rate = stats.get(f'{pii_type}_hit_rate', 0) * 100
                html += f"""                <tr>
                    <td><strong>{pii_type}</strong></td>
                    <td>{count:,}</td>
                    <td>{hit_rate:.2f}%</td>
                </tr>
"""
            
            html += """            </tbody>
        </table>
"""
            if 'pii_hits.png' in charts:
                html += f"""        <div class="chart">
            <div class="chart-title">PII Detection Results</div>
            <img src="{charts['pii_hits.png']}" alt="PII Hits">
        </div>
"""
            
            html += """    </div>

    <div class="section">
        <h2>🔄 Duplicate Detection</h2>
        <div class="summary-box">
            <strong>Duplicates Found:</strong> """ + f"{stats.get('duplicate_count', 0):,}" + """ | 
            <strong>Unique Items:</strong> """ + f"{stats.get('unique_count', 0):,}" + """ | 
            <strong>Duplicate Rate:</strong> """ + f"{stats.get('duplicate_rate', 0) * 100:.2f}%" + """
        </div>
"""
            if 'duplicates.png' in charts:
                html += f"""        <div class="chart">
            <div class="chart-title">Duplicate Analysis</div>
            <img src="{charts['duplicates.png']}" alt="Duplicates">
        </div>
"""
            
            html += """    </div>

    <div class="footer">
        <p>Generated by MainPipe Data Processing Pipeline</p>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            html_path = reports_dir / 'pipeline_report.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"HTML report generated: {html_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}", exc_info=True)
    
    def _save_checkpoint(self, processed_data: List[Dict], checkpoint_path: Path):
        """
        Save intermediate processing checkpoint.
        
        Args:
            processed_data: Processed data to save
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
    
    def run(self):
        """
        Run the complete pipeline end-to-end with chunked processing for scalability.
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
            
            # Step 3: Process data in chunks if chunk_size is specified
            chunk_size = self.config.get('chunk_size', None)
            
            if chunk_size and len(texts) > chunk_size:
                logger.info(f"Processing {len(texts)} texts in chunks of {chunk_size}...")
                processed_data = []
                
                # Check for existing checkpoints to resume from
                output_dir = Path(self.config.get('output_dir', 'data/processed'))
                resume_file = output_dir / 'resume_state.json'
                start_chunk = 0
                
                if resume_file.exists():
                    with open(resume_file, 'r') as f:
                        resume_state = json.load(f)
                        start_chunk = resume_state.get('last_completed_chunk', 0) + 1
                        logger.info(f"Resuming from chunk {start_chunk + 1}")
                        
                        # Load existing checkpoints
                        for i in range(start_chunk):
                            checkpoint_path = output_dir / f'checkpoint_{i + 1}.jsonl'
                            if checkpoint_path.exists():
                                with open(checkpoint_path, 'r') as cf:
                                    for line in cf:
                                        processed_data.append(json.loads(line))
                        logger.info(f"Loaded {len(processed_data)} items from previous checkpoints")
                
                total_chunks = (len(texts) + chunk_size - 1) // chunk_size
                for chunk_idx in range(start_chunk, total_chunks):
                    i = chunk_idx * chunk_size
                    chunk_end = min(i + chunk_size, len(texts))
                    chunk_texts = texts[i:chunk_end]
                    logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({i+1}-{chunk_end}/{len(texts)})")
                    
                    chunk_processed = self.process_data(chunk_texts)
                    processed_data.extend(chunk_processed)
                    
                    # Save checkpoint and update resume state
                    checkpoint_path = output_dir / f'checkpoint_{chunk_idx + 1}.jsonl'
                    self._save_checkpoint(chunk_processed, checkpoint_path)
                    
                    # Update resume state
                    with open(resume_file, 'w') as f:
                        json.dump({'last_completed_chunk': chunk_idx, 'total_chunks': total_chunks}, f)
                    
                    logger.info(f"Checkpoint {chunk_idx + 1} saved. Safe to stop/resume.")
                
                # Clean up resume file when complete
                if resume_file.exists():
                    resume_file.unlink()
                    logger.info("All chunks processed. Resume file cleaned up.")
            else:
                # Process all at once if no chunking
                processed_data = self.process_data(texts)
            
            # Step 4: Tokenize data
            tokenized_data = self.tokenize_data(processed_data)
            
            # Step 5: Generate inspections
            self.generate_inspections(processed_data)
            
            # Step 6: Export data
            if tokenized_data:
                self.export_data(tokenized_data)
            
            # Step 7: Generate HTML report
            self.generate_html_report()
            
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
