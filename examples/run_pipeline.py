"""
Example: Running the pipeline with custom configuration.
"""
import sys
sys.path.insert(0, '../src')

from mainpipe.pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define configuration
config = {
    'data_dir': 'data/raw',
    'local_files': ['data/raw/sample.jsonl'],
    'cleaning': {
        'min_length': 50,
        'max_length': 100000,
    },
    'language': 'en',
    'allowed_languages': ['en'],
    'remove_duplicates': True,
    'remove_pii': False,
    'tokenizer_type': 'bpe',
    'vocab_size': 5000,
    'train_tokenizer': True,
    'output_dir': 'data/processed',
    'reports_dir': 'data/reports',
    'shard_size': 50,
}

# Create and run pipeline
pipeline = Pipeline(config)
pipeline.run()

print("\n✓ Pipeline completed successfully!")
print(f"✓ Processed {pipeline.pipeline_stats.get('total_processed', 0)} texts")
print(f"✓ Generated {pipeline.pipeline_stats.get('unique_items', 0)} unique items")
print(f"✓ Average token length: {pipeline.pipeline_stats.get('avg_sequence_length', 0):.2f}")
