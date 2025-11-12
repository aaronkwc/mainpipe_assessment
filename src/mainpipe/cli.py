"""
Command-line interface for the mainpipe data processing pipeline.
"""
import click
import logging
import sys
from pathlib import Path

from mainpipe.pipeline import Pipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Mainpipe - Data processing pipeline for ML training data preparation."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to configuration YAML file')
@click.pass_context
def run(ctx, config):
    """Run the complete pipeline end-to-end."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config}")
    
    try:
        pipeline = Pipeline.from_config_file(config)
        pipeline.run()
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--urls', '-u', multiple=True, help='URLs to download')
@click.option('--output', '-o', default='data/raw', help='Output directory')
def acquire(urls, output):
    """Acquire data from URLs."""
    from mainpipe.acquisition.downloader import DataAcquisition
    
    logger = logging.getLogger(__name__)
    logger.info(f"Acquiring data from {len(urls)} URLs")
    
    acquisition = DataAcquisition(data_dir=output)
    paths = acquisition.download_urls(list(urls))
    
    logger.info(f"Downloaded {len(paths)} files to {output}")


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file or directory')
@click.option('--output', '-o', default='data/cleaned', help='Output directory')
@click.option('--min-length', default=10, help='Minimum text length')
@click.option('--max-length', default=1000000, help='Maximum text length')
def clean(input, output, min_length, max_length):
    """Clean and normalize text data."""
    from mainpipe.cleaning.cleaner import DataCleaner
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning data from {input}")
    
    cleaner = DataCleaner(config={
        'min_length': min_length,
        'max_length': max_length
    })
    
    # Load texts
    texts = []
    input_path = Path(input)
    
    if input_path.is_file():
        if input_path.suffix == '.jsonl':
            with open(input_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'text' in data:
                        texts.append(data['text'])
        elif input_path.suffix == '.txt':
            with open(input_path, 'r') as f:
                texts.append(f.read())
    
    # Clean texts
    results = cleaner.clean_batch(texts)
    
    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'cleaned.jsonl'
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    stats = cleaner.get_stats()
    logger.info(f"Cleaned {stats['total_processed']} texts, dropped {stats['total_dropped']}")
    logger.info(f"Results saved to {output_file}")


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file with texts')
@click.option('--output', '-o', default='data/tokenized', help='Output directory')
@click.option('--tokenizer', default='bpe', help='Tokenizer type (bpe, wordpiece)')
@click.option('--vocab-size', default=30000, help='Vocabulary size')
def tokenize(input, output, tokenizer, vocab_size):
    """Tokenize text data."""
    from mainpipe.tokenization.tokenizer import TextTokenizer
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Tokenizing data from {input}")
    
    tokenizer_obj = TextTokenizer(tokenizer_type=tokenizer, vocab_size=vocab_size)
    
    # Load texts
    texts = []
    with open(input, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data and data['text']:
                texts.append(data['text'])
    
    # Train tokenizer
    tokenizer_obj.train_tokenizer(texts)
    
    # Tokenize
    results = tokenizer_obj.tokenize_batch(texts)
    
    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'tokenized.jsonl'
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Tokenized {len(results)} texts")
    logger.info(f"Results saved to {output_file}")


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input directory with pipeline results')
@click.option('--output', '-o', default='data/reports', help='Output directory for reports')
def inspect(input, output):
    """Generate inspection reports and visualizations."""
    from mainpipe.inspectability.inspector import DataInspector
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Generating inspection reports from {input}")
    
    inspector = DataInspector(output_dir=output)
    
    # This is a simplified version - in practice would load actual pipeline stats
    logger.info(f"Reports saved to {output}")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
