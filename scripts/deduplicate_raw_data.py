"""
Standalone script to deduplicate raw JSONL data using MinHash LSH.
Run this BEFORE the main pipeline to remove duplicates from the input data.
Uses the same MinHash duplicate detector as the pipeline for consistent results.
"""
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mainpipe.utils.duplicate_detector import DuplicateDetector
from src.mainpipe.cleaning.cleaner import DataCleaner


def deduplicate_jsonl(input_file: str, output_file: str, config: dict = None):
    """
    Deduplicate a JSONL file using MinHash LSH (same as pipeline).
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output deduplicated JSONL file
        config: Optional config dict for cleaner and duplicate detector
    """
    if config is None:
        config = {
            'min_length': 50,
            'max_length': 100000,
            'remove_urls': True,
            'remove_emails': True,
            'dup_method': 'minhash',
            'dup_threshold': 0.8,
            'ngram_size': 5,
            'num_perm': 128
        }
    
    # Initialize cleaner and duplicate detector
    cleaner = DataCleaner(config)
    dup_detector = DuplicateDetector(
        method=config.get('dup_method', 'minhash'),
        threshold=config.get('dup_threshold', 0.8),
        ngram_size=config.get('ngram_size', 5),
        num_perm=config.get('num_perm', 128)
    )
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Config: method={config.get('dup_method')}, threshold={config.get('dup_threshold')}")
    
    # Count total lines
    print("\nCounting total lines...")
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"Total lines: {total_lines:,}")
    
    # Process file
    print("\nDeduplicating with MinHash LSH...")
    unique_count = 0
    duplicate_count = 0
    too_short_count = 0
    too_long_count = 0
    empty_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile):
            if idx % 100 == 0:
                print(f"  Processed {idx:,} / {total_lines:,} ({idx/total_lines*100:.1f}%) - "
                      f"Unique: {unique_count:,}, Duplicates: {duplicate_count:,}")
            
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                if not text:
                    empty_count += 1
                    continue
                
                # Clean the text (same as pipeline)
                cleaned_text, drop_reason = cleaner.clean_text(text)
                
                if cleaned_text is None:
                    if 'too_short' in drop_reason:
                        too_short_count += 1
                    elif 'too_long' in drop_reason:
                        too_long_count += 1
                    elif 'empty' in drop_reason:
                        empty_count += 1
                    continue
                
                # Check for duplicates using MinHash
                is_duplicate = dup_detector.is_duplicate(cleaned_text)
                
                if is_duplicate:
                    duplicate_count += 1
                else:
                    # Write unique text to output
                    unique_count += 1
                    outfile.write(line)
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: Error parsing JSON at line {idx}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {idx}: {e}")
                continue
    
    # Print statistics
    print("\n" + "="*60)
    print("DEDUPLICATION COMPLETE")
    print("="*60)
    print(f"Total processed:      {total_lines:,}")
    print(f"Unique (kept):        {unique_count:,} ({100*unique_count/total_lines:.2f}%)")
    print(f"Duplicates (dropped): {duplicate_count:,} ({100*duplicate_count/total_lines:.2f}%)")
    print(f"Too short:            {too_short_count:,}")
    print(f"Too long:             {too_long_count:,}")
    print(f"Empty:                {empty_count:,}")
    print("="*60)
    print(f"\nOutput saved to: {output_path}")
    
    # Save statistics to JSON file
    stats = {
        'total': total_lines,
        'unique': unique_count,
        'duplicates': duplicate_count,
        'too_short': too_short_count,
        'too_long': too_long_count,
        'empty': empty_count,
        'duplicate_rate': 100 * duplicate_count / total_lines if total_lines > 0 else 0
    }
    
    stats_path = output_path.parent / f"{output_path.stem}_dedup_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Deduplication statistics saved to: {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL file using MinHash LSH (same as pipeline)"
    )
    parser.add_argument(
        'input_file',
        help='Input JSONL file path'
    )
    parser.add_argument(
        'output_file',
        help='Output deduplicated JSONL file path'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.8,
        help='Duplicate detection threshold (default: 0.8)'
    )
    parser.add_argument(
        '--ngram-size', type=int, default=5,
        help='N-gram size for MinHash (default: 5)'
    )
    parser.add_argument(
        '--num-perm', type=int, default=128,
        help='Number of permutations for MinHash (default: 128)'
    )
    parser.add_argument(
        '--min-length', type=int, default=50,
        help='Minimum text length (default: 50)'
    )
    parser.add_argument(
        '--max-length', type=int, default=100000,
        help='Maximum text length (default: 100000)'
    )
    
    args = parser.parse_args()
    
    config = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'remove_urls': True,
        'remove_emails': True,
        'dup_method': 'minhash',
        'dup_threshold': args.threshold,
        'ngram_size': args.ngram_size,
        'num_perm': args.num_perm
    }
    
    deduplicate_jsonl(args.input_file, args.output_file, config)


if __name__ == '__main__':
    main()
