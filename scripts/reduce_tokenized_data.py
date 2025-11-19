#!/usr/bin/env python
"""Reduce tokenized data to only token_ids for submission."""

import json
import sys
from pathlib import Path

def reduce_tokenized_data(input_path: str, output_path: str):
    """
    Extract only token_ids from tokenized data.
    
    Args:
        input_path: Path to full tokenized_data.jsonl
        output_path: Path to output minimal JSONL
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            
            # Keep only token_ids
            minimal_data = {
                'token_ids': data['token_ids']
            }
            
            outfile.write(json.dumps(minimal_data) + '\n')
            processed += 1
            
            if processed % 10000 == 0:
                print(f"Processed {processed:,} records...")
    
    print(f"\nCompleted! Processed {processed:,} records")
    print(f"Output: {output_path}")
    
    # Show size comparison
    input_size_mb = input_file.stat().st_size / (1024 * 1024)
    output_size_mb = output_file.stat().st_size / (1024 * 1024)
    reduction_pct = ((input_size_mb - output_size_mb) / input_size_mb) * 100
    
    print(f"\nSize comparison:")
    print(f"  Input:  {input_size_mb:.2f} MB")
    print(f"  Output: {output_size_mb:.2f} MB")
    print(f"  Reduction: {reduction_pct:.1f}%")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python reduce_tokenized_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    reduce_tokenized_data(sys.argv[1], sys.argv[2])
