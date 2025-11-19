# Mainpipe - Data Processing Pipeline

Technical assessment for Maincode - A containerized data processing pipeline for ML training data preparation.

## Overview

Mainpipe is an end-to-end data processing pipeline that handles:
- **Data acquisition** from URLs and local files
- **Data cleaning, normalization, and tokenization**
- **Training-ready exports** (tokenized data, shards, mixtures)
- **Inspectability** with comprehensive statistics and visualizations
- **Scalable architecture** designed for production workloads

## Features

### Data Processing
- Multi-source data acquisition (URLs, local files)
- Text cleaning and normalization
- Unicode normalization and whitespace handling
- Configurable length filtering

### Quality Control
- Language detection with confidence scoring
- Duplicate detection and removal
- PII (Personally Identifiable Information) detection
- Configurable filtering rules

### Tokenization
- BPE (Byte Pair Encoding) tokenizer
- WordPiece tokenizer
- Support for pre-trained tokenizers (e.g., from HuggingFace)
- Custom vocabulary training

### Export Formats
- JSONL format for training data
- Automatic data sharding for distributed training
- Dataset mixture creation with configurable weights
- Pickle and NumPy array exports

### Inspectability
- Length distribution histograms
- Language score distributions
- Duplicate detection statistics
- PII hit rates by type
- Drop reason analysis
- Comprehensive JSON reports

## Requirements

- Python 3.10+
- Docker and Docker Compose (for containerized execution)

## Installation

### Docker Installation

```bash
# Build the Docker image
docker build -t mainpipe .

# Or use Docker Compose
docker-compose build
```

## Usage

### Running the Pipeline

#### Pre-processing: Deduplication (Recommended First Step)

Run deduplication as a separate pre-processing step before the main pipeline. This process uses MinHash LSH on a single thread so it can be slow.

```bash
# Step 1: Run standalone deduplication with MinHash (single-threaded, finds all duplicates)
# Bash/Linux:
docker compose run --rm mainpipe python /app/scripts/deduplicate_raw_data.py \
  /app/data/raw/mainpipe_data_v1.jsonl \
  /app/data/raw/mainpipe_data_v1_deduped.jsonl \
  --threshold 0.8

# PowerShell (use backticks for line continuation):
docker compose run --rm mainpipe python /app/scripts/deduplicate_raw_data.py `
  /app/data/raw/mainpipe_data_v1.jsonl `
  /app/data/raw/mainpipe_data_v1_deduped.jsonl `
  --threshold 0.8

# Optional parameters:
#   --threshold: Jaccard similarity threshold (default: 0.8)
#   --ngram-size: N-gram size for MinHash (default: 5)
#   --num-perm: Number of permutations (default: 128)
#   --min-length: Minimum text length (default: 50)
#   --max-length: Maximum text length (default: 100000)

# Step 2: Run the main pipeline with the deduplicated file (disable duplicate detection)
# Bash/Linux:
docker compose run --rm mainpipe mainpipe run \
  --local-file data/raw/mainpipe_data_v1_deduped.jsonl \
  --remove-duplicates false \
  -c config.yaml

# PowerShell:
docker compose run --rm mainpipe mainpipe run `
  --local-file data/raw/mainpipe_data_v1_deduped.jsonl `
  --remove-duplicates false `
  -c config.yaml

# Note: --remove-duplicates false prevents re-running duplicate detection and 
# preserves the deduplication statistics from Step 1

# Step 3 (Optional): Reduce file size for submission (keeps only token_ids)
# If the full tokenized_data.jsonl exceeds size limits, create a minimal version:
docker compose run --rm mainpipe python /app/scripts/reduce_tokenized_data.py \
  /app/data/processed/tokenized_data.jsonl \
  /app/data/processed/tokenized_data_minimal.jsonl

# This reduces file size by ~75% by keeping only token_ids
# Original: ~1.5GB with all metadata → Minimal: ~378MB with token_ids only
```

## Pipeline Architecture

```
┌─────────────────┐
│ Pre-processing  │  (Optional but recommended)
│  - Deduplication│  Run: scripts/deduplicate_raw_data.py
│  - MinHash LSH  │  Single-threaded, finds all duplicates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Acquisition│
│  - URLs         │
│  - Local Files  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Cleaning   │
│  - Normalization│
│  - Length Filter│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Checks  │
│  - Language     │
│  - Duplicates   │  (If not pre-processed)
│  - PII          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │
│  - BPE/WordPiece│
│  - Vocabulary   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export          │
│  - Shards       │
│  - JSONL        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inspectability  │
│  - Statistics   │
│  - Visualizations│
└─────────────────┘
```

## Output Structure

After running the pipeline, you'll find:

```
data/
├── raw/              # Downloaded/input data
├── processed/        # Tokenized data and shards
│   ├── tokenized_data.jsonl         # Full tokenized data with metadata
│   ├── tokenized_data_minimal.jsonl # Minimal version (token_ids only)
│   ├── shard_00000.jsonl
│   ├── shard_00001.jsonl
│   └── tokenizer.json
└── reports/          # Analysis reports and visualizations
    ├── pipeline_report.json
    ├── pipeline_report.html
    ├── text_lengths.png
    ├── token_lengths.png
    ├── drop_reasons.png
    ├── language_scores.png
    ├── duplicates.png
    └── pii_hits.png
```

## Scaling Considerations

### Conceptual Plan for Scaling

#### 1. Horizontal Scaling
- **Distributed Processing**: Use Apache Spark or Dask for parallel data processing
- **Message Queues**: Implement RabbitMQ or Kafka for task distribution
- **Worker Pools**: Scale processing workers independently using Kubernetes

#### 2. Data Sharding Strategy
- **Input Sharding**: Split large datasets into manageable chunks
- **Parallel Processing**: Process shards concurrently across multiple workers
- **Shard Size**: Configurable shard sizes (default: 10,000 samples)

#### 3. Storage Optimization
- **Object Storage**: Use S3/GCS for raw and processed data
- **Caching**: Implement Redis for intermediate results
- **Compression**: Use gzip/lz4 for data compression

#### 4. Performance Optimization
- **Batch Processing**: Process data in configurable batch sizes
- **Async I/O**: Use async operations for network and disk I/O
- **Memory Management**: Stream processing for large files
- **GPU Acceleration**: Use GPU for tokenization (via CUDA)

#### 5. Monitoring and Observability
- **Metrics**: Prometheus for pipeline metrics
- **Logging**: Centralized logging with ELK stack
- **Tracing**: Distributed tracing with Jaeger
- **Alerts**: Configure alerts for pipeline failures

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

## Author

Aaron Kwok
