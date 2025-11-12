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

### Local Installation

```bash
# Clone the repository
git clone https://github.com/aaronkwc/mainpipe_assessment.git
cd mainpipe_assessment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Docker Installation

```bash
# Build the Docker image
docker build -t mainpipe .

# Or use Docker Compose
docker-compose build
```

## Usage

### Configuration

Create a `config.yaml` file (see `config.yaml` for a template):

```yaml
# Data sources
data_dir: data/raw
data_urls:
  - https://example.com/dataset.jsonl
local_files:
  - data/raw/sample.txt

# Processing parameters
cleaning:
  min_length: 50
  max_length: 100000

allowed_languages:
  - en
  - es

remove_duplicates: true
remove_pii: false

# Tokenization
tokenizer_type: bpe
vocab_size: 30000
train_tokenizer: true

# Output settings
output_dir: data/processed
shard_size: 10000
```

### Running the Pipeline

#### Local Execution

```bash
# Run the complete pipeline
mainpipe run -c config.yaml

# Run individual steps
mainpipe acquire -u https://example.com/data.jsonl -o data/raw
mainpipe clean -i data/raw/data.jsonl -o data/cleaned
mainpipe tokenize -i data/cleaned/cleaned.jsonl -o data/tokenized
mainpipe inspect -i data/processed -o data/reports
```

#### Docker Execution

```bash
# Run with Docker Compose
docker-compose up

# Or run with Docker directly
docker run -v $(pwd)/data:/app/data -v $(pwd)/config.yaml:/app/config.yaml mainpipe mainpipe run -c config.yaml
```

### Example: Processing Sample Data

```bash
# Create sample data
mkdir -p data/raw
echo '{"text": "This is a sample text for processing."}' > data/raw/sample.jsonl
echo '{"text": "Another example text with more content."}' >> data/raw/sample.jsonl

# Update config.yaml to use local file
# Then run the pipeline
mainpipe run -c config.yaml
```

## Pipeline Architecture

```
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
│  - Duplicates   │
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
│   ├── tokenized_data.jsonl
│   ├── shard_00000.jsonl
│   ├── shard_00001.jsonl
│   └── tokenizer.json
└── reports/          # Analysis reports and visualizations
    ├── pipeline_report.json
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

#### 6. Deployment Architecture

```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐      ┌────▼────┐
    │Worker 1 │       │Worker 2 │      │Worker N │
    └────┬────┘       └────┬────┘      └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼───────┐
                    │ Message Queue│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Data Storage │
                    └──────────────┘
```

#### 7. Scalability Targets
- **Throughput**: Process 1M+ documents per hour
- **Latency**: < 1s per document for real-time processing
- **Availability**: 99.9% uptime with auto-recovery
- **Cost**: Optimize compute costs with spot instances

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=mainpipe tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## License

MIT License

## Author

Aaron Kwong - Technical Assessment for Maincode
