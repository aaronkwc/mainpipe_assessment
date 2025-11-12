# Quick Start Guide

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/aaronkwc/mainpipe_assessment.git
cd mainpipe_assessment
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Pipeline

### Quick Test

Generate sample data and run the pipeline:

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Run the pipeline
mainpipe run -c config-test.yaml
```

### Custom Configuration

1. **Edit the configuration file** (`config.yaml`):

```yaml
# Specify your data sources
local_files:
  - data/raw/your_data.jsonl

# Configure processing parameters
cleaning:
  min_length: 50
  max_length: 100000

# Set language filters
allowed_languages:
  - en

# Configure tokenization
tokenizer_type: bpe
vocab_size: 30000
```

2. **Run the pipeline**:

```bash
mainpipe run -c config.yaml
```

## Output

After running the pipeline, you'll find:

```
data/
├── processed/
│   ├── tokenized_data.jsonl    # Tokenized data ready for training
│   └── tokenizer.json          # Trained tokenizer
└── reports/
    ├── pipeline_report.json     # Complete statistics
    ├── text_lengths.png         # Length distribution
    ├── token_lengths.png        # Token length distribution
    ├── language_scores.png      # Language detection scores
    ├── duplicates.png           # Duplicate analysis
    └── pii_hits.png            # PII detection results
```

## Individual Commands

You can also run individual pipeline steps:

```bash
# Acquire data from URL
mainpipe acquire -u https://example.com/data.jsonl -o data/raw

# Clean data
mainpipe clean -i data/raw/data.jsonl -o data/cleaned

# Tokenize data
mainpipe tokenize -i data/cleaned/cleaned.jsonl -o data/tokenized

# Generate inspection reports
mainpipe inspect -i data/processed -o data/reports
```

## Using Python API

You can also use the pipeline programmatically:

```python
from mainpipe.pipeline import Pipeline

# Load configuration from file
pipeline = Pipeline.from_config_file('config.yaml')

# Or create configuration in code
config = {
    'local_files': ['data/raw/sample.jsonl'],
    'tokenizer_type': 'bpe',
    'vocab_size': 5000,
    # ... other config options
}
pipeline = Pipeline(config)

# Run the pipeline
pipeline.run()

# Access statistics
print(pipeline.pipeline_stats)
```

## Docker (Note)

A Dockerfile is provided for containerization. However, note that Docker builds may require SSL certificate configuration in restricted environments. The local installation is fully functional and recommended for testing.

To build and run with Docker (in environments with proper SSL configuration):

```bash
# Build the image
docker build -t mainpipe .

# Run with docker-compose
docker-compose up

# Or run directly
docker run -v $(pwd)/data:/app/data -v $(pwd)/config.yaml:/app/config.yaml mainpipe mainpipe run -c config.yaml
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=mainpipe tests/
```

## Troubleshooting

### Issue: Presidio requires language models

If you see warnings about missing language models, they will be automatically downloaded on first use. For faster startup, you can pre-download:

```bash
python -m spacy download en_core_web_sm
```

### Issue: Import errors

Make sure the package is installed:

```bash
pip install -e .
```

### Issue: Permission errors

Ensure you have write permissions to the `data/` directory:

```bash
chmod -R u+w data/
```

## Next Steps

- Modify `config.yaml` to process your own data
- Adjust cleaning and filtering parameters
- Experiment with different tokenizer configurations
- Review the generated reports and visualizations
- Scale the pipeline for larger datasets
