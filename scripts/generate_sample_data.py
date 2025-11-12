"""
Sample data generator for testing the pipeline.
"""
import json
from pathlib import Path


def generate_sample_data(output_file: str = "data/raw/sample.jsonl", num_samples: int = 100):
    """Generate sample text data for testing."""
    
    sample_texts = [
        "Natural language processing is a branch of artificial intelligence that focuses on the interaction between computers and human language.",
        "Machine learning algorithms can learn patterns from data without being explicitly programmed.",
        "Deep learning models use neural networks with multiple layers to learn hierarchical representations.",
        "Data preprocessing is a crucial step in any machine learning pipeline.",
        "Tokenization breaks down text into smaller units called tokens for processing.",
        "Transfer learning allows models trained on one task to be adapted for another task.",
        "Neural networks are inspired by the structure and function of biological neurons.",
        "Training data quality has a significant impact on model performance.",
        "Overfitting occurs when a model learns the training data too well and fails to generalize.",
        "Cross-validation is a technique for assessing model performance on unseen data.",
        "The attention mechanism allows models to focus on relevant parts of the input.",
        "Transformer architectures have revolutionized natural language processing.",
        "Pre-trained language models like BERT and GPT have achieved state-of-the-art results.",
        "Data augmentation techniques can help improve model robustness.",
        "Batch normalization helps stabilize and speed up neural network training.",
        "Dropout is a regularization technique that prevents overfitting.",
        "Learning rate scheduling can improve training convergence.",
        "Ensemble methods combine multiple models to improve prediction accuracy.",
        "Feature engineering is the process of creating new features from raw data.",
        "Model interpretability is important for understanding how models make predictions.",
    ]
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    with open(output_file, 'w') as f:
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            # Add variation
            if i % 3 == 0:
                text = text.upper()
            elif i % 3 == 1:
                text = text.lower()
            
            # Add some duplicates intentionally
            if i > 0 and i % 10 == 0:
                text = sample_texts[0]  # Duplicate
            
            data = {
                "text": text,
                "id": i,
                "source": "sample_generator"
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"Generated {num_samples} samples in {output_file}")


if __name__ == "__main__":
    generate_sample_data()
