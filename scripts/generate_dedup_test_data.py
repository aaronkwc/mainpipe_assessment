#!/usr/bin/env python3
"""
Generate test data specifically for testing MinHash deduplication.
Creates texts with varying degrees of similarity to validate duplicate detection.
"""

import json
from pathlib import Path

def generate_dedup_test_data():
    """Generate test data with known duplicates and near-duplicates."""
    
    test_data = []
    
    # 1. Exact duplicates (should be detected as duplicates)
    base_text_1 = "The quick brown fox jumps over the lazy dog in the sunny meadow."
    for i in range(3):
        test_data.append({
            "text": base_text_1,
            "category": "exact_duplicate",
            "instance": i + 1
        })
    
    # 2. Near-duplicates with minor word changes (should be detected with default 0.8 threshold)
    base_text_2 = "Machine learning algorithms require large amounts of training data to achieve good performance."
    near_duplicate_variants = [
        "Machine learning algorithms require large amounts of training data to achieve good performance.",  # Exact
        "Machine learning algorithms need large amounts of training data to achieve good performance.",     # 1 word change
        "Machine learning models require large amounts of training data to achieve good performance.",       # 1 word change
        "Machine learning algorithms require huge amounts of training data to achieve good performance.",    # 1 word change
        "Deep learning algorithms require large amounts of training data to achieve good performance.",      # 1 word change
    ]
    for i, text in enumerate(near_duplicate_variants):
        test_data.append({
            "text": text,
            "category": "near_duplicate_group_1",
            "instance": i + 1
        })
    
    # 3. Another near-duplicate group with sentence reordering
    base_sentences = [
        "Natural language processing has made significant advances in recent years.",
        "Deep learning models have revolutionized the field.",
        "Transformer architectures are now the standard approach."
    ]
    # Original order
    test_data.append({
        "text": " ".join(base_sentences),
        "category": "sentence_order_group",
        "instance": 1
    })
    # Reordered (high similarity, may or may not be detected depending on threshold)
    test_data.append({
        "text": " ".join([base_sentences[2], base_sentences[0], base_sentences[1]]),
        "category": "sentence_order_group",
        "instance": 2
    })
    # With one sentence changed
    test_data.append({
        "text": " ".join([base_sentences[0], "Neural networks have transformed the landscape.", base_sentences[2]]),
        "category": "sentence_order_group",
        "instance": 3
    })
    
    # 4. Similar topic but different content (should NOT be duplicates)
    different_ml_texts = [
        "Supervised learning involves training models on labeled datasets where the correct output is known for each input example.",
        "Unsupervised learning works with unlabeled data to discover hidden patterns and structures without predefined categories.",
        "Reinforcement learning trains agents through trial and error using rewards and penalties to optimize decision-making.",
        "Transfer learning leverages knowledge from pre-trained models to improve performance on new related tasks.",
    ]
    for i, text in enumerate(different_ml_texts):
        test_data.append({
            "text": text,
            "category": "similar_topic_different_content",
            "instance": i + 1
        })
    
    # 5. Near-duplicates with additional content (test sensitivity)
    base_text_3 = "Python is a versatile programming language used for web development, data science, and automation."
    extended_variants = [
        base_text_3,  # Original
        base_text_3 + " It has a simple syntax that makes it easy to learn.",  # +1 sentence
        base_text_3 + " It has a simple syntax that makes it easy to learn. Many companies use Python for production systems.",  # +2 sentences
        "Python is a versatile programming language used for web development, data science, automation, and machine learning.",  # Extended list
    ]
    for i, text in enumerate(extended_variants):
        test_data.append({
            "text": text,
            "category": "progressive_extension",
            "instance": i + 1
        })
    
    # 6. Character-level changes (test robustness)
    base_text_4 = "Artificial intelligence is transforming industries across healthcare, finance, and transportation sectors."
    char_variants = [
        base_text_4,  # Original
        base_text_4.replace("transforming", "transformimg"),  # Typo
        base_text_4.replace("healthcare", "health care"),      # Space added
        base_text_4.replace(".", "!"),                          # Punctuation change
        base_text_4.upper(),                                    # Case change
    ]
    for i, text in enumerate(char_variants):
        test_data.append({
            "text": text,
            "category": "character_variations",
            "instance": i + 1
        })
    
    # 7. Unique texts (should NOT be duplicates)
    unique_texts = [
        "Climate change poses significant challenges to global sustainability and environmental conservation efforts worldwide.",
        "Quantum computing promises exponential speedups for certain computational problems using quantum mechanical phenomena.",
        "Blockchain technology enables decentralized and transparent record-keeping systems without intermediaries.",
        "Renewable energy sources like solar and wind power are becoming increasingly cost-competitive with fossil fuels.",
        "Gene editing technologies such as CRISPR offer new possibilities for treating genetic diseases and disorders.",
    ]
    for i, text in enumerate(unique_texts):
        test_data.append({
            "text": text,
            "category": "unique",
            "instance": i + 1
        })
    
    # 8. Paraphrased content (moderate similarity, borderline case)
    original = "The coronavirus pandemic has fundamentally changed how people work, with remote work becoming the new normal."
    paraphrases = [
        original,
        "Remote work has become the new standard as the COVID-19 pandemic fundamentally altered working patterns.",
        "Due to the coronavirus outbreak, working from home has shifted from exception to norm in many industries.",
        "The global health crisis transformed workplace dynamics, making telecommuting a permanent fixture for many.",
    ]
    for i, text in enumerate(paraphrases):
        test_data.append({
            "text": text,
            "category": "paraphrased",
            "instance": i + 1
        })
    
    return test_data

def save_test_data(test_data, output_path):
    """Save test data to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(test_data)} test samples")
    print(f"Saved to: {output_path}")
    
    # Print summary
    categories = {}
    for item in test_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nTest data breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} samples")
    
    print("\nExpected duplicates (with default threshold 0.8):")
    print("  - exact_duplicate: 2 duplicates (3 total, first is unique)")
    print("  - near_duplicate_group_1: 4 duplicates (5 total, variations of same sentence)")
    print("  - character_variations: 4 duplicates (5 total, minor typos/case changes)")
    print("  - Other groups: Depends on similarity threshold")

if __name__ == "__main__":
    test_data = generate_dedup_test_data()
    output_path = "data/raw/dedup_test_data.jsonl"
    save_test_data(test_data, output_path)
