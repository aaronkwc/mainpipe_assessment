"""
Generate test data with various scenarios to verify pipeline reporting.
"""
import json
from pathlib import Path

# Create test data with different drop scenarios
test_data = []

# 1. Valid texts that should pass (10 texts)
valid_texts = [
    "This is a valid text with sufficient length and proper content for processing.",
    "Another valid text that meets all the requirements and should not be dropped.",
    "The quick brown fox jumps over the lazy dog multiple times in this sentence.",
    "Scientific research shows that proper data cleaning is essential for quality.",
    "Machine learning models require high-quality training data to perform well.",
    "Natural language processing has made significant advances in recent years.",
    "Deep learning architectures have revolutionized computer vision tasks.",
    "Text classification is an important application of machine learning.",
    "Data preprocessing steps are crucial for achieving good model performance.",
    "Feature engineering can significantly improve prediction accuracy in models."
]

for text in valid_texts:
    test_data.append({"text": text})

# 2. Texts that are too short (5 texts)
short_texts = [
    "Too short",
    "Hi",
    "Yes",
    "OK",
    "No way"
]

for text in short_texts:
    test_data.append({"text": text})

# 3. Texts with PII that should be redacted (5 texts)
pii_texts = [
    "John Smith lives at 123 Main Street and his email is john.smith@email.com",
    "Contact Dr. Sarah Johnson at (555) 123-4567 for more information about this.",
    "The meeting with Robert Brown is scheduled for January 15, 2024 in New York.",
    "Please send the documents to mary.williams@company.com by next Friday.",
    "Mr. David Miller from London will be visiting our Boston office next week."
]

for text in pii_texts:
    test_data.append({"text": text})

# 4. Duplicate texts (3 duplicates of the first valid text)
for _ in range(3):
    test_data.append({"text": valid_texts[0]})

# 5. Non-English texts (3 texts)
foreign_texts = [
    "Das ist ein deutscher Text der herausgefiltert werden sollte wenn Englisch erforderlich ist.",
    "Ceci est un texte français qui devrait être filtré si l'anglais est requis pour le pipeline.",
    "Este es un texto en español que debería ser filtrado si se requiere inglés."
]

for text in foreign_texts:
    test_data.append({"text": text})

# 6. Very long text (1 text) - assuming max length is 10000
long_text = "This is a very long text. " * 500  # About 13,000 characters
test_data.append({"text": long_text})

# Save to file
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "test_data.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"Generated {len(test_data)} test samples:")
print(f"  - {len(valid_texts)} valid texts")
print(f"  - {len(short_texts)} short texts (should be dropped)")
print(f"  - {len(pii_texts)} texts with PII (should be redacted)")
print(f"  - 3 duplicate texts (should be marked as duplicates)")
print(f"  - {len(foreign_texts)} non-English texts (should be filtered)")
print(f"  - 1 very long text (should be dropped)")
print(f"\nSaved to: {output_file}")
print(f"\nExpected results:")
print(f"  - Total processed: {len(test_data)}")
print(f"  - Should drop: {len(short_texts)} (too short) + 3 (duplicates) + {len(foreign_texts)} (language) + 1 (too long) = {len(short_texts) + 3 + len(foreign_texts) + 1}")
print(f"  - Should retain: {len(valid_texts) + len(pii_texts)} = {len(valid_texts) + len(pii_texts)} (with PII redacted)")
