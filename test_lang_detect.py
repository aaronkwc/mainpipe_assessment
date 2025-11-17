import json
from langdetect import detect_langs

data = [json.loads(line) for line in open('data/raw/dedup_test_data.jsonl')]
print(f"Testing language detection on {len(data)} samples:")
for i, d in enumerate(data):
    try:
        result = detect_langs(d['text'])
        print(f"{i+1}: {result[0]} - Category: {d['category']}")
    except Exception as e:
        print(f"{i+1}: ERROR - {e} - Category: {d['category']}")
