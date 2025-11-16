#!/usr/bin/env bash
set -euo pipefail

# Smoke test for mainpipe pipeline using config.test.yaml and sample data.
# Behavior:
# - If docker is available, run pipeline inside the 'mainpipe' service via docker compose.
# - Otherwise attempt to run CLI directly with python.
# - Verify that tokenized_data.jsonl exists and is non-empty after run.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=config.test.yaml
OUT=data/processed/tokenized_data.jsonl

# Ensure sample exists
if [ ! -f data/raw/sample.jsonl ]; then
  echo "sample.jsonl missing; generating sample data..."
  python3 scripts/generate_sample_data.py
fi

# Choose runner
if command -v docker >/dev/null 2>&1; then
  echo "Running smoke test inside Docker (docker detected)"
  docker compose run --rm mainpipe mainpipe run -c "$CONFIG"
else
  echo "Docker not found; running pipeline CLI directly in host Python"
  python3 -m mainpipe.cli run -c "$CONFIG"
fi

# Checks
if [ ! -s "$OUT" ]; then
  echo "Smoke test FAILED: output $OUT missing or empty"
  exit 2
fi

echo "Smoke test PASSED: $OUT exists and is non-empty"
exit 0
