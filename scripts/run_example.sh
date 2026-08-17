#!/usr/bin/env bash
set -e
set -u

PROJECT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${1:-"$PROJECT_PATH/results/example"}

# API keys are read from $PROJECT_PATH/.env (see .env.example).

python3 "$PROJECT_PATH/main.py" \
    -i "$PROJECT_PATH/examples/example.jsonl" \
    -o "$OUTPUT_DIR" \
    -lp en-ja \
    -s 0 \
    -e 3 \
    -mode zero \
    -max 10

python3 "$PROJECT_PATH/main.py" \
    -i "$PROJECT_PATH/examples/example.jsonl" \
    -o "$OUTPUT_DIR" \
    -lp en-ja \
    -s 0 \
    -e 3 \
    -r 3 \
    -mode som \
    -max 10
