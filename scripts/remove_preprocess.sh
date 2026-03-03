# TODO: Add docs

#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"
DATA_DIR="$SCRIPT_DIR/../data"

rm -f "$DATA_DIR/testing.csv"
rm -f "$DATA_DIR/training.csv"
rm -f "$DATA_DIR/validation.csv"
rm -f "$DATA_DIR/mappings.json"