#TODO: Add docs...

#!/bin/bash
set -e # exits immediately after failure

SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_DIR/build"

source "$SCRIPT_DIR/config.sh"

if [ -z "$LIBTORCH_PATH" ]; then
    echo "Usage: bash scripts/build.sh /path/to/libtorch"
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH"

# parallelize over 4 cpu cores :P
make -j4

echo "Build successful."
cd ..