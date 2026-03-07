#!/bin/bash
set -e

DOWNLOADS_DIR="$( cd "$( dirname "$0")" && pwd)"
SCRIPT_DIR="$DOWNLOADS_DIR/.."
ROOT_DIR="$SCRIPT_DIR/.."
THIRD_PARTY_DIR="$ROOT_DIR/third_party"

cd "$THIRD_PARTY_DIR"

curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz
tar -xzf onnxruntime.tgz
rm onnxruntime.tgz

echo "ONNX Runtime downloaded to third_party."
cd "$ROOT_DIR"