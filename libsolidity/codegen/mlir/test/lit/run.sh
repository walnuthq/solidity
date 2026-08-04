#!/usr/bin/env bash
# Runs the MLIR ladder's lit suite.
#
#   ./libsolidity/codegen/mlir/test/lit/run.sh [extra lit args]
#
# FileCheck ships with LLVM but is not always on PATH, and lit is a package
# without a __main__, so both are located here rather than in every invocation.
set -euo pipefail

root="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export MLIR_TOOL_DIR="${MLIR_TOOL_DIR:-$root/build/libsolidity/codegen/mlir/tools}"
export FILECHECK="${FILECHECK:-$(command -v FileCheck || echo /opt/homebrew/opt/llvm/bin/FileCheck)}"
export MLIR_LIT_OUTPUT_DIR="${MLIR_LIT_OUTPUT_DIR:-$(dirname "$MLIR_TOOL_DIR")/test/lit}"
mkdir -p "$MLIR_LIT_OUTPUT_DIR"

exec python3 -c "from lit.main import main; main()" "$@" "$(dirname "$0")"
