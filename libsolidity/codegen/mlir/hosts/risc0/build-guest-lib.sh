#!/usr/bin/env bash
# Builds lib/libevmcontract.a - the MLIR-ladder-compiled landmine functions
# (with by-pointer __test_* wrappers) plus evm-rt, as RV32IM objects ready to
# link into the RISC Zero guest.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MLIR_DIR="$DIR/../.."                                   # libsolidity/codegen/mlir
REPO_ROOT="$(cd "$MLIR_DIR/../../.." && pwd)"
BUILD="${SOLC_BUILD_DIR:-$REPO_ROOT/build}"
YUL2RV="$BUILD/libsolidity/codegen/mlir/tools/yul2rv"
LLVM_BIN="${LLVM_BIN_DIR:-/opt/homebrew/opt/llvm@21/bin}"

[ -x "$YUL2RV" ] || { echo "yul2rv not found at $YUL2RV (build target 'yul2rv' first)"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/landmines.yul" <<'EOF'
{
	function div2(a, b) -> r { r := div(a, b) }
	function sdiv2(a, b) -> r { r := sdiv(a, b) }
	function mod2(a, b) -> r { r := mod(a, b) }
	function smod2(a, b) -> r { r := smod(a, b) }
	function exp2(a, b) -> r { r := exp(a, b) }
	function shl2(a, b) -> r { r := shl(a, b) }
	function shr2(a, b) -> r { r := shr(a, b) }
	function sar2(a, b) -> r { r := sar(a, b) }
	function byte2(a, b) -> r { r := byte(a, b) }
	function se2(a, b) -> r { r := signextend(a, b) }
	function am3(a, b, c) -> r { r := addmod(a, b, c) }
	function mm3(a, b, c) -> r { r := mulmod(a, b, c) }
	function addmul(a, b) -> r { r := add(mul(a, 3), b) }
}
EOF

"$YUL2RV" "$TMP/landmines.yul" --wrappers --obj-dir "$TMP" | grep -q "stage=riscv" \
	|| { echo "ladder did not reach the riscv stage"; exit 1; }

"$LLVM_BIN/clang" --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 \
	-c "$MLIR_DIR/runtime/evm-rt/evm_rt.ll" -o "$TMP/evm_rt.o"

mkdir -p "$DIR/lib"
"$LLVM_BIN/llvm-ar" rcs "$DIR/lib/libevmcontract.a" "$TMP/object.o" "$TMP/evm_rt.o"
"$LLVM_BIN/llvm-nm" "$DIR/lib/libevmcontract.a" | grep -q "__test_div2" \
	|| { echo "wrappers missing from the archive"; exit 1; }

echo "wrote $DIR/lib/libevmcontract.a"
