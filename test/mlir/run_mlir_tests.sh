#!/bin/bash

# MLIR Test Runner Script
# Tests each .sol file in test/mlir/ with:
# 1) --print-mlir (MLIR dialect output)
# 2) --asm (assembly generated)
# 3) --bin (binary)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Path to solc binary
SOLC="../../build/solc/solc"

# Check if solc exists
if [ ! -f "$SOLC" ]; then
    echo -e "${RED}Error: solc binary not found at $SOLC${NC}"
    echo "Please build the project first with: make -C ../../build solc"
    exit 1
fi

# Create output directory for test results
OUTPUT_DIR="mlir_test_output"
mkdir -p "$OUTPUT_DIR"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .sol)
    local test_type=$2
    local flag=$3
    
    echo -n "  Testing $test_type... "
    
    local output_file="$OUTPUT_DIR/${test_name}_${test_type}.txt"
    local error_file="$OUTPUT_DIR/${test_name}_${test_type}.err"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Run the test (redirect stderr to stdout for --print-mlir since it outputs to stderr)
    if [ "$test_type" == "mlir" ]; then
        if $SOLC $flag "$test_file" > "$output_file" 2>&1; then
            SUCCESS=true
        else
            SUCCESS=false
        fi
    else
        if $SOLC $flag "$test_file" > "$output_file" 2> "$error_file"; then
            SUCCESS=true
        else
            SUCCESS=false
        fi
    fi
    
    if [ "$SUCCESS" = true ]; then
        # Check if output is non-empty (for --bin and --asm)
        if [ "$test_type" != "mlir" ] && [ ! -s "$output_file" ]; then
            echo -e "${RED}FAILED${NC} - Empty output"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
        
        # For MLIR, check if it contains expected MLIR operations
        if [ "$test_type" == "mlir" ]; then
            if grep -q "solidity\." "$output_file" 2>/dev/null || grep -q "builtin.module" "$output_file" 2>/dev/null; then
                echo -e "${GREEN}PASSED${NC}"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                echo -e "${RED}FAILED${NC} - No MLIR operations found"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                return 1
            fi
        else
            echo -e "${GREEN}PASSED${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        echo -e "${RED}FAILED${NC} - Compilation error"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "Error output:" >> "$error_file"
        cat "$error_file" | head -20 >> "$error_file"
        return 1
    fi
}

# Function to test with --mlir-optimize flag
run_mlir_optimize_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .sol)
    local test_type=$2
    local flag=$3
    
    echo -n "  Testing $test_type with --mlir-optimize... "
    
    local output_file="$OUTPUT_DIR/${test_name}_mlir_optimize_${test_type}.txt"
    local error_file="$OUTPUT_DIR/${test_name}_mlir_optimize_${test_type}.err"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Run the test with --mlir-optimize
    if $SOLC --mlir-optimize $flag "$test_file" > "$output_file" 2> "$error_file"; then
        # Check if output is non-empty
        if [ ! -s "$output_file" ]; then
            echo -e "${RED}FAILED${NC} - Empty output"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
        echo -e "${GREEN}PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}FAILED${NC} - Compilation error"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Main test loop
echo "========================================="
echo "MLIR Solidity Compiler Test Suite"
echo "========================================="
echo ""

# Find all .sol files in the current directory
for sol_file in *.sol; do
    if [ -f "$sol_file" ]; then
        echo -e "${YELLOW}Testing: $sol_file${NC}"
        
        # Test 1: --print-mlir (MLIR dialect output) - requires --mlir-optimize
        run_test "$sol_file" "mlir" "--mlir-optimize --print-mlir"
        
        # Test 2: --asm (assembly generated)
        run_test "$sol_file" "asm" "--asm"
        
        # Test 3: --bin (binary)
        run_test "$sol_file" "bin" "--bin"
        
        # Additional tests with --mlir-optimize flag
        echo "  With --mlir-optimize flag:"
        run_mlir_optimize_test "$sol_file" "asm" "--asm"
        run_mlir_optimize_test "$sol_file" "bin" "--bin"
        
        echo ""
    fi
done

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Total tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Check $OUTPUT_DIR/ for details.${NC}"
    exit 1
fi
