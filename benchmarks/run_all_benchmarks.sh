#!/bin/bash
# Master script to run all MIND patent benchmarks
# Usage: ./run_all_benchmarks.sh

set -e  # Exit on error

echo "========================================================================"
echo "MIND Patent Benchmarks - Running All Comparisons"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
RESULTS_DIR="$(pwd)/benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run a benchmark
run_benchmark() {
    local name="$1"
    local dir="$2"
    local script="$3"

    echo "========================================================================"
    echo "Running: $name"
    echo "========================================================================"
    echo ""

    cd "$dir"

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install -q -r requirements.txt
        echo ""
    fi

    # Run benchmark
    if python "$script"; then
        echo -e "${GREEN}✓ $name completed successfully${NC}"

        # Copy results
        cp *.json "$RESULTS_DIR/" 2>/dev/null || echo "No JSON results found"
    else
        echo -e "${RED}✗ $name failed${NC}"
    fi

    echo ""
    cd - > /dev/null
}

# 1. PyTorch 2.0 Compilation Comparison (CRITICAL)
if command -v python3 &> /dev/null; then
    run_benchmark \
        "PyTorch 2.0 Compilation Comparison" \
        "pytorch_comparison" \
        "benchmark_pytorch_compile.py"
else
    echo -e "${RED}✗ Python 3 not found, skipping PyTorch benchmark${NC}"
    echo ""
fi

# 2. Determinism Proof (CRITICAL)
echo -e "${YELLOW}NOTE: Determinism benchmark requires MIND CLI to be built${NC}"
echo -e "${YELLOW}Skipping for now - run manually with: cd determinism && python benchmark_determinism.py${NC}"
echo ""

# 3. Autograd Comparison
run_benchmark \
    "Autograd Comparison (MIND vs PyTorch)" \
    "autograd_comparison" \
    "benchmark_autograd.py"

# 4. JAX Compilation Comparison
if pip show jax &> /dev/null || pip install -q jax jaxlib; then
    run_benchmark \
        "JAX Compilation Comparison" \
        "jax_comparison" \
        "benchmark_jax_compile.py"
else
    echo -e "${YELLOW}⚠ JAX installation failed, skipping JAX benchmark${NC}"
    echo ""
fi

# 5. Inference Speed Benchmark
run_benchmark \
    "Inference Speed Benchmark" \
    "inference" \
    "benchmark_inference.py"

# Summary
echo "========================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "JSON files:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "No results found"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Update patent application with real numbers"
echo "  3. Run determinism benchmark manually if needed"
echo ""
echo "For determinism proof:"
echo "  cd benchmarks/determinism"
echo "  cargo build --release --bin mind  # Build MIND CLI first"
echo "  python benchmark_determinism.py"
echo ""
