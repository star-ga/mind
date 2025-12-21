#!/bin/bash
##===----------------------------------------------------------------------===##
#
# Mojo Compilation Benchmark Runner
# Compares Mojo compilation speed with MIND
#
##===----------------------------------------------------------------------===##

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Mojo Compilation Benchmark Runner"
echo "================================================================================"
echo ""

# Check if mojo is installed
if ! command -v mojo &> /dev/null; then
    echo -e "${RED}ERROR: 'mojo' command not found!${NC}"
    echo ""
    echo "Please install Mojo SDK:"
    echo "  1. Visit: https://docs.modular.com/mojo/manual/get-started/"
    echo "  2. Run: curl https://get.modular.com | sh -"
    echo "  3. Authenticate: modular auth <your-key>"
    echo "  4. Install: modular install mojo"
    echo ""
    echo "Then add to PATH:"
    echo "  export MODULAR_HOME=\"\$HOME/.modular\""
    echo "  export PATH=\"\$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:\$PATH\""
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Mojo found:${NC}"
mojo --version
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: 'python3' command not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found:${NC}"
python3 --version
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running benchmarks..."
echo ""

# Run benchmark script
python3 benchmark_mojo_compilation.py

echo ""
echo "================================================================================"
echo "Benchmark complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $SCRIPT_DIR/mojo_results.json"
echo ""
echo "To compare with MIND, see:"
echo "  - MIND results: ../../docs/benchmarks/compiler_performance.md"
echo "  - This comparison: mojo_results.json"
echo ""
echo "Share your results by opening a GitHub issue with:"
echo "  1. Your system specs (CPU, OS, RAM)"
echo "  2. Mojo version (shown above)"
echo "  3. The mojo_results.json file"
echo ""
