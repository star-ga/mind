#!/bin/bash
# MLIR/LLVM Pipeline Demonstration
# Shows the complete compilation flow: MIND → MLIR → LLVM IR → Binary
#
# Prerequisites:
# - mind compiler with mlir-lowering feature
# - mlir-opt (MLIR optimizer)
# - llc (LLVM compiler)
# - clang (for linking)

set -e  # Exit on error

echo "=== MIND → MLIR → LLVM → Binary Pipeline Demo ==="
echo

# ============================================================================
# Step 1: Create a simple MIND program
# ============================================================================

cat > /tmp/example.mind <<'EOF'
// Simple matrix multiplication example
fn matmul_example(a: tensor<f32[2, 3]>, b: tensor<f32[3, 4]>) -> tensor<f32[2, 4]> {
    return matmul(a, b);
}

fn main() {
    let a: tensor<f32[2, 3]> = [[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]];
    let b: tensor<f32[3, 4]> = [[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0],
                                 [9.0, 10.0, 11.0, 12.0]];
    let result = matmul_example(a, b);
    print("Result:", result);
}
EOF

echo "1. Created MIND source: /tmp/example.mind"
cat /tmp/example.mind
echo

# ============================================================================
# Step 2: MIND → Core IR
# ============================================================================

echo "2. Compiling MIND to Core IR..."
mind compile --emit-ir /tmp/example.mind > /tmp/example.ir
echo "   Generated: /tmp/example.ir"
head -20 /tmp/example.ir
echo

# ============================================================================
# Step 3: Core IR → MLIR
# ============================================================================

echo "3. Lowering Core IR to MLIR..."
mind compile --emit-mlir --features=mlir-lowering /tmp/example.mind > /tmp/example.mlir
echo "   Generated: /tmp/example.mlir"
echo
echo "   MLIR snippet:"
head -30 /tmp/example.mlir
echo

# ============================================================================
# Step 4: MLIR Optimization
# ============================================================================

echo "4. Optimizing MLIR with mlir-opt..."
mlir-opt /tmp/example.mlir \
    --linalg-bufferize \
    --arith-bufferize \
    --tensor-bufferize \
    --func-bufferize \
    --finalizing-bufferize \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    > /tmp/example_opt.mlir

echo "   Generated: /tmp/example_opt.mlir"
echo "   Optimized MLIR snippet:"
head -30 /tmp/example_opt.mlir
echo

# ============================================================================
# Step 5: MLIR → LLVM IR
# ============================================================================

echo "5. Translating MLIR to LLVM IR..."
mlir-translate --mlir-to-llvmir /tmp/example_opt.mlir > /tmp/example.ll

echo "   Generated: /tmp/example.ll"
echo "   LLVM IR snippet:"
head -40 /tmp/example.ll
echo

# ============================================================================
# Step 6: LLVM IR → Object File
# ============================================================================

echo "6. Compiling LLVM IR to object code..."
llc -filetype=obj /tmp/example.ll -o /tmp/example.o

echo "   Generated: /tmp/example.o"
ls -lh /tmp/example.o
echo

# ============================================================================
# Step 7: Link to Executable
# ============================================================================

echo "7. Linking to executable binary..."

# Check for MIND runtime library path
if [ -z "${MIND_RUNTIME_LIB:-}" ]; then
    echo "   Warning: MIND_RUNTIME_LIB not set, using placeholder path"
    echo "   Set MIND_RUNTIME_LIB to link with actual runtime library"
    echo "   Example: export MIND_RUNTIME_LIB=/opt/mind/runtime/lib"
    MIND_RUNTIME_LIB="/path/to/mind/runtime/lib"
fi

if [ ! -d "$MIND_RUNTIME_LIB" ]; then
    echo "   Note: Runtime library directory does not exist: $MIND_RUNTIME_LIB"
    echo "   Skipping link step (would fail without actual runtime)"
    echo "   To complete this step, install MIND runtime and set MIND_RUNTIME_LIB"
else
    clang /tmp/example.o -o /tmp/example_binary \
        -L"$MIND_RUNTIME_LIB" \
        -lmind_runtime \
        -lm

    echo "   Generated: /tmp/example_binary"
    ls -lh /tmp/example_binary
fi
echo

# ============================================================================
# Step 8: Execute
# ============================================================================

echo "8. Running the compiled binary..."
if [ -f /tmp/example_binary ]; then
    /tmp/example_binary
else
    echo "   Binary not created (runtime library not available)"
    echo "   Pipeline demonstration complete through LLVM IR generation"
fi
echo

# ============================================================================
# Summary
# ============================================================================

echo "=== Pipeline Summary ==="
echo "Source:       /tmp/example.mind      (MIND surface language)"
echo "Core IR:      /tmp/example.ir        (Canonical SSA IR)"
echo "MLIR:         /tmp/example.mlir      (MLIR tensor dialect)"
echo "Optimized:    /tmp/example_opt.mlir  (After linalg/bufferization passes)"
echo "LLVM IR:      /tmp/example.ll        (LLVM intermediate representation)"
echo "Object:       /tmp/example.o         (Machine code, unlinked)"
echo "Binary:       /tmp/example_binary    (Executable with runtime)"
echo

echo "=== Key MLIR Optimization Passes ==="
echo "1. linalg-bufferize:     Convert tensors to memrefs (buffers)"
echo "2. convert-linalg-to-loops: Lower linalg ops to explicit loops"
echo "3. convert-scf-to-cf:    Structured control flow → CFG"
echo "4. convert-arith-to-llvm: Arithmetic ops → LLVM intrinsics"
echo "5. convert-func-to-llvm: Function signatures → LLVM calling convention"
echo

echo "=== Performance Notes ==="
echo "- MLIR enables high-level optimizations (fusion, tiling)"
echo "- LLVM provides backend optimizations (vectorization, scheduling)"
echo "- Total compilation time for this example: ~1-2 seconds"
echo "- Binary size: ~10-50 KB (depending on runtime linking)"
echo

echo "Demo complete!"
