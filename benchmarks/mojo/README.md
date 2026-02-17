# Mojo Compilation Benchmarks

This directory contains equivalent Mojo programs for comparing compilation performance with MIND.

## Quick Start

### Prerequisites

1. **Install Mojo SDK**:
   ```bash
   curl https://get.modular.com | sh -
   modular auth <your-auth-key>
   modular install mojo
   ```

2. **Add Mojo to PATH**:
   ```bash
   export MODULAR_HOME="$HOME/.modular"
   export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"
   ```

3. **Verify installation**:
   ```bash
   mojo --version
   ```

### Run Benchmarks

```bash
# Make executable
chmod +x benchmark_mojo_compilation.py

# Run benchmarks
python3 benchmark_mojo_compilation.py
```

## Benchmark Programs

All programs are equivalent to MIND's `benches/simple_benchmarks.rs`:

| File | MIND Equivalent | Description |
|------|-----------------|-------------|
| `scalar_math.mojo` | `scalar_math` | Simple arithmetic: `1 + 2 * 3 - 4 / 2` |
| `small_matmul.mojo` | `small_matmul` | Matrix mult: `[10,20] × [20,30]` |
| `medium_matmul.mojo` | `medium_matmul` | Matrix mult: `[128,256] × [256,512]` |
| `large_matmul.mojo` | `large_matmul` | Matrix mult: `[512,1024] × [1024,512]` |

## MIND v0.2.1 Results (Baseline)

From Criterion in-process benchmarks (February 2026):

```
Benchmark         MIND Compilation Time
─────────────────────────────────────────
scalar_math       1.77 µs
small_matmul      2.95 µs
medium_matmul     2.95 µs
large_matmul      2.95 µs
```

## Interpreting Results

The benchmark script (`benchmark_mojo_compilation.py`) will:

1. **Compile each Mojo program** using `mojo build`
2. **Measure compilation time** (not execution time)
3. **Run 20 samples** with 3 warmup runs
4. **Compare with MIND** baseline results
5. **Calculate speedup ratio**:
   - Speedup < 1.0: MIND is faster
   - Speedup > 1.0: Mojo is faster
   - Speedup = 1.0: Equal performance

### Verified Output (February 2026)

```
COMPILATION TIME COMPARISON: MIND v0.2.1 vs Mojo 0.26.1
================================================================================

Benchmark            MIND (µs)       Mojo (ms)       Ratio
--------------------------------------------------------------------------------
scalar_math          1.77            810             458,000×
matmul               2.95            827             280,000×
mlp                  6.15            829             135,000×
================================================================================
```

## Important Notes

### What We're Measuring

- ✅ **Compilation time**: Parse → Type-check → Codegen → Binary
- ❌ **Not execution time**: We're not running the compiled programs

This matches MIND's benchmarks which measure:
```
Parse → Type-check → IR lowering → Verification
```

### Fair Comparison Considerations

1. **Scope Difference**:
   - MIND: Source → IR (no LLVM backend in open-core)
   - Mojo: Source → Optimized binary (full LLVM pipeline)

2. **If Mojo is slower**: This is expected! Mojo does more work:
   - LLVM optimization passes
   - Binary generation
   - Potential JIT warmup

3. **If Mojo is faster**: Consider:
   - Mojo may cache compilation artifacts
   - Different scope of "compilation"
   - JIT vs AOT differences

### For Honest Comparison

To be fair to both:

**MIND should measure**:
```bash
cargo bench --features mlir-lowering --bench compiler
```
This includes MLIR lowering (closer to Mojo's scope).

**Mojo should measure**:
- Just front-end compilation (if possible to isolate)
- Or compare full MIND compilation including MLIR → LLVM

## Troubleshooting

### Mojo not found

```bash
# Check PATH
echo $PATH | grep modular

# Add to PATH
export MODULAR_HOME="$HOME/.modular"
export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"

# Verify
mojo --version
```

### Compilation errors

If Mojo programs fail to compile, check:
1. Mojo SDK version (update if old)
2. Syntax changes in latest Mojo (API is evolving)
3. Missing imports (Tensor, TensorShape, etc.)

### Permission denied

```bash
chmod +x benchmark_mojo_compilation.py
```

## Extending Benchmarks

To add more benchmarks:

1. **Create `.mojo` file** with equivalent program
2. **Add to `BENCHMARKS` dict** in `benchmark_mojo_compilation.py`
3. **Add MIND baseline** to `compare_with_mind()` function
4. **Run benchmarks**

## Results

After running, results are saved to:
- `mojo_results.json` - Raw timing data
- Console output - Comparison table

Share results by:
1. Opening a GitHub issue
2. Including system specs (CPU, OS, Mojo version)
3. Attaching `mojo_results.json`

## Context: Why This Matters

Verified February 2026 results:
> "MIND frontend is 135,000-458,000× faster than Mojo 0.26.1 full compilation"

**Scope note**: MIND measures frontend only (parse + typecheck + IR). Mojo `mojo build` performs full LLVM compilation to native binary. Different scopes of work.

**Goal**: Honest, reproducible comparison to inform technical due diligence.

---

**Status**: Verified (Mojo 0.26.1, February 2026)
**Maintained by**: MIND benchmarking team
**Last updated**: 2026-02-17
