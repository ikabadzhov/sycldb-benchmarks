# SYCLDB Benchmarks

This repository benchmarks several GPU execution styles for Star Schema Benchmark queries `Q1.1` and `Q2.1`. It compares native CUDA baselines with multiple SYCL implementations, including modular kernels, fused kernels, and AdaptiveCpp JIT-fusion variants.

## Repository Layout

- `src/cuda/`: native CUDA baselines (`mordred`)
- `src/hardcoded/`: fused SYCL implementations
- `src/modular/`: split-kernel SYCL implementations
- `src/adaptive/`: AdaptiveCpp JIT-fusion implementations
- `scripts/`: benchmark drivers, plotting utilities, and verification helpers
- `results/`: measured benchmark outputs, plots, and analysis artifacts
- `outputs/`: raw profiling or benchmark text output

## Prerequisites

- Python 3
- AdaptiveCpp compiler
- CUDA toolkit with `nvcc`
- SSB columnar dataset compatible with the benchmark binaries

## Configuration

The Python tooling now prefers explicit CLI flags and falls back to environment variables.

- `SYCLDB_SSB_PATH`: dataset root for the SSB columnar files
- `SYCLDB_ACPP`: path to the AdaptiveCpp compiler binary
- `SYCLDB_NVCC`: path to `nvcc`
- `-d` / `--device`: SYCL device id to target at runtime for SYCL benchmarks

If `SYCLDB_SSB_PATH` is not set, the scripts try these dataset locations in order:

- `/media/ssb/sf100_columnar`
- `/media/ssb/s100_columnar`

## Common Workflows

### List SYCL devices

```bash
make list-devices
```

This compiles and runs `bin/sycl_ls`, a small `sycl-ls`-style helper that prints stable device ids and names. Use those ids with `-d`, for example `-d 3`.

### Build and benchmark

```bash
make benchmark
```

This runs `python3 scripts/bench_all.py` and writes measured output to `results/benchmark_data.json`.

To run benchmarks on a specific SYCL device:

```bash
python3 scripts/bench_all.py -d 3
```

The `-d` flag is forwarded only to the SYCL benchmarks. CUDA `mordred` binaries still use the CUDA runtime's default device selection.

### Verify final results match across variants

```bash
make verify
```

This runs `python3 scripts/verify_results.py`. It requires benchmark binaries to exist in `bin/`. In a fresh worktree without compiled binaries, the script fails explicitly with a missing-binary error rather than silently passing.

### Generate measured plots

```bash
make plot-measured
```

This generates `results/measured_comparison.png` directly from `results/benchmark_data.json`.

## Direct Script Usage

Each main script supports explicit CLI configuration:

```bash
python3 scripts/bench_all.py --dataset /path/to/ssb --acpp /path/to/acpp --nvcc /path/to/nvcc --repetitions 10 -d 3
python3 scripts/run_adaptive.py --dataset /path/to/ssb --repetitions 10 -d 3
python3 scripts/verify_results.py --dataset /path/to/ssb --queries q11 q21
python3 scripts/plot_measured.py
```

## Plotting Policy

- `scripts/plot_measured.py` is the measured-data plotting entry point.
- `scripts/plot_final.py` is presentation-only and uses synthetic or normalized values. Do not treat it as raw benchmark output.

## Results And Analysis

- `results/benchmark_data.json`: raw per-run timing samples for each benchmark entry
- `results/measured_comparison.png`: measured comparison chart generated from JSON data
- `results/bottleneck_analysis.md`: profiling-based narrative analysis

## Notes

This repository is an experiment and benchmark suite, not a reusable query engine. The kernels intentionally keep loading, execution, and timing close together so that execution-model differences remain easy to inspect.
