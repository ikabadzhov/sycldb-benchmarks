# SYCL vs Native CUDA Tiled Execution Benchmarks

This repository contains standalone implementations of Star Schema Benchmark (SSB) queries **Q1.1** and **Q2.1**, comparing standard execution methods against explicit execution topologies via both native CUDA and generic SYCL (AdaptiveCpp). 

The goal of this suite is to analyze performance scaling and JIT-compiled assembly behavior under different scheduling instructions (flat threads vs 128-bit vectorization vs manual thread blocking).

## Repository Structure

```
├── README.md
├── src/
│   ├── q11_sycldb.cpp           # Standard flat SYCL array map strategy
│   ├── q11_sycldbcoalesced.cpp  # SYCL 128-bit vectors (sycl::int4) mapping
│   ├── q11_sycldbtiled.cpp      # SYCL Block-striped unrolling with subgroup reductions
│   ├── q11_mordred.cu           # Native CUDA reference (block-striped unrolling)
│   ├── q21_sycldb.cpp           
│   ├── q21_sycldbcoalesced.cpp  
│   ├── q21_sycldbtiled.cpp      
│   └── q21_mordred.cu           
├── scripts/
│   ├── parse.py                 # Parses ncu traces for max-throughput kernel logs
│   ├── parse_ncu.py             # Parses general ncu traces
│   └── plot_roofline.py         # Matplotlib scatter script mapping Nsight Roofline data
└── results/                     # Nsight Compute (ncu) trace outputs & generated plots
    ├── bottleneck_analysis.md   # Final comparative breakdown & compiler claims
    └── roofline_plot.png        # Rendered hardware limit chart (L40S)
```

## Compilation

To compile the SYCL benchmarks using AdaptiveCpp (targeting NVIDIA backends generically via `ptx`):

```bash
acpp -O3 -std=c++20 --acpp-targets=generic src/q11_sycldbtiled.cpp -o bin/q11_sycldbtiled
```

To compile the native CUDA references:
```bash
nvcc -O3 -arch=native src/q11_mordred.cu -o bin/q11_mordred
```

## Execution Options

Each compiled binary accepts flags for iteration counts and table inputs:
- `-r <runs>`: Number of benchmark repetitions to execute to measure average and std-dev. Let convergence warm up.
- `-p <path>`: Root directory containing the SSB SF100 binary columnar arrays (default: `/media/ssb/sf100_columnar`).

**Example:**
```bash
./bin/q21_sycldbtiled -r 10 -p /datasets/ssb/sf100
```

## Key Findings & Core Takeaway

As detailed in `results/bottleneck_analysis.md`, the primary execution bottleneck for SSb queries across the board is **Global Memory Bandwidth**.
The default `sycl::parallel_for` achieves perfect L1 caching but starves waiting for small thread transactions. By upgrading the SYCL implementation to explicitly map `ITEMS_PER_THREAD * BLOCK_THREADS` via `sycl::nd_range` with group-level reductions (`sycldbtiled`), we force optimal register placement across perfectly ordered 128-bit global load bursts. 

This explicit tiling approach allows pure C++ / SYCL to completely eclipse hand-written Native CUDA.
