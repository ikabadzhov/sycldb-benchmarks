# Benchmark Revalidation Report

Date: 2026-04-02

## Scope

This report compares the benchmark behavior of:

- baseline checkout: `/home/eugenio/sycldb-benchmarks`
- candidate checkout: `/home/eugenio/sycldb-benchmarks/.worktrees/repo-reproducibility-cleanup`

The goal is to document:

- what changed in the implementation on `repo-reproducibility-cleanup`
- what the fresh benchmark campaign measured
- why the results got better, worse, or stayed flat

## Environment

- GPU: `NVIDIA L40S`
- Dataset: `/media/ssb/s100_columnar`
- Repetitions: `10`
- SYCL compiler: `/media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp`
- CUDA compiler: `/usr/local/cuda-12.6/bin/nvcc`

## Commands Run

Baseline checkout:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
python3 scripts/plot_20bars.py
```

Candidate checkout:

```bash
python3 scripts/plot_20bars.py --nvcc /usr/local/cuda-12.6/bin/nvcc
python3 scripts/verify_results.py
```

## Executive Summary

The branch improves reproducibility and correctness, but it is not performance-neutral across the board.

What changed in measured performance:

- Q1.1 is effectively unchanged across all variants
- Q2.1 Hardcoded and Modular are also effectively unchanged
- Q2.1 JIT Fusion regresses noticeably:
  - `q21_sycldb`: `+13.69%`
  - `q21_sycldbtiled`: `+13.71%`
- Q2.1 CUDA (`mordred`) regresses modestly: `+4.71%`

What improved:

- the benchmark runner now resolves dataset/tool paths robustly
- changed sources are rebuilt automatically
- all major variants now print a comparable `Final result`
- the branch adds a correctness verifier, and it passes for both `q11` and `q21`

The short version is: the branch makes the benchmark suite more trustworthy, but the Q2.1 adaptive fixes introduce extra cost.

## Implementation Changes

### 1. Benchmark harness cleanup

Files:

- `scripts/benchmark_config.py`
- `scripts/bench_all.py`
- `scripts/plot_20bars.py`
- `scripts/plot_measured.py`
- `scripts/verify_results.py`
- `Makefile`
- `README.md`

Key changes:

- added centralized benchmark configuration and path discovery
- added explicit CLI flags for dataset, `acpp`, `nvcc`, and repetitions
- changed rebuild logic from "build only if missing" to "rebuild when source is newer"
- split the old plotting flow so measured plots are generated from actual `benchmark_data.json`
- added `verify_results.py` to check that variants agree on the final numeric answer
- documented and exposed the workflow through `Makefile` and updated README guidance

Impact:

- these changes improve reproducibility and usability
- they should not materially affect kernel runtime by themselves
- they do affect confidence in the numbers because stale binaries and hidden path assumptions are no longer silently reused

### 2. Q1.1 adaptive variants: reporting and CLI consistency

Files:

- `src/standard/q11_adaptive.cpp`
- `src/coalesced/q11_adaptive.cpp`
- `src/tiled/q11_adaptive.cpp`

Key changes:

- use runtime `repetitions` instead of fixed loop counts
- accept the dataset path consistently
- print `Final result: ...`

Impact:

- these are instrumentation and interface fixes, not algorithmic changes
- the benchmark confirms that Q1.1 performance is unchanged to within noise

### 3. Q2.1 adaptive variants: correctness-oriented data sizing and bounds fixes

Files:

- `src/standard/q21_adaptive.cpp`
- `src/coalesced/q21_adaptive.cpp`
- `src/tiled/q21_adaptive.cpp`

Key changes:

- replaced hard-coded table sizes with dataset-derived row counts from `get_file_rows(...)`
- resized `d_p_filter` and `d_p_brand` from a fixed oversized domain to `n_part + 1`
- resized `d_s_filter` from a fixed oversized domain to `n_supp + 1`
- changed the date-map range from a compact `2556` range to the full date-key span:
  - old: `D_RANGE = 2556`
  - new: `D_RANGE = 19981231 - 19920101 + 1 = 61131`
- aligned date-key bounds checks with the full date-key domain
- added `Final result: ...`

Impact:

- these are real implementation changes, not just reporting changes
- they make Q2.1 semantically consistent across variants
- they also increase the working-set size of the date lookup table substantially

### 4. Q2.1 modular, hardcoded, and CUDA result reporting

Files:

- `src/standard/q21_modular.cpp`
- `src/coalesced/q21_modular.cpp`
- `src/tiled/q21_modular.cpp`
- `src/standard/q21_hardcoded.cpp`
- `src/coalesced/q21_hardcoded.cpp`
- `src/tiled/q21_hardcoded.cpp`
- `src/cuda/q21_mordred.cu`

Key changes:

- added final aggregation/result printing for comparability
- in CUDA Q2.1, updated date-hash bounds to match the wider valid range

Impact:

- most of these changes are observability changes
- the CUDA bounds change is real logic, and likely explains the smaller `mordred` slowdown

## Correctness Result

The branch passes cross-variant result verification:

```text
Verifying q11...
  ./bin/mod_q11_sycldb -> 44683458181724
  ./bin/adp_q11_sycldb -> 44683458181724
  ./bin/hrd_q11_sycldb -> 44683458181724
  ./bin/mrd_q11 -> 44683458181724
Verifying q21...
  ./bin/mod_q21_sycldb -> 17391888984506
  ./bin/adp_q21_sycldb -> 17391888984506
  ./bin/hrd_q21_sycldb -> 17391888984506
  ./bin/mrd_q21 -> 17391888984506
All selected variants produced matching final results.
```

This is an important improvement over the previous state of the repository: the benchmark results are now tied to implementations that agree on the answer.

## Measured Results

| Query | Variant | Model | Baseline ms | Worktree ms | Delta ms | Delta % |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| q11 | mordred | CUDA | 9.11586 | 9.13934 | +0.02348 | +0.26% |
| q11 | sycldb | Hardcoded | 10.23450 | 10.23200 | -0.00250 | -0.02% |
| q11 | sycldb | JIT Fusion | 7.69239 | 7.69016 | -0.00223 | -0.03% |
| q11 | sycldb | Modular | 18.28810 | 18.26550 | -0.02260 | -0.12% |
| q11 | sycldbcoalesced | Hardcoded | 6.84438 | 6.84307 | -0.00131 | -0.02% |
| q11 | sycldbcoalesced | JIT Fusion | 12.85330 | 12.84160 | -0.01170 | -0.09% |
| q11 | sycldbcoalesced | Modular | 17.04340 | 17.03890 | -0.00450 | -0.03% |
| q11 | sycldbtiled | Hardcoded | 6.02263 | 6.02112 | -0.00151 | -0.03% |
| q11 | sycldbtiled | JIT Fusion | 7.69306 | 7.69196 | -0.00110 | -0.01% |
| q11 | sycldbtiled | Modular | 9.16133 | 9.16671 | +0.00538 | +0.06% |
| q21 | mordred | CUDA | 8.52635 | 8.92781 | +0.40146 | +4.71% |
| q21 | sycldb | Hardcoded | 7.87928 | 7.87694 | -0.00234 | -0.03% |
| q21 | sycldb | JIT Fusion | 6.60706 | 7.51152 | +0.90446 | +13.69% |
| q21 | sycldb | Modular | 15.69290 | 15.69180 | -0.00110 | -0.01% |
| q21 | sycldbcoalesced | Hardcoded | 7.74133 | 7.73824 | -0.00309 | -0.04% |
| q21 | sycldbcoalesced | JIT Fusion | 12.98210 | 13.01620 | +0.03410 | +0.26% |
| q21 | sycldbcoalesced | Modular | 12.44600 | 12.43020 | -0.01580 | -0.13% |
| q21 | sycldbtiled | Hardcoded | 7.72022 | 7.71564 | -0.00458 | -0.06% |
| q21 | sycldbtiled | JIT Fusion | 6.60772 | 7.51350 | +0.90578 | +13.71% |
| q21 | sycldbtiled | Modular | 15.38610 | 15.38510 | -0.00100 | -0.01% |

## Explanation Of The Results

### Why Q1.1 stayed flat

The Q1.1 code changes are mostly benchmark-interface changes:

- runtime repetition count
- consistent dataset-path handling
- final-result printing

Those do not alter the kernel data structures or memory-access pattern, so the near-zero deltas are exactly what should be expected.

### Why Q2.1 JIT Fusion got worse

The Q2.1 adaptive variants changed in a way that directly affects memory behavior.

The main issue is the date map:

- old code used `D_RANGE = 2556`
- new code uses the full key span `61131`

That is about `23.9x` larger.

This matters because Q2.1 uses the date map in the hot path. A larger map means:

- more device memory to initialize
- a larger structure to access during probing/aggregation
- worse cache locality
- more pressure on memory bandwidth and caches

That is consistent with the measured outcome:

- standard JIT Fusion slowed down by `13.69%`
- tiled JIT Fusion slowed down by `13.71%`
- coalesced JIT Fusion barely moved because that variant was already much slower and likely bottlenecked elsewhere

### Why Q2.1 Hardcoded and Modular barely changed

Those implementations also received result-reporting changes, but they did not receive the same level of adaptive data-structure reshaping in their hot execution path.

As a result:

- correctness observability improved
- runtime remained essentially unchanged

### Why CUDA Q2.1 got a little worse

`src/cuda/q21_mordred.cu` also changed its date-hash bounds to cover the wider valid range.

That likely increases the effective lookup footprint and slightly worsens locality, but less dramatically than the adaptive path. The measured regression of `+4.71%` fits that explanation.

## Bottom Line

`repo-reproducibility-cleanup` is a net improvement in benchmark integrity:

- better environment handling
- less stale-binary risk
- explicit correctness verification
- semantically aligned final results across variants

But it is not a free cleanup. The Q2.1 adaptive fixes make the benchmark more correct and reproducible at the cost of slower JIT Fusion performance.

If the goal is trustworthy benchmarking, this branch is better.
If the goal is preserving the old Q2.1 JIT Fusion speed, the branch currently regresses it and that regression is most likely caused by the enlarged and more correct Q2.1 lookup structures.
