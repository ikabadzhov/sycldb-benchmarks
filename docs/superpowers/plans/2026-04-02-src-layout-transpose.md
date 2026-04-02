# Source Layout Transpose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the SYCL benchmark sources from model-first directories (`adaptive`, `modular`, `hardcoded`) into access-pattern-first directories (`standard`, `coalesced`, `tiled`) while preserving current behavior, binary names, CUDA layout, verification, and plotting workflows.

**Architecture:** Keep `src/cuda/` and `src/utils/` unchanged. Move every SYCL benchmark source into one of `src/standard/`, `src/coalesced/`, or `src/tiled/`, with the model encoded in the filename (`q11_modular.cpp`, `q11_adaptive.cpp`, `q11_hardcoded.cpp`, etc.). Update the Python tooling to compile and reason from an explicit `(pattern, model, query)` mapping instead of assuming the old directory structure.

**Tech Stack:** C++20 SYCL / AdaptiveCpp, CUDA, Python 3, `unittest`, Make

---

## File Structure

### New source layout

**Create:**
- `src/standard/q11_modular.cpp`
- `src/standard/q11_adaptive.cpp`
- `src/standard/q11_hardcoded.cpp`
- `src/standard/q21_modular.cpp`
- `src/standard/q21_adaptive.cpp`
- `src/standard/q21_hardcoded.cpp`
- `src/coalesced/q11_modular.cpp`
- `src/coalesced/q11_adaptive.cpp`
- `src/coalesced/q11_hardcoded.cpp`
- `src/coalesced/q21_modular.cpp`
- `src/coalesced/q21_adaptive.cpp`
- `src/coalesced/q21_hardcoded.cpp`
- `src/tiled/q11_modular.cpp`
- `src/tiled/q11_adaptive.cpp`
- `src/tiled/q11_hardcoded.cpp`
- `src/tiled/q21_modular.cpp`
- `src/tiled/q21_adaptive.cpp`
- `src/tiled/q21_hardcoded.cpp`

**Delete after migration:**
- `src/adaptive/*.cpp`
- `src/modular/*.cpp`
- `src/hardcoded/*.cpp`

### Tooling files to modify

- `scripts/bench_all.py`
- `scripts/run_adaptive.py`
- `scripts/verify_results.py`
- `scripts/plot_measured.py`
- `README.md`
- `BENCHMARK-REVALIDATION-REPORT.md` if it references the old layout

### Tests

**Modify:**
- `tests/test_benchmark_pipeline.py`

## Task 1: Add failing tests for the transposed layout mapping

**Files:**
- Modify: `tests/test_benchmark_pipeline.py`
- Test: `tests/test_benchmark_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add tests that define the desired mapping in the benchmark driver:

```python
    def test_source_mapping_uses_pattern_directories(self):
        source = bench_all.resolve_sycl_source("q11", "standard", "Modular")
        self.assertEqual(source.as_posix(), "src/standard/q11_modular.cpp")

    def test_binary_name_stays_compatible_with_existing_prefixes(self):
        binary = bench_all.resolve_binary_name("q21", "tiled", "JIT Fusion")
        self.assertEqual(binary, "adp_q21_sycldbtiled")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: FAIL because `resolve_sycl_source()` and `resolve_binary_name()` do not exist yet.

- [ ] **Step 3: Write minimal implementation in the benchmark driver**

Add the new layout constants and helper API to `scripts/bench_all.py`:

```python
PATTERN_SUFFIX = {
    "standard": "sycldb",
    "coalesced": "sycldbcoalesced",
    "tiled": "sycldbtiled",
}

MODEL_PREFIX = {
    "Modular": "mod",
    "JIT Fusion": "adp",
    "Hardcoded": "hrd",
}

MODEL_FILENAME = {
    "Modular": "modular",
    "JIT Fusion": "adaptive",
    "Hardcoded": "hardcoded",
}

def resolve_sycl_source(query: str, pattern: str, model: str) -> Path:
    return REPO_ROOT / "src" / pattern / f"{query}_{MODEL_FILENAME[model]}.cpp"

def resolve_binary_name(query: str, pattern: str, model: str) -> str:
    return f"{MODEL_PREFIX[model]}_{query}_{PATTERN_SUFFIX[pattern]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: PASS for the new mapping tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark_pipeline.py scripts/bench_all.py
git commit -m "test: define transposed source layout mapping"
```

## Task 2: Move the SYCL sources into pattern-first directories

**Files:**
- Create: `src/standard/*.cpp`, `src/coalesced/*.cpp`, `src/tiled/*.cpp`
- Delete: `src/adaptive/*.cpp`, `src/modular/*.cpp`, `src/hardcoded/*.cpp`

- [ ] **Step 1: Copy or move files into the new directories**

Create the new tree and place each source in the matching pattern directory:

```bash
mkdir -p src/standard src/coalesced src/tiled
mv src/modular/q11_sycldb.cpp src/standard/q11_modular.cpp
mv src/adaptive/q11_sycldb.cpp src/standard/q11_adaptive.cpp
mv src/hardcoded/q11_sycldb.cpp src/standard/q11_hardcoded.cpp
```

Repeat the same mapping for:

- `q21_sycldb.cpp`
- `q11_sycldbcoalesced.cpp`
- `q21_sycldbcoalesced.cpp`
- `q11_sycldbtiled.cpp`
- `q21_sycldbtiled.cpp`

- [ ] **Step 2: Confirm no source content changed during the move**

Run:

```bash
git diff --stat -- src
```

Expected: file renames/moves only at this step, no content edits.

- [ ] **Step 3: Commit**

```bash
git add src
git commit -m "refactor: transpose SYCL source layout by access pattern"
```

## Task 3: Update compile/discovery logic to use the new layout

**Files:**
- Modify: `scripts/bench_all.py`
- Modify: `scripts/run_adaptive.py`
- Modify: `scripts/verify_results.py`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_benchmark_pipeline.py` with a compile-target test:

```python
    def test_build_compile_commands_read_new_layout(self):
        source = bench_all.resolve_sycl_source("q21", "coalesced", "Hardcoded")
        self.assertEqual(source.as_posix(), "src/coalesced/q21_hardcoded.cpp")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: FAIL if `resolve_sycl_source()` still uses the old directories or absolute-path assumptions.

- [ ] **Step 3: Implement the new pattern/model matrix**

Replace the flat `VARIANTS` list in `scripts/bench_all.py` with explicit pattern/query/model iteration:

```python
PATTERNS = ["standard", "coalesced", "tiled"]
QUERIES = ["q11", "q21"]
MODELS = ["Modular", "JIT Fusion", "Hardcoded"]

for query in QUERIES:
    for pattern in PATTERNS:
        for model in MODELS:
            source = resolve_sycl_source(query, pattern, model)
            binary = REPO_ROOT / "bin" / resolve_binary_name(query, pattern, model)
```

Update `scripts/run_adaptive.py` to run:

```python
benchmarks = [
    "adp_q11_sycldb",
    "adp_q11_sycldbcoalesced",
    "adp_q11_sycldbtiled",
    "adp_q21_sycldb",
    "adp_q21_sycldbcoalesced",
    "adp_q21_sycldbtiled",
]
```

Update `scripts/verify_results.py` only if any path assumptions reference the source layout instead of the binary naming contract.

- [ ] **Step 4: Run tests to verify the mapping passes**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: PASS.

- [ ] **Step 5: Compile the moved source files**

Run:

```bash
python3 scripts/bench_all.py --nvcc /usr/local/cuda-12.6/bin/nvcc --repetitions 1
```

Expected: all binaries compile from the new `src/{standard,coalesced,tiled}` directories.

- [ ] **Step 6: Commit**

```bash
git add scripts tests results/benchmark_data.json
git commit -m "refactor: compile benchmarks from pattern-first layout"
```

## Task 4: Update plotting and result labeling to stay stable

**Files:**
- Modify: `scripts/plot_measured.py`
- Modify: `scripts/bench_all.py`

- [ ] **Step 1: Write the failing test**

Add a test that asserts the JSON `variant` values remain `sycldb`, `sycldbcoalesced`, and `sycldbtiled` despite the directory transpose:

```python
    def test_variant_labels_remain_compatible(self):
        self.assertEqual(bench_all.PATTERN_SUFFIX["standard"], "sycldb")
        self.assertEqual(bench_all.PATTERN_SUFFIX["coalesced"], "sycldbcoalesced")
        self.assertEqual(bench_all.PATTERN_SUFFIX["tiled"], "sycldbtiled")
```

- [ ] **Step 2: Run test to verify it fails if labels drift**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: FAIL if the benchmark script has begun writing raw directory names instead of the established labels.

- [ ] **Step 3: Keep the public result schema stable**

Ensure `bench_all.py` still writes:

```python
raw_data.append(
    {
        "query": query,
        "variant": PATTERN_SUFFIX[pattern],
        "model": model_key,
        "times_ms": times_ms or [],
    }
)
```

Keep `plot_measured.py` consuming:

```python
VARIANTS = ["sycldb", "sycldbcoalesced", "sycldbtiled"]
VARIANT_LABELS = ["Standard", "Coalesced", "Tiled"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline -v
```

Expected: PASS.

- [ ] **Step 5: Regenerate the measured plot**

Run:

```bash
python3 scripts/plot_measured.py
```

Expected: `results/measured_comparison.png` regenerates without code changes outside labeling compatibility.

- [ ] **Step 6: Commit**

```bash
git add scripts tests results/measured_comparison.png
git commit -m "refactor: preserve benchmark result labels after layout transpose"
```

## Task 5: Update documentation to describe the new tree

**Files:**
- Modify: `README.md`
- Modify: `BENCHMARK-REVALIDATION-REPORT.md` if it names old source directories

- [ ] **Step 1: Update the repository layout section**

Change the README source bullets to:

```markdown
- `src/cuda/`: native CUDA baselines (`mordred`)
- `src/standard/`: standard memory-access pattern with `q{11,21}_{modular,adaptive,hardcoded}.cpp`
- `src/coalesced/`: coalesced memory-access pattern with `q{11,21}_{modular,adaptive,hardcoded}.cpp`
- `src/tiled/`: tiled memory-access pattern with `q{11,21}_{modular,adaptive,hardcoded}.cpp`
- `src/utils/`: shared SYCL device utilities
```

- [ ] **Step 2: Update narrative references to the old model-first layout**

Search and fix references:

```bash
rg -n "src/adaptive|src/modular|src/hardcoded|adaptive/|modular/|hardcoded/" README.md BENCHMARK-REVALIDATION-REPORT.md scripts
```

Expected: only intentional references remain, such as historical discussion in a report.

- [ ] **Step 3: Commit**

```bash
git add README.md BENCHMARK-REVALIDATION-REPORT.md
git commit -m "docs: describe pattern-first source layout"
```

## Task 6: Final verification on the transposed tree

**Files:**
- Verify: `src/`
- Verify: `scripts/`
- Verify: `tests/`

- [ ] **Step 1: Run unit tests**

Run:

```bash
python3 -m unittest tests.test_benchmark_pipeline
```

Expected: PASS.

- [ ] **Step 2: Run device listing**

Run:

```bash
make list-devices
```

Expected: a numbered list of SYCL devices.

- [ ] **Step 3: Run a one-repetition benchmark campaign**

Run:

```bash
python3 scripts/bench_all.py --nvcc /usr/local/cuda-12.6/bin/nvcc --repetitions 1
```

Expected: all binaries compile and execute from the new layout, and `results/benchmark_data.json` is regenerated.

- [ ] **Step 4: Run correctness verification**

Run:

```bash
python3 scripts/verify_results.py
```

Expected:

```text
All selected variants produced matching final results.
```

- [ ] **Step 5: Inspect the final tree**

Run:

```bash
find src -maxdepth 2 -type f | sort
```

Expected: SYCL sources only under `src/standard`, `src/coalesced`, `src/tiled`, with CUDA still under `src/cuda` and utilities under `src/utils`.

- [ ] **Step 6: Commit**

```bash
git add src scripts tests README.md BENCHMARK-REVALIDATION-REPORT.md results
git commit -m "refactor: transpose benchmark source layout by memory pattern"
```

## Self-Review

- Spec coverage: the plan covers source moves, compile logic, run/verify scripts, plotting compatibility, and documentation.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; every task names exact files and commands.
- Type consistency: the plan consistently uses `pattern in {standard, coalesced, tiled}` and model filenames in `{modular, adaptive, hardcoded}` while preserving public binary/result labels.
