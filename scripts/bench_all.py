import json
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import (
    REPO_ROOT,
    build_parser,
    resolve_dataset_path,
    resolve_tool_path,
)


VARIANTS = [
    "q11_sycldb",
    "q11_sycldbcoalesced",
    "q11_sycldbtiled",
    "q21_sycldb",
    "q21_sycldbcoalesced",
    "q21_sycldbtiled",
]

BINARY_PREFIX = {"Modular": "mod", "JIT Fusion": "adp", "Hardcoded": "hrd"}


def needs_rebuild(output_path, source_path):
    if not output_path.exists():
        return True
    return source_path.stat().st_mtime > output_path.stat().st_mtime


def run_bench(cmd):
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error executing {cmd[0]}: {stderr}")
        return None, None

    avg_match = re.search(r"Avg[^:]*:\s*([\d\.]+)\s*ms", stdout)
    dev_match = re.search(r"StdDev:\s*([\d\.]+)\s*ms", stdout)
    if avg_match:
        avg = float(avg_match.group(1))
        stddev = float(dev_match.group(1)) if dev_match else 0.0
        return avg, stddev

    print(f"No match in output for {cmd[0]}:\n{stdout}")
    return None, None


parser = build_parser("Compile and run benchmark variants")
args = parser.parse_args()
dataset_path = resolve_dataset_path(args.dataset)
acpp_path = resolve_tool_path(args.acpp, "SYCLDB_ACPP")
nvcc_path = resolve_tool_path(args.nvcc, "SYCLDB_NVCC")
repetitions = args.repetitions

results = {
    "Q1.1": {"JIT Fusion": [], "Modular": [], "Hardcoded": [], "CUDA": []},
    "Q2.1": {"JIT Fusion": [], "Modular": [], "Hardcoded": [], "CUDA": []},
}
raw_data = []

os.makedirs(REPO_ROOT / "bin", exist_ok=True)
os.makedirs(REPO_ROOT / "results", exist_ok=True)

compile_cmds = []
for b in VARIANTS:
    for prefix, source_dir in (("adp", "adaptive"), ("mod", "modular"), ("hrd", "hardcoded")):
        output = REPO_ROOT / "bin" / f"{prefix}_{b}"
        source = REPO_ROOT / "src" / source_dir / f"{b}.cpp"
        if needs_rebuild(output, source):
            compile_cmds.append(
                [
                    acpp_path,
                    "-O3",
                    "-std=c++20",
                    "--acpp-targets=generic",
                    str(source),
                    "-o",
                    str(output),
                ]
            )

for query in ("q11", "q21"):
    output = REPO_ROOT / "bin" / f"mrd_{query}"
    source = REPO_ROOT / "src" / "cuda" / f"{query}_mordred.cu"
    if needs_rebuild(output, source):
        compile_cmds.append(
            [
                nvcc_path,
                "-O3",
                "-arch=native",
                str(source),
                "-o",
                str(output),
            ]
        )

if compile_cmds:
    print("Compiling missing variants...")
    for cmd in compile_cmds:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

for v in VARIANTS:
    query_key = "Q1.1" if v.startswith("q11") else "Q2.1"
    for model_key, prefix in BINARY_PREFIX.items():
        avg, stddev = run_bench(
            [
                str(REPO_ROOT / "bin" / f"{prefix}_{v}"),
                "-r",
                str(repetitions),
                "-p",
                dataset_path,
            ]
        )
        results[query_key][model_key].append(avg)
        raw_data.append(
            {
                "query": v[:3],
                "variant": v[4:],
                "model": model_key,
                "avg_ms": avg,
                "stddev_ms": stddev,
            }
        )

for query_key, query in (("Q1.1", "q11"), ("Q2.1", "q21")):
    avg, stddev = run_bench(
        [str(REPO_ROOT / "bin" / f"mrd_{query}"), "-r", str(repetitions), "-p", dataset_path]
    )
    results[query_key]["CUDA"].append(avg)
    raw_data.append(
        {
            "query": query,
            "variant": "mordred",
            "model": "CUDA",
            "avg_ms": avg,
            "stddev_ms": stddev,
        }
    )

with open(REPO_ROOT / "results" / "benchmark_data.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=4)

print("\n| Query | Strategy | Hardcoded (ms) | JIT Fusion (ms) | Modular (ms) |")
print("| :--- | :--- | :--- | :--- | :--- |")
for q in ["Q1.1", "Q2.1"]:
    for i, strategy in enumerate(["Standard", "Coalesced", "Tiled"]):
        try:
            h = results[q]["Hardcoded"][i]
            a = results[q]["JIT Fusion"][i]
            m = results[q]["Modular"][i]
            print(f"| {q} | {strategy} | {h:.3f} | {a:.3f} | {m:.3f} |")
        except (TypeError, IndexError):
            print(f"| {q} | {strategy} | N/A | N/A | N/A |")
