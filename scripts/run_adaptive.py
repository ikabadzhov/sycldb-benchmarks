import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_config import build_parser, resolve_dataset_path


benchmarks = [
    "q11_sycldb",
    "q11_sycldbcoalesced",
    "q11_sycldbtiled",
    "q21_sycldb",
    "q21_sycldbcoalesced",
    "q21_sycldbtiled",
]


parser = build_parser("Run adaptive benchmark binaries")
args = parser.parse_args()
dataset_path = resolve_dataset_path(args.dataset)
repetitions = args.repetitions

results = {}

script_dir = os.path.dirname(os.path.realpath(__file__))
bin_dir = os.path.join(os.path.dirname(script_dir), "bin")

for b in benchmarks:
    print(f"Running {b} ({repetitions} internal repetitions)...")
    exe_path = os.path.join(bin_dir, b)
    process = subprocess.Popen(
        [exe_path, "-r", str(repetitions), "-p", dataset_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running {b}: {stderr}")
        continue
    match = re.search(r"Avg:\s*([\d\.]+)\s*ms", stdout)
    if match:
        results[b] = float(match.group(1))
    else:
        print(f"No result found for {b}\nOutput: {stdout}")

print("\n| Benchmark | Average (ms) |")
print("| :--- | :--- |")
for b in benchmarks:
    if b in results:
        print(f"| {b} | {results[b]:.3f} |")
    else:
        print(f"| {b} | N/A |")
