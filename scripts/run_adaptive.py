import os
import subprocess
import re

benchmarks = ["q11_sycldb", "q11_sycldbcoalesced", "q11_sycldbtiled", "q21_sycldb", "q21_sycldbcoalesced", "q21_sycldbtiled"]

results = {}

script_dir = os.path.dirname(os.path.realpath(__file__))
bin_dir = os.path.join(os.path.dirname(script_dir), "bin")

for b in benchmarks:
    print(f"Running {b} (10 internal repetitions)...")
    exe_path = os.path.join(bin_dir, b)
    process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running {b}: {stderr}")
        continue
    # Search for "Avg: X ms"
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
