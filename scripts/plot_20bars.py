import subprocess
import re
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
repetitions = 10
ssb_path = "/media/ssb/s100_columnar"
acpp_path = "/media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp"

def run_bench(cmd):
    if "-p" not in cmd and "./bin/q" not in cmd[0]:
        cmd.extend(["-p", ssb_path])
        
    print(f"  -> Running: {' '.join(cmd)}... ", end="", flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"FAILED!\nError: {stderr}")
        return None, None
    
    # Improved regex to capture both Avg and StdDev
    # Matches "Avg: 10.2 ms, StdDev: 0.005 ms" or similar formats
    avg_match = re.search(r"Avg[^:]*:\s*([\d\.]+)\s*ms", stdout)
    dev_match = re.search(r"StdDev:\s*([\d\.]+)\s*ms", stdout)
    
    if avg_match:
        avg = float(avg_match.group(1))
        dev = float(dev_match.group(1)) if dev_match else 0.0
        print(f"DONE ({avg:.3f} +/- {dev:.3f} ms)")
        return avg, dev
    
    print("DONE (Could not parse time)")
    return None, None

results = {
    "Modular": {"avg": [], "dev": []},
    "JIT Fusion": {"avg": [], "dev": []},
    "Hardcoded": {"avg": [], "dev": []},
    "Mordred": {"avg": [], "dev": []}
}

raw_data = []

os.makedirs("bin", exist_ok=True)
variants = ["q11_sycldb", "q11_sycldbcoalesced", "q11_sycldbtiled", "q21_sycldb", "q21_sycldbcoalesced", "q21_sycldbtiled"]

print("--- Phase 1: Compilation ---")
# Skip if already exists? No, force re-compile if we updated source for StdDev
for v in variants:
    print(f"Compiling {v} (Modular, JIT Fusion, Hardcoded)...")
    subprocess.run([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", f"src/modular/{v}.cpp", "-o", f"bin/mod_{v}"], check=True)
    subprocess.run([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", f"src/adaptive/{v}.cpp", "-o", f"bin/adp_{v}"], check=True)
    subprocess.run([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", f"src/hardcoded/{v}.cpp", "-o", f"bin/hrd_{v}"], check=True)

subprocess.run(["nvcc", "-O3", "-arch=native", "src/cuda/q11_mordred.cu", "-o", "bin/mrd_q11"], check=True)
subprocess.run(["nvcc", "-O3", "-arch=native", "src/cuda/q21_mordred.cu", "-o", "bin/mrd_q21"], check=True)

print("\n--- Phase 2: Benchmarking (SF100) ---")
for v in variants:
    for model_key in ["Modular", "JIT Fusion", "Hardcoded"]:
        p = "mod" if model_key == "Modular" else ("adp" if model_key == "JIT Fusion" else "hrd")
        t, d = run_bench([f"./bin/{p}_{v}", "-r", str(repetitions)])
        results[model_key]["avg"].append(t)
        results[model_key]["dev"].append(d)
        raw_data.append({"query": v[:3], "variant": v[4:], "model": model_key, "avg_ms": t, "stddev_ms": d})

# Mordred
for q in ["q11", "q21"]:
    t, d = run_bench([f"./bin/mrd_{q}", "-r", str(repetitions)])
    results["Mordred"]["avg"].append(t)
    results["Mordred"]["dev"].append(d)
    raw_data.append({"query": q, "variant": "mordred", "model": "CUDA", "avg_ms": t, "stddev_ms": d})

with open("results/benchmark_data.json", "w") as f:
    json.dump(raw_data, f, indent=4)

print("\n--- Phase 3: Plotting ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
strat_labels = ["Standard", "Coalesced", "Tiled"]
colors = {'Modular': '#e15759', 'JIT Fusion': '#4e79a7', 'Hardcoded': '#76b7b2', 'Mordred': '#edc948'}

def plot_query(ax, q_idx_start, title):
    model_keys = ['Modular', 'JIT Fusion', 'Hardcoded']
    for i, model in enumerate(model_keys):
        avgs = [v if v is not None else 0 for v in results[model]["avg"][q_idx_start : q_idx_start+3]]
        devs = [v if v is not None else 0 for v in results[model]["dev"][q_idx_start : q_idx_start+3]]
        pos = np.arange(3) + (i - 1) * 0.25
        ax.bar(pos, avgs, 0.25, yerr=devs, label=model, color=colors[model], edgecolor='black', alpha=0.9, capsize=4)
    
    m_idx = 0 if q_idx_start == 0 else 1
    m_avg = results['Mordred']['avg'][m_idx]
    m_dev = results['Mordred']['dev'][m_idx]
    if m_avg:
        ax.bar(3, m_avg, 0.5, yerr=m_dev, label='Mordred (CUDA)', color=colors['Mordred'], edgecolor='black', alpha=0.9, capsize=6)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(strat_labels + ["Mordred"], fontsize=13)
    ax.set_ylabel("Execution Time (ms)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plot_query(ax1, 0, "Query 1.1 Comparison")
plot_query(ax2, 3, "Query 2.1 Comparison")

plt.suptitle("Architecture Comparison with Execution Variability (SF100)", fontsize=24, fontweight='bold', y=1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig("results/final_comparison_errorbars.png", bbox_inches='tight', dpi=300)
print("Plot saved to results/final_comparison_errorbars.png")
