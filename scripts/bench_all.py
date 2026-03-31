import subprocess
import re
import os
import numpy as np

# Configuration
repetitions = 10
ssb_path = "/media/ssb/s100_columnar"
acpp_path = "/media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp"

def run_bench(cmd):
    # Ensure ssb_path is passed if binary supports it
    if "-p" not in cmd and "./bin/q" not in cmd[0]: # adaptive binaries don't use -p in this version
        cmd.extend(["-p", ssb_path])
        
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error executing {cmd[0]}: {stderr}")
        return None
    
    # Improved regex to handle cases like "Avg execution time over 10 repetitions: 10.2296 ms"
    # It looks for "Avg", followed by anything that isn't a colon, then a colon, whitespace, and the number.
    match = re.search(r"Avg[^:]*:\s*([\d\.]+)\s*ms", stdout)
    if match:
        return float(match.group(1))
    
    # Fallback for adaptive output format if different
    match = re.search(r"Avg:\s*([\d\.]+)\s*ms", stdout)
    if match:
        return float(match.group(1))
        
    print(f"No match in output for {cmd[0]}:\n{stdout}")
    return None

results = {
    'Q1.1': {'Adaptive': [], 'Modular': [], 'Hardcoded': []},
    'Q2.1': {'Adaptive': [], 'Modular': [], 'Hardcoded': []}
}

# 1. Compile everything (Skip if already built to save time)
os.makedirs("bin", exist_ok=True)
compile_cmds = []
for b in ["q11_sycldb", "q11_sycldbcoalesced", "q11_sycldbtiled", "q21_sycldb", "q21_sycldbcoalesced", "q21_sycldbtiled"]:
    if not os.path.exists(f"bin/{b}"):
        compile_cmds.append([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", f"src/adaptive/{b}.cpp", "-o", f"bin/{b}"])
for b in ["q11_sycldb", "q11_sycldbcoalesced", "q11_sycldbtiled", "q21_sycldb", "q21_sycldbcoalesced", "q21_sycldbtiled"]:
    if not os.path.exists(f"bin/mod_{b}"):
        compile_cmds.append([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", f"src/modular/{b}.cpp", "-o", f"bin/mod_{b}"])
if not os.path.exists("bin/hard_q11"):
    compile_cmds.append([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", "../q11_hardcoded.cpp", "-o", "bin/hard_q11"])
if not os.path.exists("bin/hard_q21"):
    compile_cmds.append([acpp_path, "-O3", "-std=c++20", "--acpp-targets=generic", "../q21_hardcoded.cpp", "-o", "bin/hard_q21"])

if compile_cmds:
    print("Compiling missing variants...")
    for cmd in compile_cmds:
        subprocess.run(cmd, check=True)

# 2. Run Benchmarks
# Adaptive (Standard, Coalesced, Tiled)
for b in ["q11_sycldb", "q11_sycldbcoalesced", "q11_sycldbtiled"]:
    t = run_bench([f"./bin/{b}"])
    if t: results['Q1.1']['Adaptive'].append(t)

for b in ["q21_sycldb", "q21_sycldbcoalesced", "q21_sycldbtiled"]:
    t = run_bench([f"./bin/{b}"])
    if t: results['Q2.1']['Adaptive'].append(t)

# Modular (Standard, Coalesced, Tiled)
for b in ["mod_q11_sycldb", "mod_q11_sycldbcoalesced", "mod_q11_sycldbtiled"]:
    t = run_bench([f"./bin/{b}", "-r", str(repetitions)])
    if t: results['Q1.1']['Modular'].append(t)

for b in ["mod_q21_sycldb", "mod_q21_sycldbcoalesced", "mod_q21_sycldbtiled"]:
    t = run_bench([f"./bin/{b}", "-r", str(repetitions)])
    if t: results['Q2.1']['Modular'].append(t)

# Hardcoded
t = run_bench(["./bin/hard_q11", "-r", str(repetitions)])
if t: results['Q1.1']['Hardcoded'] = [t] * 3
t = run_bench(["./bin/hard_q21", "-r", str(repetitions)])
if t: results['Q2.1']['Hardcoded'] = [t] * 3

# Printing Table
print("\n| Query | Strategy | Hardcoded (ms) | Adaptive Fused (ms) | Modular (ms) |")
print("| :--- | :--- | :--- | :--- | :--- |")
for q in ['Q1.1', 'Q2.1']:
    for i, s in enumerate(['Standard', 'Coalesced', 'Tiled']):
        try:
            h = results[q]['Hardcoded'][i]
            a = results[q]['Adaptive'][i]
            m = results[q]['Modular'][i]
            print(f"| {q} | {s} | {h:.3f} | {a:.3f} | {m:.3f} |")
        except:
            print(f"| {q} | {s} | N/A | N/A | N/A |")
