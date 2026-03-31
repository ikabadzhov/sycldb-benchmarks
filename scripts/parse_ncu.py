import glob

with open('bottleneck_analysis.md', 'w') as out:
    out.write("# SYCLDB vs Mordred Bottleneck Analysis\n\n")
    for f in sorted(glob.glob('*_ncu.txt')):
        with open(f) as file: content = file.read()
        kernels = content.split("==PROF== Profiling")
        if len(kernels) < 2: continue
        probe_kernel = kernels[-1] # The probe loop is always the last executed kernel
        
        try: name = probe_kernel.split('"')[1]
        except: name = "Unknown Kernel"

        out.write(f"## {f.replace('_ncu.txt', '')}\n")
        out.write(f"**Target Kernel:** `{name}`\n")
        
        metrics = ["Memory Throughput", "Compute (SM) Throughput", "L1/TEX Hit Rate", "L2 Hit Rate", "^    Achieved Occupancy", "Registers Per Thread", "UncoalescedGlobalAccess"]
        
        for m in metrics:
            found = False
            for line in probe_kernel.split('\n'):
                if m.replace('^    ', '') in line and (not m.startswith('^    ') or line.startswith('    Achieved Occup')):
                    out.write(f"- **{m.replace('^    ', '')}**: `{line.strip()}`\n")
                    found = True
                    break
                    
        out.write("\n")
