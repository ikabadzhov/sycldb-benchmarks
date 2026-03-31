import matplotlib.pyplot as plt
import numpy as np

# Hardware specs for NVIDIA L40S
PEAK_BW_GBPS = 864.0        # GB/s
PEAK_COMPUTE_GINST = 91600.0 # Giga Integer Ops/sec (91.6 TINT32/s)
RIDGE_POINT = PEAK_COMPUTE_GINST / PEAK_BW_GBPS

# Data extracted from NCU: (Mem %, Compute %, Label, Color, Marker)
data = [
    (97.96, 21.56, 'q11_mordred', 'red', 'o'),
    (0.44,  0.0,   'q11_sycldbcoalesced (Setup)', 'gray', 'x'),
    (92.30, 15.00, 'q11_sycldbtiled', 'blue', 'o'),
    (45.40, 10.10, 'q11_sycldb', 'green', 'o'),
    (31.62, 4.50,  'q21_mordred', 'red', 's'),
    (88.54, 11.30, 'q21_sycldbcoalesced', 'gray', 's'),
    (91.41, 12.33, 'q21_sycldbtiled', 'blue', 's'),
    (75.00, 10.00, 'q21_sycldb', 'green', 's'),
]

# Filter out setup kernels with ~0 compute
data = [d for d in data if d[1] > 0.0]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Roofline
x_vals = np.logspace(0, 3, 500)
# Mem Bound Performance: BW * Arithmetic Intensity
y_mem_bound = x_vals * PEAK_BW_GBPS
# Compute Bound Performance: Peak Compute
y_compute_bound = np.full_like(x_vals, PEAK_COMPUTE_GINST)
# Actual Roof (minimum of the two)
y_roof = np.minimum(y_mem_bound, y_compute_bound)

ax.plot(x_vals, y_roof, color='black', linewidth=2, label='L40S Hardware Roof')
ax.plot(x_vals, y_mem_bound, color='gray', linestyle='--', alpha=0.5)

# Plot Points
for mem_pct, comp_pct, label, color, marker in data:
    achieved_bw = (mem_pct / 100.0) * PEAK_BW_GBPS
    achieved_compute = (comp_pct / 100.0) * PEAK_COMPUTE_GINST
    ai = achieved_compute / achieved_bw if achieved_bw > 0 else 0
    
    ax.scatter(ai, achieved_compute, color=color, marker=marker, s=100, label=label, zorder=5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 1000)
ax.set_ylim(100, PEAK_COMPUTE_GINST * 1.5)

ax.set_xlabel('Arithmetic Intensity (Integer Ops / Byte)')
ax.set_ylabel('Performance (Giga-Ops / Second)')
ax.set_title('Roofline Analysis (NVIDIA L40S) - SYCLDB vs Mordred (SF100)')
ax.grid(True, which="both", ls="--", alpha=0.5)

# Add Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.0, 0.0))

plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=300)
print('Saved roofline_plot.png')
