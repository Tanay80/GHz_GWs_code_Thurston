#This script plots the Solv geometry's source amplitude for each ('l', 'c') pair.
#Each 'l' value has its own unique set of 'c' values, so three separate figures are generated, one per 'l'.

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

datasets = {
    2.0: [
        (3.30, 2.310e-8, 'green', 'o'),
        (3.45, 1.473e-8, 'blue',  's'),
        (3.60, 9.155e-9, 'red',   '^'),
    ],
    4.0: [
        (2.9, 2.738e-9, 'green', 'o'),
        (3.0, 1.587e-9, 'blue',  's'),
        (3.1, 8.999e-10, 'red',  '^'),
    ],
    6.0: [
        (2.7, 5.415e-10, 'green', 'o'),
        (2.8, 2.492e-10, 'blue',  's'),
        (2.9, 1.111e-10, 'red',   '^'),
    ],
}

for l_val, points in datasets.items():
    plt.figure(figsize=(8, 6))

    c_vals = np.array([pt[0] for pt in points])
    amp_vals = np.array([pt[1] for pt in points])

    c_smooth = np.linspace(c_vals.min(), c_vals.max(), 300)
    spline = make_interp_spline(c_vals, amp_vals, k=2)
    amp_smooth = spline(c_smooth)

    plt.plot(c_smooth, amp_smooth, color='gray', linestyle='-', linewidth=1.5, zorder=1)

    for c_val, amp, color, marker in points:
        plt.plot(c_val, amp, marker=marker, linestyle='none',
                  color=color, markersize=10, label=f'c = {c_val}', zorder=2)

    plt.xlabel('c', fontsize=12)
    plt.ylabel('Solv source amplitude', fontsize=12)
    plt.title(f"Variation of Solv source with 'c' (l = {l_val})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(f'solv_source_variation_l{str(l_val).replace(".", "_")}.png', dpi=300)
    plt.close()
