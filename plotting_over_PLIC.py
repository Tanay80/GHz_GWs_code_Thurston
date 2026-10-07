#This script plots the reference PLICs contours only

import numpy as np
import matplotlib.pyplot as plt

ns = {}
src = open("mhz.py").read().split("target_models = [")[0]
exec(src, ns)
calculate_Y = ns["calculate_Y"]
c_values    = ns["c_values"]
colors      = ns["colors"]
HI, MPl     = ns["HI"], ns["MPl"]

refdir = "data/"
COL    = 1
YTOP   = 1e-4
YMIN   = 1e-18
f_array = ns["f_array"]

det = {
    "SKA":      ("#483d8b",   (1.5e-9, 6e-16)),
    "IPTA":     ("royalblue", (1.5e-9, 1.2e-13)),
    "PPTA":     ("seagreen",  (8e-9,   3e-12)),
    "EPTA":     ("sienna",    (1.5e-8, 2e-11)),
    "NANOGrav": ("darkorange",(3e-8,   2e-10)),
    "LISA":     ("teal",      (1e-5,   3e-12)),
    "DECIGO":   ("purple",    (3e-2,   1e-15)),
    "BBO":      ("darkred",   (1e-1,   1e-16)),
    "ET":       ("firebrick", (1e1,    1e-14)),
    "CE":       ("c",         (5e1,    1e-15)),
    "HLV":      ("slategray", (1e0,    1e-8)),
}

fig, ax = plt.subplots(figsize=(11, 6.5))

for name, (c, (lx, ly)) in det.items():
    d = np.loadtxt(f"{refdir}plis_{name}.dat")
    f, om = 10**d[:, 0], 10**d[:, COL]
    ax.loglog(f, om, color=c, lw=1.2, zorder=2)
    ax.fill_between(f, om, YTOP, color=c, alpha=0.12, lw=0, zorder=1)
    ax.text(lx, ly, name, color=c, fontsize=8, fontweight="bold")

ybbn = 1.2e-6
ax.axhline(ybbn, color="k", ls="--", lw=1.5, zorder=2)
ax.axhspan(ybbn, YTOP, color="gray", alpha=0.2, zorder=1)
ax.text(1.2e-10, 1.6e-6, "BBN Bound", fontweight="bold")

ax.axvspan(3e8, 3e10, color="gold", alpha=0.15, zorder=1)
ax.text(1.5e9, 3e-16, "UHF-GW Regime\n(Microwave Cavities)", ha="center",
        color="darkgoldenrod", fontweight="bold")

for c_val, color in zip(c_values, colors):
    Y = calculate_Y("RH2S2", c_val, f_array)
    m = Y > 0
    ax.plot(f_array[m], Y[m], color=color, lw=2, label=f"c = {c_val}", zorder=10)

h0, OmR, g1, g2 = 0.7, 1e-5, 106.75, 106.75
Y_pref = (1/24)*h0*h0*OmR*(g1/3.363)*(3.909/g2)**(4/3)
Y_vac  = Y_pref * 2*(HI/(np.pi*MPl))**2
ax.axhline(Y_vac, color="gold", ls="--", lw=2, label="Pure tensor vacuum", zorder=9)

ax.set_xlim(1e-10, 1e12)
ax.set_ylim(YMIN, 1e-5)
ax.set_xlabel("f [Hz]")
ax.set_ylabel(r"$h_0^2 \Omega_{GW}$")
ax.grid(alpha=0.2)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.savefig("overlay_l=2.0.png", dpi=300)
