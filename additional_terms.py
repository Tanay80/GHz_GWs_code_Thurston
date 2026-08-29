#This script plots the Log10 of those additional terms (\mathcal{D}) that modify the P(k) of anisotropic geometries other than RH2S2 as: 'P(k) - \mathcal{D}'. against Log10[f]

import numpy as np
import matplotlib.pyplot as plt

log_f = np.linspace(3, 10, 500)
f = 10**log_f

D_UH2 = (np.sqrt(3) / (2 * np.pi * f)) * (1e-26 / 1.26)
D_Nil = (np.sqrt(6) / (4 * np.pi * f)) * (1e-26 / 3.911)
D_Solv = (np.sqrt(6) / (2 * np.pi * f)) * (1e-26 / 9.587)

log_D_UH2 = np.log10(np.abs(D_UH2))
log_D_Nil = np.log10(np.abs(D_Nil))
log_D_Solv = np.log10(np.abs(D_Solv))

plt.figure(figsize=(8, 6))
plt.plot(log_f, log_D_UH2, label=r'$\widetilde{U (\mathbb{H}^2)}$ upper limit')
plt.plot(log_f, log_D_Nil, label='Nil')
plt.plot(log_f, log_D_Solv, label='Solv')

plt.xlabel(r'Frequency $\log_{10} \, \text{f}$')
plt.ylabel(r'P(k) perturbations $\log_{10} \left|\mathcal{D} \right|$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('mathcal_D_plot.png', dpi=300)
