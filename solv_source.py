import matplotlib.pyplot as plt

c_vals = [3.3, 3.45, 3.6]

solv_l20 = [2.310e-8, 1.473e-8, 9.155e-9]
solv_l25 = [7.603e-9, 4.272e-9, 2.326e-9]
solv_l30 = [2.503e-9, 1.239e-9, 5.911e-10]

plt.figure(figsize=(8, 6))

plt.plot(c_vals, solv_l20, marker='o', linestyle='-', label=r'$l = 2.0$')
plt.plot(c_vals, solv_l25, marker='s', linestyle='-', label=r'$l = 2.5$')
plt.plot(c_vals, solv_l30, marker='^', linestyle='-', label=r'$l = 3.0$')

plt.xlabel('c', fontsize=12)
plt.ylabel('Solv source amplitude', fontsize=12)
plt.title("Variation of Solv source with parameter 'c'", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('solv_source_variation.png', dpi=300)
