#A simple script to calculate EM field tensors and F^2 term used by me in the calculations

import sympy as sp

t, x, y, z = sp.symbols('t x y z')

a = sp.Function('a')(t)
b = sp.Function('b')(t)
k = sp.Function('k')(t)

E1, E2, E3, B1, B2, B3 = sp.symbols('E1 E2 E3 B1 B2 B3', real=True)

F_cov = sp.Matrix([
    [0, E1, E2, E3],
    [-E1, 0, -B3, B2],
    [-E2, B3, 0, -B1],
    [-E3, -B2, B1, 0]
])

#S = sp.sinh(chi * sp.sqrt(-k)) / sp.sqrt(-k)

g_cov = sp.Matrix([
    [-1, 0, 0, 0],
    [0, (a**2) * sp.exp(2 * z * sp.sqrt(-k)), 0, 0],
    [0, 0, (a**2) * sp.exp(-2 * z * sp.sqrt(-k)), 0],
    [0, 0, 0, b**2]
])

g_inv = g_cov.inv()
F_con = sp.simplify(g_inv * F_cov * g_inv.T)
invariant = sp.simplify(sum(F_cov.multiply_elementwise(F_con)))

print("\nContravariant EM field tensor F^{mu nu}:")
sp.pprint(F_con)

print("\nInvariant F_{mu nu} F^{mu nu}:")
sp.pprint(invariant)
