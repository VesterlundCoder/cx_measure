"""Synthetic checks of C_X: what does it reward?"""
import numpy as np
from coherence import coherence

rng = np.random.default_rng(0)
n, d = 30, 20
origin = np.zeros(d)

rows = []
# A: no covariation (isotropic), sitting at the universal prototype
A = rng.normal(0, 1, (n, d))
# B: no covariation (isotropic), but far from the universal prototype
B = rng.normal(0, 1, (n, d)) + 4.0
# C: strong covariation (one latent factor drives all features), at the prototype
f = rng.normal(0, 1, (n, 1))
Cm = f @ np.ones((1, d)) + 0.3 * rng.normal(0, 1, (n, d))
# D: same strong covariation, far from prototype
D = Cm + 4.0
for name, X in [("A isotropic, near", A), ("B isotropic, far", B),
                ("C correlated, near", Cm), ("D correlated, far", D)]:
    cu = coherence(X, ref=origin)
    co = coherence(X, ref="own")
    rows.append((name, cu["C"], co["C"], cu["dist_share"]))
print(f"{'case':22s} {'C_univ':>7s} {'C_own':>7s} {'dist_share':>10s}")
for r in rows:
    print(f"{r[0]:22s} {r[1]:7.3f} {r[2]:7.3f} {r[3]:10.3f}")
