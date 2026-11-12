"""Conceptual coherence C_X as defined by Osta-Vélez & Gärdenfors (2026, PhiMiSci 7).

Definition (their Sect. 3.3):
  * exemplars of concept X are points x_k in a conceptual space with d dimensions
  * each dimension is multiplied by a salience weight s_j (no standardisation)
  * the covariance is taken around a "universal prototype" u (mean of all objects),
    not around the concept's own mean
  * PCA on that salience-weighted matrix gives eigenvalues l_1 >= ... >= l_d
  * p_i = l_i / sum(l),  C_X = sum_i w_i p_i,  w_i = (1-r) r^(i-1), normalised to sum 1

The matrix around u is a second-moment matrix:
  S_u = Cov_own + m m^T,   m = mean(X) - u
so C_X mixes genuine covariation (Cov_own) with distinctiveness (m). We therefore
also compute the variant centred on the concept's own mean ("own"), which is the
pure covariation version, and the decomposition share ||m||^2 / tr(S_u).
"""
from __future__ import annotations

import numpy as np


def geometric_weights(d: int, r: float) -> np.ndarray:
    w = (1 - r) * r ** np.arange(d)
    return w / w.sum()


def eigvals_second_moment(Z: np.ndarray) -> np.ndarray:
    """Eigenvalues of (1/n) Z^T Z, computed via the n x n Gram matrix (d may be >> n)."""
    n = Z.shape[0]
    G = Z @ Z.T / n
    ev = np.linalg.eigvalsh(G)[::-1]
    return np.clip(ev, 0, None)


def coherence(X: np.ndarray, salience: np.ndarray | None = None,
              ref: np.ndarray | str = "own", r: float = 0.5) -> dict:
    """C_X for exemplar matrix X (n x d).

    ref: 'own' -> centre on concept mean (standard PCA covariance)
         array -> centre on that point (e.g. the universal prototype)
    """
    X = np.asarray(X, float)
    n, d = X.shape
    s = np.ones(d) if salience is None else np.asarray(salience, float)
    mu = X.mean(0)
    u = mu if (isinstance(ref, str) and ref == "own") else np.asarray(ref, float)
    Z = (X - u) * s
    ev = eigvals_second_moment(Z)[: min(n, d)]
    tot = ev.sum()
    if tot <= 0:
        return dict(C=np.nan, p=ev, dist_share=np.nan)
    p = ev / tot
    w = geometric_weights(d, r)[: len(p)]
    m = (mu - u) * s
    return dict(C=float(np.dot(w, p)), p=p, p1=float(p[0]),
                dist_share=float(m @ m / tot), total=float(tot))


def pc_scores(X, salience=None, ref="own", k=1):
    """Exemplar scores on the first k principal axes of the (weighted) second-moment matrix.
    Sign of each axis is oriented so the concept mean projects positively."""
    X = np.asarray(X, float)
    s = np.ones(X.shape[1]) if salience is None else np.asarray(salience, float)
    mu = X.mean(0)
    u = mu if (isinstance(ref, str) and ref == "own") else np.asarray(ref, float)
    Z = (X - u) * s
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:k].T
    m = (mu - u) * s
    sign = np.sign(m @ V)
    sign[sign == 0] = 1
    V = V * sign
    return Z @ V, Z
