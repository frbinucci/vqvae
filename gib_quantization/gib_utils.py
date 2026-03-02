import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

# ---------- basic covariance helpers ----------
def center(X):
    return X - X.mean(axis=0, keepdims=True)

def cov(X):
    Xc = center(X)
    return (Xc.T @ Xc) / Xc.shape[0]

def cross_cov(X, Y):
    Xc, Yc = center(X), center(Y)
    return (Xc.T @ Yc) / Xc.shape[0]

def _sym(A):
    return 0.5 * (A + A.T)

def logdet_spd(A):
    # stable log|A| for SPD matrices
    sign, ld = np.linalg.slogdet(_sym(A))
    if sign <= 0:
        raise ValueError("Matrix not SPD (non-positive determinant). Add more jitter.")
    return ld



import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

def gib_fit(X, Y, n_comp, jitter=1e-9, max_dim=None,
            beta_margin=1e-10, numer_tol=None):
    """
    Fit Gaussian Information Bottleneck mapping T = A X + ξ, with ξ~N(0, I).

    Robust version: returns exactly n_comp components (when available),
    avoiding borderline floating point issues around beta_critical.
    """

    Sxx = cov(X)
    Syy = cov(Y)
    Sxy = cross_cov(X, Y)

    Sxx = _sym(Sxx)
    Syy = _sym(Syy)

    nx = Sxx.shape[0]

    cyy = cho_factor(Syy + jitter * np.eye(Syy.shape[0]), lower=True, check_finite=False)
    Syy_inv_Syx = cho_solve(cyy, Sxy.T, check_finite=False)          # Σ_y^{-1} Σ_yx
    Sx_given_y = Sxx - Sxy @ Syy_inv_Syx                             # Σ_{x|y}
    Sx_given_y = _sym(Sx_given_y)

    evals, evecs = eigh(
        Sx_given_y,
        Sxx + jitter * np.eye(nx),
        check_finite=False
    )
    idx = np.argsort(evals)  # ascending λ
    evals, evecs = evals[idx], evecs[:, idx]

    # r_i = v_i^T Sxx v_i  (vectorized)
    r = np.sum(evecs * (Sxx @ evecs), axis=0)

    # beta critical
    beta_critical = 1.0 / np.maximum(1e-15, (1.0 - evals))

    denom_all = evals * r
    activable = (denom_all > 0) & np.isfinite(beta_critical)

    n_avail = int(np.sum(activable))

    if n_comp is None:
        raise ValueError("n_comp must be an integer >= 0")

    n_comp = int(n_comp)
    if n_comp < 0:
        raise ValueError("n_comp must be >= 0")
    if n_comp > n_avail:
        raise ValueError(f"Requested n_comp={n_comp} but only {n_avail} components are activable.")

    # handle zero components
    if n_comp == 0:
        A = np.zeros((0, nx))
        V = np.zeros((nx, 0))
        info = {
            "Sx_given_y": Sx_given_y,
            "eigenvalues": evals,
            "alpha": np.zeros_like(evals),
            "active_dims": 0,
            "beta_critical": beta_critical,
            "beta_used": 0.0,
            "activable_dims": n_avail,
        }
        return A, V, info

    # pick exactly the first n_comp activable dimensions (smallest evals first)
    activable_idx = np.flatnonzero(activable)
    sel = activable_idx[:n_comp]

    # choose beta slightly above critical of the last selected component
    beta0 = beta_critical[sel[-1]]
    beta = beta0 * (1.0 + beta_margin)

    # tolerance for "numer > 0" test (used only for diagnostics/clip)
    if numer_tol is None:
        numer_tol = 100 * np.finfo(float).eps

    # compute alpha only on selected dims
    alpha = np.zeros_like(evals)

    numer_sel = beta * (1.0 - evals[sel]) - 1.0
    denom_sel = evals[sel] * r[sel]

    # clip tiny negatives to zero (floating point noise around threshold)
    numer_sel = np.maximum(numer_sel, 0.0)

    # if numer is *really* zero (beta too close), bump it minimally
    # (rare, but makes output deterministic)
    zeroish = numer_sel <= numer_tol
    if np.any(zeroish):
        numer_sel[zeroish] = numer_tol

    alpha[sel] = np.sqrt(numer_sel / denom_sel)

    # build A, V with exactly n_comp columns/rows
    V = evecs[:, sel]                 # nx x n_comp
    W = np.diag(alpha[sel])           # n_comp x n_comp
    A = W @ V.T                       # n_comp x nx

    # max_dim behaviour: truncate deterministically (keeps first dims)
    if max_dim is not None:
        k = int(max_dim)
        V = V[:, :k]
        A = A[:k, :]
        # keep alpha consistent with truncation
        # (alpha array still full-length, but active dims reflect truncation)
        active_dims = k
    else:
        active_dims = n_comp

    info = {
        "Sx_given_y": Sx_given_y,
        "eigenvalues": evals,
        "alpha": alpha,
        "active_dims": int(active_dims),
        "beta_critical": beta_critical,
        "beta_used": float(beta),
        "activable_dims": int(n_avail),
        "selected_idx": sel,  # utile per debug
    }
    return A, V, info



def gib_mutual_infos(Sxx, Syy, Sxy, A, jitter=1e-9):
    """
    Compute I(X;T) and I(T;Y) for T = A X + ξ, ξ~N(0,I).
    """
    Sxx = _sym(Sxx)
    Syy = _sym(Syy)
    k = A.shape[0]
    if k == 0:
        return 0.0, 0.0

    # Σ_{x|y}
    cyy = cho_factor(Syy + jitter * np.eye(Syy.shape[0]), lower=True, check_finite=False)
    Syy_inv_Syx = cho_solve(cyy, Sxy.T, check_finite=False)
    Sx_given_y = _sym(Sxx - Sxy @ Syy_inv_Syx)

    I_k = np.eye(k)

    St   = _sym(A @ Sxx @ A.T + I_k)              # Σ_t
    St_y = _sym(A @ Sx_given_y @ A.T + I_k)       # Σ_{t|y}

    Ixt = 0.5 * logdet_spd(St)                   # since |Σ_ξ| = |I| = 1
    Ity = 0.5 * (logdet_spd(St) - logdet_spd(St_y))
    return float(Ixt), float(Ity)

# ---------- example usage (from samples) ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n = 5000
    nx, ny = 5, 3

    # make a simple correlated Gaussian model
    X = rng.standard_normal((n, nx))
    B = rng.standard_normal((nx, ny))
    Y = X @ B + 0.5 * rng.standard_normal((n, ny))

    Syy = cov(Y)
    Sxy = cross_cov(X, Y)

    beta = 10.0
    A, info = gib_fit_from_cov(Sxx, Syy, Sxy, beta=beta, jitter=1e-8)

    Ixt, Ity = gib_mutual_infos(Sxx, Syy, Sxy, A)
    print("active dims:", info["active_dims"])
    print("I(X;T) =", Ixt, " nats")
    print("I(T;Y) =", Ity, " nats")

    # apply the encoder: T = A X + noise
    Xc = center(X)
    T = Xc @ A.T + rng.standard_normal((n, A.shape[0]))
    print("T shape:", T.shape)
