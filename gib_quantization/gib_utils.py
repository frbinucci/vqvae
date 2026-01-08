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

def _logdet_spd(A):
    # stable log|A| for SPD matrices
    sign, ld = np.linalg.slogdet(_sym(A))
    if sign <= 0:
        raise ValueError("Matrix not SPD (non-positive determinant). Add more jitter.")
    return ld



def gib_fit(X, Y, n_comp, jitter=1e-9, max_dim=None):
    """
    Fit Gaussian Information Bottleneck mapping T = A X + ξ, with ξ~N(0, I).
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

    r = np.sum(evecs * (Sxx @ evecs), axis=0)


    beta_critical = 1.0 / np.maximum(1e-15, (1.0 - evals))

    denom = evals * r
    activable = (denom > 0) & np.isfinite(beta_critical)

    n_avail = int(np.sum(activable))

    if n_comp is None:
        raise ValueError("n_comp must be an integer >= 0")

    n_comp = int(n_comp)
    if n_comp < 0:
        raise ValueError("n_comp must be >= 0")
    if n_comp > n_avail:
        raise ValueError(f"Requested n_comp={n_comp} but only {n_avail} components are activable.")

    if n_comp == 0:
        beta = 0.0
    else:
        beta_c = beta_critical[activable]
        beta = np.nextafter(beta_c[n_comp - 1], np.inf)

    # ---- FINE NEW ----

    numer = beta * (1.0 - evals) - 1.0
    denom = evals * r

    alpha = np.zeros_like(evals)
    active = (numer > 0) & (denom > 0)
    alpha[active] = np.sqrt(numer[active] / denom[active])

    active_idx = np.where(alpha > 0)[0]

    if max_dim is not None:
        active_idx = active_idx[:max_dim]

    if max_dim is None:
        active_idx = active_idx[:n_comp]

    if active_idx.size == 0:
        A = np.zeros((0, nx))
        V = np.zeros((nx, 0))
    else:
        V = evecs[:, active_idx]              # nx x k
        W = np.diag(alpha[active_idx])        # k x k
        A = W @ V.T                           # k x nx  (rows are scaled eigenvectors)

    info = {
        "Sx_given_y": Sx_given_y,
        "eigenvalues": evals,
        "alpha": alpha,
        "active_dims": int(active_idx.size),
        "beta_critical": beta_critical,
        "beta_used": float(beta),
        "activable_dims": int(n_avail),
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

    Ixt = 0.5 * _logdet_spd(St)                   # since |Σ_ξ| = |I| = 1
    Ity = 0.5 * (_logdet_spd(St) - _logdet_spd(St_y))
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
