from sklearn.cluster import KMeans
import numpy as np

def fit_scalar_lloyd_codebooks(Z, bits, seed=0):
    """
    Z: array (N, nz) = campioni nel dominio dei modi (già decorrelati/whitened se vuoi)
    bits: array (nz,) con b_i >= 0
    """
    Z = np.asarray(Z)
    nz = Z.shape[1]
    codebooks = []
    thresholds = []

    for i in range(nz):
        b = int(bits[i])
        if b <= 0:
            # 0 bit: una sola ricostruzione (es. 0 o media)
            c = np.array([0.0])
            t = np.array([])
        else:
            K = 1 << b
            xi = Z[:, i].reshape(-1, 1)
            km = KMeans(n_clusters=K, algorithm="lloyd", n_init="auto", random_state=seed)
            km.fit(xi)
            c = np.sort(km.cluster_centers_.ravel())
            t = 0.5 * (c[:-1] + c[1:])   # soglie = punti medi
        codebooks.append(c)
        thresholds.append(t)

    return codebooks, thresholds


def quantize_components(Z, codebooks, thresholds):
    Z = np.asarray(Z)
    N, nz = Z.shape
    idx = np.zeros((N, nz), dtype=np.int32)
    Zq  = np.zeros_like(Z, dtype=float)

    for i in range(nz):
        c = codebooks[i]
        t = thresholds[i]
        if t.size == 0:
            idx[:, i] = 0
            Zq[:, i] = c[0]
        else:
            j = np.digitize(Z[:, i], t)  # 0..K-1
            idx[:, i] = j
            Zq[:, i] = c[j]
    return idx, Zq
