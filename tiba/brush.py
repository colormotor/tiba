import numpy as np

def width_profile_directional(P, theta, minw, maxw):
    D = np.diff(P, axis=0)
    D = np.vstack([D[0], D])
    D = (D.T / (kine.norms(D)+1e-10)).T
    d = np.array([np.cos(theta), np.sin(theta)])
    w = abs(np.dot(D, d))
    return minw + w*(maxw-minw)

def width_profile_bump(n, minw, maxw, mean=0.5, sigma=0.5):
    from scipy.stats import multivariate_normal
    t = np.linspace(0, 1, n)
    w = multivariate_normal.pdf(t, mean=0.5, cov=sigma**2)
    w = w / np.max(w)
    return minw + w*(maxw-minw)
