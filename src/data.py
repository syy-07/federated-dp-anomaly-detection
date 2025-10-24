from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

def make_synthetic_anomaly_data(n=10000, dim=32, anomaly_rate=0.02, seed=42):
    rng = np.random.default_rng(seed)
    X_normal = rng.normal(0, 1, size=(int(n*(1-anomaly_rate)), dim))
    X_anom = rng.normal(0, 6, size=(int(n*anomaly_rate), dim))  # larger variance
    X = np.vstack([X_normal, X_anom])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anom))])
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return (X_train, y_train), (X_test, y_test)

def split_by_clients(X, y, clients=5, iid=False, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    if iid:
        idx = rng.permutation(n)
        parts = np.array_split(idx, clients)
    else:
        # non-IID: sort by a latent projection, then split contiguous blocks
        proj = X @ rng.normal(size=(X.shape[1],))
        idx = np.argsort(proj)
        parts = np.array_split(idx, clients)
    return [(X[p], y[p]) for p in parts]
