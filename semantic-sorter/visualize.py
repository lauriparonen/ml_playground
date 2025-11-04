import numpy as np
import matplotlib.pyplot as plt

def pca_2d(X, n_components=2):
    # mean-center
    X_centered = X - X.mean(axis=0)
    # compute covariance
    cov = np.cov(X_centered, rowvar=False)
    # eigen-decomp
    eigvals, eigvecs = np.linalg.eigh(cov)
    # pick top components
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    return X_centered @ components