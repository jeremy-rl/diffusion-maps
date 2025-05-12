"""Runs Principal Component Analysis (PCA)."""

import sklearn.decomposition


def run(points, dim):
    """Runs the PCA algorithm.

    points: the input to the algorithm
    dim: the target dimension to reduce to"""
    pca = sklearn.decomposition.PCA(n_components=dim)
    return pca.fit_transform(points)
