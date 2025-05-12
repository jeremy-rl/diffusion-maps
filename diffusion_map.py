"""Implementation of a basic diffusion map algorithm. Based on de La Porte et al
and Coifman et al (see bibliography)."""

import numpy as np
from scipy import linalg
import sklearn
import sklearn.metrics
import sklearn.preprocessing

T = 1  # Diffusion time step to compute output for


def compute_gaussian_kernel(points):
    """Computes the kernel matrix using the Gaussian kernel.

    points: the data points to compute the kernel matrix for"""
    distances = sklearn.metrics.pairwise.euclidean_distances(points, squared=True)
    epsilon = np.median(distances)
    return np.exp(-(distances) / epsilon)


def reduce_to_one(points):
    """Reduces two dimensional diffusion map output to 1 dimension.

    points: the input to the algorithm
    """
    return points[:, 0:1]


def run(points, dim):
    """Runs the basic diffusion map algorithm.

    points: the input to the algorithm
    dim: the target dimension to reduce to"""
    num_points = np.size(points, 0)
    # Compute kernel matrix and normalized diffusion matrix
    kernel_matrix = compute_gaussian_kernel(points)
    row_sums = np.sum(kernel_matrix, 1)
    diffusion_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            diffusion_matrix[i, j] = kernel_matrix[i, j] / np.sqrt(
                row_sums[i] * row_sums[j]
            )
    # Compute eigenvalues and eigenvectors of diffusion matrix in descending order
    eigenvalues, eigenvectors = linalg.eigh(diffusion_matrix)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, 1)
    # Use eigenvalues and eigenvectors to map input points to new points
    mapped_points = np.zeros((num_points, dim))
    for i in range(num_points):
        mapped_point = np.zeros(dim)
        for j in range(dim):
            coordinate = eigenvectors[i, j + 1] / eigenvectors[i, 0]
            mapped_point[j] = eigenvalues[j + 1] ** T * coordinate
        mapped_points[i] = mapped_point
    return mapped_points
