"""Implementation of a modified diffusion map algorithm. Based on de La Porte et al
and Coifman et al (see bibliography)."""

import numpy as np
import numpy.ma as ma
from scipy import linalg
import sklearn
import sklearn.metrics
import sklearn.preprocessing

T = 1  # Diffusion time step to compute output for


def compute_gaussian_kernel(points):
    """Computes the kernel matrix using the Gaussian kernel.

    points: the data points to compute the kernel matrix for"""
    distances = sklearn.metrics.pairwise.euclidean_distances(points, squared=True)
    nonzero_distances = ma.masked_array(distances, mask=distances == 0)
    epsilon = np.max(np.amin(nonzero_distances, axis=1))
    return np.exp(-(distances) / epsilon)


def run(points, dim):
    """Runs the modified diffusion map algorithm.

    points: the input to the algorithm
    dim: the target dimension to reduce to"""
    num_points = np.size(points, 0)
    # Compute kernel matrix and normalized diffusion matrix
    kernel_matrix = compute_gaussian_kernel(points)
    kernel_sums = np.sum(kernel_matrix, 1)
    normalized_kernel = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            normalized_kernel[i, j] = kernel_matrix[i, j] / np.sqrt(
                kernel_sums[i] * kernel_sums[j]
            )
    normalized_sums = np.sum(normalized_kernel, 1)
    diffusion_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            diffusion_matrix[i, j] = normalized_kernel[i, j] / np.sqrt(
                normalized_sums[i] * normalized_sums[j]
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
