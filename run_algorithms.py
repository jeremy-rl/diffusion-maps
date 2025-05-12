"""Run each of the algorithms on each dataset and plot the results."""

import numpy as np
import pca
import diffusion_map
import modified_diffusion_map
import matplotlib.pyplot as plt

# Constants
PCA = "pca"
DIFFUSION = "diffusion"
MODIFIED = "modified"
ALGORITHMS = [PCA, DIFFUSION, MODIFIED]
DATASETS = ["clusters", "linear", "s_curve", "swiss_roll"]


def read_dataset(name):
    """Read the dataset.

    name: the name of the file to read
    Returns: an array of the data points and an array of the corresponding
    colors/labels"""
    data = np.loadtxt("datasets/" + name + ".csv", delimiter=",")
    return data[:, 0:3], data[:, 3]


def plot_output(x_coords, y_coords, colors, name):
    """Plot the output of the algorithm.

    x_coords: the x coordinates to plot
    y_coords: the y coordinates to plot
    colors: the colors of the data points from the original dataset
    name: the name of the file to save"""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("$x$")
    if y_coords is None:
        ax.scatter(x_coords, np.zeros(x_coords.size), c=colors)
    else:
        ax.set_ylabel("$y$")
        ax.scatter(x_coords, y_coords, c=colors)
    plt.tight_layout()
    plt.savefig(name + ".png", dpi=300)
    plt.close()


def run_algorithms():
    """Run each algorithm on each dataset for target dimensions 1 and 2."""
    for algorithm in ALGORITHMS:
        for name in DATASETS:
            for dim in 2, 1:
                points, labels = read_dataset(name)
                if algorithm == DIFFUSION or algorithm == MODIFIED:
                    if dim == 2:
                        transformed_points = (
                            diffusion_map.run(points, dim)
                            if algorithm == DIFFUSION
                            else modified_diffusion_map.run(points, dim)
                        )
                        diffusion_2d = transformed_points
                    else:
                        transformed_points = diffusion_map.reduce_to_one(diffusion_2d)
                else:
                    transformed_points = pca.run(points, dim)
                x_coords = transformed_points[:, 0]
                y_coords = transformed_points[:, 1] if dim == 2 else None
                plot_output(
                    x_coords,
                    y_coords,
                    labels,
                    algorithm + "_plots/" + name + "_" + algorithm + "_" + str(dim),
                )


if __name__ == "__main__":
    run_algorithms()
