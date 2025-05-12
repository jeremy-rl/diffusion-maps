"""Generate datasets to run the algorithms on."""

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

NUM_POINTS = 2500


def save_dataset(x_coords, y_coords, z_coords, colors, name):
    """Saves a dataset as a CSV file and the plot as a PNG file.

    x_coords: the x coordinates of the data points
    y_coords: the y coordinates of the data points
    z_coords: the z coordinates of the data points
    colors: the labels for each data point
    name: the name of the file to save"""
    np.savetxt(
        "datasets/" + name + ".csv",
        np.column_stack((x_coords, y_coords, z_coords, colors)),
        delimiter=",",
    )
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.scatter(x_coords, y_coords, z_coords, c=colors)
    plt.tight_layout()
    plt.savefig("data_plots/" + name + ".png", dpi=300)
    plt.close()


def generate_clusters():
    """Generates a clustered dataset."""
    points, labels = sklearn.datasets.make_blobs(
        n_samples=NUM_POINTS, n_features=3, centers=4, cluster_std=0.5, random_state=1
    )
    save_dataset(points[:, 0], points[:, 1], points[:, 2], labels, "clusters")


def generate_linear():
    """Generates a linear dataset."""
    x_coords, y_coords = sklearn.datasets.make_regression(
        n_samples=NUM_POINTS, n_features=2, noise=1, random_state=1
    )
    save_dataset(x_coords[:, 0], x_coords[:, 1], y_coords, y_coords, "linear")


def generate_s_curve():
    """Generates an s-curve dataset."""
    points, positions = sklearn.datasets.make_s_curve(
        n_samples=NUM_POINTS, noise=0.05, random_state=1
    )
    save_dataset(points[:, 0], points[:, 1], points[:, 2], positions, "s_curve")


def generate_swiss_roll():
    """Generates a swiss roll dataset."""
    points, positions = sklearn.datasets.make_swiss_roll(
        n_samples=NUM_POINTS, noise=0.05, random_state=1
    )
    save_dataset(points[:, 0], points[:, 1], points[:, 2], positions, "swiss_roll")


if __name__ == "__main__":
    generate_clusters()
    generate_s_curve()
    generate_swiss_roll()
    generate_linear()
