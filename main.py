# Tutorial on finding the best fit line
# Dmitrii Khizbullin, 2021

import math
import numpy as np
from typing import Optional, Any
import matplotlib.pyplot as plt


def generate_points(axis: Optional[Any] = None) -> np.ndarray:
    """
    Function to generate an array of points approximately lying on
    on a segment of a straight line.

    :param axis: Collection of axis objects to plot onto
    :return: Numpy array of shape [N, 2] of points on a straight line
    """

    # Let's generate 10 points laying on a straight line
    num_points = 10
    p_initial = np.array((-2, 3))
    speed = 1.0
    angle_degrees = -30
    angle_radians = angle_degrees / 180 * math.pi
    velocity = speed * np.array((math.cos(angle_radians),
                                 math.sin(angle_radians)))
    ideal_points = np.expand_dims(p_initial, 0) + \
                   np.outer(np.arange(0, num_points), velocity)
    # and add some noise to points to simulate real measurements
    noise = 0.1 * np.random.randn(num_points, 2)
    points = ideal_points + noise

    if axis is not None:
        for ax in axis:
            ax.set_title("Input point set")
            ax.plot(points[:, 0], points[:, 1], 'or')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return points


def least_squares(points: np.ndarray, axis: Optional[Any] = None) -> np.ndarray:
    """
    Function to approximate a set of points by a straight line
    using least squares method.

    :param points: an input array of points of shape [N, 2]
    :param axis: Collection of axis objects to plot onto
    :return: Numpy array of shape [N, 2] of points on a straight line
    """

    x = points[:, 0]
    y = points[:, 1]
    # For least squares method we need X to be a matrix containing
    # 1-s in the first column and x-s in the second
    X = np.vstack((np.ones(x.shape[0]), x)).T
    # We compute normal matrix and moment matrix as a part of
    # formula to compute intercept and slope
    normal_matrix = np.dot(X.T, X)
    moment_matrix = np.dot(X.T, y)
    # beta_hat is a vector [intercept, slope], we need to invert
    # the normal matrix and compute cross product with the moment matrix
    beta_hat = np.dot(np.linalg.inv(normal_matrix), moment_matrix)
    intercept = beta_hat[0]
    slope = beta_hat[1]
    # Now when we know the parameters of the line, computing
    # y coordinates is straightforward
    y_hat = intercept + slope * x
    # Let's combine x and y into a single matrix that we want to return
    points_hat = np.vstack((x, y_hat)).T

    if axis is not None:
        for ax in axis:
            ax.set_title("Least squares")
            ax.plot(x, y, 'or')
            ax.plot(x, y_hat, 'o-', mfc='none')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return points_hat


def ransac(points: np.ndarray,
           min_inliers: int = 4,
           max_distance: float = 0.15,
           outliers_fraction: float = 0.5,
           probability_of_success: float = 0.99,
           axis: Optional[Any] = None) -> Optional[np.ndarray]:
    """
    RANdom SAmple Consensus method of finding the best fit line.

    :param points: an input array of points of shape [N, 2]
    :param min_inliers: Minimum number of inliers to consider a support
    :param max_distance: Maximum distance from a support line
                         for a point to be considered as an inlier
    :param outliers_fraction: As estimated fraction of outliers
    :param probability_of_success: desired probability that the support
                                   does not contain outliers
    :param axis: Collection of axis objects to plot onto
    :return: Numpy array of shape [N, 2] of points on a straight line
    """

    # Let's calculate the required number of trials to sample a support
    num_trials = int(math.log(1 - probability_of_success) /
                     math.log(1 - outliers_fraction**2))

    best_num_inliers = 0
    best_support = None
    for _ in range(num_trials):
        # For each trial, randomly sample 2 different points
        # from the input array to form the support
        random_indices = np.random.choice(
            np.arange(0, len(points)), size=(2,), replace=False)
        assert random_indices[0] != random_indices[1]
        support = np.take(points, random_indices, axis=0)

        # Here we compute distances from all points to the line
        # defined by the support. Distances are nicely computed
        # with cross product function.
        cross_prod = np.cross(support[1, :] - support[0, :],
                              support[1, :] - points)
        support_length = np.linalg.norm(support[1, :] - support[0, :])
        # cross_prod contains signed distances thus we need modulus
        distances = np.abs(cross_prod) / support_length

        # Inliers are all the points that are close enough
        # to the support line
        num_inliers = np.sum(distances < max_distance)
        # Here we update the support with the better one
        if num_inliers >= min_inliers and num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_support = support

    # If we succeeded to find a good support
    if best_support is not None:
        # Let's project all points onto the support line
        support_start = best_support[0]
        support_vec = best_support[1] - best_support[0]
        # Dot product is the right function to compute projections
        offsets = np.dot(support_vec, (points - support_start).T)
        proj_vectors = np.outer(support_vec, offsets).T
        support_sq_len = np.inner(support_vec, support_vec)
        projected_vectors = proj_vectors / support_sq_len
        projected_points = support_start + projected_vectors

        if axis is not None:
            for ax in axis:
                ax.set_title("RANSAC")
                ax.scatter(best_support[:, 0], best_support[:, 1],
                            s=200, facecolors='none', edgecolors='k', marker='s')
                ax.plot(points[:, 0], points[:, 1], 'or')
                ax.plot(projected_points[:, 0], projected_points[:, 1],
                         'o-', mfc='none')
                ax.grid(True, linestyle='--')
                ax.axis('equal')
    else:
        projected_points = None

    return projected_points


def pca(points: np.ndarray, axis: Optional[Any] = None) -> np.ndarray:
    """
    Principal Component Analysis (PCA) method to estimate the direction
    of the maximal variance of a point set.

    :param points: an input array of points of shape [N, 2]
    :param axis: Collection of axis objects to plot onto
    :return: Numpy array of shape [N, 2] of points on a straight line
    """

    # Perform PCA to understand what the primary axis
    # of the given point set is
    mean = np.mean(points, axis=0)
    # Points have to be zero-mean
    centered = points - mean
    # np.linalg.eig takes a covariance matrix as an argument
    cov = np.cov(centered.T)
    # Call eigenvector decomposition to obtain principal components
    eigenval, eigenvec = np.linalg.eig(cov)
    # We want to parametrize target straight line
    # in the coordinate frame given by the eigenvector
    # that corresponds to the biggest eigenvalue
    argmax_eigen = np.argmax(eigenval)
    # We'll need projections of data points
    # on the primary axis
    loc_pca = np.dot(centered, eigenvec)
    loc_maxeigen = loc_pca[:, argmax_eigen]
    max_eigenval = eigenval[argmax_eigen]
    max_eigenvec = eigenvec[:, argmax_eigen]
    # Re-parametrize the line
    loc_start = mean + max_eigenvec * loc_maxeigen[0]
    loc_final = mean + max_eigenvec * loc_maxeigen[-1]
    linspace = np.linspace(0, 1, num=len(points))
    positions = loc_start + np.outer(linspace, loc_final - loc_start)

    if axis is not None:
        for ax in axis:
            ax.set_title("PCA")
            ax.plot(points[:, 0], points[:, 1], 'or')
            ax.plot(positions[:, 0], positions[:, 1], 'o-', mfc='none')
            ax.grid(True, linestyle='--')
            ax.axis('equal')

    return positions


def main():
    fig_all, axs = plt.subplots(2, 2)
    fig_all.set_size_inches(10.0, 6.0)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    points = generate_points(axis=(axs[0, 0], ax))
    fig.savefig('1_points.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    least_squares(points, axis=(axs[0, 1], ax))
    fig.savefig('2_leastsq.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    ransac(points, axis=(axs[1, 0], ax))
    fig.savefig('3_ransac.png', dpi=300)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5.0, 3.0)
    pca(points, axis=(axs[1, 1], ax))
    fig.savefig('4_pca.png', dpi=300)

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')

    for ax in axs.flat:
        ax.label_outer()

    fig_all.savefig('0_all.png', dpi=300)

    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
