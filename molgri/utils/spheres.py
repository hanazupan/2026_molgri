"""
Here we collect functions that are generally useful and have something to do with spheres, points on spheres,
spherical arches and angles, spherical polygons ...
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi

from molgri.utils.arrays import angle_between_vectors, norm_per_axis, normalise_vectors


def dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Distance between two points on a sphere is a product of the radius (has to be the same radius for both  points) and
    angle between them.

    Args:
        vector1: vector shape (n1, d) or (d,)
        vector2: vector shape (n2, d) or (d,)

    Returns:
        an array the shape (n1, n2) containing distances between both sets of points on sphere
    """

    norm1 = norm_per_axis(vector1)
    norm2 = norm_per_axis(vector2)
    # all norms the same
    flat_norm = norm1.flatten()[0]
    assert np.allclose(norm1, flat_norm)
    assert np.allclose(norm2, flat_norm)
    angle = angle_between_vectors(vector1, vector2)
    return angle * flat_norm


def random_sphere_points(n: int = 1000) -> NDArray:
    """
    Create n points that are randomly distributed across the sphere. There are better ways to do this, this is just a
    quick and easy generation scheme.

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 3)
    """
    coord = np.random.normal(size=(n, 3))
    normalized_coord = normalise_vectors(coord)
    return normalized_coord


def sort_points_on_sphere_ccw(points: NDArray) -> NDArray:
    """
    Gets an array of points on a 2D sphere; returns an array of the same points, but ordered in a counter-clockwise
    manner.

    Args:
        points (NDArray): an array in which each row is a coordinate of a point on a unit sphere (2-sphere)

    Returns:
        the same array of points, but sorted in a counter-clockwise manner. The first point remains in first position.
    """

    def is_ccw(v_0: NDArray, v_c: NDArray, v_i: NDArray) -> bool:
        """
        Checks if the smaller interior angle for the great circles connecting trajectory_universe-v and v-w is CCW (counter-clockwise)

        Args:
            v_0 (NDArray): 3D coordinate of the first point
            v_c (NDArray): 3D coordinate of the center
            v_i (NDArray): 3D coordinate of the i-th point

        Returns:
            True if the sorting is counter-clockwise
        """
        #
        return np.dot(np.cross(v_c - v_0, v_i - v_c), v_i) < 0

    vector_center = normalise_vectors(np.average(points, axis=0), length=np.linalg.norm(points, axis=1)[0])
    N = len(points)
    # angle between first point, center point, and each additional point
    alpha = np.zeros(N)  # initialize array
    for i in range(1, N):
        alpha_candidate = _get_alpha_with_spherical_cosine_law(vector_center, points[0], points[i])
        if is_ccw(points[0], vector_center, points[i]):
            alpha[i] = alpha_candidate
        else:
            alpha[i] = 2*pi - alpha_candidate
    assert np.all(alpha >= 0), alpha

    output = points[np.argsort(alpha)]
    return output


def _get_alpha_with_spherical_cosine_law(A: NDArray, B: NDArray, C: NDArray):
    """
    A, B and C are points on a sphere that form a triangle, given as vectors in cartesian coordinates. We use
    the spherical law of cosines to obtain the angle at point A.
    """
    # check that they all have the same norm (are on the same sphere)
    #assert np.allclose(np.linalg.norm(A), np.linalg.norm(B)) and np.allclose(np.linalg.norm(A), np.linalg.norm(C))
    # consider spherical triangle:
    A = normalise_vectors(A)
    B = normalise_vectors(B)
    C = normalise_vectors(C)
    # and lengths of the opposite sides a, b, c are
    a = dist_on_sphere(B, C)
    b = dist_on_sphere(C, A)
    c = dist_on_sphere(A, B)
    # using cosine law on spheres (need rounding so we don't numerically get over/under the range of arccos):
    alpha = np.arccos(np.round((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), 7))
    return alpha


def exact_area_of_spherical_polygon(vertices: NDArray, r: float = 1) -> float:
    """
    Use the formula (Todhunter, I. (1886). Spherical Trigonometry), to calculate the
    area of a spherical polygon with given
    vertices. In the formula, r, is the radius of the sphere, n the number of polygon vertices, theta_i-s are the radian
    angles of a spherical polygon.

    Args:
        vertices (NDArray): A (m, 3) array of points on a unit 2-sphere. Points must be ordered counter-clockwise and
        unique.
        r (float): radius of the sphere

    Returns:
        the area of the spherical polygon
    """
    n = len(vertices)

    sorted_vertices = sort_points_on_sphere_ccw(vertices)

    thetas = []
    for i in range(n):
        # using cosine law on spheres:
        theta = _get_alpha_with_spherical_cosine_law(sorted_vertices[i], sorted_vertices[i-1], sorted_vertices[(i + 1) % n])
        thetas.append(theta)
    area = (np.sum(thetas) - (n-2)*pi) * r**2
    # chose the smaller of two possible spherical polygons described by these vertices
    if area > 2 * pi * r**2:
        area = 4 * pi * r**2 - area
    assert area >= 0, f"Area cannot be negative!"
    return area


def circular_sector_area(shared_upper_vertices: NDArray, shared_lower_vertices: NDArray) -> float:
    """
    Find the area of either circular sector or a difference between a smaller and bigger circular sector (same angle,
    two different radii).

    Args:
        shared_upper_vertices (NDArray): coordinates of the upper arch, should be exactly two
        shared_lower_vertices (NDArray): coordinates of the lower arch, should be exactly two or one if it is (0,0,0)

    Returns:
        area in Angstrom^2
    """
    assert shared_upper_vertices.shape == (2, 3)
    assert shared_lower_vertices.shape == (2, 3) or np.allclose(shared_lower_vertices, 0.0)

    radius_smaller = np.linalg.norm(shared_lower_vertices[0])
    radius_larger = np.linalg.norm(shared_upper_vertices[0])
    angle = angle_between_vectors(shared_upper_vertices[0], shared_upper_vertices[1])

    return float((radius_larger ** 2 - radius_smaller ** 2) * angle / 2)
