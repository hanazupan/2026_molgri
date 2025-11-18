import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import pi
from scipy.linalg import svd

from molgri.utils.arrays import angle_between_vectors, is_array_with_d_dim_r_rows_c_columns


def hypersphere_voronoi_cell_volumes(voronoi_cell_centers: NDArray, N: int = int(1e7)) -> NDArray:
    """
    This is Monte-Carlo integration on hyperspere. It doesn't rely on Voronoi cell edges, just on the central points
    and the property of Voronoi cells that points belong to the cell with the closest center.

    Note that the cell centers should be given for the entirety on the hypersphere, not just one hemisphere.

    Args:
        voronoi_cell_centers (NDArray): an array of shape (N_points, 4), each row a quaternion on a hypersphere
        N (int): the number of random points used in integration. More points mean longer calculation but higher
            accuracy (especially if there are many Voronoi cells)

    Returns:
        an array of length N_points, each a volume of a cell; they add up to 2pi^2
    """
    X = random_quaternions(N)
    idx = assign_closest_quaternion(X, voronoi_cell_centers)

    k = voronoi_cell_centers.shape[0]
    counts = np.bincount(idx, minlength=k)

    vols = (2 * np.pi ** 2) * counts / N
    return vols


def distance_between_quaternions(q1: NDArray, q2: NDArray) -> ArrayLike:
    """
    Calculate the distance between two unit quaternions or the pairwise distances between two arrays of unit
    quaternions. Quaternion distance is like hypersphere distance, but also considers double coverage.
    Args:
        q1 (): array either of shape (4,) or (N, 4), every row has unit length
        q2 (): array either of shape (4,) or (N, 4), every row has unit length

    Returns:
        Float or an array of shape (N,) containing distances between unit quaternions.
    """
    if q1.shape == (4,) and q2.shape == (4,):
        theta = angle_between_vectors(q1, q2)
    elif q1.shape[1] == 4 and q2.shape[1] == 4 and q1.shape[0]==q2.shape[0]:
        theta = np.diagonal(angle_between_vectors(q1, q2))
    else:
        raise ValueError("Shape of quaternions not okay")
    # if the distance would be more than half hypersphere, use the smaller distance
    return np.where(theta > pi / 2, pi-theta, theta)


def hemisphere_quaternion_set(quaternions: NDArray, upper=True) -> NDArray:
    """
    Select only the "upper half"/"bottom half" of hyperspherical points (quaternions that may be repeating).
    How selection is done:
    for all points select either q or -q, depending which is in the right hemisphere

    Args:
        quaternions: array (N, 4), each row a coordinate
        upper: if True, select the upper hemisphere, that is, demand that the first non-zero coordinate is positive

    Returns:
        quaternions: array (M <= N, 4), each row a coordinate different from all other ones
    """
    # test input
    is_array_with_d_dim_r_rows_c_columns(quaternions, d=2, c=4)

    non_repeating_quaternions = []
    for projected_point in quaternions:
        for i in range(4):
            # if this if-sentence is True, the point is in the upper hemisphere
            if np.allclose(projected_point[:i], 0) and projected_point[i] > 0:
                # the point is selected
                if upper:
                    non_repeating_quaternions.append(projected_point)
                else:
                    non_repeating_quaternions.append(find_inverse_quaternion(projected_point))
                break
        # if the loop didn't break, the point was not in upper hemisphere
        else:
            if upper:
                non_repeating_quaternions.append(find_inverse_quaternion(projected_point))
            else:
                non_repeating_quaternions.append(projected_point)

    return np.array(non_repeating_quaternions)


def remove_bottom_half_quaternions(quaternions: NDArray) -> NDArray:
    """
    Here we immediately remove (don't project to the other side) any quaternions in the bottom half.

    Args:
        quaternions: array (N, 4), each row a coordinate

    Returns:
        quaternions: array (M <= N, 4), each row a coordinate different from all other ones
    """
    is_array_with_d_dim_r_rows_c_columns(quaternions, d=2, c=4)

    non_repeating_quaternions = []
    for projected_point in quaternions:
        for i in range(4):
            # if this if-sentence is True, the point is in the upper hemisphere
            if np.allclose(projected_point[:i], 0) and projected_point[i] > 0:
                # the point is selected
                non_repeating_quaternions.append(projected_point)

    return np.array(non_repeating_quaternions)


def double_coverage_from_upper_quaternions(quaternions: NDArray) -> NDArray:
    """
    For each q in quaternions also get -q so that the resulting list is twice as long.
    """
    N_points = quaternions.shape[0]
    all_points = np.zeros((2 * N_points, 4))
    all_points[:N_points] = quaternions
    for i in range(N_points):
        inverse_q = find_inverse_quaternion(quaternions[i])
        all_points[N_points + i] = inverse_q
    return all_points


def q_in_upper_sphere(q: NDArray) -> bool:
    """
    Determine whether q in the upper part of the (hyper)sphere. This will be true if the first non-zero element of
    the vector/coordinate is positive.

    The point of all zeros is defined to be in the bottom hemisphere.

    Args:
        q: a vector/coordinate to be tested

    Returns:

    """
    assert len(q.shape) == 1
    for i, q_i in enumerate(q):
        if np.allclose(q[:i], 0) and q[i] > 0:
            return True
    return False


def find_inverse_quaternion(q: NDArray) -> NDArray:
    """
    Inverse coordinate -q = (-q0, -q1, -q2, -q3) is the coordinate that represents the same rotation as q.

    Args:
        q: a coordinate of shape (4,) whose inverse is needed

    Returns:
        another coordinate of shape (4,) with all coordinates inversed
    """
    assert q.shape == (4,)
    return -q


def quaternion_in_array(quat: NDArray, quat_array: NDArray) -> bool:
    """
    Check if a coordinate q or its equivalent complement -q is present in the coordinate array quat_array.
    """
    quat1 = quat[np.newaxis, :]
    for quat2 in quat_array:
        if two_sets_of_quaternions_equal(quat1, quat2[np.newaxis, :]):
            return True
    return False


def two_sets_of_quaternions_equal(quat1: NDArray, quat2: NDArray) -> bool:
    """
    This test is necessary because for quaternions, q and -q represent the same rotation. You therefore cannot simply
    use np.allclose to check if two sets of rotations represented with quaternions are the same. This function checks
    if all rows of two arrays are the same up to a flipped sign.
    """
    assert quat1.shape == quat2.shape
    assert quat1.shape[1] == 4
    # quaternions are the same if they are equal up to a +- sign
    # I have checked this fact and it is mathematically correct
    for q1, q2 in zip(quat1, quat2):
        if not (np.allclose(q1, q2) or np.allclose(q1, find_inverse_quaternion(q2))):
            return False
    return True


def find_shared_quaternions(array_1: NDArray, array_2: NDArray) -> ArrayLike:
    shared_vertices = []
    for row in array_1:
        if quaternion_in_array(row, array_2):
            shared_vertices.append(row)
    return np.array(shared_vertices)


def random_quaternions(n: int = 1000, only_upper=False, rotation_random_seed: float = None) -> NDArray:
    """
    Create n random quaternions

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 4)
    """
    if rotation_random_seed is not None:
        rng = np.random.default_rng(rotation_random_seed)
    else:
        rng = np.random.default_rng()

    result = np.zeros((n, 4))
    random_num = rng.random((n, 3))
    result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
    result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
    result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
    result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
    assert result.shape[1] == 4

    if only_upper:
        return hemisphere_quaternion_set(result, upper=True)
    return result


def project_quaternions_to_3D(quaternion_array: NDArray) -> NDArray:
    """
    This is only for visualization, it is not volume-preserving
    Args:
        quaternion_array ():

    Returns:

    """
    three_components = quaternion_array[:, :3]
    for i, q in enumerate(quaternion_array):
        if not np.isclose(q[3], 1):
            three_components[i] /= (1-q[3])
    return three_components


def additional_vertices_hyprsphere_polygons(current_vertices, n_per_line: int = 10):
    from scipy.spatial.transform import Slerp, Rotation
    additional_points = []
    for index_1, point1 in enumerate(current_vertices):
        for point2 in current_vertices[index_1 + 1:]:
            rot = Rotation.from_quat(np.array([point1, point2]), scalar_first=True)
            my_slerp = Slerp([0,1], rot)
            t = np.linspace(0, 1, n_per_line)
            interpolated_rot = my_slerp(t)
            points = interpolated_rot.as_quat(scalar_first=True)
            #points = geometric_slerp(point1, point2, t=np.linspace(0, 1, n_per_line))
            additional_points.append(points)
    all_hull_points = np.vstack([np.vstack(additional_points), current_vertices])
    return all_hull_points


def assign_closest_quaternion(points: NDArray, available_quaternions) -> NDArray:
    """
    Args:
        points ():
        available_quaternions ():

    Returns:

    """
    # dot product of each sample with each site
    dots = points @ available_quaternions.T           # shape (N, k)
    return np.argmax(dots, axis=1)   # closest site (max dot = smallest angle)


def cut_off_constant_dimension_quat(my_array: NDArray):
    u, s, vh = svd(my_array)
    # rotate till last dimension is only zeros, then cut off the redundant dimension. Now we can correctly
    # calculate borders using lower-dimensional tools
    rotated_points = np.dot(my_array, vh.T)
    assert np.allclose(rotated_points[:, -1], 0.0)
    return rotated_points[:, :-1]
