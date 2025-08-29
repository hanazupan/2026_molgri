
import numpy as np
from numpy.typing import NDArray, ArrayLike

UNIQUE_TOL = 5

def which_row_is_k(my_array: NDArray, k: NDArray) -> ArrayLike:
    """
    returns all indices of rows in my_array that are equal (within floating point errors) to my_array.
    Args:
        my_array:
        k:

    Returns:

    """
    return np.nonzero(np.all(np.isclose(k, my_array), axis=1))[0]


def all_rows_unique(my_array: NDArray, tol: int = UNIQUE_TOL):
    """
    Check if all rows of the array are unique up to tol number of decimal places.
    """
    my_unique = np.unique(my_array.round(tol), axis=0)
    difference = np.abs(len(my_array) - len(my_unique))
    assert len(my_array) == len(my_unique), f"{difference} elements of an array are not unique up to tolerance."


def norm_per_axis(array: NDArray, axis: int = None) -> NDArray:
    """
    Returns the norm of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur

    Returns:
        an array of the same shape as the input array where each value is the norm of the corresponding
        vector/row/column
    """
    if axis is None:
        if len(array.shape) > 1:
            axis = 1
        else:
            axis = 0
    my_norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.repeat(my_norm, array.shape[axis], axis=axis)


def angle_between_vectors(central_vec: np.ndarray, side_vector: np.ndarray) -> np.array:
    """
    Having two vectors or two arrays in which each row is a vector, calculate all angles between vectors.
    For arrays, returns an array giving results like those:

    ------------------------------------------------------------------------------------
    | angle(central_vec[0], side_vec[0])  | angle(central_vec[0], side_vec[1]) | ..... |
    | angle(central_vec[1], side_vec[0])  | angle(central_vec[1], side_vec[1]  | ..... |
    | ..................................  | .................................  | ..... |
    ------------------------------------------------------------------------------------

    Angle between vectors equals the distance between two points measured on a surface of an unit sphere!

    Args:
        central_vec: first vector or array of vectors
        side_vector: second vector or array of vectors

    Returns:

    """
    assert central_vec.shape[-1] == side_vector.shape[-1], f"Last components of shapes of both vectors are not equal:" \
                                                     f"{central_vec.shape[-1]}!={side_vector.shape[-1]}"
    v1_u = normalise_vectors(central_vec)
    v2_u = normalise_vectors(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
    return angle_vectors


def normalise_vectors(array: NDArray, axis: int = None, length: float = 1) -> NDArray:
    """
    Returns the unit vector of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur
        length: desired new length for all vectors in the array

    Returns:
        an array of the same shape as the input array where vectors are normalised, now all have length 'length'
    """
    assert length >= 0, "Length of a vector cannot be negative"
    my_norm = norm_per_axis(array=array, axis=axis)
    return length * np.divide(array, my_norm)

def dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Distance between two points on a sphere is a product of the radius (has to be the same for both) and angle
    between them.

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
        quaternions: array (N, 4), each row a quaternion
        upper: if True, select the upper hemisphere, that is, demand that the first non-zero coordinate is positive

    Returns:
        quaternions: array (M <= N, 4), each row a quaternion different from all other ones
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

def q_in_upper_sphere(q: NDArray) -> bool:
    """
    Determine whether q in the upper part of the (hyper)sphere. This will be true if the first non-zero element of
    the vector/quaternion is positive.

    The point of all zeros is defined to be in the bottom hemisphere.

    Args:
        q: a vector/quaternion to be tested

    Returns:

    """
    assert len(q.shape) == 1
    for i, q_i in enumerate(q):
        if np.allclose(q[:i], 0) and q[i] > 0:
            return True
    return False