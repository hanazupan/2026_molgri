"""
Here we collect functions that are generally useful and have something to do with arrays: checking if arrays are similar
enough, calculating particular norms ... Most of this is just to give better names to numpy functunalities I cannot remember how to use correctly.
"""


from __future__ import annotations

import numbers
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from molgri.constants import UNIQUE_TOL


def check_equality(arr1: NDArray, arr2: NDArray, atol: float = None, rtol: float = None) -> bool:
    """
    Use the numpy function np.allclose to compare two arrays and return True if they are all equal. This function
    is a wrapper where I can set my preferred absolute and relative tolerance
    """
    if atol is None:
        atol = 1e-8
    if rtol is None:
        rtol = 1e-5
    return np.allclose(arr1, arr2, atol=atol, rtol=rtol)


def is_array_with_d_dim_r_rows_c_columns(my_array: NDArray, d: int = None, r: int = None, c: int = None) -> bool:
    """
    Assert that the object is an array. If you specify d, r, c, it will check if this number of dimensions, rows, and/or
    columns are present.
    """
    assert type(my_array) == np.ndarray, "The first argument is not an array"
    # only check if dimension if d specified
    if d is not None:
        assert len(my_array.shape) == d, f"The dimension of an array is not d: {len(my_array.shape)}=!={d}"
    if r is not None:
        assert my_array.shape[0] == r, f"The number of rows is not r: {my_array.shape[0]}=!={r}"
    if c is not None:
        assert my_array.shape[1] == c, f"The number of columns is not c: {my_array.shape[1]}=!={c}"
    return True


def which_row_is_k(my_array: NDArray, k: NDArray) -> NDArray:
    """
    returns all indices of rows in my_array that are equal (within floating point errors) where row==k.

    Args:
        my_array (NDArray): a 2D array of shape (n, m)
        k (NDArray): a 1D array of shape (m)

    Returns:
        an array of indices, might be an empty array if row k is not present in my_array
    """
    return np.nonzero(np.all(np.isclose(k, my_array), axis=1))[0]


def find_shared_rows(array_1: NDArray, array_2: NDArray) -> NDArray:
    """
    Among two arrays, find the rows that exist in both (within floating point errors). They need not be in the same
    position in both arrays.

    Args:
        array_1 (NDArray): a 2D array of shape (n1, m)
        array_2 (NDArray): a 2D array of shape (n1, m)

    Returns:
        a 2D array with rows that occur in both arrays
    """
    shared_vertices = []
    for row in array_1:
        if which_row_is_k(array_2, row).size > 0:
            shared_vertices.append(row)
    return np.array(shared_vertices)


def all_rows_unique(my_array: NDArray, tol: int = UNIQUE_TOL) -> None:
    """
    Check if all rows of the array are unique up to tol number of decimal places.
    """
    my_unique = np.unique(my_array.round(tol), axis=0)
    difference = np.abs(len(my_array) - len(my_unique))
    assert len(my_array) == len(my_unique), f"{difference} elements of an array are not unique up to tolerance."


def k_argmin_in_array(my_array: NDArray, k: int) -> NDArray:
    """
    Of all the values in the array, find the indices of the k smallest values.

    The indices are not necessarily sorted!

    Args:
        my_array (NDArray): array in which to search
        k (int): number of results

    Returns:
        an array with k indices indicating the smallest item up to the kth smallest item, possibly unsorted
    """

    idx = np.argpartition(my_array, k)
    return idx[:k]


def k_argmax_in_array(my_array: NDArray, k: int) -> NDArray:
    """
    Of all the values in the array, find the indices of the k largest values.

    The indices are not necessarily sorted!

    Args:
        my_array (): array in which to search
        k (): number of results

    Returns:
        an array with k indices indicating the largest item up to the kth largest item, possibly unsorted
    """
    return np.argpartition(my_array, -k)[-k:]


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


def all_row_norms_equal_k(my_array: NDArray, k: float, atol: float = None, rtol: float = None) -> NDArray:
    """
    Same as all_row_norms_similar, but also test that all row norms equals k.
    """
    my_norms = all_row_norms_similar(my_array=my_array, atol=atol, rtol=rtol)
    assert check_equality(my_norms, np.array(k), atol=atol, rtol=rtol), "The norms are not equal to k"
    return my_norms


def all_row_norms_similar(my_array: NDArray, atol: float = None, rtol: float = None) -> NDArray:
    """
    Assert that in an 2D array each row has the same norm (up to the floating point tolerance).

    Returns:
        the array of norms in the same shape as my_array
    """
    is_array_with_d_dim_r_rows_c_columns(my_array, d=2)
    axis = 1
    all_norms = norm_per_axis(my_array, axis=axis)
    average_norm = np.average(all_norms)
    assert check_equality(all_norms, np.array(average_norm), atol=atol, rtol=rtol), ("The norms of all rows are not "
    "equal")
    return all_norms


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


def angle_between_vectors(central_vec: np.ndarray, side_vector: np.ndarray) -> NDArray:
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
        for two vectors, returns a single ange, for two 2D arrays returns a 2D array of angles for all combinations
        of input vectors
    """
    assert central_vec.shape[-1] == side_vector.shape[-1], f"Last components of shapes of both vectors are not equal:" \
                                                     f"{central_vec.shape[-1]}!={side_vector.shape[-1]}"
    v1_u = normalise_vectors(central_vec)
    v2_u = normalise_vectors(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
    return angle_vectors


def nested_numpy_types_to_python_types(obj: Sequence | numbers.Number) -> Sequence | numbers.Number:
    """
    Convert a list or tuple of any depth that contains numbers (potentially numpy numerical types) to generic float and
    int types. Uses recursion for nested lists or tuples of any depth.

    This is useful as some converters have problems processing numpy types.

    All elements must be numerical.

    Args:
        obj (list, tuple or number): object to be converted, if a sequence then iteratively.

    Returns:
        The starting object where every element is a float or int type.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(nested_numpy_types_to_python_types(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return nested_numpy_types_to_python_types(obj.tolist())
    else:
        return obj


def iter_elements_nested(obj: Iterable) -> Iterable[Any]:
    """
    Yield all elements of a (nested) list, tuple or array. This function uses recursion. If you are encountering a
    flat object, return all elements sequentially. If you are encountering an object with subobjects, repeat the
    function on subobjects.

    Args:
        obj (Iterable): list, tuple or array nested at as many levels as wanted

    Yields:
        sequentially each element of obj that is the final element, not a container for other elements
    """
    if isinstance(obj, (list, tuple, np.ndarray)):
        for x in obj:
            yield from iter_elements_nested(x)
    else:
        yield obj
