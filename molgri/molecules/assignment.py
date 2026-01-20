"""
This file deals with assigning trajectory frames to their closest grid positions and orientations
"""
import pandas as pd
from numpy.typing import NDArray
import numpy as np

def assign_1D(points_to_assign: NDArray, grid_points: NDArray, limits: list, is_periodic: bool):
    """

    Args:
        points_to_assign ():
        grid_points ():
        limits ():
        is_periodic ():

    Returns:

    """
    N_gridpoints = len(grid_points)
    distance_gridpoints = np.array(grid_points[1]-grid_points[0])

    if is_periodic:
        periodic_box_len = np.abs(limits[1] - limits[0])
        points_to_assign = points_to_assign % periodic_box_len + limits[0]
        assigned_indices = np.floor(points_to_assign / distance_gridpoints + 0.5) % N_gridpoints
    else:
        assigned_indices = np.argmin(np.abs(np.subtract.outer(points_to_assign, grid_points)), axis=1)
    return assigned_indices


def assign_to_cartesian_translation_grid(points_to_assign: NDArray, subgrids: list, subgrid_limits: list,
                                         periodic_in: list) -> tuple:
    """
    How to assign to cartesian grid: assign coordinates individually (x to closest x grid point and so on,
    then add x_index * (N_y_gridpoints * N_z_gridpoints) + y_index * (N_z_gridpoints) + z_index

    Args:
        points_to_assign (NDArray): a (N_points, 3) array of coordinates to be assigned
        subgrids (list): a list of sublists of the form [x_gridpoints, y_gridpoints, z_gridpoints]
        subgrid_limits (list): a list of the form [[x_min, x_max], [y_min, y_max], [z_min, z_max]] - limits are
            important for periodic grids
        periodic_in (list): a list that tells you which dimension is periodic eg [True, True, False]

    Returns:
        a (N_points,) array of translation indices
    """
    assert len(subgrids) == len(subgrid_limits) == len(periodic_in) == 3
    N_x, N_y, N_z = [len(subgrid) for subgrid in subgrids]

    xyz_assignments = np.zeros(points_to_assign.shape)

    for i in range(3):
        xyz_assignments[:, i] = assign_1D(points_to_assign[:, i].T, subgrids[i], subgrid_limits[i], periodic_in[i])

    x_assignments, y_assignments, z_assignments = xyz_assignments.T

    combined_assignments = x_assignments*N_y*N_z + y_assignments*N_z + z_assignments
    return x_assignments, y_assignments, z_assignments, combined_assignments