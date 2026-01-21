"""
This file deals with assigning trajectory frames to their closest grid positions and orientations
"""
from functools import partial
from multiprocessing import Pool

import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.lib.distances import calc_bonds
from numpy.typing import NDArray
import numpy as np

from molgri.utils.quaternions import assign_closest_quaternion


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

def _calc_distance_along_traj(frame_index, u, reference_coordinates):
    # index the trajectory to set it to the frame_index frame
    u.trajectory[frame_index]
    distance_between = calc_bonds(u.atoms, reference_coordinates)
    return np.sum(distance_between)


def loop_slowly_over_trajectories(u, u_ref, ag, ag_ref, N_rotations, limit_to):
    all_rmsds = np.zeros((limit_to, N_rotations))
    for ts in u_ref.trajectory[:N_rotations]:
        for traj_ts in u.trajectory[:limit_to]:
            distance_between = calc_bonds(ag,ag_ref)
            distance_between = np.sum(distance_between)
            all_rmsds[traj_ts.frame, ts.frame] = distance_between
    return all_rmsds


def loop_fast_over_trajectories(u, u_ref, N_rotations, limit_to=None):
    if limit_to is None:
        limit_to = len(u.trajectory)
    all_rmsds = np.zeros((limit_to, N_rotations))
    for ts in u_ref.trajectory[:N_rotations]:
        run_per_frame = partial(_calc_distance_along_traj,
            u=u,
            reference_coordinates=u_ref.atoms.positions)
        frame_values = np.arange(limit_to)  #len(trajectory_universe.trajectory)
        with Pool(1) as worker_pool:
            resulting_distances = worker_pool.map(run_per_frame,frame_values)
        resulting_distances = np.array(resulting_distances)
        all_rmsds[:, ts.frame] = resulting_distances
    return all_rmsds

if __name__ == '__main__':
    path_str2 = "nobackup/graphene_xylene/auto_20/simulation/gromacs/molecule2.gro"
    path_traj = "nobackup/graphene_xylene/auto_20/simulation/gromacs/m2_trajectory_centered.xtc"
    path_pseudotraj = "nobackup/graphene_xylene/auto_20/pseudosimulation/gromacs/m2_trajectory_centered.xtc"

    uni = Universe(path_str2, path_traj)
    uni_ag = uni.select_atoms(f"all")

    uni_ref = Universe(path_str2, path_pseudotraj)
    uni_ref_ag = uni_ref.select_atoms(f"all")

    all_rmsds = loop_fast_over_trajectories(uni, uni_ref, N_rotations=30, limit_to=50)

    best_indices = np.argmin(all_rmsds, axis=1)
    print(best_indices)