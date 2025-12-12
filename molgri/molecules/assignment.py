"""
This file deals with assigning trajectory frames to their closest grid positions and orientations
"""
from MDAnalysis import Universe
from numpy.typing import NDArray
import numpy as np
from scipy.spatial import KDTree

class AssignmentTool:
    """
    This tool is used to assign trajectory frames to grid cells.
    """

    def __init__(self, full_7d_grid: NDArray, full_universe: Universe, n_atoms_m1: int):
        self.grid = full_7d_grid
        self.universe = full_universe
        print(full_7d_grid)


if __name__ == "__main__":
    path_structure = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/simulation/production.gro"
    path_trajectory = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/simulation/production.xtc"
    path_grid = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/full_network/grid.npy"

    my_grid = np.load(path_grid)

    n_m1 = 1056

    my_traj = Universe(path_structure, path_trajectory)

    a = AssignmentTool(my_grid, my_traj, n_atoms_m1=n_m1)