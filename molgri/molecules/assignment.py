"""
This file deals with assigning trajectory frames to their closest grid positions and orientations
"""
import pickle

import pandas as pd
from MDAnalysis import Universe, Writer
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial import KDTree

class AssignmentTool:
    """
    This tool is used to assign trajectory frames to grid cells.
    """

    def __init__(self, full_network, full_universe: Universe, n_atoms_m1: int):

        # todo: simplify grid so you don't have duplicates for every rotation
        position_nodes = full_network.list_of_position_nodes[:, 0]
        self.positions_set = np.array([position_node.translation_node.coordinate for position_node in position_nodes])
        only_one_z = self.positions_set[::20]
        only_one_z[:, 2] = 0.5
        rotation_nodes = full_network.list_of_position_nodes[0, :]
        rotation_set =  np.array([rotation_node.rotation_node.coordinate for rotation_node in rotation_nodes])


        self.universe = full_universe
        self.selection_text_molecule1 = f"bynum 1:{n_atoms_m1}"
        self.selection_text_molecule2 = f"bynum {n_atoms_m1+1}:{len(self.universe.atoms)}"
        self.atom_group_molecule1 = self.universe.select_atoms(self.selection_text_molecule1)
        self.atom_group_molecule2 = self.universe.select_atoms(self.selection_text_molecule2)

        self.align_molecule1()
        initial_com_molecule2 = self.com_molecule2_along_trajectory()
        com_molecule2 = self.move_to_rectangular_cell(initial_com_molecule2.copy())

        # explanation: this must stay this way, if you create a new array MDAnalysis keeps complaining about type
        # mismatch
        rectangular_unit_cell = self.universe.dimensions
        rectangular_unit_cell[0] = 2.46700664
        rectangular_unit_cell[1] = 4.27298084
        rectangular_unit_cell[2] = 5

        from molgri.plotting import draw_points
        fig = draw_points(only_one_z, color="red", equal_aspect=True)

        def periodic_metric(u, v):
            Lx = 2.46700664
            Ly = 4.27298084
            dx = u[0] - v[0]
            dy = u[1] - v[1]

            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)

            return np.sqrt(dx * dx + dy * dy)

        cdist_array = cdist(com_molecule2, only_one_z, metric=periodic_metric)
        idx = np.argmin(cdist_array, axis=1)

        from plotly.express.colors import qualitative
        color_list = [qualitative.Light24[i%24] for i in idx]

        fig = draw_points(initial_com_molecule2, color=color_list, fig=fig, show=True, equal_aspect=True)
        # print(dists)
        # #idx = tree.get_indices()
        # print(idx)
        # import plotly.express as px
        # fig = px.histogram(
        #     x=idx,
        #     nbins=len(set(idx)),  # one bin per integer value
        #     labels={"x": "Index", "count": "Occurrences"}
        # )
        #
        # fig.show()

    def align_molecule1(self):
        """
        As a first step for consistent assignment align molecule1 to the initial structure, which is a structure
        where the center of mass of that molecule is moved to (0,0,0).
        """
        self.universe.trajectory.add_transformations(trans.translate(-self.atom_group_molecule1.center_of_mass()))

        # ag = self.universe.select_atoms("all")
        # with Writer("/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/simulation/aligned.xtc",
        #             len(self.universe.atoms)) as w:
        #     for ts in self.universe.trajectory:
        #         w.write(ag)

    def com_molecule2_along_trajectory(self) -> NDArray:
        """
        Find the center of mass of the second molecule for each frame.
        """
        com_analysis = AnalysisFromFunction(lambda ag: ag.center_of_mass(), self.universe.trajectory,
            self.atom_group_molecule2)
        com_analysis.run()

        original_timeseries = com_analysis.results['timeseries']
        return original_timeseries

    def move_to_rectangular_cell(self, original_timeseries):
        # todo: periodic boundary conditions
        # where is the com relative to the rectangular cell
        original_timeseries[:, 0]  = original_timeseries[:, 0] % 2.46700664
        original_timeseries[:, 1] = original_timeseries[:, 1] % 4.27298084
        original_timeseries[:, 2] = 0.0
        return original_timeseries




if __name__ == "__main__":
    path_structure = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/simulation/trajectory.gro"
    path_trajectory = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/simulation/trajectory.xtc"
    path_grid = "/home/hanaz63/2026_molgri/nobackup/graphene_xylene/auto_grid_20/full_network/network.pkl"

    with open(path_grid, "rb") as f:
        my_network = pickle.load(f)

    n_m1 = 1056

    my_traj = Universe(path_structure, path_trajectory)

    a = AssignmentTool(my_network, my_traj, n_atoms_m1=n_m1)

    # import numpy as np
    # from MDAnalysis.lib.pkdtree import PeriodicKDTree
    #
    # box = my_traj.universe.dimensions
    # box[0] = 50
    # box[1] = 50
    #
    # # Random coordinates in the box
    # coords = np.random.rand(100, 3) * box[:3]
    #
    # # Build tree
    # tree = PeriodicKDTree(box=box)
    # tree.set_coords(coords, cutoff=20.0)
    #
    # # Query neighbors within 5 Å of the first atom
    # neighbors = tree.search(coords[0], radius=5.0)
    #
    # print("Neighbors of atom 0:", neighbors)