"""
Rotation grid is always based on the cube4D algorithm; the only input is the number of rotations N_rot.
"""

from numpy.typing import NDArray
from molgri.polytope import Cube4DPolytope


class RotationObject():

    def __init__(self, N_rot: int):
        self.polytope = Cube4DPolytope()
        self.N_rot = N_rot

        while len(self.polytope.get_half_of_hypercube()) < self.N_rot:
            self.polytope.divide_edges()
        self.grid = self.polytope.get_half_of_hypercube(N=self.N_rot, projection=True)

    def get_grid(self) -> NDArray:
        pass

    def get_adjacencies(self) -> NDArray:
        pass

    def get_hulls(self):
        pass