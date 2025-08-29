"""
Rotation grid is always based on the ico algorithm and an equally spaced radial distribution; the inputs are the
number of rays N_ray, and the radial factors R_min, R_max and N_rad.

Units of R_min and R_max are in ANGSTROM (new)
"""

import numpy as np
from numpy.typing import NDArray

from molgri.polytope import IcosahedronPolytope


class TranslationObject():

    def __init__(self, N_ray: int, R_min: float, R_max: float, N_rad: int):
        self.radial_points = np.linspace(R_min, R_max, num=N_rad, endpoint=True)

        self.polytope = IcosahedronPolytope()
        self.polytope.create_exactly_N_points(N_ray)
        self.ray_points = self.polytope.get_nodes(projection=True)




    def get_grid(self) -> NDArray:
        # create a product of ray_points and radial_points

        pass

    def get_adjacencies(self) -> NDArray:
        pass

    def get_hulls(self):
        pass

if __name__ == "__main__":
    to = TranslationObject(20, 2, 7, 15)
    print(to.polytope.get_nodes(projection=False))