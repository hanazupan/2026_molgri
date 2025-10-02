"""
A FullNetwork consists of
- StructureNodes that carry at least a global index but can also have the attributes of: position grid index,
orientation grid index, energy, volume, radius
- StructureEdges that carry information on: edge distance, surface area, diffusion, calculated rates.
"""
from functools import cached_property

from numpy.typing import NDArray
import numpy as np

from molgri.transgrid import circular_sector_area
from molgri.utils import dist_on_sphere,  distance_between_quaternions

# todo: structure node saves the PositionGridNode and the RotationGridNode and only accesses their properties
class StructureNode:
    def __init__(self, node_index: int, coordinate_7D: NDArray, node_type: str):
        assert node_type in ["position_grid", "rotation_grid"]
        self.node_index = node_index
        self.node_type = node_type

        self.coordinate_7D = coordinate_7D
        self.position_3d = coordinate_7D[:3]
        self.quaternion = coordinate_7D[3:]

        self.radius = np.linalg.norm(self.position_3d)

    @cached_property
    def volume(self):
        pass

class PositionGridNode:

    def __init__(self, node_index: int, coordinate_3d: NDArray, node_type: str):
        self.coordinate_3d = coordinate_3d
        self.position_index = node_index
        self.radius = np.linalg.norm(self.coordinate_3d)
        self.hull
        self.unit_voronoi_area

    @cached_property
    def volume(self) -> float:
        """
        Get the volume of the hull for each grid point.

        Returns:
            An array of volumes of shape (N_points,) in Angstrom^3

        """

        hull_smaller, hull_larger = self.hull
        radius_smaller = np.linalg.norm(hull_smaller[0])
        radius_larger = np.linalg.norm(hull_larger[0])

        # how much of the unit surface is this spherical surface
        if self.num_directions > 1:
            percentage = self.unit_voronoi_area / (4 * np.pi)
        else:
            percentage = 1.0
        # the same percentage of the volume is this cell
        volume = 4 / 3 * np.pi * (radius_larger ** 3 - radius_smaller ** 3) * percentage
        return volume


    def change_radius(self, new_radius):
        assert not np.isclose(new_radius, 0.0), "The radius of zero makes no sense"
        # change the radius
        self.radius = new_radius

        # rescale the coordinate
        self.coordinate = self.coordinate / np.linalg.norm(self.coordinate) * new_radius

        # adapt the hull


class RotationGridNode(StructureNode):

    def __init__(self, node_index: int, coordinate: NDArray, node_type: str):
        self.quaternion
        self.hull
        self.rotation_index


class StructureEdge:
    def __init__(self, node1: StructureNode, node2: StructureNode, edge_type: str):
        assert edge_type in ["radial", "directional", "orientational"]
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2

    @cached_property
    def edge_distance(self):
        if self.edge_type == "radial":
            # these are straight edges between layers of points in translational grid
            return np.linalg.norm(self.node1.position_3d - self.node2.position_3d)
        elif self.edge_type == "directional":
            return dist_on_sphere(self.node1.position_3d, self.node2.position_3d)
        else:
            return distance_between_quaternions(self.node1.quaternion, self.node2.quaternion)

    @cached_property
    def surface_area(self):
        if self.edge_type == "radial":
            # radial neighbours, surfaces are spherical polygons
            current_spherical_index = row % self.num_directions
            unit_area = spherical_voronoi_areas[current_spherical_index]
            # scale to a radius between both layers
            two_points = np.array([self.node1.position_3d, self.node2.position_3d])
            radius = np.mean(np.linalg.norm(two_points, axis=1))
            return radius ** 2 * unit_area
        elif self.edge_type == "directional":
            shared_upper = find_shared_vertices(all_hulls[row][1], all_hulls[col][1])
            shared_lower = find_shared_vertices(all_hulls[row][0], all_hulls[col][0])
            return circular_sector_area(shared_upper, shared_lower)
        else:
            # quaternion surface ares obtained as spherical polygons
            pass


    @cached_property
    def diffusion_coefficient(self):
        pass

    def calculate_rate_matrix_element(self):
        pass

def create_position_network(spherical_points, radial_points):
    pass


def create_full_network(network_of_one_layer, radial_distances):
    # you need to repeat the network_of_one_layer at different radii and introduce radial eges between them