"""
Rotation grid is always based on the ico algorithm and an equally spaced radial distribution; the inputs are the
number of rays N_directions, and the radial factors R_min, R_max and N_radial.

Units of R_min and R_max are in ANGSTROM (new)

TranslationObject can be plotted for visualization and its properties can be accessed through properties grid,
adjacency, distances, surfaces and volumes.
"""
from __future__ import annotations

from copy import copy
from itertools import combinations
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import SphericalVoronoi
from scipy.sparse import bmat, coo_array, diags
import plotly.graph_objects as go

from molgri.polytope import IcosahedronPolytope
from molgri.constants import UNIQUE_TOL
from molgri.utils import all_rows_unique, angle_between_vectors, dist_on_sphere, which_row_is_k
from molgri.plotting import draw_curve, draw_line_between, draw_points, draw_spherical_polygon


class TranslationObject():
    """
    This object gives you access to the purely translational part of the grid (directions from ico algorithm and
    radial distances from linear distribution of points between R_min and R_max.
    """

    def __init__(self, N_directions: int, R_min: float, R_max: float, N_radial: int):
        """
        Create the grid with N_directions spherical points repeated  N_radial times at radii between R_min and R_max.

        Args:
            N_directions (int): the number of directions to be created with Icosahedron subdivision
            R_min (float): minimal radius of grid points (in Angstrom)
            R_max (float): maximal radius of grid points (in Angstrom)
            N_radial (int): the number of radii to be used between R_min and R_max
        """

        # make sure input data is sensible
        if N_radial == 1: assert np.isclose(R_min, R_max)

        # create radial and directional grids separately
        self.radial_grid = np.linspace(R_min, R_max, num=N_radial, endpoint=True)
        self.num_radial_points = N_radial

        self.polytope = IcosahedronPolytope()
        self.polytope.create_exactly_N_points(N_directions)
        self.unit_sphere_grid = self.polytope.get_nodes(projection=True)
        self.num_directions = N_directions

    # PROPERTIES

    @cached_property
    def grid(self) -> NDArray:
        """
        Create a product of unit_sphere_grid and radial_grid. We always do this in the same way:
            - firstly, all orientations at first distance
            - then all orientations at second distance
            - ....

        Outputs an array of len n_o*n_t, can have shape 1 or higher depending on the my_property

        Returns:
            a (N_points, 3) numpy array where N_points is the total number of grid points and each row is the 3D
            coordinate of the point.

        """
        if len(self.unit_sphere_grid.shape) > 1:
            tiled_o = np.tile(self.unit_sphere_grid, reps=(self.num_radial_points, 1))
            tiled_t = np.repeat(self.radial_grid, self.num_directions)[:, np.newaxis]
            result = tiled_o * tiled_t
        # special case - only radial grid
        else:
            tiled_o = np.tile(self.unit_sphere_grid, reps=self.num_radial_points)
            tiled_t = np.repeat(self.radial_grid, self.num_directions)[np.newaxis, :]
            result = (tiled_o * tiled_t)[0]
        assert len(result) == self.num_directions * self.num_radial_points
        return result

    @cached_property
    def unit_voronoi(self) -> SphericalVoronoi | None:
        """
        Getter for spherical Voronoi on the unit sphere - must allow access to properties regions and vertices and
        method calculate_areas().

        Returns:
            a spherical Voronoi from scipy or a mock object for tiny grids
        """
        unit_voronoi = ReducedSphericalVoronoi(self.unit_sphere_grid)
        return unit_voronoi

    @cached_property
    def adjacency(self) -> coo_array:
        """
        Get an adjacency array for the TranslationObject.

        Returns:
            A boolean sparse array with value True at position (i, j) if the points i and j are neighbours.
        """
        return self.neighbour_types.copy().astype(bool)

    @cached_property
    def distances(self) -> coo_array:
        """
        Get distances between neighbouring points calculated either as straight line distances (in radial direction) or
        curved lines (on the sphere)

        Returns:
            a float sparse array with value d_ij at position (i, j) if i and j are neighbours and their distance is d_ij
        """
        distance_array = self.neighbour_types.copy().astype(float).tocsr()

        all_points = self.grid

        for row, col, value in zip(self.neighbour_types.row, self.neighbour_types.col, self.neighbour_types.data):
            if value == 1:
                distance_array[row, col] = dist_on_sphere(all_points[row], all_points[col])
            elif value == 2:
                distance_array[row, col] = np.linalg.norm(all_points[row] - all_points[col])
            else:
                raise ValueError(f"Type of neighbourhood should only be 1 or 2, not {value}")

        distance_array = distance_array.tocoo()
        return distance_array

    @cached_property
    def surfaces(self) -> coo_array:
        """
        Get surfaces between neighbouring points calculated either as spherical polygons (stacked neighbours one
        above the other in radial direction) or circular sectors (side-by-side neighbours on the sphere).

        Returns:
            a float sparse array with value S_ij at position (i, j) if i and j are neighbours and their dividing
            surface is S_ij
        """
        surface_array = self.neighbour_types.copy().astype(float).tocsr()
        spherical_voronoi_areas = self.unit_voronoi.calculate_areas()

        all_hulls = self.get_hulls()

        for row, col, value in zip(self.neighbour_types.row, self.neighbour_types.col, self.neighbour_types.data):
            if value == 2:
                # radial neighbours, surfaces are spherical polygons
                current_spherical_index = row % self.num_directions
                unit_area = spherical_voronoi_areas[current_spherical_index]
                # scale to a radius between both layers
                two_points = np.array([self.grid[row], self.grid[col]])
                radius = np.mean(np.linalg.norm(two_points, axis=1))
                surface_array[row, col] = radius ** 2 * unit_area
            elif value == 1:
                shared_upper = find_shared_vertices(all_hulls[row][1], all_hulls[col][1])
                shared_lower = find_shared_vertices(all_hulls[row][0], all_hulls[col][0])
                surface_array[row, col] = circular_sector_area(shared_upper, shared_lower)
            else:
                raise ValueError(f"Type of neighbourhood should only be 1 or 2, not {value}")

        surface_array = surface_array.tocoo()
        return surface_array

    @cached_property
    def volumes(self) -> NDArray:
        """
        Get the volume of the hull for each grid point.

        Returns:
            An array of volumes of shape (N_points,) in Angstrom^3

        """
        spherical_voronoi_areas = self.unit_voronoi.calculate_areas()
        points = self.grid
        hulls = self.get_hulls()

        all_volumes = []

        for i, point in enumerate(points):
            hull_smaller, hull_larger = hulls[i]
            radius_smaller = np.linalg.norm(hull_smaller[0])
            radius_larger = np.linalg.norm(hull_larger[0])
            direction_index = i % self.num_directions
            # how much of the unit surface is this spherical surface
            if self.num_directions > 1:
                percentage = spherical_voronoi_areas[direction_index] / (4 * np.pi)
            else:
                percentage = 1.0
            # the same percentage of the volume is this cell
            volume = 4 / 3 * np.pi * (radius_larger ** 3 - radius_smaller ** 3) * percentage
            all_volumes.append(volume)
        return np.array(all_volumes)

    @cached_property
    def neighbour_types(self) -> coo_array:
        """
        Return a sparse matrix where:
        - 0 means elements are not neighbours
        - 1 means elements are neighbours on a sphere
        - 2 means elements are neighbours in radial direction

        Returns:
            an integer sparse array with True at (i, j) if i and j are neighbours; i and j between 0 and
            self.num_directions
        """
        n_points = self.number_of_points
        n_o = self.num_directions
        n_t = self.num_radial_points

        # First you have neighbours that occur from being at subsequent radii and the same ray
        # Since the position grid has all orientations at first r, then all at second r ...
        # the points index_center and index_center+n_o will
        # always be neighbours, so we need the off-diagonals by n_o and -n_o
        # Most points have two neighbours this way, first and last layer have only one

        # the off-diagonals are True because they are the neighbours in radial direction
        # points i and i+num_directions are always neighbours
        same_ray_neighbours = diags((2,), offsets=self.num_directions, shape=(n_points, n_points), dtype=int,
                                    format="coo")
        same_ray_neighbours += diags((2,), offsets=-self.num_directions, shape=(n_points, n_points), dtype=int,
                                     format="coo")

        one_layer_neighbours = self._layer_adjacency()

        # Now we also want neighbours on the same level based on Voronoi discretisation
        # We first focus on the first n_o points since the set-up repeats at every radius

        # can't create Voronoi grid with <= 4 points, but then they are just all neighbours (except with itself)
        # if n_o <= 4:
        #     neig = np.ones((n_o, n_o), dtype=my_dtype) ^ np.eye(n_o, dtype=my_dtype)
        # else:
        # neig = self.o_rotations.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()

        # in case there are several translation distances, the array neig repeats along the diagonal n_t times
        if n_t > 1:
            my_blocks = [one_layer_neighbours]
            my_blocks.extend([None, ] * n_t)
            my_blocks = my_blocks * n_t
            my_blocks = my_blocks[:-n_t]
            my_blocks = np.array(my_blocks, dtype=object)
            my_blocks = my_blocks.reshape(n_t, n_t)
            same_radius_neighbours = bmat(my_blocks, dtype=int)

            for ind_n_t in range(n_t):
                smallest_row = ind_n_t * n_o <= same_radius_neighbours.row
                largest_row = same_radius_neighbours.row < (ind_n_t + 1) * n_o
                smallest_column = ind_n_t * n_o <= same_radius_neighbours.col
                largest_column = same_radius_neighbours.col < (ind_n_t + 1) * n_o
                mask = smallest_row & largest_row & smallest_column & largest_column
                same_radius_neighbours.data[mask] *= np.full(n_t, 1)[ind_n_t]
        else:
            same_radius_neighbours = coo_array(one_layer_neighbours) * np.full(n_t, 1)
        all_neighbours = same_ray_neighbours + same_radius_neighbours
        return all_neighbours.tocoo()

    @cached_property
    def number_of_points(self) -> int:
        """
        Returns:
            The number of points in a grid
        """
        return len(self.grid)

    # OTHER USEFUL FUNCTIONS

    def __str__(self):
        return f"Translational grid with {self.num_directions} directions, each of them with {self.num_radial_points} radial points."

    def plot(self, show_node_numbers: bool = False, show_distances: bool = False, show_hulls: bool = False,
             show_hulls_of_cells: list = None) -> None:
        """
        All of the plotting needs should be covered here.

        Args:
            show_node_numbers (bool): if True, each grid point is labelled by its index
            show_distances (bool): if True, arch- and straight distances between points are plotted
            show_hulls (bool): if True, the hulls (vertices and edges) of all Voronoi cells are shown
            show_hulls_of_cells (list): a possibility to only plot the hulls of some points by providing their indices
        """
        points = self.grid
        fig = go.Figure()
        # grid points
        draw_points(points, fig, label_by_index=show_node_numbers)

        if show_distances:
            color_distances = "blue"
            neighbour_types = self.neighbour_types
            for row, col, value in zip(neighbour_types.row, neighbour_types.col, neighbour_types.data):
                if value == 1:
                    # spherical arch connects the points
                    draw_curve(fig, points[row], points[col], color=color_distances)
                elif value == 2:
                    # straight line connects the points
                    draw_line_between(fig, points[row], points[col], color=color_distances)

        if show_hulls or show_hulls_of_cells:
            vertices_color = "green"
            vertices = self.get_hulls()
            # in case you don't want to plot all but only selected ones
            if show_hulls_of_cells:
                selected_vertices = [el for i, el in enumerate(vertices) if i in show_hulls_of_cells]
            else:
                selected_vertices = vertices
            for el in selected_vertices:
                lower_bond, upper_bond = el
                # plot the vertex points
                draw_points(lower_bond, fig, color=vertices_color)
                draw_points(upper_bond, fig, color=vertices_color)

                # draw curves
                draw_spherical_polygon(fig, upper_bond, color=vertices_color)
                draw_spherical_polygon(fig, lower_bond, color=vertices_color)

                # draw straight lines
                smaller_radius = np.linalg.norm(lower_bond[0])
                larger_radius = np.linalg.norm(upper_bond[0])
                for direction in upper_bond:
                    draw_line_between(fig, smaller_radius / larger_radius * direction, direction, color=vertices_color)

        fig.show()

    def get_hulls(self):
        """
        We save the vertices of the hull for each point in the grid. For each gridpoint we save a tuple of two arrays: (
        lower_border, upper_border). The lower_border array saves all vertices of this hull with the radius smaller
        than the gridpoint and the upper_border all vertices of this hull with a larger radius.

        Returns:
            A list of length N_gridpoints, each element is a tuple of two arrays, each array contains coordinates of
            vertices in rows
        """

        all_points = self.grid
        between_radii = self._get_between_radii()

        all_vertices = []

        for point_i, point in enumerate(all_points):
            direction_index = point_i % self.num_directions
            radial_index = point_i // self.num_directions
            # find vertices on a unit sphere

            unit_vertices = self.unit_voronoi.vertices[self.unit_voronoi.regions[direction_index]]
            # scale these vertices to in-between radii
            radius_smaller = between_radii[radial_index]
            radius_larger = between_radii[radial_index + 1]

            vertices = []
            # add bottom vertices
            if np.isclose(radius_smaller, 0.0):
                vertices.append(np.zeros((1, 3)))
            else:
                vertices.append(unit_vertices * np.linalg.norm(radius_smaller))
            # add upper vertices
            vertices.append(unit_vertices * np.linalg.norm(radius_larger))
            all_vertices.append(vertices)

        # cannot be an array because different points can have a different number of vertices!
        return all_vertices

    # HELPER FUNCTIONS

    def _get_between_radii(self) -> NDArray:
        """
        If your radial points are [R_1, R_1+dR, R_1+2dR ... R_1+NdR], the in-between radii are
        [0, R_1+1/2 dR, R_1+3/2 dR ... R_1+(N+1/2)dR]

        Returns:
            a 1D array of in-berween radii (in Angstrom)
        """
        if self.num_radial_points == 1:
            increment = self.radial_grid[0]
        else:
            increment = self.radial_grid[1] - self.radial_grid[0]

        between_radii = self.radial_grid + increment / 2

        between_radii = np.concatenate([[0, ], between_radii])
        return between_radii

    def _layer_adjacency(self) -> coo_array:
        """
        On the unit sphere, find which riection points are neighbours of each other.

        Returns:
            a boolean sparse array with True at (i, j) if i and j are neighbours; i and j between 0 and self.num_directions
        """
        # prepare for adjacency matrix
        rows = []
        columns = []
        elements = []

        # neighbours have at least two spherical Voronoi vertices in common
        for index_tuple in combinations(list(range(self.num_directions)), 2):
            set_1 = set(self.unit_voronoi.regions[index_tuple[0]])
            set_2 = set(self.unit_voronoi.regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= 2:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                elements.extend([True, True])

        adj_matrix = coo_array((elements, (rows, columns)), shape=(self.num_directions, self.num_directions))
        return adj_matrix


class ReducedSphericalVoronoi(SphericalVoronoi):
    """
    This layer on top of SphericalVoronoi has two purposes:
    - removing vertices that repeat
    - allowing the choice of exactly one spherical point
    """

    def __init__(self, points, radius=1.0, threshold=10 ** -UNIQUE_TOL):
        assert len(points.shape) == 2, "Must provide a 2D array of points"
        self.num_dimensions = points.shape[1]
        num_points = len(points)
        if num_points == 1:
            # this is mocked
            self.points = points
            self.vertices = np.array([[0, 0, 1]])
            self.regions = [[0]]
            # each point is assigned proportional areas
            if self.num_dimensions == 3:
                self.areas = np.array([4*np.pi])
        elif 1 < num_points <= 4:
            raise ValueError(f"For technical reasons, the number of direction can be either 1 or >4, your choice of "
                             f"{num_points} is not supported.")
        else:
            super().__init__(points, radius=radius, threshold=threshold)
            if self.num_dimensions == 3:
                self.areas = super().calculate_areas()
            self._purge_redundant_voronoi_vertices()
            # make sure no repeated vertices now
            all_rows_unique(self.vertices)

    def calculate_areas(self) -> NDArray:
        """
        This is overwritten with previous values so that the regions are not messed up again after calculating areas.

        Returns:
            an array of areas the same length as the number of points
        """
        return self.areas

    def get_adjacency_matrix(self) -> coo_array:
        """
        Adjacent points share at least dimension-1 vertices.
        """
        if len(self.points) == 1:
            return coo_array(np.zeros((1,1)))

        num_points = len(self.points)
        rows = []
        columns = []
        elements = []

        # neighbours have at least two spherical Voronoi vertices in common
        for index_tuple in combinations(list(range(num_points)), 2):
            set_1 = set(self.regions[index_tuple[0]])
            set_2 = set(self.regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= self.num_dimensions - 1:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                elements.extend([True, True])

        adj_matrix = coo_array((elements, (rows, columns)), shape=(num_points, num_points))
        return adj_matrix

    def get_hulls(self):
        return [self.vertices[region] for region in self.regions]

    def _purge_redundant_voronoi_vertices(self):
        original_vertices = copy(self.vertices)
        # correctly determines which lines to use
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        new_vertices = np.array([original_vertices[index] for index in sorted(indexes)])

        # regions
        # correctly assigns
        old2new = {old_i: which_row_is_k(new_vertices, old)[0] for old_i, old in enumerate(original_vertices)}
        old_regions = self.regions
        new_regions = []
        for i, region in enumerate(old_regions):
            fresh_region = []
            for j, el in enumerate(region):
                fresh_region.append(old2new[el])
            new_regions.append(list(set(fresh_region)))

        # now we can overwrite
        self.vertices = new_vertices
        self.regions = new_regions


def find_shared_vertices(vertices1: NDArray, vertices2: NDArray) -> NDArray:
    """
    Given coordinates of vertices around point 1 (vertices1) and coordinates of vertices around point 2 (vertices2),
    find the intersection (vertices that belong to both sets).

    Args:
        vertices1 (NDArray): array of coordinates, shape (N_1, 3)
        vertices2 (NDArray): array of coordinates, shape (N_2, 3)

    Returns:
        array of coordinates, shape (N_3, 3) where N_3 <= N_1 and N_3 <= N_2
    """

    # coordinates of points must be converted to tuples so that intersection of sets can be used
    vertices_point1 = (tuple(i) for i in vertices1)
    vertices_point2 = (tuple(i) for i in vertices2)
    border_vertices = np.array(list(set(vertices_point1).intersection(set(vertices_point2))))
    return border_vertices


def circular_sector_area(shared_upper_vertices: NDArray, shared_lower_vertices: NDArray) -> float:
    """
    Find the area of either circular sector or a difference between a smaller and bigger circular sector (same angle,
    two different radii).

    Args:
        shared_upper_vertices (NDArray): coordinates of the upper arch, should be exactly two
        shared_lower_vertices (NDArray): coordinates of the lower arch, should be exactly two or one if it is (0,0,0)

    Returns:
        area in Angstrom^2
    """
    assert shared_upper_vertices.shape == (2, 3)
    assert shared_lower_vertices.shape == (2, 3) or np.allclose(shared_lower_vertices, 0.0)

    radius_smaller = np.linalg.norm(shared_lower_vertices[0])
    radius_larger = np.linalg.norm(shared_upper_vertices[0])
    angle = angle_between_vectors(shared_upper_vertices[0], shared_upper_vertices[1])

    return (radius_larger ** 2 - radius_smaller ** 2) * angle / 2


if __name__ == "__main__":
    from molgri.plotting import show_array
    # demonstrating the use of the object
    to = TranslationObject(20, 2, 7, 3)
    to.plot(show_node_numbers=True, show_hulls=False, show_distances=False, show_hulls_of_cells=[0, 33])

    print("Volumes\n", to.volumes)

    show_array(to.adjacency.toarray(), "Adjacency")
    show_array(to.neighbour_types.toarray(), "Neighbour types")
    show_array(to.distances.toarray(), "Distances")
    show_array(to.surfaces.toarray(), "Surfaces")


