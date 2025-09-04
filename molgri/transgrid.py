"""
Rotation grid is always based on the ico algorithm and an equally spaced radial distribution; the inputs are the
number of rays N_ray, and the radial factors R_min, R_max and N_rad.

Units of R_min and R_max are in ANGSTROM (new)
"""
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Voronoi, SphericalVoronoi, Delaunay
from scipy.sparse import bmat, coo_array, diags
import plotly.graph_objects as go

from molgri.polytope import IcosahedronPolytope
from molgri.constants import UNIQUE_TOL
from molgri.utils import angle_between_vectors, dist_on_sphere, sort_points_on_sphere_ccw
from molgri.plotting import draw_curve, draw_spherical_polygon


class TranslationObject():

    def __init__(self, N_ray: int, R_min: float, R_max: float, N_rad: int):
        self.radial_points = np.linspace(R_min, R_max, num=N_rad, endpoint=True)
        self.num_radial_points = N_rad

        self.polytope = IcosahedronPolytope()
        self.polytope.create_exactly_N_points(N_ray)
        self.ray_points = self.polytope.get_nodes(projection=True)
        self.num_directions = N_ray
        self.unit_voronoi = None

        self.saved_grid = None

    def __str__(self):
        return f"Translational grid with {self.num_directions} directions, each of them with {self.num_radial_points} radial points."

    def get_number_of_points(self):
        return len(self.get_grid())

    def get_grid(self) -> NDArray:
        """
        Create a product of ray_points and radial_points. We always do this in the same way:
            - firstly, all orientations at first distance
            - then all orientations at second distance
            - ....

        Outputs an array of len n_o*n_t, can have shape 1 or higher depending on the my_property

        """
        if self.saved_grid is not None:
            return self.saved_grid

        n_t = len(self.radial_points)
        n_o = len(self.ray_points)

        if len(self.ray_points.shape) > 1:
            tiled_o = np.tile(self.ray_points, reps=(n_t, 1))
            tiled_t = np.repeat(self.radial_points, n_o)[:, np.newaxis]
            result = tiled_o * tiled_t
        # special case - only radial grid
        else:
            tiled_o = np.tile(self.ray_points, reps=n_t)
            tiled_t = np.repeat(self.radial_points, n_o)[np.newaxis, :]
            result = (tiled_o * tiled_t)[0]
        assert len(result) == n_o * n_t

        self.saved_grid = result
        return self.saved_grid

    def plot(self, show_node_numbers: bool = False, show_distances: bool = False, show_vertices: bool = False,
             show_hulls_of_cells: list = None):
        points = self.get_grid()
        fig = go.Figure()
        text = list(range(self.get_number_of_points()))
        if show_node_numbers:
            fig.add_trace(
                go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], text=text, mode="text+markers",
                                                                                    marker=dict(
                    color='black')))
        else:
            fig.add_trace(
                go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], mode="markers", marker=dict(
                    color='black')))
        if show_distances:
            neighbour_types = self.get_type_of_neighbourhood()
            for row, col, value in zip(neighbour_types.row, neighbour_types.col, neighbour_types.data):
                if value == 1:
                    # spherical arch connects the points
                    draw_curve(fig, points[row], points[col], color="blue")
                elif value == 2:
                    # straight line connects the points
                    fig.add_trace(
                        go.Scatter3d(x=[points[row][0], points[col][0]],
                                     y=[points[row][1], points[col][1]],
                                     z=[points[row][2], points[col][2]],
                                     mode="lines",
                                     marker=dict(color='blue')))

        if show_vertices or show_hulls_of_cells:
            vertices = self.get_hulls()
            if show_hulls_of_cells:
                selected_vertices = [el for i, el in enumerate(vertices) if i in show_hulls_of_cells]
            else:
                selected_vertices = vertices
            for el in selected_vertices:
                vertices_color="green"
                lower_bond, upper_bond = el
                # plot the vertex points
                fig.add_trace(
                    go.Scatter3d(x=lower_bond.T[0], y=lower_bond.T[1], z=lower_bond.T[2], mode="markers",
                                 marker=dict(color=vertices_color)))
                fig.add_trace(
                    go.Scatter3d(x=upper_bond.T[0], y=upper_bond.T[1], z=upper_bond.T[2], mode="markers",
                                 marker=dict(color=vertices_color)))
                # draw curves
                draw_spherical_polygon(fig, upper_bond, color=vertices_color)
                draw_spherical_polygon(fig, lower_bond, color=vertices_color)

                # draw straight lines
                smaller_radius = np.linalg.norm(lower_bond[0])
                larger_radius = np.linalg.norm(upper_bond[0])
                for direction in upper_bond:
                    fig.add_trace(
                        go.Scatter3d(x=[smaller_radius/larger_radius*direction[0], direction[0]],
                                     y=[smaller_radius/larger_radius*direction[1], direction[1]],
                                     z=[smaller_radius/larger_radius*direction[2], direction[2]],
                                     mode="lines", marker=dict(color=vertices_color)))



        fig.show()


    def get_unit_voronoi(self):
        if self.unit_voronoi is not None:
            return self.unit_voronoi

        if self.num_directions <= 3:
            return None
        else:
            self.unit_voronoi = SphericalVoronoi(self.ray_points, radius=1.0, threshold=10**-UNIQUE_TOL)
            return self.unit_voronoi

    def _get_between_radii(self):
        if self.num_radial_points == 1:
            increment = self.radial_points[0]
        else:
            increment = self.radial_points[1]-self.radial_points[0]


        between_radii = self.radial_points + increment /2

        between_radii = np.concatenate([[0, ], between_radii])
        return between_radii

    def _layer_adjacency(self) -> coo_array:
        regions = self.get_unit_voronoi().regions
        num_regions = len(regions)

        # prepare for adjacency matrix
        rows = []
        columns = []
        elements = []

        # neighbours have at least two spherical Voronoi vertices in common
        for index_tuple in combinations(list(range(num_regions)), 2):
            set_1 = set(regions[index_tuple[0]])
            set_2 = set(regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= 2:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                elements.extend([True, True])

        adj_matrix = coo_array((elements, (rows, columns)), shape=(self.num_directions, self.num_directions))
        return adj_matrix


    def get_type_of_neighbourhood(self) -> coo_array:
        """
        Return a matrix where:
        - 0 means elements are not neighbours
        - 1 means elements are neighbours on a sphere
        - 2 means elements are neighbours in radial direction
        """
        n_points = self.get_number_of_points()
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

    def get_adjacencies(self) -> coo_array:
        neighbour_types = self.get_type_of_neighbourhood()
        return neighbour_types.astype(bool)

    def get_distances(self) -> coo_array:
        """
        Between neighbouring points you will either calculate straight lines or curved lines
        """
        neighbour_types = self.get_type_of_neighbourhood()
        distance_array = neighbour_types.copy().astype(float).tocsr()

        all_points = self.get_grid()

        for row, col, value in zip(neighbour_types.row, neighbour_types.col, neighbour_types.data):
            if value==1:
                distance_array[row, col] = dist_on_sphere(all_points[row], all_points[col])
            elif value==2:
                distance_array[row, col] = np.linalg.norm(all_points[row]-all_points[col])
            else:
                raise ValueError(f"Type of neighbourhood should only be 1 or 2, not {value}")

        distance_array = distance_array.tocoo()
        return distance_array

    def get_surfaces(self) -> coo_array:
        """
        Between neighbouring points you will either calculate pieces of a circle or spherical polygons
        """
        neighbour_types = self.get_type_of_neighbourhood()
        surface_array = neighbour_types.copy().astype(float).tocsr()

        all_points = self.get_grid()

        spherical_voronoi_areas = self.get_unit_voronoi().calculate_areas()
        all_hulls = self.get_hulls()

        for row, col, value in zip(neighbour_types.row, neighbour_types.col, neighbour_types.data):
            if value==2:
                # radial neighbours, surfaces are spherical polygons
                current_spherical_index = row % self.num_directions
                unit_area = spherical_voronoi_areas[current_spherical_index]

                # scale to a radius between both layers
                two_points = np.array([all_points[row], all_points[col]])
                radius = np.mean(np.linalg.norm(two_points, axis=1))
                surface_array[row, col] = radius**2 * unit_area
            elif value==1:
                # shared upper points
                upper_vertices_point1 = (tuple(i) for i in all_hulls[row][1])
                upper_vertices_point2 = (tuple(i) for i in all_hulls[col][1])
                upper_border_vertices = np.array(list(set(upper_vertices_point1).intersection(set(upper_vertices_point2))))
                if upper_border_vertices.shape == (2, 3):
                    radius = np.linalg.norm(upper_border_vertices[0])
                    angle = angle_between_vectors(upper_border_vertices[0], upper_border_vertices[1])
                    area = radius**2 * angle / 2
                else:
                    area=0
                    print("Warning", row, col)

                # now subtract the smaller circular sector unles zero
                lower_vertices_point1 = (tuple(i) for i in all_hulls[row][0])
                lower_vertices_point2 = (tuple(i) for i in all_hulls[col][0])
                if not np.allclose(all_hulls[row][0][0], np.zeros(3)):
                    lower_border_vertices = np.array(list(set(lower_vertices_point1).intersection(set(
                        lower_vertices_point2))))
                    if lower_border_vertices.shape == (2, 3):
                        radius = np.linalg.norm(lower_border_vertices[0])
                        angle = angle_between_vectors(lower_border_vertices[0], lower_border_vertices[1])
                        area -= radius ** 2 * angle / 2

                surface_array[row, col] = area
            else:
                raise ValueError(f"Type of neighbourhood should only be 1 or 2, not {value}")

        surface_array = surface_array.tocoo()
        return surface_array


    def get_hulls(self):
        unit_voronoi_vertices = self.get_unit_voronoi().vertices
        unit_voronoi_regions = self.get_unit_voronoi().regions

        all_points = self.get_grid()
        between_radii = self._get_between_radii()

        all_vertices = []

        for point_i, point in enumerate(all_points):
            direction_index = point_i % self.num_directions
            radial_index = point_i // self.num_directions
            # find vertices on a unit sphere
            unit_vertices = unit_voronoi_vertices[unit_voronoi_regions[direction_index]]
            # scale these vertices to in-between radii
            radius_smaller = between_radii[radial_index]
            radius_larger = between_radii[radial_index+1]

            vertices = []
            # add bottom vertices
            if np.isclose(radius_smaller, 0.0):
                vertices.append(np.zeros((1,3)))
            else:
                vertices.append(unit_vertices*np.linalg.norm(radius_smaller))
            # add upper vertices
            vertices.append(unit_vertices*np.linalg.norm(radius_larger))

            all_vertices.append(vertices)

        # cannot be an array because different points can have a different number of vertices!
        return all_vertices




if __name__ == "__main__":
    import plotly.express as px

    np.random.seed(1)
    to = TranslationObject(20, 2, 7, 3)
    print(to.get_surfaces())
    #fig = px.imshow(to.get_surfaces().toarray())
    #fig = px.imshow(to.get_adjacencies().toarray())
    to.plot(show_node_numbers=True, show_vertices=False, show_distances=False, show_hulls_of_cells=[1, 33])
    #fig.show()