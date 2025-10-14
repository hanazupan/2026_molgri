"""
In this file, the only goal is creating and selecting points on (hyper) spheres - we do not do any hulls, spherical
Voronois etc.

The main algorithm that may be subject to change in the future is select_a_node_to_delete, a system to decide which
nodes should be removed first.
"""

from abc import ABC, abstractmethod
from itertools import product
from typing import Hashable, Iterable, List

import networkx as nx
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.constants import pi, golden
from scipy.sparse import coo_array
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from molgri.plotting import draw_curve, draw_line_between, draw_points
from molgri.utils import (find_inverse_quaternion, normalise_vectors, remove_bottom_half_quaternions, which_row_is_k,
                          q_in_upper_sphere)


class NewPolytope(ABC):

    """
    A polytope is a d-dim object consisting of a set of nodes (vertices) and connections between them (edges) saved
    in self.G (graph).

    The nodes are integers, useful for us are their attributes "node" (3D coordinate on the polygon) and "projection"
    (coordinate projected on a sphere of radius 1) as well as "level" (on which level of subdivision was this node
    added). Edges have no special properties.

    The getter for all nodes and their attributes is get_nodes().

    The basic polytope is created when the object is initiated. All further divisions can be performed
    with the function self.divide_edges(), but if you know exactly how many points you want in the end, use
    create_exactly_N_points.

    The function plot() is very useful for visualization
    """

    np.random.seed(1)

    def __init__(self, d: int = 3):
        self.G = nx.Graph()
        self.current_level = 0
        self.side_len = 0
        self.d = d


        # function that depends on the object being created
        self._create_level0()

    def __str__(self):
        return f"Polytope up to level {self.current_level}"

    def get_nodes(self, projection: bool = False, indices: bool = False) -> NDArray:
        """
        The default option returns an array of polygon node coordinates. The only getter to use for nodes,
        don't access self.G.nodes directly.


        Args:
            projection (bool): if True, return projected node coordinates instead
            indices (bool): if True, return node indices instead

        Returns:
            a) all polygon node coordinates as a (N, 3)-dimensional numpy array OR
            b) all projected node coordinates as a (N, 3)-dimensional numpy array OR
            c) all indices of nodes as a (N,)-dimensional numpy array
        """
        if indices:
            # node indices are mostly consecutive integers, but this is helpful in case some have been removed
            return np.array([key for key, value in sorted(self.G.nodes.items())])
        if projection:
            attribute_name = "projection"
        else:
            attribute_name = "node"

        # we always use this function as a getter to make sure nodes are always sorted in the same manner
        sorted_nodes = [value[attribute_name] for key, value in sorted(self.G.nodes.items())]
        return np.array(sorted_nodes)

    def create_exactly_N_points(self, N: int) -> None:
        """
        Start the initial distribution, keep dividing the edges until you have more points than N, then remove points and
        reconnect edges until you have exactly N points.
        """
        while self.G.number_of_nodes() < N:
            self.divide_edges()
        while self.G.number_of_nodes() > N:
            node_to_remove = select_a_node_to_delete(self.G)
            remove_and_reconnect(self.G, node_to_remove)

    def plot(self, show_nodes: bool = True, show_projected_nodes: bool = True,
             show_vertices: bool = True, show_node_numbers: bool = False):
        """
       The general-use plotting function showing polygon- and/or projected node coordinates and straight or curved
       edges between them.
        """
        fig = go.Figure()
        nodes = self.get_nodes(projection=False)
        projected_nodes = self.get_nodes(projection=True)
        indices = self.get_nodes(projection=False, indices=True)
        adjacencies = nx.adjacency_matrix(self.G)
        if show_nodes:
            draw_points(nodes, fig, label_by_index=show_node_numbers, custom_labels=indices)
            # potentially show straight-line edges
            if show_vertices:
                rows, columns = adjacencies.nonzero()
                for row, col in zip(rows, columns):
                    draw_line_between(fig, nodes[row], nodes[col])

        if show_projected_nodes:
            draw_points(projected_nodes, fig, label_by_index=show_node_numbers, custom_labels=indices, color="green")
            # potentially show curved-line edges
            if show_vertices:
                rows, columns = adjacencies.nonzero()
                for row, col in zip(rows, columns):
                    draw_curve(fig, projected_nodes[row], projected_nodes[col], color="green")
        fig.show()


########################################################################################################################
#
#               CREATION & SUBDIVISION METHODS
#
########################################################################################################################

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""
        self._end_of_divison()

    def divide_edges(self):
        """
        Subdivide once by putting a new point at mid-point of each existing edge and replacing this sub-edge with
        two edges from the two old to the new point.
        """
        self._add_mid_edge_nodes()
        self._end_of_divison()


    def _end_of_divison(self):
        """
        Use at the end of _create_level0 or divide edges. This automatically insreases the attribute level (of
        division) of novel nodes and halves the side legth.
        """
        self.current_level += 1
        self.side_len = self.side_len / 2

    def _add_mid_edge_nodes(self):
        """
        For each edge in the system, add a new point in the middle of the edge and replace the previous long edge with
        two shorter ones.
        """
        indices_to_add = list(self.G.edges())
        self._add_average_point_and_edges(indices_to_add, add_edges=True)
        # remove the old connection: old_point1 -------- old_point2
        for old1, old2 in indices_to_add:
            self.G.remove_edge(old1, old2)


    def _add_edges_of_len(self, edge_len: float, wished_levels: List[int] = None, only_seconds: bool = True,
                          only_face: bool = True):
        """
        Finds and adds all possible edges of specifies length between existing nodes (optionally only between nodes
        fulfilling face/level/second neighbour condition to save computational time).

        In order to shorten the time to search for appropriate points, it is advisable to make use of filters:
         - only_seconds will only search for connections between points that are second neighbours of each other
         - wished_levels defines at which division level the points between which the edge is created should be
         - only_face: only connections between points on the same face will be created

        Args:
            edge_len: length of edge on the polyhedron surface that is condition for adding edges
            wished_levels: at what level of division should the points that we want to connect be?
            only_seconds: if True, search only among the points that are second neighbours
            only_face: if True, search only among the points that lie on the same face
        """
        if wished_levels is None:
            wished_levels = [self.current_level, self.current_level]
        else:
            wished_levels.sort(reverse=True)
        assert len(wished_levels) == 2

        # central indices of all points within wished level
        selected_level = [n for n, d in self.G.nodes(data=True) if d['level'] == wished_levels[0]]
        for new_node in selected_level:
            # searching only second neighbours of the node if this option has been selected
            if only_seconds:
                sec_neighbours = list(second_neighbours(self.G, new_node))
                sec_neighbours = [n for n in sec_neighbours if self.G.nodes[n]["level"] == wished_levels[1]]
            else:
                sec_neighbours = [n for n in self.G.nodes if self.G.nodes[n]["level"] == wished_levels[1]]
            for other_node in sec_neighbours:
                new_node_point = self.G.nodes[new_node]["node"]
                other_node_point = self.G.nodes[other_node]["node"]
                node_dist = np.linalg.norm(new_node_point-other_node_point)
                # check face criterion
                if not only_face or self._find_face([new_node, other_node]):
                    # check distance criterion
                    if np.isclose(node_dist, edge_len):
                        self.G.add_edge(new_node, other_node)


    def _add_polytope_point(self, polytope_point: ArrayLike, face: set = None, face_neighbours_indices=None):
        """
        This is the only method that adds nodes to self.G.

        Args:
            polytope_point (ArrayLike): coordinates on the surface of polyhedron
            face (set or None): a set of integers that assign this point to one or more faces of the polyhedron
            face_neighbours_indices (): a list of neighbours from which we can determine the face(s) of the new point (
            overriden by face if given)

        Returns:

        """
        if face_neighbours_indices is None:
            face_neighbours_indices = []
        if face is None:
            face = self._find_face(face_neighbours_indices)

        # make sure that this node doesn't exist yet
        all_existing_coordinates = self.get_nodes()
        # don't add the point if it is there already
        if len(all_existing_coordinates) > 0 and len(which_row_is_k(all_existing_coordinates, polytope_point)) > 0:
            return

        node_value = self.G.number_of_nodes()
        self.G.add_node(node_value, level=self.current_level, node=polytope_point,
                        face=face, projection=normalise_vectors(polytope_point))
        return node_value

    def _add_average_point_and_edges(self, list_old_indices: list, add_edges=True):
        """
        A helper function to _add_mid_edge_nodes.
        Given a list in which every item is a set of nodes, add a new point that is the average of them and also add
        new edges between the newly created points and all of the old ones.

        It adds new edges and does not delete any existing ones.

        Args:
            list_old_indices: each item is a list of already existing nodes - the average of their polytope_points
            should be added as a new point
        """
        # indices that are currently in a sublist will be averaged and that will be a new point
        for old_points in list_old_indices:
            all_coordinates = []
            for old_point in old_points:
                all_coordinates.append(self.G.nodes[old_point]["node"])

            # new node is the midpoint of the old ones
            new_coordinate = np.average(np.array(all_coordinates), axis=0)

            node_value = self._add_polytope_point(new_coordinate, face_neighbours_indices=old_points)

            if add_edges and node_value is not None:
                for old_point in old_points:
                    self.G.add_edge(node_value, old_point)

########################################################################################################################
#
#               OTHER HELP FUNCTIONS
#
########################################################################################################################

    def _find_face(self, node_list: list) -> set:
        """
        Find the face that is common between all nodes in the list. (May be an empty set.)

        Args:
            node_list: a list in which each element is a tuple of coordinates that has been added to self.G as a node

        Returns:
            the set of faces (may be empty) that all nodes in this list share
        """
        face = set(self.G.nodes[node_list[0]]["face"])
        for neig in node_list[1:]:
            faces_neighbour_vector = self.G.nodes[neig]["face"]
            face = face.intersection(set(faces_neighbour_vector))
        return face





class PolyhedronFromG(NewPolytope):

    """
    A mock polyhedron created from an existing graph. No guarantee that it actually represents a polyhedron. Useful for
    side tasks like visualisations, searches ...
    """

    def __init__(self, G: nx.Graph):
        super().__init__()
        self.G = G

    def _create_level0(self):
        pass



class Cube4DPolytope(NewPolytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube, its center or the center of its face. It is possible to subdivide the sides, in that case
    the volumes of sub-cubes are fully sub-divided.

    Special: the attribute "face" that in 3D polyhedra actually means face here refers to the cell (3D object) to which
    this specific node belongs.
    """

    def __init__(self):
        super().__init__(d=4)

    def create_exactly_N_points(self, N: int) -> NDArray:
        """
        Start the initial distribution, keep dividing the edges until you have more points than N, then remove points and
        reconnect edges until you have exactly N points.
        """
        all_in_upper_sphere = []
        while len(all_in_upper_sphere) < N:
            self.divide_edges()
            current_nodes = self.get_nodes(projection=True)
            all_in_upper_sphere = remove_bottom_half_quaternions(current_nodes)
        while len(all_in_upper_sphere) > N:
            node_to_remove = select_a_node_to_delete(self.G)
            remove_and_reconnect(self.G, node_to_remove)
            current_nodes = self.get_nodes(projection=True)
            all_in_upper_sphere = remove_bottom_half_quaternions(current_nodes)
        return np.array(all_in_upper_sphere)

    def get_nodes(self, projection: bool = False, indices: bool = False, only_upper: bool = True) -> NDArray:
        result = super().get_nodes(projection=projection, indices=indices)
        if only_upper and projection:
            return remove_bottom_half_quaternions(result)
        if not only_upper and projection:
            # first remove, then add exact copies to the bottom
            remaining_nodes = remove_bottom_half_quaternions(result)
            N = len(remaining_nodes)
            all_points = np.zeros((2 * N, 4))
            all_points[:N] = remaining_nodes
            for i in range(N):
                inverse_q = find_inverse_quaternion(remaining_nodes[i])
                all_points[N + i] = inverse_q
            return all_points
        return result


    def plot(self, show_nodes: bool = True, show_projected_points: bool = False, show_vertices: bool = True,
             show_node_numbers: bool = False):
        """
        The plot of a 4D object must be a bit special. We show the 8 cubes of the hypercube like we would show the
        six faces of a cube separately. Of course this means that the same edge vertices and edges are repeated!

        - the 8 cube vertices repeat on 4 different cubes
        - the points on the edges of cubes repeat on 3 cubes
        - the points on the faces repeat on 2 cubes
        - the points inside the faces do not repeat
        """
        if show_projected_points:
            print("Cannot display projected points in 3D.")

        individual_cells = self.get_all_cells()

        fig = make_subplots(rows=2, cols=4, specs=[[{'type': 'scene'}]*4,[{'type': 'scene'}]*4])

        if show_nodes:
            for i, cell in enumerate(individual_cells):
                row = 1 if i < 4 else 2
                col = i + 1 if i < 4 else i + 1 - 4
                draw_points(cell.get_nodes(), fig, label_by_index=show_node_numbers, custom_labels=cell.get_nodes(
                    indices=True), row=row, col=col)

                if show_vertices:
                    adjacency = nx.adjacency_matrix(cell.G).astype(bool).tocoo()
                    for i, j in zip(adjacency.row, adjacency.col):
                        coo_1 = cell.get_nodes()[i]
                        coo_2 = cell.get_nodes()[j]
                        draw_line_between(fig, coo_1, coo_2, row=row, col=col)
        fig.show()


    def __str__(self):
        return f"Cube4D up to level {self.current_level}"

    def _create_level0(self):
        self.side_len = 2 * np.sqrt(1/4)
        # create vertices
        vertices = list(product((-self.side_len/2, self.side_len/2), repeat=4))
        faces = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 8, 9, 10, 11], [0, 1, 4, 5, 8, 9, 12, 13],
                 [0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15], [2, 3, 6, 7, 10, 11, 14, 15],
                 [4, 5, 6, 7, 12, 13, 14, 15], [8, 9, 10, 11, 12, 13, 14, 15]]

        assert len(vertices) == 16
        assert np.all([np.isclose(x, 0.5) or np.isclose(x, -0.5) for row in vertices for x in row])
        assert len(set(vertices)) == 16
        vertices = np.array(vertices)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            belongs_to = [face_i for face_i, face in enumerate(faces) if i in face]
            self._add_polytope_point(vert, face=set(belongs_to))
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        self._add_edges_of_len(self.side_len*np.sqrt(2), wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        self._add_edges_of_len(self.side_len * np.sqrt(3), wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        super()._create_level0()

    def divide_edges(self):
        """Before or after dividing edges, make sure all relevant connections are present. Then perform a division of
        all connections (point in the middle, connection split in two), increase the level and halve the side_len."""
        super().divide_edges()
        print("BEFORE EDGES", self.G.number_of_edges())
        self._add_edges_of_len(2*self.side_len, wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=False)
        print("BEFORE FACE DIAGONALS", self.G.number_of_edges())
        len_square_diagonals = 2*self.side_len*np.sqrt(2)
        self._add_edges_of_len(len_square_diagonals, wished_levels=[self.current_level-1, self.current_level-1],
                              only_seconds=False)
        print("BEFORE BODY DIAGONALS", self.G.number_of_edges())
        len_cube_diagonals = 2 * self.side_len * np.sqrt(3)
        self._add_edges_of_len(len_cube_diagonals, wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=False)
        print("AFTER ALL", self.G.number_of_edges())

########################################################################################################################
#
#               CUBE 4D-SPECIFIC METHODS
#
########################################################################################################################

    def get_half_of_hypercube(self, projection: bool = False, N: int = None) -> NDArray:
        """
        Select only half of points in a hypercube polytope in such a manner that double coverage is eliminated.

        Args:
            projection: if True, return points projected on a hypersphere
            N: if you only want to return N points, give an integer here

        Returns:
            a list of elements, each of them either a node in c4_polytope.G or its central_index

        How selection is done: select a list of all nodes (each node a 4-tuple) that have non-negative
        first coordinate. Among the nodes with first coordinate equal zero, select only the ones with
        non-negative second coordinate etc.

        Points are always returned sorted in the order of increasing central_index
        """

        projected_points = self.get_nodes(projection=True)
        unique_projected_points = [p for p in projected_points if q_in_upper_sphere(p)]

        all_ci = []
        for upp in unique_projected_points:
            all_ci.append(which_row_is_k(projected_points, upp)[0])
        all_ci.sort()

        N_available = len(all_ci)
        # can't order more points than there are
        if N is None:
            N = N_available
        if N > N_available:
            raise ValueError(f"Cannot order more points than there are! N={N} > {N_available}")

        # DO NOT use N as an argument, as you first need to select half-hypercube
        return self.get_nodes(projection=projection)[all_ci][:N]

    def get_all_cells(self, include_only: ArrayLike = None) -> List[PolyhedronFromG]:
        """
        Returns 8 sub-graphs belonging to individual cells of hyper-cube. These polytopes are re-labeled with 3D
        coordinates so that they can be plotted

        Args:
            include_only (ArrayLike): a list of nodes that should be included in result;

        Returns:
            a list, each element of a list is a 3D polyhedron corresponding to a cell in hypercube
        """
        all_subpoly = []
        if include_only is None:
            include_only = self.get_nodes(indices=True)
        for cell_index in range(8):
            nodes = [node_i for node_i, data in self.G.nodes(data=True) if cell_index in data["face"] and node_i in
                       include_only]
            subgraph = self.G.subgraph(nodes).copy()
            # find the component corresponding to the constant 4th dimension
            if subgraph.number_of_nodes() > 0:
                arr_nodes = np.array([data["node"] for node, data in subgraph.nodes(data=True)])
                num_dim = 4
                dim_to_keep = list(np.where(~np.all(arr_nodes == arr_nodes[0, :], axis=0))[0])
                removed_dim = max(set(range(num_dim)).difference(set(dim_to_keep)))
                # change the node property in the data
                dict_changes = {}
                for node_i, data in subgraph.nodes(data=True):
                    old_node = data["node"]
                    new_node = np.array([old_node[d] for d in range(num_dim) if d != removed_dim])
                    dict_changes[node_i] = {"node": new_node}
                nx.set_node_attributes(subgraph, dict_changes)
            # create a 3D polyhedron and use its plotting functions
            sub_polyhedron = PolyhedronFromG(subgraph)
            all_subpoly.append(sub_polyhedron)
        return all_subpoly

    def get_cdist_matrix(self, only_half_of_cube: bool = True, N: int = None) -> NDArray:
        """
        Update for quaternions: can decide to get cdist matrix only for half-hypersphere.

        Args:
            only_half_of_cube (bool): select True if you want only one half of the hypersphere
            N (int): number of points you want included in the cdist matrix

        Returns:
            a symmetric array (N, N) in which every item is a (hyper)sphere distance between points
        """
        if only_half_of_cube:
            only_nodes = self.get_half_of_hypercube(N=N, projection=False)
        else:
            only_nodes = self.get_nodes(N=N, projection=False)

        return super().get_cdist_matrix(only_nodes=[tuple(n) for n in only_nodes])

    def get_polytope_adj_matrix(self, include_opposing_neighbours=True, only_half_of_cube=True):
        adj_matrix = super().get_polytope_adj_matrix().toarray()

        if include_opposing_neighbours:
            ind2opp_index = dict()
            for n, d in self.G.nodes(data=True):
                ind = d["central_index"]
                opp_n = find_opposing_q(n, self.G)
                opp_ind = self.G.nodes[opp_n]["central_index"]
                if opp_n:
                    ind2opp_index[ind] = opp_ind
            for i, line in enumerate(adj_matrix):
                for j, el in enumerate(line):
                    if el:
                        adj_matrix[i][ind2opp_index[j]] = True
        if only_half_of_cube:
            available_indices = self.get_half_of_hypercube()
            available_indices = [self.G.nodes[tuple(n)]["central_index"] for n in available_indices]
            # Create a new array with the same shape as the original array
            extracted_arr = np.empty_like(adj_matrix, dtype=float)
            extracted_arr[:] = np.nan

            # Extract the specified rows and columns from the original array
            extracted_arr[available_indices, :] = adj_matrix[available_indices, :]
            extracted_arr[:, available_indices] = adj_matrix[:, available_indices]
            adj_matrix = extracted_arr
        return coo_array(adj_matrix)

    def get_neighbours_of(self, point_index, include_opposing_neighbours=True, only_half_of_cube=True):
        adj_matrix = self.get_polytope_adj_matrix(include_opposing_neighbours=include_opposing_neighbours,
                                                  only_half_of_cube=only_half_of_cube).toarray()
        if not only_half_of_cube:
            return np.nonzero(adj_matrix[point_index])[0]
        else:
            # change indices because adj matrix is smaller
            available_nodes = self.get_half_of_hypercube()
            available_is = [self.G.nodes[tuple(n)]["central_index"] for n in available_nodes]

            if point_index in available_is:
                adj_ind = np.nonzero(adj_matrix[available_is.index(point_index)])[0]

                real_ind = []

                for el in adj_ind:
                    if el in available_is:
                        real_ind.append(available_is.index(el))
                return real_ind
            else:
                return []


class IcosahedronPolytope(NewPolytope):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node
    is a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in
    the middle of each triangle side.
    """

    def __init__(self):
        super().__init__(d=3)

    def __str__(self):
        return f"Icosahedron up to level {self.current_level}"

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                 [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                 [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        self.side_len = 1 / np.sin(2 * pi / 5)
        # create vertices
        vertices = [(-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0),
                    (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden),
                    (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)]
        vertices = np.array(vertices) * self.side_len / 2
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            set_of_faces = set(faces_i for faces_i, face in enumerate(faces) if i in face)
            self._add_polytope_point(vert, face=set_of_faces)
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # perform end of creation
        super()._create_level0()

    def divide_edges(self):
        """
        Subdivide once. If previous faces are triangles, adds one point at mid-point of each edge. If they are
        squares, adds one point at mid-point of each edge + 1 in the middle of the face. New points will have a higher
        level attribute.
        """
        self._add_mid_edge_nodes()
        self._end_of_divison()
        self._add_edges_of_len(self.side_len*2, wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=True)


#######################################################################################################################
#                                          GRAPH HELPER FUNCTIONS
#######################################################################################################################


def second_neighbours(graph: nx.Graph, node: Hashable) -> Iterable:
    """
    Yield second neighbors of node in graph. Ignore second neighbours that are also first neighbours.
    Second neighbors may repeat!

    Example:

        5------6
        |      |
        2 ---- 1 ---- 3 ---- 7
               |      |
               |__8___|

    First neighbours of 1: 2, 6, 3, 8
    Second neighbours of 1: 5, 7
    """
    direct_neighbours = list(graph.neighbors(node))
    # don't repeat the same second neighbour twice
    seen_seconds = []
    for neighbor_list in [graph.neighbors(n) for n in direct_neighbours]:
        for n in neighbor_list:
            if n != node and n not in direct_neighbours and n not in seen_seconds:
                # if no edge there, put it there just to record the distance
                seen_seconds.append(n)
                yield n



def remove_and_reconnect(g: nx.Graph, node: int):
    """Remove a node and reconnect edges, adding the properties of previous edges together."""
    sources = list(g.neighbors(node))
    targets = list(g.neighbors(node))

    new_edges = list(product(sources, targets))
    # remove self-loops
    new_edges = [(edge[0], edge[1], ) for j, edge in enumerate(new_edges) if
                 edge[0] != edge[1]]
    g.add_edges_from(new_edges)
    g.remove_node(node)

def select_a_node_to_delete(G, alg="random"):
    """
    Among the nodes with the smallest number of edges, randomly select a node to delete.
    """
    if alg == "smallest_num_edges":
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=False)
        sorted_nodes = np.array(sorted_nodes)
        degrees_of_nodes = np.array([G.degree(node) for node in sorted_nodes])
        mask = degrees_of_nodes == np.min(degrees_of_nodes)
        choose_from = sorted_nodes[mask]
        return np.random.choice(choose_from)
    elif alg == "random":
        return np.random.choice(G.nodes())


def find_opposing_q(node, G):
    """
    Node is one node in graph G.

    Return the node of the opposing point if it is in G, else None
    """
    all_nodes_dict = {n: G.nodes[n]['projection'] for n in G.nodes}
    projected = all_nodes_dict[node]
    opposing_projected = - projected.copy()
    opposing_projected = tuple(opposing_projected)
    # return the non-projected point if available
    for n, d in G.nodes(data=True):
        projected_n = d["projection"]
        if np.allclose(projected_n, opposing_projected):
            return n
    return None


if __name__ == "__main__":
    import time
    from molgri.utils import all_rows_unique

    # VISUALIZE AN ICOSAHEDRON
    # t1 = time.time()
    # my_ico = IcosahedronPolytope()
    # my_ico.create_exactly_N_points(39)
    # my_nodes = my_ico.get_nodes()
    # my_p_nodes = my_ico.get_nodes(projection=True)
    # t2 = time.time()
    #
    # print(f"Time: {t2-t1} s")
    #
    # my_ico.plot(show_nodes=True, show_projected_nodes=True, show_vertices=True, show_node_numbers=True)

    # VISUALIZE A HYPERCUBE
    t1 = time.time()
    my_hypercube = Cube4DPolytope()
    #my_hypercube.divide_edges()
    my_hypercube.create_exactly_N_points(8)
    t2 = time.time()
    print(f"Time: {t2 - t1} s")

    nodes = my_hypercube.get_nodes()
    print(my_hypercube.get_nodes(projection=True, only_upper=True))
    my_hypercube.plot(show_nodes=True, show_node_numbers=False, show_vertices=True)


