from abc import abstractmethod, ABC
from functools import cached_property

import networkx as nx
import numpy as np
from numpy.typing import NDArray

class AbstractNode(ABC):

    @abstractmethod
    def __lt__(self, other: "AbstractNode") -> bool:
        pass

    @abstractmethod
    def hull(self) -> NDArray:
        pass

    @abstractmethod
    def volume(self) -> float:
        pass

class AbstractNetwork(nx.Graph, ABC):

    """
    Just a bit enhanced networkx Graph that I use for my RotationNetwork, TranslationNetwork and FullNetwork.
    The assumptions are:
        - all nodes are objects that can be sorted (have implemented __lt__)
        - all nodes have the properties hull and volume
        - edges have properties edge_type and methods are implemented to further calculate numeric_edge_type, distance
         and surface
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_all_edge_properties()

    @cached_property
    def sorted_nodes(self):
        nodes = [node for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def volumes(self) -> NDArray:
        volumes = [node.volume() for node in self.sorted_nodes]
        volumes = np.array(volumes)
        return volumes

    @cached_property
    def hulls(self):
        hulls = [node.hull for node in self.sorted_nodes]
        return hulls

    @abstractmethod
    def grid(self) -> NDArray:
        pass

    @abstractmethod
    def _distances(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate distance is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _surfaces(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate surface is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _numerical_edge_type(self) -> dict:
        """
        Must return a dict in which for every edge type a number is returned.
        """
        pass

    def calculate_all_edge_properties(self):
        df_edges = nx.to_pandas_edgelist(self)
        print(df_edges)
        # now list all properties to be calculated
        df_edges["numerical_edge_type"] = df_edges.apply(
            lambda row: self._numerical_edge_type(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["distance"] = df_edges.apply(
            lambda row: self._distances(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["surface"] = df_edges.apply(
            lambda row: self._surfaces(row.to_dict())[row["edge_type"]], axis=1)
        for attribute in ["distance", "surface", "numerical_edge_type"]:
            nx.set_edge_attributes(self, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)

    @cached_property
    def adjacency_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=bool)

    @cached_property
    def adjacency_type_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=bool, weight="numerical_edge_type")

    @cached_property
    def distance_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="distance")

    @cached_property
    def surface_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="surface")