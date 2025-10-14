
import pickle
import os

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import save_npz, sparray, load_npz
import MDAnalysis as md

def write_object(my_object, filename) -> None:
    file_extension = os.path.splitext(filename)[1]
    if isinstance(my_object,np.ndarray):
        function = _write_array
    elif isinstance(my_object, sparray):
        function = _write_sparse_array
    elif isinstance(my_object, nx.Graph):
        function = _write_network
    elif file_extension == ".xtc":
        function = _write_trajectory
    elif file_extension == ".gro":
        function = _write_structure
    else:
        raise TypeError(f"Cannot write object of type {type(my_object)} to a {file_extension} file.")

    function(my_object, filename)

def read_object(filename):
    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".npy":
        function = _read_array
    elif file_extension == ".npz":
        function = _read_sparse_array
    elif file_extension == ".pkl":
        function = _read_network
    elif file_extension == ".gro":
        function = _read_molecular_structure
    else:
        raise TypeError(f"Cannot read object from file with extension {file_extension}")
    return function(filename)

def _write_network(network, filename: str) -> nx.Graph:
    with open(filename, "wb") as f:
        pickle.dump(network, f)

def _read_network(filename: str):
    my_network = None
    with open(filename, "rb") as f:
        my_network = pickle.load(f)
    return my_network

def _write_array(array, filename: str):
    np.save(filename, array)

def _read_array(filename: str) -> NDArray:
    return np.load(filename)

def _write_sparse_array(sparse_array, filename: str) -> None:
    save_npz(filename, sparse_array)

def _read_sparse_array(filename: str) -> sparray:
    return load_npz(filename)

def _write_structure(universe, filename: str) -> None:
    universe.atoms.write(filename)

def _write_trajectory(universe, filename: str) -> None:
    with md.coordinates.XTC.XTCWriter(filename, n_atoms=universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            W.write(universe.atoms)

def _read_molecular_structure(filename: str) -> md.Universe:
    return md.Universe(filename)