
import pickle
import os

import networkx as nx
import numpy as np
import pandas as pd
from MDAnalysis import Universe
from numpy.typing import NDArray
from scipy.sparse import save_npz, sparray, load_npz
import MDAnalysis as md

def write_object(my_object, filename) -> None:
    file_extension = os.path.splitext(filename)[1]
    if isinstance(my_object,np.ndarray) and file_extension != ".txt":
        function = _write_array
    elif isinstance(my_object, sparray):
        function = _write_sparse_array
    elif isinstance(my_object, nx.Graph):
        function = _write_network
    elif file_extension == ".xtc":
        function = _write_trajectory
    elif file_extension == ".gro":
        function = _write_structure
    elif file_extension == ".csv":
        function = _write_csv
    elif file_extension == ".txt":
        function = _write_txt
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
    elif file_extension == ".xvg":
        function = _read_energy
    elif file_extension == ".csv":
        function = _read_csv
    elif file_extension == ".txt":
        function = _read_txt
    else:
        raise TypeError(f"Cannot read object from file with extension {file_extension}")
    return function(filename)

def _write_network(network, filename: str) -> nx.Graph:
    with open(filename, "wb") as f:
        pickle.dump(network, f)

def _read_network(filename: str):
    with open(filename, "rb") as f:
        my_network = pickle.load(f)
    return my_network

def _read_energy(filename: str):
    def _get_column_names(filename) -> list:
        result = ["Time [ps]"]
        with open(filename, "r") as f:
            for line in f:
                # parse column number
                for i in range(0, 10):
                    if line.startswith(f"@ s{i} legend"):
                        split_line = line.split('"')
                        result.append(split_line[-2])
                if not line.startswith("@") and not line.startswith("#"):
                    break
        return result

    column_names = _get_column_names(filename)
    # skip 13 rows commented with # and then also a variable amount of rows commented with @
    table = pd.read_csv(filename, sep=r'\s+', comment='@', skiprows=13, header=None, names=column_names)
    return table

def _write_txt(some_array: NDArray, filename: str):
    if np.issubdtype(some_array.dtype, np.integer):
        fmt="%d"
    else:
        fmt="%.12f"

    np.savetxt(filename, some_array, fmt=fmt)

def _read_txt(filename: str) -> NDArray:
    array_or_num = np.loadtxt(filename)
    if np.issubdtype(array_or_num.dtype, np.integer) or np.issubdtype(array_or_num.dtype, float):
        array_or_num = np.array([array_or_num])
    return array_or_num.reshape(-1)

def _write_csv(df, filename: str):
    df.to_csv(filename)

def _read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, index_col=0)

def _write_array(array, filename: str):
    np.save(filename, array)

def _read_array(filename: str) -> NDArray:
    return np.load(filename)

def _write_sparse_array(sparse_array, filename: str) -> None:
    save_npz(filename, sparse_array)

def _read_sparse_array(filename: str) -> sparray:
    return load_npz(filename)

def _write_structure(universe, filename: str) -> None:
    print(universe.atoms[-1].mass, filename)
    universe.atoms.write(filename)

def _write_trajectory(universe, filename: str) -> None:
    with md.coordinates.XTC.XTCWriter(filename, n_atoms=universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            W.write(universe.atoms)

def _read_molecular_structure(filename: str) -> md.Universe:
    return md.Universe(filename)

def get_num_atoms(structure_file:str) -> int:
    file = read_object(structure_file)
    return int(file.atoms.n_atoms)

def get_atomgoup_m1(universe_both: md.Universe, path_str1: str):
    n1 = get_num_atoms(path_str1)

    m1_atoms = universe_both.select_atoms(f"all")
    m1_atoms = m1_atoms[m1_atoms.indices < n1]
    return m1_atoms

def get_atomgoup_m2(universe_both: md.Universe, path_str1: str):
    n1 = get_num_atoms(path_str1)

    m2_atoms = universe_both.select_atoms(f"all")
    m2_atoms = m2_atoms[m2_atoms.indices >= n1]
    return m2_atoms