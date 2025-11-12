"""
This is the file in which we are using MDAnalysis to merge universes of static and moving parts of the system and
therefore create pseudotrajectories from atoms and frames.
"""

from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray
import numpy as np
from MDAnalysis import Universe, Merge


def combine_coordinates(static_coordinates: NDArray, moving_coordinates: NDArray):
    """
    For each frame of the trajectory, we want to stack the coordinates of the first molecule (that do not change at all)
    with the coordinates of the second molecule that change with very step.

    Args:
        static_coordinates (NDArray): array of shape (N_atoms_m1, 3)
        moving_coordinates (NDArray): array of shape (N_frames, N_atoms_m2, 3)

    Returns:
        an array of shape (N_frames, N_atoms_m1+N_atoms_m2, 3)
    """
    assert len(static_coordinates.shape) == 2 and static_coordinates.shape[-1] == 3
    assert len(moving_coordinates.shape) == 3 and moving_coordinates.shape[-1] == 3
    N_frames = moving_coordinates.shape[0]

    # here we change the shape to (1, N_atoms_m1, 3)
    static_coordinates_expanded = static_coordinates[np.newaxis, ...]
    # here we change the shape to (N_frames, N_atoms_m1, 3)
    static_coordinates_tiled = np.tile(static_coordinates_expanded, (N_frames, 1, 1))
    # and now merge both
    merged_coordinates = np.concatenate([static_coordinates_tiled, moving_coordinates], axis=1)
    return merged_coordinates

def get_bimolecular_structure(universe_static: Universe, universe_moving: Universe) -> Universe:
    """
    The goal here is to combine the atoms from both Universe objects to create a combined Universe. This will be
    exported as the structure file. Only a single frame  is returned even if universes have trajectories attached.

    Note that the atoms are not moved, so if both universes are centered at zero,
    they likely overlap. However, a structure file is a formal requirement of some programs like GROMACS or VMD.

    Args:
        universe_static (Universe): contain the atoms of molecule 1 that do not move during a pseutotrajectory
        universe_moving (Universe): contain the atoms of molecule 2 that move

    Returns:
        a Universe object where atoms from both are combined.
    """
    return Merge(universe_static.atoms, universe_moving.atoms)

def get_bimolecular_pseudotrajectory(universe_static: Universe, universe_moving: Universe, moving_coordinates: list)-> Universe:
    """
    Using two molecular structures and a list of coordinates, create a single Universe object that shows these two
    molecules in all given coordinates.

    Both molecules will be initially centered at their centre of mass.

    Args:
        universe_static (Universe): first molecular object with N_1 atoms
        universe_moving (Universe): second molecular object with N_2 atoms
        moving_coordinates (list): a list of coordinates, each entry is a (N_1+N_2, 3) array that gives the current
            position of both molecules

    Returns:
        a universe of both molecules with a trajectory of the same length as the moving_coordinates list
    """
    m1 = move_to_center(universe_static)
    m2 = move_to_center(universe_moving)

    structure = get_bimolecular_structure(m1, m2)
    full_coordinates = combine_coordinates(m1.atoms.positions, np.array(moving_coordinates))
    combined_universe = Universe(structure._topology, full_coordinates, format=MemoryReader)
    return combined_universe

def move_to_center(universe: Universe) -> Universe:
    """
    A helper function to move a molecule so that its center of mass is at (0,0,0).

    Args:
        universe (Universe): a molecular object

    Returns:
        the molecule rigidly translated to (0,0,0)
    """
    com = universe.atoms.center_of_mass()
    universe.atoms.positions -= com
    return universe