"""
Our molecular systems are always bimolecular - consist of two structures.

In this file we use MDAnalysis to modify these structures, e.g. bring m1 and m2 together as a structure or as a
pseudotrajectory and move molecules to origin.
"""

from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray
import numpy as np
from MDAnalysis import Universe, Merge
from scipy.linalg import svd


def combine_coordinates(static_coordinates: NDArray, moving_coordinates: NDArray) -> NDArray:
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

def get_bimolecular_structure(universe_static: Universe, universe_moving: Universe, z_distance: float = None) -> Universe:
    """
    The goal here is to combine the atoms from both Universe objects to create a combined Universe. This will be
    exported as the structure file. Only a single frame is returned even if universes have trajectories attached.

    Note that the atoms of molecule 2 can be moved along the z-axis so that the two molecules don't overlap. However,
    it often plays no role whether the atoms overlap since the structure file is usually used as information on atom
    types, not positions - for programs like GROMACS or VMD.

    Args:
        universe_static (Universe): contain the atoms of molecule 1 that do not move during a pseutotrajectory
        universe_moving (Universe): contain the atoms of molecule 2 that move
        z_distance (float): if not None molecule 2 will be translated in the z-direction for this amount (unit is A)

    Returns:
        a Universe object where atoms from both are combined.
    """
    if z_distance is not None:
        universe_moving.atoms.translate(np.array([0, 0, z_distance]))
    merged_universe = Merge(universe_static.atoms, universe_moving.atoms)
    merged_universe.dimensions = universe_static.dimensions
    return merged_universe

def get_bimolecular_pseudotrajectory(universe_static: Universe, universe_moving: Universe, moving_coordinates: list)-> Universe:
    """
    Using two molecular structures and a list of coordinates, create a single Universe object that shows these two
    molecules in all given coordinates.

    Both molecules are initially centered with their centre of mass moved to the origin.

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
    for ts in combined_universe.trajectory:
        combined_universe.dimensions = universe_static.dimensions
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

def move_to_box_center(universe: Universe) -> Universe:
    """
    Move the molecule so that its center of mass is the center of the simulation box.

    Args:
        universe (Universe): a molecular object

    Returns:
        the molecule rigidly translated to box center
    """
    box_center = universe.dimensions[:3] / 2
    com = universe.atoms.center_of_mass()
    universe.atoms.translate(box_center - com)
    return universe

def move_universe_to_xy_plane(universe: Universe) -> Universe:
    """
    This is a helper function that fits an universe with one molecule (presumably somewhat planar) to the xy plane as
    much as possible using only rigid rotations and translations.

    This is useful e.g. so we can have a starting position for molecule1 that is aligned with the xy plane.

    Args:
        universe (Universe): a MDAnalysis universe object

    Returns:
        the same universe but the atoms were rigidly rotated and translated so they  best fit the xy-plane
    """
    universe.atoms.translate(-universe.atoms.center_of_mass())
    a, b, c = svd(universe.atoms.positions)
    rotated_points = np.dot(universe.atoms.positions, c.T)
    universe.atoms.positions = rotated_points
    universe.atoms.translate(-universe.atoms.center_of_mass())
    return universe