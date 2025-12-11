"""
Here we use existing networks and apply them to molecules.
"""
from workflow.helpers.io import read_object, write_object
from workflow.helpers.PATHS import NAME_SIMULATION_FOLDER, PATH_INPUT_MOLECULES, NAME_PT_FOLDER, NAME_NETWORK_FOLDER


MOLECULE_1_NAME = config["pseudotrajectory"]["molecule_1"]
MOLECULE_2_NAME = config["pseudotrajectory"]["molecule_2"]
STRUCTURE_ENDING = config["pseudotrajectory"]["structure_ending"]
TRAJECTORY_ENDING = config["pseudotrajectory"]["trajectory_ending"]


rule all_pseudotrajectory:
    input:
        structure = f"{{some_path}}{NAME_PT_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{{some_path}}{NAME_PT_FOLDER}trajectory.{TRAJECTORY_ENDING}",


rule copy_molecular_files_from_input:
    """
    Here the goal is just to start a new directory and copy molecular files there.
    """
    input:
        molecule_1 = f"{PATH_INPUT_MOLECULES}{MOLECULE_1_NAME}.{STRUCTURE_ENDING}",
        molecule_2 = f"{PATH_INPUT_MOLECULES}{MOLECULE_2_NAME}.{STRUCTURE_ENDING}",
    output:
        molecule_1 = f"{{some_path}}{NAME_PT_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{{some_path}}{NAME_PT_FOLDER}molecule2.{STRUCTURE_ENDING}",
    run:
        from molgri.molecules.bimolecular import move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        # center molecules
        m1 = move_to_center(m1)
        m2 = move_to_center(m2)

        write_object(m1, output.molecule_1)
        write_object(m2, output.molecule_2)

rule create_only_structure:
    input:
        molecule_1 = f"{{some_path}}{NAME_PT_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{{some_path}}{NAME_PT_FOLDER}molecule2.{STRUCTURE_ENDING}",
    output:
        structure = f"{{some_path}}{NAME_SIMULATION_FOLDER}structure.{STRUCTURE_ENDING}",
    run:
        from molgri.molecules.bimolecular import get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        structure = get_bimolecular_structure(m1, m2)
        write_object(structure, output.structure)


rule create_pseudotrajectory:
    """
    Here we are creating a pseudotrajectory from two molecules and a network.
    """
    input:
        molecule_1 = f"{{some_path}}{NAME_PT_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{{some_path}}{NAME_PT_FOLDER}molecule2.{STRUCTURE_ENDING}",
        network = f"{{some_path}}{NAME_NETWORK_FOLDER}network.pkl"
    output:
        structure = f"{{some_path}}{NAME_PT_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{{some_path}}{NAME_PT_FOLDER}trajectory.{TRAJECTORY_ENDING}"
    run:
        from molgri.molecules.bimolecular import get_bimolecular_pseudotrajectory, get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        network = read_object(input.network)
        weights = m2.atoms.masses
        coordinates = network.create_pseudotrajectory_coordinates_from(m2.atoms.positions, weights)

        structure = get_bimolecular_structure(m1, m2)
        pt = get_bimolecular_pseudotrajectory(m1, m2, coordinates)

        write_object(structure, output.structure)
        write_object(pt, output.trajectory)


