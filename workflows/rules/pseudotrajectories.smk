from workflows.helpers.io import read_object, write_object
from workflows.helpers.PATHS import PATH_INPUT_MOLECULES, PATH_OUTPUT_PTS, PATH_OUTPUT_NETWORKS


MOLECULE_1_NAME = config["molecule_1"]
MOLECULE_2_NAME = config["molecule_2"]
STRUCTURE_ENDING = config["structure_ending"]
TRAJECTORY_ENDING = config["trajectory_ending"]
NETWORK_NAME = config["unique_network_name"]
SOME_FOLDER = f"{PATH_OUTPUT_PTS}/{MOLECULE_1_NAME}_{MOLECULE_2_NAME}/{NETWORK_NAME}/"


rule all:
    input:
        structure = f"{SOME_FOLDER}structure.{STRUCTURE_ENDING}"


rule copy_molecular_files_from_input:
    input:
        molecule_1 = f"{PATH_INPUT_MOLECULES}{MOLECULE_1_NAME}.{STRUCTURE_ENDING}",
        molecule_2 = f"{PATH_INPUT_MOLECULES}{MOLECULE_2_NAME}.{STRUCTURE_ENDING}",
    output:
        molecule_1 = f"{SOME_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{SOME_FOLDER}molecule2.{STRUCTURE_ENDING}",
    run:
        import shutil

        shutil.copy(input.molecule_1, output.molecule_1)
        shutil.copy(input.molecule_2, output.molecule_2)


rule create_pseudotrajectory:
    input:
        molecule_1 = f"{SOME_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{SOME_FOLDER}molecule2.{STRUCTURE_ENDING}",
        network = f"{PATH_OUTPUT_NETWORKS}{NETWORK_NAME}/full_network/network.pkl"
    output:
        structure = f"{SOME_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{SOME_FOLDER}trajectory.{TRAJECTORY_ENDING}"
    run:
        from molgri.molecules.pseudotrajectory import create_pseudotrajectory

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        network = read_object(input.network)

        pt = create_pseudotrajectory(m1, m2, network)

        write_object(pt, output.structure)
        write_object(pt, output.trajectory)


