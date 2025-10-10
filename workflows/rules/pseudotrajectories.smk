
SOME_FOLDER = "outputs/pseudotrajectories/H2O_H2O/first_experiment/"
STRUCTURE_ENDING = "xyz"
TRAJECTORY_ENDING = "xyz"

ALGORITHM_TRANSLATION = "radial"
N_RADIAL =


rule create_grid:
    output:
        network=f"{SOME_FOLDER}network.gml"
    run:
        pass


rule create_pseudotrajectory:
    input:
        molecule_1 = f"{SOME_FOLDER}molecule1.{STRUCTURE_ENDING}",
        molecule_2 = f"{SOME_FOLDER}molecule2.{STRUCTURE_ENDING}",
        network = f"{SOME_FOLDER}network.gml"
    output:
        structure = f"{SOME_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{SOME_FOLDER}trajectory.{TRAJECTORY_ENDING}"
    run:
        pass