
SOME_FOLDER = "outputs/pseudotrajectories/H2O_H2O/first_experiment/"

MOLECULE_1_NAME = "water"
MOLECULE_2_NAME = "water"
STRUCTURE_ENDING = "gro"
TRAJECTORY_ENDING = "xtc"

rule all:
    input:
        structure = f"{SOME_FOLDER}structure.{STRUCTURE_ENDING}"


rule copy_molecular_files_from_input:
    input:
        molecule_1 = f"inputs/one_molecule_structures/{MOLECULE_1_NAME}.{STRUCTURE_ENDING}",
        molecule_2 = f"inputs/one_molecule_structures/{MOLECULE_2_NAME}.{STRUCTURE_ENDING}",
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
        network = "output/networks/hypercube-25-cartesian_nonperiodic-test/full_network.pkl"
    output:
        structure = f"{SOME_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{SOME_FOLDER}trajectory.{TRAJECTORY_ENDING}"
    run:
        import MDAnalysis as md
        import pickle

        m1 = md.Universe(input.molecule_1)
        m2 = md.Universe(input.molecule_2)

        with open(input.network, "rb") as f:
            network = pickle.load(f)

        frames = network.get_psudotrajectory(m1, m2)

        with md.Writer(output.structure,m1.atoms.n_atoms+m2.atoms.n_atoms) as W:
            W.write(frames[0])

        with md.Writer(output.trajectory,m1.atoms.n_atoms+m2.atoms.n_atoms) as W:
            for frame in frames:
                W.write(frame)
