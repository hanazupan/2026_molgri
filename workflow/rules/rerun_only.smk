"""
Here are the rules that are specific to rerun and pseudotrajectory so they are not confused with e.g. how a trajectory
is generated in a real trajectory.
"""

from workflow.helpers.io import read_object, write_object

rule create_pseudotrajectory:
    """
    Here we are creating a pseudotrajectory from two molecules and a network.
    """
    input:
        molecule_1 = f"<outputs_gromacs>molecule1.<ext_str>",
        molecule_2 = f"<outputs_gromacs>molecule2.<ext_str>",
        network = f"<outputs_network>network.pkl"
    output:
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>"
    run:
        from molgri.molecules.bimolecular import get_bimolecular_pseudotrajectory, get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        network = read_object(input.network)
        weights = m2.atoms.masses
        coordinates = network.create_pseudotrajectory_coordinates_from(m2.atoms.positions, weights)
        pt = get_bimolecular_pseudotrajectory(m1, m2, coordinates)
        write_object(pt, output.trajectory)


rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        runfile = f"<outputs_gromacs>production.mdp",
        topology = f"<outputs_gromacs>topol.top",
        index=f"<outputs_gromacs>index.ndx",
        select_energy = f"<outputs_gromacs>select_energy",
        force_field_stuff = f"<outputs_gromacs>force_field_stuff/"
    shadow: "minimal"
    log:
        log = f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt", 1)
    output:
        energy = f"<outputs_gromacs>energy.xvg",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr  -n $(basename {input.index})
        gmx22 mdrun -s result.tpr -rerun $(basename {input.trajectory}) -g $(basename {log.log})
        gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """


rule read_in_energies:
    """
    Here we can assign the energy to the nodes of the network.
    """
    input:
        network = f"<outputs_network>network.pkl",
        energy = f"<outputs>energy.csv",
    output:
        network_energy = f"<outputs>network_energy.pkl"
    run:

        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy["Binding energy [kJ/mol]"].to_numpy()
        my_network.add_node_properties(my_energy_array,"binding_energy")
        write_object(my_network, output.network_energy)

rule create_index_csv:
    """
    Save which quaternion and position relate to which index.
    """
    input:
        network= f"<outputs_network>network.pkl",
        energy = f"<outputs>energy.csv"
    output:
        energy_csv = f"<outputs>indices_interpretation.csv"
    run:
        import pandas as pd
        import numpy as np

        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy["Binding energy [kJ/mol]"].to_numpy()

        translation_indices = my_network.get_translation_indices()
        rotation_indices = my_network.get_rotation_indices()
        coordinates = my_network.grid
        positions = coordinates[:, :3]
        quaternions = coordinates[:, 3:]

        df = pd.DataFrame(np.array([translation_indices, rotation_indices, my_energy_array]).T,
            columns=["Translation index", "Rotation index", "Binding energy [kJ/mol]"])

        df["Position"] = list(positions)
        df["Quaternion"] = list(quaternions)
        df.index.name = "Total index"
        # indices should be integers
        df["Translation index"] = df["Translation index"].astype("Int64")
        df["Rotation index"] = df["Rotation index"].astype("Int64")

        write_object(df, output.energy_csv)


rule print_indices_interpretation:
    """
    Use this rule if you want to quickly look at the indices and understand them.
    """
    input:
        indices_csv =f"<outputs>indices_interpretation.csv"
    run:
        df = read_object(input.indices_csv)
        # for example only filter the ones with specific rotation index
        df_filtered = df.loc[df["Rotation index"] == 5]
        df_filtered = df_filtered.sort_values(by="Binding energy [kJ/mol]",ascending=True)
        print(df_filtered.head(10))