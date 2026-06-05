"""
Everything here is saved to the pseudosimulation/gromacs folder: creating a structure, pseudotrajectory and calculating the energy along it.
"""
from workflow.helpers.io import read_object, write_object

rule copy_molecular_files_from_input:
    """
    Here the goal is just to start a new directory and copy molecular files there.
    """
    input:
        molecule_1 = f"<inputs_structures><molecule1>.<ext_str>",
        molecule_2 = f"<inputs_structures><molecule2>.<ext_str>",
    output:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_str>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_str>",
    run:
        from molgri.molecules.bimolecular import move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        u = mda.Universe("file.gro", format="GRO")

        # center molecules
        m1 = move_to_center(m1)
        m2 = move_to_center(m2)

        write_object(m1, output.molecule_1)
        write_object(m2, output.molecule_2)


# rule copy_molecular_files_from_input:
#     input:
#         molecule_1 = f"<inputs_structures><molecule1>.<ext_str>",
#         molecule_2 = f"<inputs_structures><molecule2>.<ext_str>"
#     output:
#         molecule_1 = f"<pseudosimulation>molecule1.<ext_str>",
#         molecule_2 = f"<pseudosimulation>molecule2.<ext_str>"
#     run:
#         import mdtraj as md
#         from molgri.molecules.bimolecular import move_to_center

#         traj1 = md.load(input.molecule_1)
#         traj1.save_gro(output.molecule_1)

#         traj2 = md.load(input.molecule_2)
#         traj2.save_gro(output.molecule_2)

#         m1 = read_object(input.molecule_1)
#         m2 = read_object(input.molecule_2)

#         # set masses manually
#         for atom in m1.atoms:
#             if atom.name.startswith("C"):
#                 atom.mass = 12.011
#             elif atom.name.startswith("H"):
#                 atom.mass = 1.008

#         for atom in m2.atoms:
#             if atom.name.startswith("C"):
#                 atom.mass = 12.011
#             elif atom.name.startswith("F"):
#                 atom.mass = 18.998

#         # center molecules
#         m1 = move_to_center(m1)
#         m2 = move_to_center(m2)

#         write_object(m1, output.molecule_1)
#         write_object(m2, output.molecule_2)

rule create_structure:
    input:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_str>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_str>",
    output:
        structure = f"<pseudosimulation>structure.<ext_str>",
    run:
        from molgri.molecules.bimolecular import get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        z_distance = float(config["grid"]["translation_subgrids_A"][-1][1])
        structure = get_bimolecular_structure(m1, m2, z_distance=z_distance)
        write_object(structure, output.structure)

rule copy_mdp_files:
    """
    Copy only mdp files that are required - e.g. for rerun you do not need a minim.mdp and nvt.mdp.
    """
    input:
        mdp = f"<inputs_gromacs>{{file_name}}.mdp",
    output:
        mdp = f"<pseudosimulation>{{file_name}}.mdp",
    run:
        import shutil
        shutil.copy(input.mdp,output.mdp)


rule copy_other_gromacs_input:
    """
    Copy the rest of necessary files to start a GROMACS calculation.
    """
    input:
        dimer_topology = f"<inputs_gromacs>topol.top",
        select_energy = f"<inputs_gromacs>select_energy",
        index = f"<inputs_gromacs>index.ndx",
        force_field_stuff = f"<inputs_gromacs>force_field_stuff/"
    output:
        dimer_topology = f"<pseudosimulation>topol.top",
        select_energy = f"<pseudosimulation>select_energy",
        index = f"<pseudosimulation>index.ndx",
        force_field_stuff = directory(f"<pseudosimulation>force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.index,output.index)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

rule create_pseudotrajectory:
    """
    Here we are creating a pseudotrajectory from two molecules and a network.
    """
    input:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_str>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_str>",
        network = f"<outputs_network>network.pkl"
    output:
        trajectory = f"<pseudosimulation>trajectory.<ext_trj>"
    run:
        from molgri.molecules.bimolecular import get_bimolecular_pseudotrajectory, move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        network = read_object(input.network)
        weights = m2.atoms.masses
        coordinates = network.create_pseudotrajectory_coordinates_from(m2.atoms.positions, weights)
        pt = get_bimolecular_pseudotrajectory(m1, m2, coordinates)
        write_object(pt, output.trajectory)


rule read_in_energies:
    """
    Here we can assign the energy to the nodes of the network.
    """
    input:
        network = f"<outputs_network>network.pkl",
        energy = f"<pseudosimulation>energy.csv",
    output:
        network_energy = f"<pseudosimulation>network_energy.pkl"
    run:
        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy["Energy [kJ/mol]"].to_numpy()
        my_network.add_node_properties(my_energy_array,"binding_energy")
        write_object(my_network, output.network_energy)
