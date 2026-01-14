"""
Here we copy individual molecules and combine them to structure.gro. Pseudotrajectory creation can be found in rerun_only.smk.
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
        molecule_1 = f"<outputs_gromacs>molecule1.<ext_str>",
        molecule_2 = f"<outputs_gromacs>molecule2.<ext_str>",
    run:
        from molgri.molecules.bimolecular import move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        # center molecules
        m1 = move_to_center(m1)
        m2 = move_to_center(m2)

        write_object(m1, output.molecule_1)
        write_object(m2, output.molecule_2)

rule create_structure:
    input:
        molecule_1 = f"<outputs_gromacs>molecule1.<ext_str>",
        molecule_2 = f"<outputs_gromacs>molecule2.<ext_str>",
    output:
        structure = f"<outputs_gromacs>structure.<ext_str>",
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
        mdp = f"<outputs_gromacs>{{file_name}}.mdp",
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
        dimer_topology = f"<outputs_gromacs>topol.top",
        select_energy = f"<outputs_gromacs>select_energy",
        index = f"<outputs_gromacs>index.ndx",
        force_field_stuff = directory(f"<outputs_gromacs>force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.index,output.index)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

