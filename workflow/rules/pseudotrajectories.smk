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

