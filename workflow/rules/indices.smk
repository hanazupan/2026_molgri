import numpy as np

from workflow.helpers.io import read_object, write_object

checkpoint all_rotations_first_position_indices:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        info_material = "<outputs_network>grid_info.yaml"
    output:
        indices= f"<outputs_indices>all_rotations_first_position.txt"
    run:
        info_grid = read_object(input.info_material)
        rotation_points = info_grid["N_rotations"]
        required_indices = np.array(list(range(0,rotation_points)))
        write_object(required_indices, output.indices)

checkpoint all_positions_first_rotation_indices:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        info_material = "<outputs_network>grid_info.yaml"
    output:
        indices= f"<outputs_indices>all_positions_first_rotation.txt"
    run:
        info_grid = read_object(input.info_material)
        position_points = np.prod(info_grid["subgrid_N_points"])
        rotation_points = info_grid["N_rotations"]
        total_N_points = position_points * rotation_points
        required_indices = np.array(list(range(0,total_N_points,rotation_points)))
        write_object(required_indices, output.indices)


rule lowest_E_indices_pseudosimulation:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        energy_csv = f"<pseudosimulation>energy.csv"
    output:
        indices= f"<outputs_indices>pseudosimulation_lowest_{{N}}_binding_energies.txt"
    run:
        df_energy = read_object(input.energy_csv)
        required_indices = np.array(df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]").index)
        write_object(required_indices, output.indices)

rule lowest_E_indices:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        energy_csv = f"<simulation>energy.csv"
    output:
        indices= f"<outputs_indices>simulation_lowest_{{N}}_binding_energies.txt"
    run:
        df_energy = read_object(input.energy_csv)
        required_indices = np.array(df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]").index)
        write_object(required_indices, output.indices)