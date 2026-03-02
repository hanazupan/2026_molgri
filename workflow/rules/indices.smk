import numpy as np

from molgri.molecules.transitions import auto_determine_eigenvector_extremes
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



checkpoint lowest_E_indices:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        energy_csv = "{path}energy.csv"
    output:
        indices= "{path}lowest_{N}_binding_energies.txt"
    run:
        df_energy = read_object(input.energy_csv)
        required_indices = np.array(df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]").index)
        write_object(required_indices, output.indices)

checkpoint find_indices_dominant_eigenvectors:
    """
    For each eigenvector find the structures that contribute the most to the eigenvector.
    """
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
    output:
        abs_e_indices = expand(f"<outputs_indices>{{tau}}/0_eigenvector_{{j}}_largest_abs_values.txt",
            j=config["msm"]["num_extremes_to_plot"], allow_missing=True),
        pos_e_indices = expand(f"<outputs_indices>{{tau}}/{{i}}_eigenvector_{{j}}_most_positive.txt",
            i=range(1, config["msm"]["num_interesting_eigenvectors"]), j=config["msm"]["num_extremes_to_plot"],
            allow_missing=True),
        neg_e_indices= expand(f"<outputs_indices>{{tau}}/{{i}}_eigenvector_{{j}}_most_negative.txt",
            i=range(1,config["msm"]["num_interesting_eigenvectors"]),j=config["msm"]["num_extremes_to_plot"],
            allow_missing=True)
    run:
        eigenvectors = read_object(input.eigenvectors)

        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]

        N_extremes_to_plot = config["msm"]["num_extremes_to_plot"]

        zeroth_eigenvector = eigenvectors[:, 0].T
        pos_e, pos_neg = auto_determine_eigenvector_extremes(zeroth_eigenvector, N_extremes_to_plot)
        abs_e = min(pos_e, pos_neg, key=len)
        # save the absolute
        write_object(np.array(abs_e),output.abs_e_indices[0])

        for eigenvector_i in range(1, N_interesting_eigenvectors):
            higher_eigenvector = eigenvectors[:, 0].T
            pos_e, neg_e = auto_determine_eigenvector_extremes(zeroth_eigenvector, N_extremes_to_plot)
            # save the positive and negative indices
            write_object(np.array(pos_e), output.pos_e_indices[eigenvector_i-1])
            write_object(np.array(neg_e),output.neg_e_indices[eigenvector_i-1])
        write_object(np.array(abs_e),output.abs_e_indices[0])