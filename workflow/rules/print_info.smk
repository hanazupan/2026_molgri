"""
These functions can be used directly to quickly access some information
"""
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigs

from molgri.molecules.rate_merger import delete_rows_columns, sqra_determine_indices_never_visited_states, \
    msm_determine_indices_never_visited_states
from molgri.molecules.transitions import DecompositionTool, SQRA, auto_determine_eigenvector_extremes
from workflow.helpers.io import read_object

rule print_lowest_energies:
    """
    Use this rule if you want to quickly look at the indices of the lowest energies.
    """
    input:
        energy_csv =f"<simulation>energy.csv"
    run:
        df = read_object(input.energy_csv)
        df = df.sort_values(by="Binding energy [kJ/mol]",ascending=True)
        print(df.head(5))

rule print_position_assignment:
    input:
        energy_csv = rules.position_assignment_csv.output.assignment_csv
    run:
        df = read_object(input.energy_csv)
        print(df.loc[83546])

rule print_assignment:
    input:
        trans_csv = f"<outputs_assignment>translation_assignment.csv",
        translation_assignment = f"<outputs_assignment>translation_assignment.npy",
        rot_assignment = f"<outputs_assignment>rotation_assignment.npy",
        full_assignment = f"<outputs_assignment>full_assignment.npy",
    run:
        my_assignments = read_object(input.rot_assignment)
        print("Rot assignment: ", my_assignments[[5390, 44854, 83546]])
        my_assignments = read_object(input.translation_assignment)
        print("Trans assignment: ", my_assignments[[5390, 44854, 83546]])
        my_assignments = read_object(input.full_assignment)
        print(my_assignments[[5390, 44854, 83546]])

        df = read_object(input.trans_csv, header = [0,1])
        print(df)
        print(df.loc[83546])


rule print_indices_interpretation:
    """
    Use this rule if you want to quickly look at the indices and understand them.
    """
    input:
        indices_csv =f"<outputs_network>indices_interpretation.csv"
    run:
        df = read_object(input.indices_csv)
        print(df.iloc[70:90])
        # for example only filter the ones with specific rotation index
        #df_filtered = df.loc[df["Rotation index"] == 5]
        #print(df_filtered.head(10))

rule print_position_subgrids:
    input:
        grid_info = rules.save_basic_grid_information.output.info_material
    run:
        grid_info = read_object(input.grid_info)
        print("X gridpoints: ", grid_info["subgrid_points"][0])
        print("Y gridpoints: ", grid_info["subgrid_points"][1])
        print("Z gridpoints: ", grid_info["subgrid_points"][2])


import numpy as np


rule print_eigenvector_statistics:
    input:
        eigenvectors = "<outputs_transitions>10/eigenvectors.npy",
        indices_csv=f"<outputs_network>indices_interpretation.csv",
    run:
        import plotly.graph_objects as go

        eigenvectors = read_object(input.eigenvectors)
        indices_csv = read_object(input.indices_csv)

        first_eigenvector = eigenvectors[:, 0].T
        lower_extremes, upper_extremes = auto_determine_eigenvector_extremes(first_eigenvector, N_extremes_to_plot=10)

        for el in lower_extremes:
            row_of_table = indices_csv.loc[str(int(el))]
            print(str(int(el)), int(row_of_table["Rotation index"]), np.round(float(row_of_table["Position"]), 2), np.round(float(row_of_table["Position.1"]), 2), np.round(float(row_of_table["Position.2"]), 2))

        for higher_i in range(1, 4):
            print("Eigenvector ", higher_i)
            first_eigenvector = eigenvectors[:, higher_i].T
            lower_extremes, upper_extremes = auto_determine_eigenvector_extremes(first_eigenvector,N_extremes_to_plot=10)

            print("Lower extremes: ", lower_extremes)
            for el in lower_extremes:
                row_of_table = indices_csv.loc[str(int(el))]
                print(str(int(el)),int(row_of_table["Rotation index"]),np.round(float(
                    row_of_table["Position"]),2),np.round(float(row_of_table["Position.1"]),2),np.round(float(
                    row_of_table["Position.2"]),2))

            print("Upper extremes: ", upper_extremes)
            for el in upper_extremes:
                row_of_table = indices_csv.loc[str(int(el))]
                print(str(int(el)),int(row_of_table["Rotation index"]),np.round(float(
                    row_of_table["Position"]),2),np.round(float(row_of_table["Position.1"]),2),np.round(float(
                    row_of_table["Position.2"]),2))


rule try_out_tiny_sqra:
    input:
        adjacency = "<outputs_network>adjacency.npz",
        energies = "<pseudosimulation>energy.csv",
    run:
        my_energy = read_object(input.energies)
        energies = my_energy["Binding energy [kJ/mol]"].to_numpy()
        transition_matrix = read_object(input.adjacency).astype(np.float64).toarray()

        #transition_matrix = np.array([[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0], [1,0,1,1,0, 0], [0,1,0,0,0,0]], dtype=np.float64)
        #energies = np.array([1, 5, 105000, 10, 70000, 7]) #[1, 5, 205000, 10, 70000, 7]
        diff_energies = energies[:, None] - energies[None, :]
        pi_exponent = np.round(diff_energies,14) / 100
        transition_matrix *= np.exp(pi_exponent)
        # normalize
        sums = transition_matrix.sum(axis=1)
        sums = np.array(sums).squeeze()
        diag_array = np.diag(-sums)
        transition_matrix = transition_matrix + diag_array
        np.set_printoptions(precision=3,suppress=True,linewidth=np.inf)
        #print(np.round(transition_matrix[:10,:10], 3))

        # for row in transition_matrix:
        #     if np.any(~np.isfinite(row)):
        #         mask = np.isfinite(row) & (row != 0)
        #
        #         count = np.count_nonzero(mask)
        #         # when it contains other elements they are extremely large or extremely small
        #         if count > 0:
        #             elements = row[mask]
        #             print("New one")
        #             print(count)
        #             print(elements)


        # too_large = np.where(transition_matrix.diagonal() < -1e100)[0]
        # not_finite = np.where(~np.isfinite(transition_matrix.diagonal()))[0]
        #
        # all_bad_ones = list(not_finite)
        # all_bad_ones.extend(list(too_large))
        # all_bad_ones.sort()
        # all_bad_ones = np.array(all_bad_ones)
        # print(all_bad_ones[:20])

        sparse_arr = csr_array(transition_matrix)


        reduced_rate_matrix, to_keep = delete_rows_columns(sparse_arr,"sqra")
        print("rate inf ",len(np.where(np.isinf(transition_matrix.data))[0]))
        print("reduced rate inf ",len(np.where(np.isinf(reduced_rate_matrix.data))[0]))
        #print(reduced_rate_matrix[10, :])
        #reduced_rate_matrix, to_keep = delete_rows_columns(reduced_rate_matrix,bad_ones,"sqra")

        reduced_rate_matrix = reduced_rate_matrix.toarray()
        #print(reduced_rate_matrix)
        #print(np.round(reduced_rate_matrix[:10,:10], 3))


rule try_out_sqra:
    input:
        energies = "<pseudosimulation>energy.csv",
        volumes = "<outputs_network>volumes.npy",
        distances = "<outputs_network>distances.npz",
        surfaces = "<outputs_network>surfaces.npz"
    params:
        T_in_K = 293,
        diffusion_coefficient = 1,
    run:
        my_energy = read_object(input.energies)
        my_energy_array = my_energy["Binding energy [kJ/mol]"].to_numpy()
        volumes = read_object(input.volumes)
        distances = read_object(input.distances)
        surfaces = read_object(input.surfaces)

        sqra = SQRA(energies=my_energy_array,volumes=volumes,distances=distances,surfaces=surfaces)


        rate_matrix = sqra.get_rate_matrix(params.diffusion_coefficient,params.T_in_K)

        reduced_rate_matrix, to_keep = delete_rows_columns(rate_matrix,"sqra")

        print("rate ",len(np.where(np.isinf(rate_matrix.data))[0]))
        print("reduced rate ",len(np.where(np.isinf(reduced_rate_matrix.data))[0]))

        dc = DecompositionTool(rate_matrix, np.arange(10000),10000)
        dc = DecompositionTool(reduced_rate_matrix, to_keep, 10000)
        eigenval, eigenvec = dc.decompose_sqra()
        print(eigenval)

        for evec in eigenvec.T:
            print("Eigenvector")
            print(pd.DataFrame(evec).describe())