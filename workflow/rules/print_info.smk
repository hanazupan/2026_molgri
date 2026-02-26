"""
These functions can be used directly to quickly access some information
"""
from molgri.molecules.transitions import auto_determine_eigenvector_extremes
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
        print(df)
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