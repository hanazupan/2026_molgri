"""
General (not molecular) plots in other_plots directory.
"""
import ast
import numpy as np
import plotly.graph_objects as go
from MDAnalysis import Universe

from molgri.plotting import draw_unit_cell
from workflow.helpers.graphene_xylene_specific import plot_graphene
from workflow.helpers.io import read_object, write_object

N_points_z_dir = int(config["grid"]["translation_subgrids_A"][-1][-1])

def input_violin_E(wc):
    if wc.pseudo_or_sim == "pseudosimulation":
        return "<pseudosimulation>energy.csv"
    else:
        return "<simulation>energy.csv"

rule violin_plot_E_distributions:
    input:
        input_violin_E
    output:
        violin_plot = "<outputs_other_plots>{pseudo_or_sim}_violin_plot_energies.png"
    run:
        from molgri.plotting import show_violin
        df = read_object(input[0])

        max_energy = config['analysis']['upper_E_limit']
        energies = df["Binding energy [kJ/mol]"]

        show_violin(energies, max_energy, "Binding Energy", save_as=output.violin_plot, show=False)


rule plot_E_per_assigned_position_grids:
    input:
        structure1 = f"<pseudosimulation>molecule1.<ext_str>",
        assignment_csv =  f"<outputs_assignment>translation_assignment.csv",
        grid_info= "<outputs_network>grid_info.yaml",
    output:
        plot = expand("<outputs_other_plots>position_grid_energy_{i}.png", i=range(N_points_z_dir))
    run:

        df = read_object(input.assignment_csv, header=[0,1])

        print(df.columns)

        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        side_lengths = np.array([subgrid_limits[0][1], subgrid_limits[1][1]])

        unique_z_indices = df[('Direction indices', 'Z_index')].unique()
        unique_z_indices.sort()
        for z_index in range(N_points_z_dir):
            filtered_df = df[df[('Direction indices', 'Z_index')] == z_index]

            x = filtered_df[('Assigned position gridpoint','x')].to_numpy()
            y = filtered_df[('Assigned position gridpoint','y')].to_numpy()
            z = filtered_df[('Assigned position gridpoint','z')].to_numpy()


            position_gridpoints = np.column_stack((x,y,z))

            fig = go.Figure()

            trajectory_universe = Universe(input.structure1)
            plot_graphene(trajectory_universe, fig, in_3d=False)

            if len(position_gridpoints) >= 1:
                x, y, z = position_gridpoints.T
                fig.add_trace(go.Scatter(x=position_gridpoints.T[0], y=position_gridpoints.T[1],
                    mode = "markers", opacity=0.7,
                    marker=dict(color=filtered_df[('Energy [kJ/mol]', 'Unnamed: 14_level_1')], size=10,
                        colorbar=dict(thickness=20, title="Energy [kJ/mol]", y=1.05,yanchor="top"),
                        colorscale="RdBu_r", cmin=-61,cmax=-50)))
            draw_unit_cell(fig, side_lengths, in_3d=False)
            fig.write_image(output.plot[z_index])