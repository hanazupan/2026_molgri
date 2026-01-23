rule position_assignment_csv:
    input:
        energy_csv = f"<outputs>energy.csv",
        com_m2= f"<outputs_assignment>m2_com.npy",
        grid_info= "<outputs_network>grid_info.yaml",
        indices_csv= f"<outputs_network>indices_interpretation.csv"
    output:
        assignment_csv =  f"<outputs_assignment>translation_assignment.csv",
        translation_assignment= f"<outputs_assignment>translation_assignment.npy",
    run:
        from molgri.molecules.assignment import assign_to_cartesian_translation_grid
        from workflow.helpers.io import read_object, write_object
        import pandas as pd
        import numpy as np

        com_m2 = read_object(input.com_m2)
        energy = read_object(input.energy_csv)["Binding energy [kJ/mol]"]
        df_indices = read_object(input.indices_csv)
        df_indices = df_indices.loc[df_indices["Rotation index"] == 0]

        grid_info = read_object(input.grid_info)
        subgrids = grid_info["subgrid_points"]
        subgrid_limits = grid_info["subgrid_limits_A"]
        periodic_in = grid_info["periodic_in"]
        N_gridpoints = grid_info["subgrid_N_points"]
        data = assign_to_cartesian_translation_grid(com_m2, subgrids, subgrid_limits, periodic_in)

        df_assignments = pd.DataFrame(np.array(data).T, columns=["X index", "Y index", "Z index", "Total position index"], dtype=int)

        df_assignments["Exact COM position"] = list(map(tuple, com_m2))

        assigned_array = df_assignments["Total position index"].to_numpy()

        rows = (
            df_indices
            .set_index("Translation index")
            .loc[assigned_array]
            .reset_index()
        )

        df_assignments["Assigned position gridpoint"] = rows["Position"].to_numpy()
        df_assignments["Energy [kJ/mol]"] = energy.to_numpy()

        print(df_assignments)

        write_object(df_assignments, output.assignment_csv)
        write_object(np.array(data[3]), output.translation_assignment)



N_points_z_dir = int(config["grid"]["translation_subgrids_A"][-1][-1])

rule plot_E_per_assigned_position_grids:
    input:
        structure1 = f"<outputs_gromacs>molecule1.<ext_str>",
        assignment_csv =  f"<outputs_assignment>assignment.csv",
        grid_info= "<outputs_network>grid_info.yaml",
    output:
        plot = expand("<outputs_other_plots>position_grid_energy_{i}.png", i=range(N_points_z_dir))
    run:
        import ast
        from workflow.helpers.graphene_xylene_specific import plot_graphene
        from molgri.plotting import draw_unit_cell
        import plotly.graph_objects as go
        from MDAnalysis import Universe

        df = read_object(input.assignment_csv)
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        side_lengths = np.array([subgrid_limits[0][1], subgrid_limits[1][1]])

        unique_z_indices = df["Z index"].unique()
        unique_z_indices.sort()
        for z_index in range(N_points_z_dir):
            filtered_df = df[df["Z index"] == z_index]
            position_gridpoints = np.array([ast.literal_eval(s) for s in filtered_df["Assigned position gridpoint"].to_numpy()])

            fig = go.Figure()

            trajectory_universe = Universe(input.structure1)
            plot_graphene(trajectory_universe, fig, in_3d=False)

            if len(position_gridpoints) >= 1:
                x, y, z = position_gridpoints.T
                fig.add_trace(go.Scatter(x=position_gridpoints.T[0], y=position_gridpoints.T[1],
                    mode = "markers", opacity=0.7,
                    marker=dict(color=filtered_df["Energy [kJ/mol]"], size=10,
                        colorbar=dict(thickness=20, title="Energy [kJ/mol]", y=1.05,yanchor="top"),
                        colorscale="RdBu_r", cmin=-61,cmax=-50)))
            draw_unit_cell(fig, side_lengths, in_3d=False)
            fig.write_image(output.plot[z_index])


from workflow.helpers.io import get_atomgoup_m2, write_object
from MDAnalysis import Universe


rule rotation_assignment:
    """
    The closest rotational fit of molecule 2 is found and reported as rotation index.
    """
    input:
        structure2 = f"<outputs_gromacs>molecule2.<ext_str>",
        trajectory=f"<outputs_gromacs>m2_trajectory_centered.<ext_trj>",
        reference_pt = f"nobackup/<molecules><network>pseudosimulation/gromacs/m2_trajectory_centered.<ext_trj>"
    benchmark:
        repeat(f"<outputs_assignment>duration_rotational_assignment.txt",1)
    output:
        assigned_trajectory = f"<outputs_assignment>rotation_assignment.npy"
    run:
        # the trajectories must only contain m2 and it needs to be already centered at origin
        trajectory_universe = Universe(input.structure2, input.trajectory)
        reference_universe = Universe(input.structure2, input.reference_pt)

        N_rotations = int(config["grid"]["N_rotations"])

        #TODO if possible: use gromacs or similar to extract position array faster

        # shape (N_frames, N_atoms, 3)
        pos = np.array([ts.positions.copy() for ts in trajectory_universe.trajectory])
        # shape (N_rotations, N_atoms, 3)
        pos_ref = np.array([ts.positions.copy() for ts in reference_universe.trajectory[:N_rotations]])

        # shape (N_rotations, N_frames)
        distances = np.empty((len(pos_ref), len(pos)))

        # TODO: see if using worker pool could be helpful here

        for i, ref_structure in enumerate(pos_ref):
            distances_to_this_ref = np.linalg.norm(pos - ref_structure, axis=2)
            total_distances = distances_to_this_ref.sum(axis=1)
            distances[i] = total_distances

        best_indices = np.argmin(distances.T, axis=1)
        write_object(best_indices, output.assigned_trajectory)

rule full_assignment:
    input:
        rotation_assignment = f"<outputs_assignment>rotation_assignment.npy",
        translation_assignment= f"<outputs_assignment>translation_assignment.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    output:
        full_assignment = f"<outputs_assignment>full_assignment.npy",
    run:
        rotation_assignments = read_object(input.rotation_assignment)
        translation_assignments = read_object(input.translation_assignment)
        grid_info = read_object(input.grid_info)
        N_rotations = int(grid_info["N_rotations"])
        #N_translations = int(np.prod(grid_info["subgrid_N_points"]))
        full_assignments = translation_assignments * N_rotations + rotation_assignments
        print(full_assignments)

        write_object(full_assignments, output.full_assignment)

rule see_assignments:
    input:
        assigned_trajectory = f"<outputs_assignment>rotation_assignment.npy"
    run:
        assignments = read_object(input.assigned_trajectory)
        print(np.unique(assignments, return_counts=True))


rule trajectory_centered_at_m2_COM:
    """
    Write the whole trajectory translated in such a way that the COM of molecule 2 is at (0,0,0) in each frame.
    This is useful so we can later assign the best rotation.
    """
    input:
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure=f"<outputs_gromacs>structure.<ext_str>",
        index=f"<outputs_gromacs>index.ndx",
    output:
        trajectory=f"<outputs_gromacs>m2_trajectory_centered.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.trajectory})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n3\n" |  gmx22 trjconv  -s $(basename {input.structure})  -f $(basename {input.trajectory}) -o $(basename {output.trajectory}) -n $(basename {input.index}) -center -boxcenter zero
        cd "$initial_dir" || exit
        """