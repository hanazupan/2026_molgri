rule position_assignment_csv:
    input:
        energy_csv = f"<outputs>energy.csv",
        com_m2= f"<outputs_assignment>m2_com.npy",
        grid_info= "<outputs_network>grid_info.yaml",
        indices_csv= f"<outputs_network>indices_interpretation.csv"
    output:
        assignment_csv =  f"<outputs_assignment>assignment.csv"
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

rule quaternion_assignment_csv:
    input:
        energy_csv = f"<outputs>energy.csv",
        m2_quaternions = f"<outputs_assignment>m2_quaternions.npy",
        grid_info= "<outputs_network>grid_info.yaml",
        indices_csv= f"<outputs_network>indices_interpretation.csv"
    output:
        assignment_csv =  f"<outputs_assignment>assignment_quaternion.csv"
    run:
        from molgri.molecules.assignment import assign_to_rotation_grid
        from workflow.helpers.io import read_object, write_object
        import pandas as pd
        import numpy as np

        m2_quaternions = read_object(input.m2_quaternions)
        energy = read_object(input.energy_csv)["Binding energy [kJ/mol]"]
        df_indices = read_object(input.indices_csv)
        df_indices = df_indices.loc[df_indices["Translation index"] == 0]

        grid_info = read_object(input.grid_info)
        quaternions = grid_info["quaternions"]

        data = assign_to_rotation_grid(m2_quaternions, np.array(quaternions))

        print(data)

        df_assignments = pd.DataFrame(np.array(data).T, columns=["Rotation index"], dtype=int)

        df_assignments["Exact quaternion"] = list(map(tuple, m2_quaternions))

        assigned_array = df_assignments["Rotation index"].to_numpy()

        rows = (
            df_indices
            .set_index("Rotation index")
            .loc[assigned_array]
            .reset_index()
        )

        df_assignments["Assigned quaternion gridpoint"] = rows["Quaternion"].to_numpy()
        df_assignments["Energy [kJ/mol]"] = energy.to_numpy()
        print(df_assignments)
        write_object(df_assignments, output.assignment_csv)


rule molecule2_quaternions:
    """
    For every frame in the trajectory, find the exact quaternion needed to go from reference structure of
    molecule2 to structure of molecule2 in this frame (this is exact if trajectory is done with rigid molecules,
    otherwise it must be an approximation).
    """
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure1 = f"<outputs_gromacs>molecule1.<ext_str>",
        structure2 = f"<outputs_gromacs>molecule2.<ext_str>",
    output:
        m2_quaternions = f"<outputs_assignment>m2_quaternions.npy",
    run:
        from workflow.helpers.io import get_atomgoup_m1, get_atomgoup_m2, write_object
        from scipy.spatial.transform import Rotation
        from MDAnalysis import Universe

        import numpy as np
        from molgri.molecules.assignment import _determine_positive_directions, _complex_mdanalysis_func

        trajectory_universe = Universe(input.structure, input.trajectory)
        ag_m2 = get_atomgoup_m2(trajectory_universe, input.structure1)
        ag_m1 = get_atomgoup_m1(trajectory_universe, input.structure1)

        reference_m2 = Universe(input.structure2)
        reference_principal_axes = reference_m2.atoms.principal_axes().T
        inverse_pa = np.linalg.inv(reference_principal_axes)
        reference_direction = _determine_positive_directions(reference_m2)
        print(reference_direction)
        print("reference principal axes\n", reference_principal_axes, "\n")

        # run_per_frame = partial(_complex_mdanalysis_func,
        #                         ag=ag_m2,
        #                         reference_direction=reference_direction)
        # frame_values = np.arange(stop=len(trajectory_universe.trajectory)) #len(trajectory_universe.trajectory)
        # with Pool(1) as worker_pool:
        #     direction_frames = worker_pool.map(run_per_frame, frame_values)

        direction_frames= []
        for ts in trajectory_universe.trajectory:
            direction_frames.append(_complex_mdanalysis_func(ts.frame, ag_m2, reference_direction))
        direction_frames= np.array(direction_frames)
        print(direction_frames)
        print("pre-matrix")
        result_as_matrix = np.matmul(direction_frames,inverse_pa)
        print("pre-quaternion")
        result_as_quat = Rotation.from_matrix(result_as_matrix).as_quat(scalar_first=True)
        print("pre-writing")
        print(result_as_quat)
        write_object(result_as_quat, output.m2_quaternions)


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
        from molgri.molecules.assignment import loop_fast_over_trajectories
        # the trajectories must only contain m2 and it needs to be already centered at origin
        trajectory_universe = Universe(input.structure2, input.trajectory)
        reference_universe = Universe(input.structure2, input.reference_pt)

        N_rotations = int(config["grid"]["N_rotations"])

        # loop over rotations and measure their distance to the set of available rotations
        all_rmsds = loop_fast_over_trajectories(trajectory_universe, reference_universe, N_rotations)

        best_indices = np.argmin(all_rmsds, axis=1)
        print(best_indices)
        write_object(best_indices, output.assigned_trajectory)

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