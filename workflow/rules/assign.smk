"""
Every rule here should save to assign/ directory. Assigning is performed on the simulated trajectory and assigns every
frame of that trajectory to the best grid point. In addition, wrapping is performed, where the simulation frames are
wrapped to the smallest cuboid cell, mostly for plotting of periodic structures.
"""

from MDAnalysis import Universe, Writer
import numpy as np
import plotly.graph_objects as go

from molgri.molecules.find_unit_cell import get_cuboid_cell_side_lengths, wrap_multiple_atoms_to_cuboid_cell, wrap_to_cuboid_cell
from molgri.molecules.assignment import assign_to_cartesian_translation_grid, assign_to_best_orientation

from workflow.helpers.io import get_num_atoms, read_object, write_object

##################################### POSITION ASSIGNMENT ####################################################

rule wrap_trajectory2cuboid_cell:
    """
    This rule takes a normal (already centered etc.) trajectory of two molecules and applies periodic boundary
    conditions of the cuboid cell to the molecule2. This is not used for assignment but for plotting trajectories.
    
    This is tested - you can look at frame_i of wrapped trajectory and frame_i of simulation to confirm.
    """
    input:
        structure = f"<simulation>structure.<ext_str>",
        trajectory = f"<simulation>trajectory.<ext_trj>",
        structure1 = f"<simulation>molecule1.<ext_str>",
    benchmark:
        "<outputs_assignment>duration_wrapping_trajectory.txt"
    output:
        wrapped_trajectory = f"<outputs_assignment>wrapped_trajectory.<ext_trj>",
    run:
        n_mol1 = get_num_atoms(input.structure1)

        u = Universe(input.structure,input.trajectory)
        ag2 = u.select_atoms("all")
        ag2 = ag2[n_mol1:]


        side_lengths = get_cuboid_cell_side_lengths(input.structure1)

        with Writer(output.wrapped_trajectory, u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                ts.positions[n_mol1:] = wrap_multiple_atoms_to_cuboid_cell(ag2.center_of_mass(),
                    ts.positions[n_mol1:],
                    side_lengths,
                    wrap_only_xy=True)

                W.write(u.atoms)

rule wrap_molecule2_COM:
    """
    This rule uses gromacs' output that contains the center of mass of molecule 2 for each frame of the trajectory. This
    center of mass is wrapped to the cuboid cell and later used for position assignment.
    """
    input:
        com_m2 = f"<simulation>COM_m2.xvg",
        structure1 = f"<simulation>molecule1.<ext_str>",
    output:
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
    run:
        com_m2 = read_object(input.com_m2)
        # NEED TO CONVERT TO ANGSTROM
        com_array_m2 = 10 * com_m2.to_numpy()[:, -3:]

        write_object(com_array_m2, output.com_m2)

        # determine cuboid cell
        side_lengths = get_cuboid_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2)
        write_object(wrapped_com_m2, output.com_m2_wrapped)

rule position_assignment_csv:
    """
    Here the position gridpoint that best fits the wrapped center of mass of molecule 2 is found for each frame.
    
    We also provide a .csv file with more information, very usefula for debugging.
    """
    input:
        energy_csv = f"<simulation>energy.csv",
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
        grid_info= "<outputs_network>grid_info.yaml",
        indices_csv= f"<outputs_network>indices_interpretation.csv"
    output:
        assignment_csv =  f"<outputs_assignment>translation_assignment.csv",
        translation_assignment= f"<outputs_assignment>translation_assignment.npy",
    run:

        import pandas as pd
        import numpy as np

        com_m2 = read_object(input.com_m2)
        com_m2_wrapped = read_object(input.com_m2_wrapped)

        energy = read_object(input.energy_csv)["Energy [kJ/mol]"]
        df_indices = read_object(input.indices_csv, header=[0, 1])

        df_indices.columns = [col[0] if col[1].startswith("Unnamed") else col[1] for col in df_indices.columns]


        df_indices = df_indices.loc[df_indices["Rotation index"] == 0]

        grid_info = read_object(input.grid_info)
        subgrids = grid_info["subgrid_points"]
        subgrid_limits = grid_info["subgrid_limits_A"]
        periodic_in = grid_info["periodic_in"]
        N_gridpoints = grid_info["subgrid_N_points"]
        data = assign_to_cartesian_translation_grid(com_m2, subgrids, subgrid_limits, periodic_in)
        assigned_array = data[-1].astype(int).T

        df_assignments = pd.DataFrame(np.array(data).T, columns=["X index", "Y index", "Z index", "Total position index"], dtype=int)

        df_assignments[["x_COM", "y_COM", "z_COM"]] = pd.DataFrame(list(map(tuple, np.round(com_m2, 5))),index=df_assignments.index)
        df_assignments[
            ["x_wCOM", "y_wCOM", "z_wCOM"]] = pd.DataFrame(list(map(tuple, np.round(com_m2_wrapped, 5))),index=df_assignments.index)

        rows = (
            df_indices
            .set_index("Translation index")
            .loc[assigned_array]
            .reset_index()
        )

        df_assignments["x"] = rows["x"]
        df_assignments["y"] = rows["y"]
        df_assignments["z"] = rows["z"]


        #df_assignments["Assigned position gridpoint"] = list(map(tuple, np.round(position_array, 5)))
        df_assignments["Energy [kJ/mol]"] = energy.to_numpy()

        tuples = [
            ("Direction indices", "X_index"),
            ("Direction indices", "Y_index"),
            ("Direction indices", "Z_index"),
            ("Total position index", ""),
            ("Exact COM position", "x_COM"),
            ("Exact COM position", "y_COM"),
            ("Exact COM position", "z_COM"),
            ("Wrapped COM position", "x_wCOM"),
            ("Wrapped COM position", "y_wCOM"),
            ("Wrapped COM position", "z_wCOM"),
            ("Assigned position gridpoint", "x"),
            ("Assigned position gridpoint", "y"),
            ("Assigned position gridpoint", "z"),
            ("Energy [kJ/mol]", ""),
        ]

        df_assignments.columns = pd.MultiIndex.from_tuples(tuples)

        write_object(df_assignments, output.assignment_csv)
        write_object(np.array(data[3]), output.translation_assignment)



##################################### ROTATION ASSIGNMENT ####################################################


rule rotation_assignment:
    """
    The closest rotational fit of molecule 2 is found and reported as rotation index.
    """
    input:
        structure2 = f"<pseudosimulation>molecule2.<ext_str>",
        trajectory=f"<simulation>m2_trajectory_centered.<ext_trj>",
        reference_pt = f"<pseudosimulation>m2_trajectory_centered.<ext_trj>"
    benchmark:
        repeat(f"<outputs_assignment>duration_rotational_assignment.txt",1)
    output:
        assigned_trajectory = f"<outputs_assignment>rotation_assignment.npy"
    run:
        # the trajectories must only contain m2 and it needs to be already centered at origin
        trajectory_universe = Universe(input.structure2, input.trajectory)
        reference_universe = Universe(input.structure2, input.reference_pt)

        N_rotations = int(config["grid"]["N_rotations"])

        # shape (N_frames, N_atoms, 3)
        pos = np.array([ts.positions.copy() for ts in trajectory_universe.trajectory])
        # shape (N_rotations, N_atoms, 3)
        pos_ref = np.array([ts.positions.copy() for ts in reference_universe.trajectory[:N_rotations]])

        best_indices = assign_to_best_orientation(pos, pos_ref)
        write_object(best_indices, output.assigned_trajectory)

##################################### FULL ASSIGNMENT ####################################################

checkpoint full_assignment:
    """
    We combine the rotation assignment and position assignment into the final, full assignment index.
    """
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
        full_assignments = translation_assignments * N_rotations + rotation_assignments
        write_object(full_assignments, output.full_assignment)


rule count_assignments:
    """
    Some quick plotting to show which states are low in energy (lots of assignments) and which are high (basically no
    assignments).
    """
    input:
        translation_assignment=f"<outputs_assignment>translation_assignment.npy",
        rot_assignment=f"<outputs_assignment>rotation_assignment.npy",
        full_assignment=f"<outputs_assignment>full_assignment.npy",
    output:
        translation_assignment = f"<outputs_other_plots>hist_translation_assignment.png",
        rot_assignment = f"<outputs_other_plots>hist_rotation_assignment.png",
        full_assignment = f"<outputs_other_plots>hist_full_assignment.png",
    run:
        for input_file, output_file in zip(input, output):
            data = read_object(input_file)
            fig = go.Figure(
                go.Histogram(
                    x=data,
                    xbins=dict(
                        start=data.min() - 0.5,
                        end=data.max() + 0.5,
                        size=1)
                )
            )

            fig.write_image(output_file, scale=3)

