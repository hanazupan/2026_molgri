"""
Every rule here should save to assign/ directory.
"""

from MDAnalysis import Universe, Writer
import numpy as np

from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths, wrap_multiple_atoms_to_cuboid_cell, wrap_to_cuboid_cell
from molgri.molecules.assignment import assign_to_cartesian_translation_grid

from workflow.helpers.io import get_num_atoms, get_atomgoup_m1, get_atomgoup_m2, read_object, write_object

##################################### POSITION ASSIGNMENT ####################################################

rule wrap_trajectory2cuboid_cell:
    """
    This rule takes a normal (already centered etc.) trajectory of two molecules and applies periodic boundary
    conditions of the cuboid cell to the molecule2. This is useful so we can assign the best position.
    
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


        side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        print(side_lengths)

        with Writer(output.wrapped_trajectory, u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                ts.positions[n_mol1:] = wrap_multiple_atoms_to_cuboid_cell(ag2.center_of_mass(),
                    ts.positions[n_mol1:],
                    side_lengths,
                    wrap_only_xy=True)

                W.write(u.atoms)

rule wrap_molecule2_COM:
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
        side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2)
        write_object(wrapped_com_m2, output.com_m2_wrapped)

rule position_assignment_csv:
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

        energy = read_object(input.energy_csv)["Binding energy [kJ/mol]"]
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

rule trajectory_centered_at_m2_COM:
    """
    Write the whole trajectory translated in such a way that the COM of molecule 2 is at (0,0,0) in each frame and
    molecule1 is not written. This is useful so we can later assign the best rotation.
    """
    input:
        trajectory = "{some_folder}trajectory.<ext_trj>",
        structure="{some_folder}structure.<ext_str>",
        index="{some_folder}index.ndx",
    output:
        trajectory="{some_folder}m2_trajectory_centered.<ext_trj>",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n3\n" |  gmx22 trjconv  -s {input.structure}  -f {input.trajectory} -o {output.trajectory} -n {input.index} -center -boxcenter zero
        """

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

        # last_atom = trajectory_universe.atoms
        #
        # trajectory_universe.trajectory[83546]
        # pos = last_atom.positions.copy()
        #
        # N_rotations = int(config["grid"]["N_rotations"])
        #
        # pos_ref = np.array([ts.positions.copy() for ts in reference_universe.trajectory[:N_rotations]])
        #
        # distances = np.empty(len(pos_ref))
        # for i, ref_structure in enumerate(pos_ref):
        #     distances_to_this_ref = np.linalg.norm(pos - ref_structure, axis=1)
        #     print(distances_to_this_ref)
        #     total_distances = distances_to_this_ref.sum(axis=0)
        #     distances[i] = total_distances
        #
        # print(distances)


        N_rotations = int(config["grid"]["N_rotations"])

        #TODO if possible: use gromacs or similar to extract position array faster

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

##################################### FULL ASSIGNMENT ####################################################

checkpoint full_assignment:
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
        print(full_assignments[27], rotation_assignments[27], translation_assignments[27])

        write_object(full_assignments, output.full_assignment)

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

rule count_assignments:
    input:
        translation_assignment=f"<outputs_assignment>translation_assignment.npy",
        rot_assignment=f"<outputs_assignment>rotation_assignment.npy",
        full_assignment=f"<outputs_assignment>full_assignment.npy",
    output:
        translation_assignment = f"<outputs_other_plots>hist_translation_assignment.png",
        rot_assignment = f"<outputs_other_plots>hist_rotation_assignment.png",
        full_assignment = f"<outputs_other_plots>hist_full_assignment.png",
    run:
        import plotly.graph_objects as go
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

