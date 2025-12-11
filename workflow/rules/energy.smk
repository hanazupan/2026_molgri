
from workflow.helpers.io import read_object, write_object
from workflow.helpers.PATHS import NAME_NETWORK_FOLDER, NAME_ENERGY_FOLDER, NAME_PLOTS, NAME_LOWEST_E_FOLDER, \
    NAME_FRAME_PLOTS, NAME_PT_FOLDER
from molgri.utils.arrays import k_argmin_in_array

MOLECULE_NAMES = f"{config['pseudotrajectory']['molecule_1']}_{config['pseudotrajectory']['molecule_2']}/"
ENERGY_TYPE = config['plotting']['energy_type_name']
NUM_STRUCTURES = config['plotting']['plot_N_lowest']
STRUCTURE_ENDING = config["pseudotrajectory"]["structure_ending"]

checkpoint create_energy_csv:
    input:
        energy=f"{{some_path}}{NAME_ENERGY_FOLDER}energy.xvg",
        m1_energy=f"{{some_path}}molecule1/energy.xvg",
        m2_energy=f"{{some_path}}molecule2/energy.xvg",
        network= f"{{some_path}}{NAME_NETWORK_FOLDER}network.pkl"
    output:
        energy_csv = f"{{some_path}}{NAME_ENERGY_FOLDER}energy.csv"
    run:
        import pandas as pd
        import numpy as np

        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy[ENERGY_TYPE].to_numpy()

        energy_m1 = read_object(input.m1_energy)[ENERGY_TYPE][0]
        energy_m2 = read_object(input.m2_energy)[ENERGY_TYPE][0]

        translation_indices = my_network.get_translation_indices()
        rotation_indices = my_network.get_rotation_indices()
        coordinates = my_network.grid
        positions = coordinates[:, :3]
        quaternions = coordinates[:, 3:]

        df = pd.DataFrame(np.array([translation_indices, rotation_indices, my_energy_array]).T,
            columns=["Translation index", "Rotation index", "Energy [kJ/mol]"])

        df["Binding energy [kJ/mol]"] = df["Energy [kJ/mol]"] - energy_m1 - energy_m2
        df["Position"] = list(positions)
        df["Quaternion"] = list(quaternions)
        df.index.name = "Total index"

        write_object(df, output.energy_csv)

        df = df.sort_values(by="Binding energy [kJ/mol]",ascending=True)
        print(df.head(300)["Rotation index"])


rule read_in_energies:
    input:
        network = f"{{some_path}}{NAME_NETWORK_FOLDER}network.pkl",
        energy = f"{{some_path}}{NAME_ENERGY_FOLDER}energy.xvg",
        m1_energy = f"{{some_path}}molecule1/energy.xvg",
        m2_energy= f"{{some_path}}molecule2/energy.xvg",
    output:
        network_energy = f"{{some_path}}{NAME_ENERGY_FOLDER}network_energy.pkl"
    run:

        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy[ENERGY_TYPE].to_numpy()

        energy_m1 = read_object(input.m1_energy)[ENERGY_TYPE][0]
        energy_m2 = read_object(input.m2_energy)[ENERGY_TYPE][0]

        my_network.add_node_properties(my_energy_array,"energy")
        my_network.add_node_properties(my_energy_array-energy_m1-energy_m2,"binding_energy")

        write_object(my_network, output.network_energy)

rule lowest_E_indices:
    input:
        energy_csv = f"{{some_path}}{NAME_ENERGY_FOLDER}energy.csv"
    output:
        indices= f"{{some_path}}{NAME_LOWEST_E_FOLDER}lowest_{{N}}.txt"
    run:
        df_energy = read_object(input.energy_csv)
        required_indices = np.array(df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]").index)
        write_object(required_indices, output.indices)


def get_lowest_energy_frames(wildcards):
    df_energy = read_object(checkpoints.create_energy_csv.get(some_path=wildcards.some_path).output.energy_csv)
    required_indices = df_energy.nsmallest(NUM_STRUCTURES, "Binding energy [kJ/mol]").index
    requested_outputs = [f"{wildcards.some_path}{NAME_FRAME_PLOTS}frame_{i}_view{wildcards.view_i}.tga" for i in required_indices]
    return requested_outputs

rule join_plots_lowestE:
    input:
        get_lowest_energy_frames
    output:
        joint_plot = f"{{some_path}}{NAME_LOWEST_E_FOLDER}all_view{{view_i}}.png"
    run:
        from molgri.images.modifying_images import trim_images_with_common_bbox, join_images
        modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
        trim_images_with_common_bbox(input,modified_paths)
        join_images(modified_paths, output.joint_plot, flip=False)

# rule join_plots_lowestE_from_simulation:
#     input:
#         get_lowest_energy_frames_from_simulation
#     output:
#         joint_plot = f"{{some_path}}simulation/all_view{{view_i}}.png"
#     run:
#         from molgri.images.modifying_images import trim_images_with_common_bbox, join_images
#         modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
#         trim_images_with_common_bbox(input,modified_paths)
#         join_images(modified_paths, output.joint_plot, flip=False)

rule violin_plot_E_distributions:
    input:
        network_energy = f"{{some_path}}{NAME_ENERGY_FOLDER}network_energy.pkl"
    output:
        violin_plot = f"{{some_path}}{NAME_PLOTS}violin_plot_energies.png"
    run:
        from molgri.plotting import show_violin
        my_network = read_object(input.network_energy)

        max_energy = config['plotting']['upper_E_limit']
        energies = my_network.get_node_properties("binding_energy")

        show_violin(energies, max_energy, "Binding Energy", save_as=output.violin_plot, show=False)


def get_ring_carbons(u):
    ring_carbons = u.select_atoms(f"type C")
    has_ring_coordinates = np.where(np.isin(ring_carbons.indices, [1056, 1058, 1060, 1061, 1063, 1065]))[0]
    ring_carbons = ring_carbons[has_ring_coordinates ]
    return ring_carbons




rule get_xylene_ring_normal:
    input:
        structure=f"{{some_path}}{NAME_PT_FOLDER}structure.gro",
        trajectory=f"{{some_path}}{NAME_PT_FOLDER}trajectory.xtc",
        structure1=f"{{some_path}}{NAME_PT_FOLDER}molecule1.gro",
        structure2=f"{{some_path}}{NAME_PT_FOLDER}molecule2.gro",
        energy_csv=f"{{some_path}}{NAME_ENERGY_FOLDER}energy.csv"
    output:
        plot=f"{{some_path}}{NAME_PLOTS}ring_vector.png"
    run:
        from molgri.utils.spheres import angle_between_vectors
        from molgri.plotting import draw_points
        from MDAnalysis import Universe
        import plotly.graph_objects as go

        from workflow.helpers.io import get_num_atoms


        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)
        u = Universe(input.structure,input.trajectory)

        first_molecule = u.select_atoms(f"type C")
        first_molecule = first_molecule[first_molecule.indices < n1]
        second_molecule = u.select_atoms(f"type C")
        second_molecule = second_molecule[second_molecule.indices >= n1]
        from MDAnalysis.topology.guessers import guess_bonds
        bonds = guess_bonds(first_molecule, first_molecule.atoms.positions)
        u.add_TopologyAttr('bonds',bonds)

        df = read_object(input.energy_csv)
        all_energies = df["Binding energy [kJ/mol]"]

        max_energy = config['plotting']['upper_E_limit']
        colors = df["Binding energy [kJ/mol]"]
        colors[colors > max_energy] = max_energy


        ring_carbons = get_ring_carbons(u)


        fig=go.Figure()

        def get_angle(positions):
            positions = positions - np.mean(positions,axis=0,keepdims=True)
            U, S, Vh = np.linalg.svd(positions)
            normal = Vh[:, -1]
            normal = normal / np.linalg.norm(normal)
            return angle_between_vectors(np.array([0,0,1]), normal)

        all_angles = []
        for ts in zip(u.trajectory):
            ring_carbons = get_ring_carbons(u)
            positions = ring_carbons.positions
            all_angles.append(np.rad2deg(get_angle(positions)))

        print(f"Smallest available angle is {np.min(all_angles)} with energy {all_energies[np.argmin(all_angles)]}")
        print(f"Smallest energy is {np.min(all_energies)} with angle {all_angles[np.argmin(all_energies)]}")

        fig.add_trace(go.Scatter(
            x=all_angles,
            y=colors,
            mode='markers',# scatter points only
            marker=dict(
                size=8,# marker size
                color='black',
            ),
        ))

        #fig.add_shape(type="line",x0=-0.5, x1=0.5,y0=max_energy, y1=max_energy,line=dict(color="red", dash="dash"))
        #fig.add_annotation(x=0,y=max_energy, text=f"cutoff = {max_energy}", showarrow=False, yshift=10, font=dict(color="red"))

        # Add axis labels
        fig.update_layout(
            xaxis_title="Angle to z-axis",
            yaxis_title="Binding energy [kJ/mol]",
            template="plotly_white",
            xaxis=dict(range=[0, 180], tickvals=[0, 30, 60, 90, 120, 150, 180])
        )
        fig.update_layout(
            font=dict(
                size=18  # adjust to your preference
            )
        )
        fig.write_image(output.plot)


rule plot_xylene_ring_center:
    input:
        structure=f"{{some_path}}{NAME_PT_FOLDER}structure.gro",
        trajectory=f"{{some_path}}{NAME_PT_FOLDER}trajectory.xtc",
        structure1=f"{{some_path}}{NAME_PT_FOLDER}molecule1.gro",
        structure2=f"{{some_path}}{NAME_PT_FOLDER}molecule2.gro",
        energy_csv= f"{{some_path}}{NAME_ENERGY_FOLDER}energy.csv"
    output:
        plot=f"{{some_path}}{NAME_PLOTS}ring_position_lowest_{{N}}.png"
    run:
        from molgri.plotting import draw_points
        from MDAnalysis import Universe
        import plotly.graph_objects as go

        from workflow.helpers.io import get_num_atoms
        molecule1 = read_object(input.structure1)
        molecule2 = read_object(input.structure2)

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)

        u = Universe(input.structure, input.trajectory)
        first_molecule = u.select_atoms(f"type C")
        first_molecule = first_molecule[first_molecule.indices < n1]

        from MDAnalysis.topology.guessers import guess_bonds
        bonds = guess_bonds(first_molecule, first_molecule.atoms.positions)
        u.add_TopologyAttr('bonds',bonds)
        print(u.bonds)


        fig = go.Figure()


        ring_carbons = get_ring_carbons(u)


        df = read_object(input.energy_csv)

        result = (
            df.loc[df.groupby("Translation index")["Binding energy [kJ/mol]"].idxmin()]
            .nsmallest(int(wildcards.N),"Binding energy [kJ/mol]")
        )
        selected_indices = result.index.to_numpy()


        # todo: plot positions of ring at min energy
        max_energy = config['plotting']['upper_E_limit']
        colors = df["Binding energy [kJ/mol]"]
        colors[colors > max_energy] = max_energy

        collect_points = []
        for i, ts in zip(selected_indices, u.trajectory[selected_indices]):
            collect_points.append(ring_carbons.center_of_geometry())
        collect_points = np.array(collect_points)
        fig.add_trace(go.Scatter3d(x=collect_points.T[0], y=collect_points.T[1], z=collect_points.T[2], mode = "markers",
            marker=dict(color=colors[selected_indices],
                colorbar=dict(thickness=20, title="Energy [kJ/mol]", y=1.05,yanchor="top"),
                colorscale="RdBu_r", cmin=-52,cmax=-50),))

        fig.update_layout(
            scene_camera=dict(eye=dict(x=0,y=0,z=0.5))
        )

        positions = first_molecule.atoms.positions
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Create bond line segments
        bond_traces = []
        for bond in u.bonds:
            i, j = bond.atoms.indices
            bond_traces.append(
                go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode='lines',
                    line=dict(color='gray',width=3),
                    hoverinfo='skip'
                )
            )

        for bt in bond_traces:
            fig.add_trace(bt)
        fig.update_layout(showlegend=False)
        fig = draw_points(first_molecule.atoms.positions, fig=fig, show=False, save_as=output.plot, equal_aspect=True)