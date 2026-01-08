

def get_ring_carbons(u):
    ring_carbons = u.select_atoms(f"type C")
    has_ring_coordinates = np.where(np.isin(ring_carbons.indices, [1056, 1058, 1060, 1061, 1063, 1065]))[0]
    ring_carbons = ring_carbons[has_ring_coordinates ]
    return ring_carbons


"""

0 gmx22 trjconv -f trajectory.xtc -s structure.tpr -o pbc_trajectory1.xtc -pbc mol
2 0 gmx22 trjconv -f pbc_trajectory1.xtc -s structure.tpr -o pbc_trajectory2.xtc -center


2 2 0 gmx22 trjconv -f pbc_trajectory1.xtc -s structure.tpr -o pbc_trajectory2.xtc -fit rot+trans -center



        
"""



rule get_xylene_ring_normal:
    input:
        structure=f"<outputs_gromacs>structure.gro",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
        structure1=f"<outputs_gromacs>molecule1.gro",
        structure2=f"<outputs_gromacs>molecule2.gro",
        energy_csv=f"<outputs>energy.csv"
    output:
        plot=f"<outputs_other_plots>ring_vector.png"
    run:
        from molgri.utils.spheres import angle_between_vectors
        from molgri.plotting import draw_points
        from MDAnalysis import Universe
        import plotly.graph_objects as go
        import numpy as np

        from workflow.helpers.io import get_num_atoms, read_object


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
        structure=f"<outputs_gromacs>structure.gro",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
        structure1=f"<outputs_gromacs>molecule1.gro",
        structure2=f"<outputs_gromacs>molecule2.gro",
        indices_csv = "<outputs_network>indices_interpretation.csv",
        energy_csv= f"<outputs>energy.csv"
    output:
        plot=f"<outputs_other_plots>ring_position_lowest_{{N}}.png"
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


        df_energy = read_object(input.energy_csv)
        df_index = read_object(input.indices_csv)
        df_combined = df_index.join(df_energy)


        result = (
            df_combined.loc[df_combined.groupby("Translation index")["Binding energy [kJ/mol]"].idxmin()]
            .nsmallest(int(wildcards.N),"Binding energy [kJ/mol]")
        )
        selected_indices = result.index.to_numpy()


        # todo: plot positions of ring at min energy
        max_energy = config['plotting']['upper_E_limit']
        colors = df_combined["Binding energy [kJ/mol]"]
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