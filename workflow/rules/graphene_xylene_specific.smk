

def get_ring_carbons(u):
    ring_carbons = u.select_atoms(f"type C")
    has_ring_coordinates = np.where(np.isin(ring_carbons.indices, [1056, 1058, 1060, 1061, 1063, 1065]))[0]
    ring_carbons = ring_carbons[has_ring_coordinates ]
    return ring_carbons


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


rule wrap_ring_center:
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure1=f"<outputs_gromacs>molecule1.<ext_str>",
    output:
        com_m2 = f"<outputs_assignment>ring_center.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_ring_center.npy",
    run:
        from workflow.helpers.io import write_object
        from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths, wrap_to_cuboid_cell
        from MDAnalysis import Universe

        # determine com of 2nd molecule
        u = Universe(input.structure, input.trajectory)
        ag_ring = get_ring_carbons(u)

        com_ring = np.zeros((len(u.trajectory), 3))
        for i, ts in enumerate(u.trajectory):
            com_ring[i] = ag_ring.center_of_mass()
        write_object(com_ring, output.com_m2)

        # determine cuboid cell
        side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_ring)
        write_object(wrapped_com_m2, output.com_m2_wrapped)


rule plot_xylene_ring_center:
    input:
        structure1=f"<outputs_gromacs>molecule1.gro",
        energy_csv= f"<outputs>energy.csv",
        #com_m2= f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped= f"<outputs_assignment>cuboid_wrapped_ring_center.npy",
    output:
        plot=f"<outputs_other_plots>ring_position_lowest_{{N}}.png"
    run:
        from molgri.plotting import draw_points, draw_line_between
        from MDAnalysis import Universe
        import plotly.graph_objects as go
        from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths
        my_sides = get_rectangular_cell_side_lengths(input.structure1)
        my_sides[-1] = 0.1
        start_at = np.array([0,0, 0])

        u = Universe(input.structure1)

        from MDAnalysis.topology.guessers import guess_bonds
        bonds = guess_bonds(u.select_atoms("all"), u.atoms.positions)
        u.add_TopologyAttr('bonds',bonds)

        # plot bonds
        fig = go.Figure()
        positions = u.atoms.positions
        x, y = positions[:, 0], positions[:, 1]
        ring_carbons = get_ring_carbons(u)
        bond_traces = []
        for bond in u.bonds:
            i, j = bond.atoms.indices
            bond_traces.append(
                go.Scatter(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    mode='lines+markers',
                    line=dict(color='gray',width=3),
                    hoverinfo='skip'
                )
            )

        for bt in bond_traces:
            fig.add_trace(bt)
        fig.update_layout(showlegend=False)

        # plot COMs
        coms = read_object(input.com_m2_wrapped)
        df_energy = read_object(input.energy_csv)
        rows_smallest_energy = rows = df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]")
        selected_indices = rows_smallest_energy.index.to_numpy()
        selected_coms = coms[selected_indices]

        colors = df_energy["Binding energy [kJ/mol]"]
        inverse_coms = selected_coms[::-1]
        inverse_indices = selected_indices[::-1]
        fig.add_trace(go.Scatter(x=inverse_coms.T[0], y=inverse_coms.T[1], mode = "markers", opacity=0.7,
            marker=dict(color=colors[inverse_indices], size=10,
                colorbar=dict(thickness=20, title="Energy [kJ/mol]", y=1.05,yanchor="top"),
                colorscale="RdBu_r", cmin=-61,cmax=-59)))

        Lx, Ly, Lz = my_sides
        draw_line_between(fig, np.array([0, 0]), np.array([Lx, 0]), color="green")
        draw_line_between(fig, np.array([0, 0]), np.array([0, Ly]), color="green")
        draw_line_between(fig,np.array([Lx, Ly]),np.array([Lx, 0]),color="green")
        draw_line_between(fig,np.array([Lx, Ly]),np.array([0, Ly]),color="green")
        fig.update_xaxes(showgrid=True, range=[-Lx/2, 3*Lx/2])
        fig.update_yaxes(showgrid=True, range=[-Ly/2, 3*Ly/2])
        fig.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1
            )
        )
        fig.write_image(output.plot)