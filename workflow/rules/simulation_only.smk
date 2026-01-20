import numpy as np
import pandas as pd

rule gromacs_equilibration:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure=f"<outputs_gromacs>structure.<ext_str>",
        runfile_minim=f"<outputs_gromacs>minim.mdp",
        runfile_nvt=f"<outputs_gromacs>nvt.mdp",
        index=f"<outputs_gromacs>index.ndx",
        topology=f"<outputs_gromacs>topol.top",
        force_field_stuff=f"<outputs_gromacs>force_field_stuff/"
    shadow: "minimal"
    output:
        energy=f"<outputs_gromacs>nvt.gro",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile_minim})
        echo $(pwd)
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile_minim}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o em.tpr -r $(basename {input.structure})
        gmx22 mdrun -v -deffnm em

        gmx22 grompp -f $(basename {input.runfile_nvt}) -c em.gro -r em.gro -p $(basename {input.topology}) -o nvt.tpr -n $(basename {input.index})
        gmx22 mdrun -v -deffnm nvt

        cd "$initial_dir" || exit
        """


rule gromacs_production:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure=f"<outputs_gromacs>nvt.<ext_str>",
        runfile=f"<outputs_gromacs>production.mdp",
        topology=f"<outputs_gromacs>topol.top",
        select_energy=f"<outputs_gromacs>select_energy",
        force_field_stuff=f"<outputs_gromacs>force_field_stuff/",
        index=f"<outputs_gromacs>index.ndx",
    log:
        log=f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
    shadow: "minimal"
    output:
        structure_tpr=f"<outputs_gromacs>structure.tpr",
        energy=f"<outputs_gromacs>energy.xvg",
        original_trajectory=f"<outputs_gromacs>raw_trajectory.<ext_trj>"
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -r $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index})
        gmx22 mdrun -v -deffnm structure -g $(basename {log.log})
        mv structure.xtc $(basename {output.original_trajectory})
        gmx22 energy -f structure.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """


rule postprocess_gromacs:
    input:
        original_trajectory = f"<outputs_gromacs>raw_trajectory.<ext_trj>",
        structure_tpr=f"<outputs_gromacs>structure.tpr",
        index=f"<outputs_gromacs>index.ndx",
    log:
        log=f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
    shadow: "minimal"
    output:
        trajectory=f"<outputs_gromacs>trajectory.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.original_trajectory})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        # now fit to first frame
        echo "2\n0\n" |  gmx22 trjconv -f $(basename {input.original_trajectory}) -s  $(basename {input.structure_tpr}) -pbc mol -center -o centered_trajectory.xtc -n $(basename {input.index})
        echo "2\n0\n" |  gmx22 trjconv -fit rot+trans -f centered_trajectory.xtc -o $(basename {output.trajectory}) -s  $(basename {input.structure_tpr}) -n $(basename {input.index})
        cd "$initial_dir" || exit
        """



rule wrap_molecule2_COM:
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure1 = f"<outputs_gromacs>molecule1.<ext_str>",
    output:
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
    run:
        from workflow.helpers.io import get_atomgoup_m1, get_atomgoup_m2, write_object
        from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths, wrap_to_cuboid_cell
        from MDAnalysis import Universe

        # determine com of 2nd molecule
        u = Universe(input.structure, input.trajectory)
        ag_m2 = get_atomgoup_m2(u, input.structure1)
        ag_m1 = get_atomgoup_m1(u, input.structure1)

        com_array_m1 = np.zeros((len(u.trajectory), 3))
        com_array_m2 = np.zeros((len(u.trajectory), 3))
        for i, ts in enumerate(u.trajectory):
            shift = ag_m1.center_of_mass()
            u.atoms.translate(-shift)
            com_array_m1[i] = ag_m1.center_of_mass()
            com_array_m2[i] = ag_m2.center_of_mass()

        write_object(com_array_m2, output.com_m2)
        # assert com of m1 not changing
        assert np.max(com_array_m1 - com_array_m1[0]) < 0.01, "Molecule 1 seems to be moving - is it not fitted to the reference or just very flexible?"


        # determine cuboid cell
        side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2)
        write_object(wrapped_com_m2, output.com_m2_wrapped)


# rule assign_com2position_grid:
#     input:
#         com_m2 = f"<outputs_assignment>m2_com.npy",
#         grid_info = "<outputs_network>grid_info.yaml"
#     output:
#         position_assignment = f"<outputs_assignment>position_assignment.npy",
#     run:
#         from molgri.molecules.assignment import assign_to_cartesian_translation_grid
#         from workflow.helpers.io import read_object, write_object
#         com_m2 = read_object(input.com_m2)
#
#         grid_info = read_object(input.grid_info)
#         subgrids = grid_info["subgrid_points"]
#         subgrid_limits = grid_info["subgrid_limits_A"]
#         periodic_in = grid_info["periodic_in"]
#         N_gridpoints = grid_info["subgrid_N_points"]
#         full_assignments = assign_to_cartesian_translation_grid(com_m2, subgrids, subgrid_limits, periodic_in)
#
#         write_object(full_assignments, output.position_assignment)

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

            u = Universe(input.structure1)
            plot_graphene(u, fig, in_3d=False)

            if len(position_gridpoints) >= 1:
                x, y, z = position_gridpoints.T
                fig.add_trace(go.Scatter(x=position_gridpoints.T[0], y=position_gridpoints.T[1],
                    mode = "markers", opacity=0.7,
                    marker=dict(color=filtered_df["Energy [kJ/mol]"], size=10,
                        colorbar=dict(thickness=20, title="Energy [kJ/mol]", y=1.05,yanchor="top"),
                        colorscale="RdBu_r", cmin=-61,cmax=-50)))
            draw_unit_cell(fig, side_lengths, in_3d=False)
            fig.write_image(output.plot[z_index])

