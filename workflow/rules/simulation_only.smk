import numpy as np

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
        original_trajectory=f"<outputs_gromacs>raw_trajectory.xtc",
        trajectory=f"<outputs_gromacs>trajectory.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o raw_trajectory.tpr -n $(basename {input.index})
        gmx22 mdrun -v -deffnm raw_trajectory -g $(basename {log.log})
        gmx22 energy -f raw_trajectory.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        # now fit to first frame
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index}) 
        echo "2\n0\n" |  gmx22 trjconv -f raw_trajectory.xtc -s  $(basename {output.structure_tpr}) -pbc mol -center -o centered_trajectory.xtc -n $(basename {input.index})
        echo "2\n0\n" |  gmx22 trjconv -fit rot+trans -f centered_trajectory.xtc -o $(basename {output.trajectory}) -s  $(basename {output.structure_tpr}) -n $(basename {input.index})
        cd "$initial_dir" || exit
        """

rule wrap_molecule2_COM:
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure1=f"<outputs_gromacs>molecule1.<ext_str>",
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


rule assign_com2position_grid:
    input:
        structure1 = f"<outputs_gromacs>molecule1.<ext_str>",
        com_m2 = f"<outputs_assignment>m2_com.npy",
        df_indices = f"<outputs_network>indices_interpretation.csv",
        grid= "<outputs_network>grid.npy",
    output:
        position_assignment = f"<outputs_assignment>position_assignment.npy",
    run:
        from workflow.helpers.io import read_object, write_object
        from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths
        import matplotlib.pyplot as plt

        com_m2 = read_object(input.com_m2)
        position_grid = read_object(input.grid)[:, :3]
        x_grid = np.unique(position_grid[:, 0].T)
        x_grid.sort()



        delta_x = x_grid[1] - x_grid[0]
        N = len(x_grid)

        Lx, Ly, Lz = get_rectangular_cell_side_lengths(input.structure1)

        fig, ax = plt.subplots(figsize=(5, 2))
        #ax.scatter(x_grid, np.ones(x_grid.shape))
        #plt.xlim(0, Lx)


        x_trajectory = com_m2[:, 0].T
        wrapped_x_trajectory = x_trajectory % Lx
        x_indices = np.floor(wrapped_x_trajectory  / delta_x + 0.5) % N
        write_object(x_indices, output.position_assignment)

        for index_num in range(N):
            assigned_to = x_trajectory[np.where(x_indices == index_num)[0]]
            ax.scatter(assigned_to,np.full(assigned_to.shape, 1.0+0.1*index_num), s=1, alpha=0.4)
        ax.vlines([k*Lx for k in range(10, 33)], ymin=1, ymax=2, linestyles="dashed", color="black")
        ax.vlines([k * Lx + delta_x for k in range(10,33)],ymin=1,ymax=2,linestyles="dotted",color="black", lw=1)
        ax.vlines([k * Lx + 9*delta_x for k in range(10,33)],ymin=1,ymax=2,linestyles="dotted",color="black",lw=1)
        ax.set_xlim(50,55)
        plt.savefig("outputs/image.png", dpi=600)


        #print(np.min(position_grid,axis=0))

        # determine cuboid cell
        # side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        # origin = np.zeros(3)
        #
        # # wrap
        # wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2)
        # write_object(wrapped_com_m2, output.com_m2_wrapped)